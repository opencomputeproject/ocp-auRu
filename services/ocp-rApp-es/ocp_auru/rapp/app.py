# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 UGECO Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""es-rApp HTTP service (FastAPI) — HLD §3.7.

Routes: `POST /r1/pm-delivery` (ICS job delivery of O1 PM, VES measurement envelope),
`POST /cycle` (one decision cycle, optional forced action for demos), `GET /health`,
`GET /policy`, `GET /decisions`, `GET /pm`, `GET /metrics` (incl. `twin_checks`, `twin_blocks`,
`last_twin_verdict`, `forecast_divergent_decisions`, and the per-carrier-day actuation budget: `actuations_today`
per carrier, `actuation_alarm` / `actuation_budget_reached` carrier counts, cumulative carrier-days and refusals;
`RAPP_MAX_ACTUATIONS_PER_DAY` / `RAPP_ACTUATION_ALARM_PER_DAY`). Every decision record carries both forecast
estimators at the same instant (`p_low_pm_window`, `p_low_profile`, `forecast_divergence`). Every SLEEP is put to the physics twin first (`twin.py`, ADR-7) and the
verdict is recorded on the decision record (`/decisions[i].twin`). A background thread runs the
cycle every `RAPP_CYCLE_S` when `RAPP_AUTO_CYCLE` is on. `create_app()` accepts injected
model/R1/twin objects so tests run without the platform.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ocp_auru.common.config import Settings, settings
from ocp_auru.common.events import (Active, CarrierPm, CarrierState, CarrierStatus, CmCarrierRequest, RuAppPolicy,
                                VesEnvelope)

from .data import load as load_data
from .features import pm_to_kpi_row
from .model import FORECAST_DIVERGENCE_FLAG, LowLoadModel, load_or_train
from .policy_engine import Decision, PolicyState, actuations_today, count_actuation, decide, roll_actuation_day
from .r1 import R1Client
from .twin import TwinGuard

log = logging.getLogger(__name__)


class CycleRequest(BaseModel):
    force: Optional[Literal["SLEEP", "ACTIVE", "INACTIVE"]] = None
    carrier: Optional[str] = None


class RappState:
    """Everything the rApp remembers: latest PM + short history per carrier, decisions, CM results."""

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.lock = threading.RLock()
        self.ru_id: str = cfg.ru_id
        self.pm_latest: dict[str, dict[str, Any]] = {}
        self.pm_history: dict[str, deque] = {}
        self.wake_stats: dict[str, dict[str, Any]] = {}
        self.policy: dict[str, PolicyState] = {}
        self.p_low: dict[str, float] = {}
        self.last_cm: dict[str, dict[str, Any]] = {}
        self.decisions: list[dict[str, Any]] = []
        self.last_cm_rtt_ms: Optional[float] = None
        self.last_netconf_rtt_ms: Optional[float] = None
        self.last_sim_time: Optional[str] = None
        self.sleep_windows = 0
        self.pm_reports = 0
        self.notifications = 0
        self.cycles = 0
        self.divergent_decisions = 0   # decisions where |p_low(pm-window) - p_low(profile)| > FORECAST_DIVERGENCE_FLAG
        # actuation budget, cumulative since start: decisions the budget refused, carrier-days that crossed the
        # alarm threshold and carrier-days that reached the budget (each counted once, when the WARNING is logged)
        self.budget_refusals = 0
        self.alarm_carrier_days = 0
        self.budget_carrier_days = 0
        self._inflight: set[str] = set()  # carriers with a CM request in progress (lock released meanwhile)

    # ---- PM ingest (R1 delivery) -----------------------------------------------------------
    def ingest(self, env: VesEnvelope) -> dict[str, Any]:
        ev = env.event
        hdr = ev.commonEventHeader
        with self.lock:
            if hdr.domain == "measurement" and ev.measurementFields:
                fields = ev.measurementFields
                sim_time = str(fields.get("simTime") or time.strftime("%Y-%m-%dT%H:%M:%S"))
                self.last_sim_time = sim_time
                self.ru_id = hdr.sourceName or self.ru_id
                n = 0
                for obj in fields.get("additionalObjects") or []:
                    try:
                        pm = CarrierPm.model_validate(obj)
                    except Exception as exc:
                        log.warning("skipping malformed carrier PM object: %s", exc)
                        continue
                    self._record_pm(pm, sim_time)
                    n += 1
                self.pm_reports += 1
                return {"accepted": True, "domain": "measurement", "carriers": n, "sim_time": sim_time}
            if hdr.domain == "notification":
                self.notifications += 1
                self._apply_notification(ev.notificationFields or {})
                return {"accepted": True, "domain": "notification"}
            return {"accepted": True, "domain": hdr.domain, "ignored": True}

    def _record_pm(self, pm: CarrierPm, sim_time: str) -> None:
        row = pm.model_dump(mode="json")
        row["sim_time"] = sim_time
        self.pm_latest[pm.name] = row
        hist = self.pm_history.setdefault(pm.name, deque(maxlen=self.cfg.rapp_history_len))
        # the row carries the carrier's `active` so the profile level ratio (model.py) skips asleep windows;
        # the feature builder ignores the extra key
        hist.append({**pm_to_kpi_row(pm, sim_time), "active": pm.active.value})
        self.wake_stats[pm.name] = {"wake_events": pm.wake_events, "prach_detections": pm.prach_detections,
                                    "wake_latency_us": pm.wake_latency_us.model_dump()}
        self.policy.setdefault(pm.name, PolicyState())

    def _apply_notification(self, fields: dict[str, Any]) -> None:
        add = fields.get("additionalFields") or {}
        carrier = add.get("carrier")
        if carrier in self.pm_latest:
            for key in ("active", "state"):
                if add.get(key):
                    self.pm_latest[carrier][key] = add[key]

    # ---- decision cycle -------------------------------------------------------------------
    # Locking: the state lock is held only to *plan* (read PM, predict, decide) and to *commit*
    # results; the R1 CM request and the twin call run with the lock released, so PM delivery
    # and the read routes never wait on a slow SMO/RU hop (RAPP_CM_TIMEOUT_S = 30 s).
    # A per-carrier in-flight set keeps the auto-cycle and POST /cycle from actuating the same
    # carrier concurrently.
    def run_cycle(self, model: LowLoadModel, r1: R1Client, twin: TwinGuard, *, force: Optional[str] = None,
                  carrier: Optional[str] = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        plans: list[tuple[str, dict[str, Any], Decision]] = []
        with self.lock:
            self.cycles += 1
            names = [carrier] if carrier else list(self.pm_latest)
            for name in names:
                pm = self.pm_latest.get(name)
                if pm is None:
                    out.append({"carrier": name, "error": "no PM seen for carrier"})
                    continue
                if name in self._inflight:
                    out.append({"carrier": name, "error": "actuation in flight for carrier (previous cycle); skipped"})
                    continue
                record, d = self._plan(name, pm, model, force)
                out.append(record)
                if d.action in ("SLEEP", "ACTIVE", "INACTIVE"):
                    self._inflight.add(name)
                    plans.append((name, record, d))
                else:
                    self._finish(name, record, d)
        for name, record, d in plans:
            try:
                self._actuate(name, record, d, r1, twin)
            finally:
                with self.lock:
                    self._inflight.discard(name)
        return out

    def _observe(self, name: str, state: PolicyState, status: CarrierStatus, sim_now: Optional[str]) -> None:
        """Caller holds the lock. Reconcile the chatter-control clocks with what PM says BEFORE deciding:
        a carrier that is not asleep has no running sleep-dwell clock (a rejected/failed SLEEP, a ruApp
        local wake, an operator write: the clock is cleared, never measured from a stale stamp), and a
        SLEEP -> ACTIVE transition the rApp did not write (or a grown `wake_events` counter) is a ruApp
        LOCAL wake, stamped in `local_wake_sim` for the post-local-wake hold. The actuation counter rolls over
        when the sim day of the PM moved on (so /metrics reads the current day's count, not yesterday's)."""
        if roll_actuation_day(state, sim_now) and state.last_action is not None:
            log.info("%s: sim day %s: actuation counter reset (budget %d, alarm %d per carrier-day)", name, state.actuations_day,
                     self.cfg.rapp_max_actuations_per_day, self.cfg.rapp_actuation_alarm_per_day)
        wake_events = self.wake_stats.get(name, {}).get("wake_events")
        if status.active == Active.ACTIVE:
            grew = (wake_events is not None and state.last_wake_events is not None and int(wake_events) > int(state.last_wake_events))
            transition = state.last_seen_active == Active.SLEEP.value and state.last_action != "ACTIVE"
            if grew or transition:
                state.local_wake_sim = sim_now
                log.info("%s: ruApp local wake observed in PM (wake_events %s -> %s) at sim %s", name, state.last_wake_events,
                         wake_events, sim_now)
        if status.active != Active.SLEEP and state.sleep_entered_sim is not None:
            log.info("%s: carrier %s while the dwell clock ran (%s); clearing it", name, status.active.value, state.sleep_entered_sim)
            state.sleep_entered_sim = None
        state.last_seen_active = status.active.value
        state.last_wake_events = None if wake_events is None else int(wake_events)

    def _plan(self, name: str, pm: dict[str, Any], model: LowLoadModel, force: Optional[str]) -> tuple[dict[str, Any], Decision]:
        """Caller holds the lock. Predict + decide for one carrier; no I/O."""
        now = time.time()
        state = self.policy.setdefault(name, PolicyState())
        status = CarrierStatus(name=name, active=Active(pm["active"]), state=CarrierState(pm["state"]), ruapp_policy=state.envelope)
        # sim_now feeds the sleep-dwell clock, the post-local-wake hold and the one-vote-per-window rule
        # (simulated/data time, never wall time); None when no PM has carried a simTime yet, in which
        # case those rules do not apply.
        sim_now_raw = pm.get("sim_time") or self.last_sim_time
        sim_now = str(sim_now_raw) if sim_now_raw else None
        self._observe(name, state, status, sim_now)
        # A SLEEP/INACTIVE carrier reports PRB 0 by construction; forecast from the expected load
        # at this time of day instead (HLD §3.7 pre-emptive wake), see model.py. BOTH estimators are
        # evaluated at the same instant on every decision and recorded side by side (diagnostic: the
        # estate-pooled profile vs the carrier's own PM, the root cause of the chatter); the one the
        # carrier's state calls for is the decision's p_low.
        asleep = status.active != Active.ACTIVE
        basis = "expected-hour-profile" if asleep else "pm-window"
        hist = list(self.pm_history[name])
        p_pm: Optional[float] = None
        p_prof: Optional[float] = None
        prof_info: dict[str, Any] = {}
        try:
            p_pm = float(model.predict_low_load_probability(hist))
        except Exception as exc:
            log.warning("pm-window forecast failed for %s: %s", name, exc)
        if sim_now is not None:
            try:
                p_prof, prof_info = model.asleep_forecast(sim_now, hist, self.cfg.rapp_profile_scale)
            except Exception as exc:
                log.warning("expected-hour-profile forecast failed for %s: %s", name, exc)
        p_low = p_prof if asleep else p_pm
        if p_low is None:
            log.warning("prediction failed for %s (%s); p_low=0", name, basis)
            p_low = 0.0
            basis += "-failed"
        self.p_low[name] = p_low
        d = decide(status, pm, p_low, now, state, low_threshold=model.low_threshold, cfg=self.cfg, sim_now=sim_now)
        # the engine is pure: the vote count and its window live here between cycles
        state.wake_votes, state.last_vote_sim = d.wake_votes, d.vote_sim
        d.sim_time = pm.get("sim_time")
        d.forecast_basis = basis
        d.p_low_pm_window, d.p_low_profile = p_pm, p_prof
        if p_pm is not None and p_prof is not None:
            d.forecast_divergence = p_pm - p_prof
            # flagged on AWAKE decisions only: a sleeping carrier's own-PM estimate reads its zero counters
            # (recorded, not judged); the offline evaluation judges every window from the recorded series
            d.forecast_divergent = (not asleep) and abs(d.forecast_divergence) > FORECAST_DIVERGENCE_FLAG
            if d.forecast_divergent:
                self.divergent_decisions += 1
        d.profile_scale = prof_info.get("profile_scale")
        if not force and d.action == "NONE" and d.reason.startswith("actuation budget"):
            self.budget_refusals += 1     # a decision the budget held (a forced override is not a refusal)
        if force:
            d = d.model_copy(update={"action": force, "forced": True, "reason": f"forced by operator via POST /cycle (policy said {d.action}: {d.reason})"})
            log.warning("FORCED decision %s for %s (demo path)", force, name)
        record: dict[str, Any] = {"decision": d.model_dump(mode="json"), "twin": None, "cm": None}
        return record, d

    def _actuate(self, name: str, record: dict[str, Any], d: Decision, r1: R1Client, twin: TwinGuard) -> None:
        """Lock released: twin guardrail (SLEEP only) and the R1 CM request; then commit under the lock."""
        if d.action == "SLEEP":
            # ADR-7: the RF twin is asked before EVERY SLEEP (natural or forced); its verdict is
            # recorded on the decision. A block stops a policy decision; a forced decision is an
            # explicit operator override of the policy and is let through, recorded as such.
            allowed, detail = twin.evaluate_sleep(name)
            record["twin"] = detail
            if not allowed and not d.forced:
                d = d.model_copy(update={"action": "NONE", "reason": f"twin guardrail blocked SLEEP: coverage_rate delta {detail.get('coverage_rate_delta')} "
                                                                     f"< {detail.get('threshold')} (cell {detail.get('cell_id')}, {detail.get('rung')})"})
                record["decision"] = d.model_dump(mode="json")
                with self.lock:
                    self._finish(name, record, d)
                return
            if not allowed and d.forced:
                detail["overridden"] = True
                d = d.model_copy(update={"reason": d.reason + f" | twin guardrail verdict BLOCKED (coverage_rate delta {detail.get('coverage_rate_delta')}) overridden by the forced decision"})
                record["decision"] = d.model_dump(mode="json")
                log.warning("twin block OVERRIDDEN by forced SLEEP for %s (operator demo path)", name)
        req = CmCarrierRequest(ru_id=self.ru_id, carrier=name, active=Active(d.action), ruapp_policy=d.envelope, reason=d.reason)
        t0 = time.perf_counter()
        resp = None
        try:
            resp = r1.cm_request(req)
            cm = resp.model_dump(mode="json")
            cm["rapp_rtt_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            log.info("CM %s %s -> ok=%s cm_rtt=%.1fms netconf_rtt=%s", name, d.action, resp.ok, resp.cm_rtt_ms, resp.netconf_rtt_ms)
        except Exception as exc:
            cm = {"ok": False, "error": str(exc), "rapp_rtt_ms": round((time.perf_counter() - t0) * 1000.0, 3)}
            log.error("CM request %s %s failed: %s", name, d.action, exc)
        with self.lock:
            state = self.policy.setdefault(name, PolicyState())
            if resp is not None:
                self.last_cm_rtt_ms = resp.cm_rtt_ms
                self.last_netconf_rtt_ms = resp.netconf_rtt_ms
                if resp.ok:
                    if name in self.pm_latest:
                        self.pm_latest[name]["active"] = resp.active.value  # optimistic until the next PM confirms
                    state.envelope = d.envelope
                    if d.action == "SLEEP":
                        self.sleep_windows += 1
            state.last_decision_wall = time.time()
            state.last_action = d.action
            # chatter control, on an APPLIED write only: the votes reset; a SLEEP starts the dwell clock at
            # the window's simulated time (the PM simTime the decision was taken on), an ACTIVE/INACTIVE
            # clears it. A failed/rejected write leaves the clocks alone (a failed SLEEP must not start a
            # dwell on an ACTIVE carrier; a failed ACTIVE keeps its votes so the retry needs no
            # re-confirmation). A lost response on a SLEEP the RU did apply yields one un-dwelled wake at
            # most (service-first); the next PM reconciles the state in `_observe`.
            if resp is not None and resp.ok:
                state.wake_votes, state.last_vote_sim = 0, None
                state.sleep_entered_sim = (d.sim_time or self.last_sim_time) if d.action == "SLEEP" else None
                state.last_seen_active = resp.active.value
                # actuation budget: an APPLIED write counts on the sim day of the window it was decided on
                # (forced ones too); the alarm and the budget are each announced ONCE per carrier-day, on the write
                # that reaches them (the counter only grows within a day; a restart starts a fresh count)
                self._count_actuation(name, state, d.sim_time or self.last_sim_time, d)
            record["cm"] = cm
            self.last_cm[name] = cm
            self._finish(name, record, d)

    def _count_actuation(self, name: str, state: PolicyState, sim_now: Optional[str], d: Decision) -> int:
        """Caller holds the lock. Count one applied actuation for the carrier's sim day; log the operator-visible
        WARNING when the count reaches the alarm threshold and when it reaches the budget (once each per carrier-day)."""
        n = count_actuation(state, sim_now)
        cap, alarm = int(self.cfg.rapp_max_actuations_per_day), int(self.cfg.rapp_actuation_alarm_per_day)
        day = state.actuations_day or "unknown sim day"
        forced = " (forced)" if d.forced else ""
        if alarm > 0 and n == alarm:
            self.alarm_carrier_days += 1
            log.warning("%s: actuation ALARM: %d of %d actuations on sim day %s (alarm threshold %d; budget %s)%s", name, n,
                        cap, day, alarm, cap if cap > 0 else "off", forced)
        if cap > 0 and n == cap:
            self.budget_carrier_days += 1
            log.warning("%s: actuation BUDGET reached: %d of %d actuations on sim day %s; no further rApp SLEEP / pre-emptive "
                        "ACTIVE until the sim day changes, the ruApp remains the wake path%s", name, n, cap, day, forced)
        return n

    def _finish(self, name: str, record: dict[str, Any], d: Decision) -> None:
        """Caller holds the lock: log and append the record to the decision history."""
        log.info("decision %s: %s (p_low=%.2f, %s) %s", name, d.action, d.p_low, d.forecast_basis, d.reason)
        self.decisions.append(record)

    # ---- views ------------------------------------------------------------------------------
    def policy_settings_view(self) -> dict[str, Any]:
        """The policy engine's thresholds and the chatter-control settings (config.py), as applied."""
        cfg = self.cfg
        return {"sleep_threshold": cfg.rapp_sleep_threshold, "wake_threshold": cfg.rapp_wake_threshold,
                "wake_margin": cfg.rapp_wake_margin, "wake_threshold_effective": cfg.rapp_wake_threshold_effective,
                "wake_confirm_windows": cfg.rapp_wake_confirm_windows, "min_sleep_dwell_min": cfg.rapp_min_sleep_dwell_min,
                "min_hold_s": cfg.rapp_min_hold_s, "allow_inactive": cfg.rapp_allow_inactive,
                "profile_scale": cfg.rapp_profile_scale, "post_local_wake_hold_min": cfg.rapp_post_local_wake_hold_min,
                "max_actuations_per_day": cfg.rapp_max_actuations_per_day, "actuation_alarm_per_day": cfg.rapp_actuation_alarm_per_day}

    def policy_view(self, model: LowLoadModel) -> dict[str, Any]:
        with self.lock:
            return {name: {"active": self.pm_latest.get(name, {}).get("active"), "state": self.pm_latest.get(name, {}).get("state"),
                           "envelope": st.envelope.model_dump(), "last_action": st.last_action,
                           "last_decision_wall": st.last_decision_wall, "p_low": self.p_low.get(name),
                           "low_threshold": model.low_threshold, "last_cm": self.last_cm.get(name),
                           # chatter control: the dwell clock, the wake-vote count and its window, the last
                           # observed ruApp local wake, plus the settings
                           "sleep_entered_sim": st.sleep_entered_sim, "wake_votes": st.wake_votes, "last_vote_sim": st.last_vote_sim,
                           "local_wake_sim": st.local_wake_sim,
                           # actuation budget: the sim day the counter belongs to and the applied actuations on it
                           "actuations_day": st.actuations_day, "actuations_today": st.actuations_today,
                           "settings": self.policy_settings_view()} for name, st in self.policy.items()}

    def pm_view(self) -> dict[str, Any]:
        with self.lock:
            return {"ru_id": self.ru_id, "sim_time": self.last_sim_time, "reports": self.pm_reports,
                    "carriers": {n: {"latest": pm, "history_len": len(self.pm_history.get(n, ()))} for n, pm in self.pm_latest.items()}}

    def metrics_view(self) -> dict[str, Any]:
        with self.lock:
            return {"last_cm_rtt_ms": self.last_cm_rtt_ms, "last_netconf_rtt_ms": self.last_netconf_rtt_ms,
                    "wake_latency_us": {n: s["wake_latency_us"] for n, s in self.wake_stats.items()},
                    "wake_events": {n: s["wake_events"] for n, s in self.wake_stats.items()},
                    "decisions_count": len(self.decisions), "sleep_windows": self.sleep_windows,
                    "pm_reports": self.pm_reports, "notifications": self.notifications, "cycles": self.cycles,
                    "forecast_divergent_decisions": self.divergent_decisions,
                    "in_flight": sorted(self._inflight),
                    # actuation budget: per-carrier count on the sim day of the latest PM (as decide() reads it);
                    # carriers currently at/over the alarm threshold and at/over the budget (0 = off for either);
                    # cumulative carrier-days and refusals
                    **self._budget_metrics()}

    def _budget_metrics(self) -> dict[str, Any]:
        """Caller holds the lock. The per-carrier count is read AS THE ENGINE READS IT for the latest PM simTime
        (`policy_engine.actuations_today`: 0 when the stored day is another day), so after sim midnight a carrier
        that reached the budget yesterday is not reported as at the budget until its first cycle rolls the counter;
        `actuations_day` is the raw stored day for the record (the raw pair is on /policy)."""
        cap, alarm = int(self.cfg.rapp_max_actuations_per_day), int(self.cfg.rapp_actuation_alarm_per_day)
        today = {n: actuations_today(st, self.last_sim_time) for n, st in self.policy.items()}
        return {"actuations_today": today,
                "actuations_day": {n: st.actuations_day for n, st in self.policy.items()},
                "actuation_alarm": sum(1 for n in today.values() if alarm > 0 and n >= alarm),
                "actuation_budget_reached": sum(1 for n in today.values() if cap > 0 and n >= cap),
                "actuation_alarm_carrier_days": self.alarm_carrier_days,
                "actuation_budget_carrier_days": self.budget_carrier_days,
                "actuation_budget_refusals": self.budget_refusals,
                "max_actuations_per_day": cap, "actuation_alarm_per_day": alarm}


def create_app(cfg: Optional[Settings] = None, *, model: Optional[LowLoadModel] = None, r1: Optional[R1Client] = None,
               twin: Optional[TwinGuard] = None) -> FastAPI:
    cfg = cfg or settings
    state = RappState(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cfg = cfg
        app.state.rapp = state
        app.state.model = model or load_or_train(lambda: load_data(cfg.rapp_data, cfg.rapp_data_dir, cfg), cfg)
        app.state.r1 = r1 or R1Client(cfg)
        app.state.twin = twin or TwinGuard(cfg)
        app.state.r1.start_background()
        if hasattr(app.state.twin, "start_background"):
            app.state.twin.start_background()   # probe /health + discover the cell; never blocks startup
        stop = threading.Event()
        app.state.stop = stop

        def _auto() -> None:
            while not stop.wait(cfg.rapp_cycle_s):
                try:
                    state.run_cycle(app.state.model, app.state.r1, app.state.twin)
                except Exception as exc:  # keep the loop alive
                    log.error("auto cycle failed: %s", exc)

        if cfg.rapp_auto_cycle:
            threading.Thread(target=_auto, name="rapp-cycle", daemon=True).start()
        log.info("es-rApp ready: profile=%s data=%s auto_cycle=%s cycle=%.0fs", cfg.profile, cfg.rapp_data, cfg.rapp_auto_cycle, cfg.rapp_cycle_s)
        try:
            yield
        finally:
            stop.set()
            app.state.r1.stop()

    app = FastAPI(title="es-rApp (energy saving, Non-RT RIC)", version="0.1.0", lifespan=lifespan)

    def _twin_view() -> dict[str, Any]:
        t = app.state.twin
        return t.health() if hasattr(t, "health") else {"enabled": t.enabled}

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "component": "es-rapp", "profile": cfg.profile, "model": app.state.model.metrics,
                "r1": app.state.r1.status(), "twin": _twin_view(),
                "auto_cycle": cfg.rapp_auto_cycle, "carriers": sorted(state.pm_latest),
                "policy": state.policy_settings_view()}

    @app.post("/r1/pm-delivery")
    def pm_delivery(body: VesEnvelope | list[VesEnvelope]) -> dict[str, Any]:
        envs = body if isinstance(body, list) else [body]
        results = [state.ingest(e) for e in envs]
        return results[0] if len(results) == 1 else {"accepted": True, "count": len(results), "results": results}

    @app.post("/cycle")
    def cycle(req: Optional[CycleRequest] = None) -> dict[str, Any]:
        req = req or CycleRequest()
        if req.force == "INACTIVE" and not cfg.rapp_allow_inactive:
            raise HTTPException(status_code=403, detail="INACTIVE is a hard-off outside the envelope; set RAPP_ALLOW_INACTIVE=1 to permit it")
        if req.carrier and req.carrier not in state.pm_latest:
            raise HTTPException(status_code=404, detail=f"no PM seen for carrier {req.carrier!r}")
        decisions = state.run_cycle(app.state.model, app.state.r1, app.state.twin, force=req.force, carrier=req.carrier)
        return {"forced": req.force, "decisions": decisions}

    @app.get("/policy")
    def policy() -> dict[str, Any]:
        return state.policy_view(app.state.model)

    @app.get("/decisions")
    def decisions() -> list[dict[str, Any]]:
        with state.lock:
            return list(state.decisions)

    @app.get("/pm")
    def pm() -> dict[str, Any]:
        return state.pm_view()

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        t = app.state.twin
        return {**state.metrics_view(), **(t.metrics() if hasattr(t, "metrics") else {})}

    return app
