#!/usr/bin/env python
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
"""The demo narrative (HLD §3.8), driven over the components' HTTP APIs.

Called by `scripts/run_demo.sh` once every process is healthy. Steps:

  1. show the O-RU carrier (ACTIVE/READY)                          - mock O-RU mgmt API
  2. wait for the es-rApp model + R1 registrations + first PM      - R1 job delivery
  3. one rApp cycle -> policy envelope active=SLEEP -> RU DISABLED  - R1 CM -> SMO -> M-Plane (NETCONF) -> O-RU
     (falls back to a forced SLEEP if the model does not decide it; said loudly, recorded)
  3a. the physics twin's verdict on that SLEEP (cell, delta_vs_baseline, rung, allowed/blocked)  - ADR-7
  4. print the management-plane round trip (cm_rtt_ms / netconf_rtt_ms)
  5. inject PRACH energy -> the ruApp wakes the carrier locally (wake_latency_us)
  6. one PM interval later: notification + measurement at the SMO, PM at the rApp, twin ingest
  7. R1 platform view: ICS job list and rApp Catalogue (real NONRTRIC in `full`, SMO routes in `lite`)
  8. write <evidence>/summary.json and print the two measured numbers

Every URL comes from `ocp_auru.common.config.settings` (env overrides respected).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from ocp_auru.common.config import display_path, settings

TIMEOUT = httpx.Timeout(10.0, read=40.0)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def say(msg: str = "") -> None:
    print(msg, flush=True)


def step(n: int | str, title: str) -> None:
    say()
    say(f"--- [{n}] {title}")


def wait_for(desc: str, fn: Callable[[], Optional[Any]], timeout_s: float, every_s: float = 0.5) -> Any:
    """Poll `fn` until it returns a truthy value; raise with `desc` on timeout."""
    t0 = time.monotonic()
    last_exc: Optional[BaseException] = None
    while True:
        try:
            v = fn()
            if v:
                return v
        except Exception as exc:  # noqa: BLE001 - keep polling until the deadline
            last_exc = exc
        if time.monotonic() - t0 >= timeout_s:
            raise TimeoutError(f"{desc}: not satisfied within {timeout_s:.0f}s" + (f" (last error: {last_exc})" if last_exc else ""))
        time.sleep(every_s)


class Narrative:
    def __init__(self, evidence_dir: Path, natural_wait_s: float, carrier: Optional[str]) -> None:
        self.http = httpx.Client(timeout=TIMEOUT)
        self.evidence_dir = evidence_dir
        self.natural_wait_s = natural_wait_s
        self.carrier = carrier or settings.ru_carriers[0]
        self.ru = settings.ru_mgmt_url
        self.smo = settings.smo_url
        self.rapp = settings.rapp_url
        self.ics = settings.ics_url            # real ICS in `full`, the SMO's compatible routes in `lite`
        self.catalogue = settings.catalogue_url
        self.ts: dict[str, str] = {"start": now_iso()}
        self.wake_source, self.external_wake_wait_s, self.ru_seq_asleep = "script", 120.0, 0
        self.twin_url = settings.twin_url
        self.summary: dict[str, Any] = {"profile": settings.profile, "forced": False, "carrier": self.carrier}

    # ---- small helpers ---------------------------------------------------------------------
    def get(self, url: str, **params: Any) -> Any:
        r = self.http.get(url, params=params or None)
        r.raise_for_status()
        return r.json()

    def post(self, url: str, body: Optional[dict] = None) -> Any:
        r = self.http.post(url, json=body if body is not None else {})
        r.raise_for_status()
        return r.json()

    def ru_state(self) -> dict[str, Any]:
        return self.get(f"{self.ru}/state")["carriers"][self.carrier]

    def show_ru(self, label: str) -> dict[str, Any]:
        st = self.ru_state()
        say(f"  O-RU {settings.ru_id} carrier {self.carrier}: active={st['active']} state={st['state']} "
            f"(sim time {self.get(f'{self.ru}/health')['sim_time_iso']}, load {st['sim'].get('prb_util_dl_pct')}% PRB, "
            f"{st['sim'].get('power_w')} W illustrative) <- {label}")
        return st

    # ---- steps -----------------------------------------------------------------------------
    def step1_initial(self) -> None:
        step(1, "O-RU initial state (M-Plane `tx-array-carriers/active` + `state`)")
        st = self.show_ru("GET /state on the mock O-RU")
        self.summary["initial"] = {"active": st["active"], "state": st["state"]}

    def step2_rapp_ready(self) -> None:
        step(2, "es-rApp: model trained, R1 registrations done, first PM delivered through the R1 job")

        def ready() -> Optional[dict]:
            h = self.get(f"{self.rapp}/health")
            r1 = h.get("r1", {})
            if h.get("model") and r1.get("registered") and r1.get("job_created"):
                return h
            return None

        h = wait_for("rApp health (model + catalogue + ICS job)", ready, timeout_s=90)
        m = h["model"]
        say(f"  model: {m.get('model_type')} auc={m.get('auc')} acc={m.get('accuracy')} "
            f"n_train={m.get('n_train')} low_threshold={m.get('low_threshold'):.2f}% PRB ({m.get('low_threshold_source')}, data={m.get('data_kind')})")
        say(f"  R1: catalogue {h['r1']['catalogue_url']}/services/{h['r1']['service']}  |  "
            f"ICS job {h['r1']['job_id']} (type {h['r1']['info_type_id']}) at {h['r1']['ics_url']} -> {h['r1']['job_result_uri']}")
        pm = wait_for("first PM report at the rApp (R1 job delivery)", lambda: self._rapp_pm(min_wake_events=0),
                      timeout_s=max(60.0, settings.ru_pm_interval_s * 4))
        latest = pm["carriers"][self.carrier]["latest"]
        say(f"  PM at rApp: reports={pm['reports']} sim_time={pm['sim_time']} {self.carrier}: active={latest['active']} "
            f"prb_dl={latest['prb_util_dl_pct']}% rrc={latest['rrc_conn']} power_w_avg={latest['power_w_avg']} (illustrative)")
        self.ts["rapp_ready"] = now_iso()

    def _print_decisions(self, decisions: list[dict]) -> None:
        for rec in decisions:
            d = rec.get("decision") or {}
            if "error" in rec:
                say(f"  decision: {rec}")
                continue
            say(f"  decision {d.get('carrier')}: action={d.get('action')} p_low={d.get('p_low', 0):.2f} ({d.get('forecast_basis')}) "
                f"load={d.get('prb_util_dl_pct')}% thr={d.get('low_threshold'):.1f}% forced={d.get('forced')}  - {d.get('reason')}")
            tw = rec.get("twin")
            if tw and tw.get("enabled"):
                say(f"    twin: verdict={tw.get('verdict')} cell={tw.get('cell_id')} coverage_rate delta={tw.get('coverage_rate_delta')} "
                    f"(threshold {tw.get('threshold')}) [{tw.get('rung') or tw.get('status')}]{' OVERRIDDEN by forced decision' if tw.get('overridden') else ''}")
            cm = rec.get("cm")
            if cm:
                say(f"    CM: ok={cm.get('ok')} cm_rtt_ms={cm.get('cm_rtt_ms')} netconf_rtt_ms={cm.get('netconf_rtt_ms')} {cm.get('detail') or cm.get('error') or ''}")

    def _rapp_pm(self, min_wake_events: int) -> Optional[dict]:
        """The rApp's PM view once it has >= 1 report for the carrier with >= min_wake_events wakes."""
        p = self.get(f"{self.rapp}/pm")
        latest = p.get("carriers", {}).get(self.carrier, {}).get("latest")
        if p.get("reports", 0) >= 1 and latest and (latest.get("wake_events") or 0) >= min_wake_events:
            return p
        return None

    def _ready(self) -> Optional[dict]:
        st = self.ru_state()
        return st if st["state"] == "READY" and st["active"] == "ACTIVE" else None

    def _asleep(self) -> Optional[dict]:
        st = self.ru_state()
        return st if st["active"] == "SLEEP" and st["state"] == "DISABLED" else None

    def step3_sleep(self) -> None:
        step(3, "slow loop: rApp policy -> R1 CM request -> SMO CM adapter -> M-Plane NETCONF edit-config active=SLEEP on the O-RU")
        if self._asleep():
            say("  the rApp's own cycle already put the carrier to SLEEP (natural decision, auto-cycle)")
        else:
            say("  POST /cycle on the rApp (one decision cycle now)")
            self._print_decisions(self.post(f"{self.rapp}/cycle", {}).get("decisions", []))
            t0 = time.monotonic()
            while time.monotonic() - t0 < self.natural_wait_s:
                if self._asleep():
                    break
                time.sleep(1.0)
                if int(time.monotonic() - t0) % 5 == 0:
                    st = self.ru_state()
                    say(f"    waiting for a natural SLEEP decision ... RU active={st['active']} state={st['state']} "
                        f"({int(time.monotonic() - t0)}/{int(self.natural_wait_s)}s)")
        if not self._asleep():
            say("")
            say("  !!! NO NATURAL SLEEP DECISION within the wait window - FORCING active=SLEEP via POST /cycle {\"force\":\"SLEEP\"}")
            say("  !!! (the model's forecast was not confident enough or the load was not below the low threshold;")
            say("  !!!  the CM path below is still the real one: R1 -> SMO -> NETCONF edit-config -> RU)")
            say("")
            self.summary["forced"] = True
            self._print_decisions(self.post(f"{self.rapp}/cycle", {"force": "SLEEP", "carrier": self.carrier}).get("decisions", []))
            wait_for("RU carrier SLEEP/DISABLED after forced SLEEP", self._asleep, timeout_s=15)
        self.ts["asleep"] = now_iso()
        self.ru_seq_asleep = int(self.get(f"{self.ru}/events", since=0).get("next", 0))
        st = self.show_ru("after the CM request settled (BUSY -> DISABLED)")
        say(f"  ruApp policy envelope on the RU: {st['ruapp_policy']}")

    def step3a_twin(self) -> None:
        step("3a", "twin guardrail: the RF twin evaluated the cell switched off before the SLEEP (POST /v1/evaluate on_off=false, ADR-7)")
        tv = self.get(f"{self.rapp}/health").get("twin") or {}
        summary: dict[str, Any] = {"url": self.twin_url or None, "enabled": bool(tv.get("enabled")), "status": tv.get("status"),
                                   "cell_id": tv.get("cell_id"), "cell_source": tv.get("cell_source"), "rung": tv.get("rung"),
                                   "verdict": None, "delta_vs_baseline": None, "latency_ms": None,
                                   "checks": tv.get("checks"), "blocks": tv.get("blocks"), "error": tv.get("last_error")}
        if not tv.get("enabled"):
            say("  twin guardrail DISABLED (TWIN_URL empty; run_demo.sh TWIN=0 or no twin at TWIN_DIR): no RF check before SLEEP")
            say("  the SMO's twin-ingest JSONL (step 6) still runs; it is the NDT feed, not the RF twin's input")
            self.summary["twin"] = summary
            return
        # the SLEEP decision(s) of this run carry the verdict; take the last one that reached the twin
        recs = [r for r in self.get(f"{self.rapp}/decisions") if (r.get("twin") or {}).get("enabled")]
        last = recs[-1]["twin"] if recs else {}
        say(f"  twin {self.twin_url}  status={tv.get('status')}  cell={tv.get('cell_id')} ({tv.get('cell_source')}, {tv.get('cells_listed')} cells listed)")
        if tv.get("rung_warning"):
            say(f"  !!! {tv['rung_warning']}")
        if not last:
            say(f"  no SLEEP decision reached the twin in this run (checks={tv.get('checks')}); last error: {tv.get('last_error')}")
            self.summary["twin"] = summary
            return
        # the rung is the rung of the VERDICT: never fall back to the twin's /health rung when no evaluation happened
        summary.update({"verdict": last.get("verdict"), "delta_vs_baseline": last.get("delta_vs_baseline"), "latency_ms": last.get("latency_ms"),
                        "rung": last.get("rung") if last.get("status") == "ok" else None, "cell_id": last.get("cell_id"), "status": last.get("status"),
                        "error": last.get("error"), "overridden": last.get("overridden", False), "threshold": last.get("threshold"),
                        "guardrail_kpis": last.get("guardrail_kpis"), "baseline_kpis": last.get("baseline_kpis")})
        if last.get("status") != "ok":
            say(f"  twin {last.get('status').upper()}: {last.get('error')}")
            say(f"  verdict: {last.get('verdict')} - the guardrail is advisory; nothing about the RF twin is faked here")
            self.summary["twin"] = summary
            return
        d = last.get("delta_vs_baseline") or {}
        say(f"  rung: {last.get('rung')}")
        say(f"  delta_vs_baseline (cell {last.get('cell_id')} off): rsrp_p5={d.get('rsrp_p5')} dB  sinr_p5={d.get('sinr_p5')} dB  "
            f"outage_rate={d.get('outage_rate')}  coverage_rate={d.get('coverage_rate')}  (block if coverage_rate < {last.get('threshold')})")
        say(f"  verdict: {str(last.get('verdict')).upper()}{'  (OVERRIDDEN by the forced decision, recorded)' if last.get('overridden') else ''}  "
            f"latency {last.get('latency_ms')} ms  checks={tv.get('checks')} blocks={tv.get('blocks')}")
        say("  note: the RF twin consumes topology/config (which cell is off), not PM; the SMO's twin-ingest JSONL (step 6) is the NDT feed")
        self.summary["twin"] = summary

    def step4_cm_rtt(self) -> None:
        step(4, "management-plane round trip as seen by the rApp (GET /policy, GET /metrics)")
        pol = self.get(f"{self.rapp}/policy").get(self.carrier, {})
        last_cm = pol.get("last_cm") or {}
        met = self.get(f"{self.rapp}/metrics")
        cm_rtt = last_cm.get("cm_rtt_ms", met.get("last_cm_rtt_ms"))
        nc_rtt = last_cm.get("netconf_rtt_ms", met.get("last_netconf_rtt_ms"))
        say(f"  cm_rtt_ms (whole SMO-side round trip: NETCONF session setup + edit-config + close, measured at the SMO): {cm_rtt}")
        say(f"  netconf_rtt_ms (edit-config only, measured by the SMO's M-Plane client): {nc_rtt}")
        say(f"  rapp_rtt_ms (whole R1 CM request as seen by the rApp): {last_cm.get('rapp_rtt_ms')}")
        self.summary["cm_rtt_ms"] = cm_rtt
        self.summary["netconf_rtt_ms"] = nc_rtt
        self.summary["rapp_rtt_ms"] = last_cm.get("rapp_rtt_ms")
        self.summary["envelope"] = pol.get("envelope")

    def step5_wake(self) -> None:
        step(5, "fast loop: PRACH energy -> ruApp wakes the carrier locally (SLEEP -> ACTIVE, no network on the path)")
        if self.wake_source == "external":
            self._external_wake()
            return
        st = self.ru_state()
        if st["active"] != "SLEEP":
            say(f"  carrier is {st['active']} (the rApp may have pre-emptively woken it); re-forcing SLEEP once")
            self.summary["forced"] = True
            self._print_decisions(self.post(f"{self.rapp}/cycle", {"force": "SLEEP", "carrier": self.carrier}).get("decisions", []))
            wait_for("RU carrier SLEEP/DISABLED", self._asleep, timeout_s=15)
        say("  POST /sim/prach {\"energy_db\": 20} on the mock O-RU (the demo trigger; threshold is the envelope's 10 dB)")
        res = self.post(f"{self.ru}/sim/prach", {"carrier": self.carrier, "energy_db": 20})
        say(f"  ruApp decision: {res['decision']}  wake_latency_us={res['wake_latency_us']}  active={res['active']} state={res['state']}")
        if res["decision"] != "wake":
            raise RuntimeError(f"ruApp did not wake the carrier: {res}")
        self.summary["wake_latency_us"] = res["wake_latency_us"]
        self.ts["woken"] = now_iso()
        wait_for("carrier READY after the local wake", self._ready, timeout_s=10, every_s=0.05)
        st = self.show_ru("after the ruApp's local wake settled (BUSY -> READY)")
        say(f"  RU now rejects a new SLEEP for hold_remaining_s={st['hold_remaining_s']} (min-hold-after-wake from the envelope)")
        self.summary["ruapp"] = st.get("ruapp")

    def _external_wake(self) -> None:
        """The wake trigger comes from outside the script (for example a stimulus seen by radio hardware that then
        raises PRACH energy on the O-RU). The narrative only watches the O-RU event stream for the ruApp's decision."""
        say(f"  wake source: EXTERNAL - waiting up to {self.external_wake_wait_s:.0f}s for the ruApp's wake decision (no trigger sent from here)")

        def wake_decision() -> Optional[dict]:
            for e in self.get(f"{self.ru}/events", since=self.ru_seq_asleep).get("events", []):
                ev = e.get("event") or {}
                if e.get("kind") == "RuAppDecision" and ev.get("carrier") == self.carrier and ev.get("decision") == "wake":
                    return ev
            return None

        ev = wait_for("ruApp wake decision from the external source", wake_decision, timeout_s=self.external_wake_wait_s, every_s=0.2)
        lat = None if ev.get("wake_latency_ns") is None else round(ev["wake_latency_ns"] / 1000.0, 3)
        say(f"  ruApp decision: wake  wake_latency_us={lat}  prach_energy_db={ev.get('prach_energy_db')}  (trigger: external)")
        self.summary["wake_latency_us"] = lat
        self.summary["wake_source"] = "external"
        self.ts["woken"] = now_iso()
        wait_for("carrier READY after the local wake", self._ready, timeout_s=10, every_s=0.05)
        st = self.show_ru("after the ruApp's local wake settled (BUSY -> READY)")
        say(f"  RU now rejects a new SLEEP for hold_remaining_s={st['hold_remaining_s']} (min-hold-after-wake from the envelope)")
        self.summary["ruapp"] = st.get("ruapp")

    def step6_reports(self) -> None:
        wait_s = settings.ru_pm_interval_s + 1.0
        step(6, f"north-bound O-RU push (hybrid M-Plane, VES-style PM/notifications): waiting one PM interval + 1 s ({wait_s:.0f}s) for the reports to land")
        time.sleep(wait_s)

        def wake_notification() -> Optional[list[dict]]:
            evs = self.get(f"{self.smo}/o1/events", since=0).get("events", [])
            hits = []
            for e in evs:
                nf = (e.get("event") or {}).get("notificationFields") or {}
                add = nf.get("additionalFields") or {}
                if add.get("carrier") != self.carrier:
                    continue
                if nf.get("changeType") == "tx-array-carriers-state-change" and add.get("cause") == "ruapp-wake":
                    hits.append({"seq": e["seq"], "state": add.get("state"), "active": add.get("active"), "cause": add.get("cause")})
                if nf.get("changeType") == "ruapp-decision" and add.get("decision") == "wake":
                    hits.append({"seq": e["seq"], "decision": "wake", "wake_latency_us": add.get("wake_latency_us")})
            return hits or None

        hits = wait_for("SMO notifications for the local wake (tx-array-carriers-state-change cause=ruapp-wake, ruapp-decision)",
                        wake_notification, timeout_s=wait_s + 5)
        say(f"  SMO GET /o1/events: {len(hits)} wake-related notification(s): {hits[:3]}")
        pm_latest = self.get(f"{self.smo}/o1/pm/latest")
        entry = pm_latest.get(settings.ru_id) or next(iter(pm_latest.values()), {})
        objs = ((entry.get("event") or {}).get("measurementFields") or {}).get("additionalObjects") or []
        car = next((o for o in objs if o.get("name") == self.carrier), objs[0] if objs else {})
        say(f"  SMO GET /o1/pm/latest: {self.carrier} active={car.get('active')} state={car.get('state')} wake_events={car.get('wake_events')} "
            f"wake_latency_us={car.get('wake_latency_us')} prach_detections={car.get('prach_detections')} energy_wh={car.get('energy_wh')} (illustrative)")

        rpm = wait_for("PM with wake_events >= 1 delivered to the rApp (R1 job)", lambda: self._rapp_pm(min_wake_events=1),
                       timeout_s=settings.ru_pm_interval_s * 2 + 2)
        rl = rpm["carriers"][self.carrier]["latest"]
        say(f"  rApp GET /pm: reports={rpm['reports']} {self.carrier} active={rl['active']} wake_events={rl['wake_events']} wake_latency_us={rl['wake_latency_us']}")
        twin = self.get(f"{self.smo}/twin/ingest/stats")
        say(f"  twin ingest sink: {twin['lines']} JSON lines at {twin['path']}")
        self.summary["smo_wake_notifications"] = hits
        self.summary["smo_pm_latest_carrier"] = car
        self.summary["rapp_pm_latest_carrier"] = rl
        self.summary["twin_ingest_lines"] = twin["lines"]
        self.summary["twin_ingest_path"] = twin["path"]
        self.ts["reported"] = now_iso()

    def step7_platform(self) -> None:
        where = "real NONRTRIC containers" if settings.profile == "full" else "the SMO's ICS/catalogue-compatible routes (lite)"
        step(7, f"R1 platform services view - {where}")
        jobs = self.get(f"{self.ics}/data-consumer/v1/info-jobs")
        say(f"  ICS  GET {self.ics}/data-consumer/v1/info-jobs -> {jobs}")
        job = self.get(f"{self.ics}/data-consumer/v1/info-jobs/es-rapp-pm")
        say(f"  ICS  job es-rapp-pm: type={job.get('info_type_id')} owner={job.get('job_owner')} result_uri={job.get('job_result_uri')}")
        try:
            status = self.get(f"{self.ics}/data-consumer/v1/info-jobs/es-rapp-pm/status")
            say(f"  ICS  job status: {status}")
        except httpx.HTTPError as exc:
            say(f"  ICS  job status: n/a ({exc})")
        services = self.get(f"{self.catalogue}/services")
        names = [s.get("name") for s in services] if isinstance(services, list) else services
        say(f"  catalogue GET {self.catalogue}/services -> {names}")
        if "es-rapp-pm" not in jobs:
            raise RuntimeError("ICS job list does not contain es-rapp-pm")
        if "es-rapp" not in (names or []):
            raise RuntimeError("catalogue does not contain es-rapp")
        smo_h = self.get(f"{self.smo}/health")
        say(f"  SMO R1 producer: registration={smo_h['r1']['registration'].get('state')} jobs={smo_h['r1']['jobs']} delivery={smo_h['r1']['delivery']}")
        self.summary["ics_jobs"] = jobs
        self.summary["catalogue_services"] = services
        self.summary["smo_r1"] = smo_h["r1"]

    def step8_summary(self) -> Path:
        step(8, "evidence")
        self.summary["decisions"] = self.get(f"{self.rapp}/decisions")
        events = self.get(f"{self.ru}/events", since=0).get("events", [])
        self.summary["ru_state_timeline"] = [
            {"seq": e["seq"], **{k: e["event"].get(k) for k in ("wall_us", "active", "state", "cause")}}
            for e in events if e.get("kind") == "CarrierStateChanged" and e["event"].get("carrier") == self.carrier
        ]
        self.summary["ru_decisions"] = [
            {"seq": e["seq"], **{k: e["event"].get(k) for k in ("wall_us", "decision", "prach_energy_db", "wake_latency_ns")}}
            for e in events if e.get("kind") == "RuAppDecision" and e["event"].get("carrier") == self.carrier
        ]
        self.ts["end"] = now_iso()
        self.summary["timestamps"] = self.ts
        self.summary["urls"] = {"ru_mgmt": self.ru, "smo": self.smo, "rapp": self.rapp, "ics": self.ics, "catalogue": self.catalogue,
                                "cm_adapter": settings.cm_adapter, "ru_netconf": settings.ru_netconf_target}
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        out = self.evidence_dir / "summary.json"
        out.write_text(json.dumps(self.summary, indent=2, default=str))
        say(f"  wrote {display_path(out, settings.var_dir)}")  # relative: evidence dirs are shared, no local account names
        return out

    def banner(self) -> None:
        wake = self.summary.get("wake_latency_us")
        cm = self.summary.get("cm_rtt_ms")
        nc = self.summary.get("netconf_rtt_ms")
        forced = self.summary.get("forced")
        say()
        say("=" * 78)
        say(f"  profile: {settings.profile}   sleep decision: {'FORCED (no natural decision in time)' if forced else 'natural (rApp model)'}")
        say(f"  local wake (ruApp): {wake} µs")
        say(f"  management-plane CM round trip (SMO -> NETCONF -> RU, measured at the SMO): {cm} ms")
        say(f"    (rApp-observed R1 CM request incl. the rApp -> SMO hop: {self.summary.get('rapp_rtt_ms')} ms; NETCONF edit-config alone: {nc} ms)")
        tw = self.summary.get("twin") or {}
        if not tw.get("enabled"):
            say("  twin guardrail: disabled (no RF check before SLEEP)")
        elif tw.get("status") == "ok" and tw.get("verdict"):
            d = tw.get("delta_vs_baseline") or {}
            say(f"  twin guardrail: {tw.get('verdict')}  cell {tw.get('cell_id')}  coverage_rate delta {d.get('coverage_rate')}  [{tw.get('rung')}]")
        elif tw.get("verdict"):
            say(f"  twin guardrail: {tw.get('verdict')} - twin {tw.get('status')}, no RF verdict ({str(tw.get('error'))[:80]})")
        else:
            say(f"  twin guardrail: status={tw.get('status')} - no verdict recorded ({tw.get('error')})")
        say("=" * 78)

    def run(self) -> int:
        self.step1_initial()
        self.step2_rapp_ready()
        self.step3_sleep()
        self.step3a_twin()
        self.step4_cm_rtt()
        self.step5_wake()
        self.step6_reports()
        self.step7_platform()
        self.step8_summary()
        self.banner()
        return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evidence-dir", required=True, type=Path)
    ap.add_argument("--natural-wait-s", type=float, default=30.0, help="how long to wait for a natural SLEEP decision before forcing")
    ap.add_argument("--carrier", default=None)
    ap.add_argument("--wake-source", choices=["script", "external"], default="script",
                    help="script: the narrative injects PRACH energy itself; external: wait for a wake triggered elsewhere")
    ap.add_argument("--external-wake-wait-s", type=float, default=120.0)
    args = ap.parse_args(argv)
    n = Narrative(args.evidence_dir, args.natural_wait_s, args.carrier)
    n.wake_source, n.external_wake_wait_s = args.wake_source, args.external_wake_wait_s
    try:
        return n.run()
    except (TimeoutError, RuntimeError, httpx.HTTPError) as exc:
        say()
        say(f"NARRATIVE FAILED: {exc}")
        try:
            n.step8_summary()
        except Exception as exc2:  # noqa: BLE001
            say(f"(could not write summary: {exc2})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
