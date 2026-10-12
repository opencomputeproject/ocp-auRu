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
"""Per-carrier-day actuation budget / alarm (a "max toggles per day" guard):
policy engine, es-rApp service and offline evaluation. Synthetic data only.

Shipped defaults: RAPP_MAX_ACTUATIONS_PER_DAY=6, RAPP_ACTUATION_ALARM_PER_DAY=4 (0 = off), counted in
SIMULATED/data time per carrier per calendar day of the PM simTime; a clean night is exactly 2.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ocp_auru.common.config import Settings
from ocp_auru.common.events import (Active, CarrierPm, CarrierState, CarrierStatus, CmCarrierRequest, CmCarrierResponse, PmReport,
                                RuAppPolicy, WakeLatencyStats)
from ocp_auru.common.ves import measurement_event
from ocp_auru.rapp import eval as ev
from ocp_auru.rapp import model as model_mod
from ocp_auru.rapp.app import create_app
from ocp_auru.rapp.data import synthetic
from ocp_auru.rapp.policy_engine import (PolicyState, actuations_today, count_actuation, decide, roll_actuation_day, sim_day)

pytestmark = pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")


class FakeR1:
    """The R1 client double of test_rapp_app (ok / rejected / raising CM requests)."""

    def __init__(self, ok=True, fail=False):
        self.requests: list[CmCarrierRequest] = []
        self.ok, self.fail, self.started, self.stopped = ok, fail, False, False

    def start_background(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def status(self):
        return {"registered": True, "job_created": True, "fake": True}

    def cm_request(self, req: CmCarrierRequest) -> CmCarrierResponse:
        self.requests.append(req)
        if self.fail:
            raise httpx.ConnectError("smo down")
        return CmCarrierResponse(ok=self.ok, ru_id=req.ru_id, carrier=req.carrier, active=req.active, cm_rtt_ms=12.5, netconf_rtt_ms=4.25, detail="fake")


@pytest.fixture(scope="module")
def trained():
    return model_mod.train(synthetic.generate(n_cells=2, days=10, seed=11), seed=11)


def pm_envelope(active=Active.ACTIVE, state=CarrierState.READY, prb=3.0, sim_time="2024-01-10T03:00:00", wake_events=0, lat=None):
    car = CarrierPm(name="txcarrier0", active=active, state=state, tx_power_dbm=24.0, prb_util_dl_pct=prb, prb_util_ul_pct=prb / 2,
                    rrc_conn=1, power_w_avg=255.0, energy_wh=0.35, prach_detections=1, wake_events=wake_events,
                    wake_latency_us=WakeLatencyStats(last=lat, p50=lat, max=lat, count=wake_events))
    rep = PmReport(ru_id="oru-test", interval_s=5.0, sim_time_iso=sim_time, carriers=[car], epoch_us=1_700_000_000_000_000)
    return measurement_event(rep).model_dump(mode="json")

THR = 10.0
DAY = "2024-01-10"
ROOT = Path(__file__).resolve().parents[3] / "demo"   # the CLI (scripts/eval_energy.py) lives with the demo
BUDGET_ENV = ("RAPP_MAX_ACTUATIONS_PER_DAY", "RAPP_ACTUATION_ALARM_PER_DAY")


def status(active=Active.ACTIVE, state=CarrierState.READY):
    return CarrierStatus(name="txcarrier0", active=active, state=state, ruapp_policy=RuAppPolicy())


AWAKE, ASLEEP = status(), status(Active.SLEEP, CarrierState.DISABLED)


def _cfg(monkeypatch, **env):
    for name in BUDGET_ENV:
        monkeypatch.delenv(name, raising=False)
    base = {"RAPP_SLEEP_THRESHOLD": "0.7", "RAPP_WAKE_THRESHOLD": "0.3", "RAPP_MIN_HOLD_S": "0", "RAPP_ALLOW_INACTIVE": "0",
            "RAPP_MIN_SLEEP_DWELL_MIN": "0", "RAPP_POST_LOCAL_WAKE_HOLD_MIN": "0"}
    for k, v in {**base, **env}.items():
        monkeypatch.setenv(k, str(v))
    return Settings()


def _state(count: int, day: str = DAY, **kw) -> PolicyState:
    return PolicyState(actuations_day=day, actuations_today=count, **kw)


def _sleep(cfg, st, sim_now=f"{DAY}T03:00:00"):
    """The ACTIVE branch with a forecast that would SLEEP."""
    return decide(AWAKE, {"prb_util_dl_pct": 3.0}, 0.9, 1000.0, st, low_threshold=THR, cfg=cfg, sim_now=sim_now)


def _wake(cfg, st, sim_now=f"{DAY}T10:00:00"):
    """The SLEEP branch with a forecast that would wake pre-emptively."""
    return decide(ASLEEP, {"prb_util_dl_pct": 0.0}, 0.1, 1000.0, st, low_threshold=THR, cfg=cfg, sim_now=sim_now)


# ---- config -----------------------------------------------------------------------------------------------------
def test_shipped_defaults_and_clamping(monkeypatch):
    for name in BUDGET_ENV:
        monkeypatch.delenv(name, raising=False)
    s = Settings()
    assert s.rapp_max_actuations_per_day == 6 and s.rapp_actuation_alarm_per_day == 4
    monkeypatch.setenv("RAPP_MAX_ACTUATIONS_PER_DAY", "-3")
    monkeypatch.setenv("RAPP_ACTUATION_ALARM_PER_DAY", "-1")
    s = Settings()
    assert s.rapp_max_actuations_per_day == 0 and s.rapp_actuation_alarm_per_day == 0   # clamped to off, never negative
    assert replace(s, rapp_max_actuations_per_day=-7).rapp_max_actuations_per_day == 0
    # the alarm is the signal BELOW the budget: an alarm above a cap > 0 would fire on forced writes only, so it is
    # clamped down to the cap; with the cap off the alarm stands alone
    monkeypatch.setenv("RAPP_MAX_ACTUATIONS_PER_DAY", "3")
    monkeypatch.setenv("RAPP_ACTUATION_ALARM_PER_DAY", "5")
    s = Settings()
    assert (s.rapp_max_actuations_per_day, s.rapp_actuation_alarm_per_day) == (3, 3)
    assert replace(s, rapp_max_actuations_per_day=0).rapp_actuation_alarm_per_day == 3          # cap off: the alarm stays
    monkeypatch.setenv("RAPP_MAX_ACTUATIONS_PER_DAY", "0")
    assert Settings().rapp_actuation_alarm_per_day == 5
    assert replace(Settings(), rapp_max_actuations_per_day=6, rapp_actuation_alarm_per_day=9).rapp_actuation_alarm_per_day == 6


# ---- policy engine ----------------------------------------------------------------------------------------------
def test_sim_day_and_counter_helpers():
    assert sim_day(f"{DAY}T23:59:59") == DAY and sim_day("2026-01-01T02:00:00Z") == "2026-01-01"
    assert sim_day("2024-01-10T23:30:00+05:00") == DAY                    # the day as written, no tz conversion
    assert sim_day(None) is None and sim_day("") is None and sim_day("garbage") is None
    st = PolicyState()
    assert actuations_today(st, f"{DAY}T03:00:00") == 0 and roll_actuation_day(st, None) is False
    assert count_actuation(st, f"{DAY}T03:00:00") == 1 and (st.actuations_day, st.actuations_today) == (DAY, 1)
    assert count_actuation(st, f"{DAY}T10:00:00") == 2
    assert actuations_today(st, f"{DAY}T12:00:00") == 2 and actuations_today(st, "2024-01-11T00:00:00") == 0   # other day reads 0
    assert roll_actuation_day(st, f"{DAY}T13:00:00") is False and st.actuations_today == 2                    # same day: untouched
    assert roll_actuation_day(st, "2024-01-11T00:00:00") is True and (st.actuations_day, st.actuations_today) == ("2024-01-11", 0)
    assert count_actuation(st, "2024-01-12T02:00:00") == 1 and st.actuations_day == "2024-01-12"           # counting rolls too
    # an unknown day counts under None, which decide() never reads
    st2 = PolicyState()
    assert count_actuation(st2, None) == 1 and st2.actuations_day is None and actuations_today(st2, f"{DAY}T03:00:00") == 0


def test_budget_blocks_sleep_and_preemptive_active_after_n_applied_actuations(monkeypatch):
    c = _cfg(monkeypatch, RAPP_MAX_ACTUATIONS_PER_DAY=2, RAPP_ACTUATION_ALARM_PER_DAY=0)
    # below the budget: the usual decisions, the count reported, the flag off
    for n in (0, 1):
        d = _sleep(c, _state(n))
        assert d.action == "SLEEP" and d.actuations_today == n and d.budget_reached is False and d.alarm is False
        d = _wake(c, _state(n))
        assert d.action == "ACTIVE" and d.actuations_today == n and d.budget_reached is False
    # at the budget: no new SLEEP, no new pre-emptive ACTIVE; the carrier stays as it is
    st = _state(2)
    d = _sleep(c, st)
    assert d.action == "NONE" and d.budget_reached is True and d.actuations_today == 2
    assert d.reason.startswith(f"actuation budget: 2 of 2 for {DAY} reached; ruApp remains the wake path") and "SLEEP refused" in d.reason
    d = _wake(c, st)
    assert d.action == "NONE" and d.budget_reached is True
    assert d.reason.startswith(f"actuation budget: 2 of 2 for {DAY} reached; ruApp remains the wake path") and "pre-emptive ACTIVE refused" in d.reason
    # the budget is the LAST guard of both branches: a NONE for any other reason names that reason
    assert "sleep threshold" in decide(AWAKE, {"prb_util_dl_pct": 3.0}, 0.1, 1000.0, st, low_threshold=THR, cfg=c, sim_now=f"{DAY}T03:00:00").reason
    assert "stay SLEEP" in decide(ASLEEP, {"prb_util_dl_pct": 0.0}, 0.9, 1000.0, st, low_threshold=THR, cfg=c, sim_now=f"{DAY}T03:00:00").reason
    assert "BUSY" in decide(status(Active.ACTIVE, CarrierState.BUSY), {"prb_util_dl_pct": 3.0}, 0.9, 1000.0, st, low_threshold=THR, cfg=c,
                            sim_now=f"{DAY}T03:00:00").reason
    # past the budget (forced writes counted through it): still refused, the count is reported honestly
    assert "actuation budget: 5 of 2" in _sleep(c, _state(5)).reason
    # the engine never mutates the state
    assert (st.actuations_day, st.actuations_today) == (DAY, 2)
    # a refused wake keeps its (confirmed) vote so no re-confirmation is needed once the day rolls over
    c2 = replace(c, rapp_wake_confirm_windows=2)
    d = _wake(c2, _state(2, wake_votes=1, last_vote_sim=f"{DAY}T09:30:00"))
    assert d.action == "NONE" and "actuation budget" in d.reason and d.wake_votes == 2 and d.vote_sim == f"{DAY}T10:00:00"
    # the RAPP_ALLOW_INACTIVE restore of a hard-off carrier is service-first, not budgeted
    ci = _cfg(monkeypatch, RAPP_MAX_ACTUATIONS_PER_DAY=2, RAPP_ALLOW_INACTIVE=1)
    d = decide(status(Active.INACTIVE, CarrierState.DISABLED), {"prb_util_dl_pct": 0.0}, 0.1, 1000.0, _state(2), low_threshold=THR, cfg=ci,
               sim_now=f"{DAY}T10:00:00")
    assert d.action == "ACTIVE" and d.budget_reached is True


def test_budget_resets_at_sim_midnight_and_ignores_unknown_days(monkeypatch):
    c = _cfg(monkeypatch, RAPP_MAX_ACTUATIONS_PER_DAY=2)
    st = _state(2)
    assert _sleep(c, st, sim_now=f"{DAY}T23:59:00").action == "NONE"
    d = _sleep(c, st, sim_now="2024-01-11T00:00:00")          # the next sim day: a fresh budget (the engine reads 0)
    assert d.action == "SLEEP" and d.actuations_today == 0 and d.budget_reached is False
    assert _wake(c, st, sim_now="2024-01-11T06:00:00").action == "ACTIVE"
    # a stamp from the day BEFORE the counter's day is another day too (never a stale count)
    assert _sleep(c, st, sim_now="2024-01-09T23:00:00").action == "SLEEP"
    # unknown day (no PM simTime yet, unparseable): the budget does not apply, as with the other sim-time rules
    for sim in (None, "garbage"):
        d = _sleep(c, st, sim_now=sim)
        assert d.action == "SLEEP" and d.actuations_today == 0 and d.budget_reached is False and d.alarm is False
    # the caller's roll + count sequence as app.py does it around midnight
    st = _state(2)
    roll_actuation_day(st, "2024-01-11T00:00:00")
    assert st.actuations_today == 0 and count_actuation(st, "2024-01-11T00:00:00") == 1
    assert _sleep(c, st, sim_now="2024-01-11T00:30:00").action == "SLEEP"


def test_zero_is_off_and_the_alarm_flag(monkeypatch):
    c0 = _cfg(monkeypatch, RAPP_MAX_ACTUATIONS_PER_DAY=0, RAPP_ACTUATION_ALARM_PER_DAY=0)
    d = _sleep(c0, _state(100))
    assert d.action == "SLEEP" and d.actuations_today == 100 and d.budget_reached is False and d.alarm is False
    assert _wake(c0, _state(100)).action == "ACTIVE"
    # the alarm is a flag below the budget: it never changes the decision
    c = _cfg(monkeypatch, RAPP_MAX_ACTUATIONS_PER_DAY=6, RAPP_ACTUATION_ALARM_PER_DAY=4)
    for n, alarm, reached, action in ((3, False, False, "SLEEP"), (4, True, False, "SLEEP"), (5, True, False, "SLEEP"), (6, True, True, "NONE")):
        d = _sleep(c, _state(n))
        assert (d.alarm, d.budget_reached, d.action) == (alarm, reached, action), n
    # the shipped defaults (6 / 4) on a clean night: two actuations, no alarm, no budget
    shipped = _cfg(monkeypatch)
    assert shipped.rapp_max_actuations_per_day == 6 and shipped.rapp_actuation_alarm_per_day == 4
    st = PolicyState()
    d = _sleep(shipped, st)
    assert d.action == "SLEEP" and not d.alarm and not d.budget_reached
    count_actuation(st, d.sim_time or f"{DAY}T03:00:00")
    d = _wake(shipped, st)
    assert d.action == "ACTIVE" and d.actuations_today == 1 and not d.alarm and not d.budget_reached
    count_actuation(st, f"{DAY}T10:00:00")
    assert st.actuations_today == 2 and _sleep(shipped, st, sim_now=f"{DAY}T22:00:00").action == "SLEEP"
    # alarm off, budget on (and the other way round) are independent
    ca = _cfg(monkeypatch, RAPP_MAX_ACTUATIONS_PER_DAY=0, RAPP_ACTUATION_ALARM_PER_DAY=2)
    d = _sleep(ca, _state(3))
    assert d.action == "SLEEP" and d.alarm is True and d.budget_reached is False
    cb = _cfg(monkeypatch, RAPP_MAX_ACTUATIONS_PER_DAY=2, RAPP_ACTUATION_ALARM_PER_DAY=0)
    d = _sleep(cb, _state(3))
    assert d.action == "NONE" and d.alarm is False and d.budget_reached is True


def test_legacy_decisions_unchanged_by_the_budget_fields(monkeypatch):
    """With an empty counter (a fresh PolicyState, as before the budget existed) every action and reason is identical
    under the shipped defaults and with the budget off; the new fields only add to the Decision."""
    on, off = _cfg(monkeypatch), _cfg(monkeypatch, RAPP_MAX_ACTUATIONS_PER_DAY=0, RAPP_ACTUATION_ALARM_PER_DAY=0)
    for active in (Active.ACTIVE, Active.SLEEP, Active.INACTIVE):
        for p in (0.0, 0.29, 0.3, 0.7, 0.9):
            for prb in (1.0, 50.0):
                for sim in (None, f"{DAY}T02:30:00"):
                    a = decide(status(active, CarrierState.READY), {"prb_util_dl_pct": prb}, p, 1000.0, PolicyState(), low_threshold=THR, cfg=on, sim_now=sim)
                    b = decide(status(active, CarrierState.READY), {"prb_util_dl_pct": prb}, p, 1000.0, PolicyState(), low_threshold=THR, cfg=off, sim_now=sim)
                    assert (a.action, a.reason) == (b.action, b.reason)
                    assert a.actuations_today == 0 and a.budget_reached is False and a.alarm is False


# ---- es-rApp service --------------------------------------------------------------------------------------------
@pytest.fixture
def app_cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("VAR_DIR", str(tmp_path))
    monkeypatch.setenv("RAPP_PORT", "28101")
    monkeypatch.setenv("SMO_PORT", "28103")
    monkeypatch.setenv("RAPP_AUTO_CYCLE", "0")
    monkeypatch.setenv("RAPP_MIN_HOLD_S", "0")
    monkeypatch.setenv("TWIN_URL", "")
    monkeypatch.setenv("RAPP_MIN_SLEEP_DWELL_MIN", "0")       # the dwell would otherwise name itself before the budget
    monkeypatch.setenv("RAPP_POST_LOCAL_WAKE_HOLD_MIN", "0")
    monkeypatch.setenv("RAPP_MAX_ACTUATIONS_PER_DAY", "3")
    monkeypatch.setenv("RAPP_ACTUATION_ALARM_PER_DAY", "2")
    return Settings()


def _cycle(c, **body):
    return c.post("/cycle", json=body).json()["decisions"][0]


def _pm(c, **kw):
    assert c.post("/r1/pm-delivery", json=pm_envelope(**kw)).status_code == 200


def test_counter_increments_only_on_applied_cm_and_is_exposed(app_cfg, trained, caplog):
    """The count grows on an APPLIED CM write only (natural and forced), the alarm and the budget are each logged
    ONCE per carrier-day, the budget refuses a natural pre-emptive wake, a forced action bypasses it but counts,
    /metrics and /policy expose the state, and the next sim day starts a fresh count."""
    r1 = FakeR1()
    caplog.set_level(logging.WARNING, logger="ocp_auru.rapp.app")
    with TestClient(create_app(app_cfg, model=trained, r1=r1)) as c:
        h = c.get("/health").json()["policy"]
        assert h["max_actuations_per_day"] == 3 and h["actuation_alarm_per_day"] == 2
        m = c.get("/metrics").json()
        assert m["actuations_today"] == {} and m["actuation_alarm"] == 0 and m["actuation_budget_reached"] == 0
        assert m["max_actuations_per_day"] == 3 and m["actuation_alarm_per_day"] == 2
        # 1: a natural SLEEP at 03:00
        _pm(c, prb=3.0, sim_time=f"{DAY}T03:00:00")
        rec = _cycle(c)
        assert rec["decision"]["action"] == "SLEEP" and rec["cm"]["ok"] is True
        assert rec["decision"]["actuations_today"] == 0 and rec["decision"]["budget_reached"] is False and rec["decision"]["alarm"] is False
        pol = c.get("/policy").json()["txcarrier0"]
        assert (pol["actuations_day"], pol["actuations_today"]) == (DAY, 1)
        m = c.get("/metrics").json()
        assert m["actuations_today"] == {"txcarrier0": 1} and m["actuations_day"] == {"txcarrier0": DAY}
        assert m["actuation_alarm"] == 0 and m["actuation_alarm_carrier_days"] == 0 and not caplog.records
        # 2: the natural pre-emptive wake at the morning ramp reaches the alarm threshold -> one WARNING
        _pm(c, active=Active.SLEEP, state=CarrierState.DISABLED, prb=0.0, sim_time=f"{DAY}T10:00:00")
        rec = _cycle(c)
        assert rec["decision"]["action"] == "ACTIVE" and rec["decision"]["actuations_today"] == 1 and rec["decision"]["alarm"] is False
        m = c.get("/metrics").json()
        assert m["actuations_today"] == {"txcarrier0": 1 + 1} and m["actuation_alarm"] == 1 and m["actuation_alarm_carrier_days"] == 1
        assert m["actuation_budget_reached"] == 0 and m["actuation_budget_carrier_days"] == 0
        alarms = [r for r in caplog.records if "actuation ALARM" in r.getMessage()]
        assert len(alarms) == 1 and alarms[0].levelno == logging.WARNING and f"2 of 3 actuations on sim day {DAY}" in alarms[0].getMessage()
        # 3: a forced SLEEP (operator) counts and reaches the budget -> one WARNING
        _pm(c, active=Active.ACTIVE, prb=55.0, sim_time=f"{DAY}T10:30:00")
        rec = _cycle(c, force="SLEEP")
        assert rec["decision"]["action"] == "SLEEP" and rec["decision"]["forced"] is True and rec["cm"]["ok"] is True
        m = c.get("/metrics").json()
        assert m["actuations_today"] == {"txcarrier0": 3} and m["actuation_budget_reached"] == 1 and m["actuation_budget_carrier_days"] == 1
        budgets = [r for r in caplog.records if "actuation BUDGET reached" in r.getMessage()]
        assert len(budgets) == 1 and f"3 of 3 actuations on sim day {DAY}" in budgets[0].getMessage() and "(forced)" in budgets[0].getMessage()
        # the budget now refuses the natural pre-emptive wake: no CM request, the carrier stays asleep, the ruApp is the wake path
        _pm(c, active=Active.SLEEP, state=CarrierState.DISABLED, prb=0.0, sim_time=f"{DAY}T11:00:00")
        rec = _cycle(c)
        d = rec["decision"]
        assert d["action"] == "NONE" and d["reason"].startswith(f"actuation budget: 3 of 3 for {DAY} reached; ruApp remains the wake path"), d
        assert d["budget_reached"] is True and d["alarm"] is True and d["actuations_today"] == 3 and rec["cm"] is None
        assert len(r1.requests) == 3 and c.get("/metrics").json()["actuation_budget_refusals"] == 1
        assert c.get("/policy").json()["txcarrier0"]["active"] == "SLEEP"
        # a forced ACTIVE bypasses the budget (operator intent) but still counts; no second WARNING for the day
        rec = _cycle(c, force="ACTIVE")
        assert rec["decision"]["action"] == "ACTIVE" and rec["decision"]["forced"] is True and rec["cm"]["ok"] is True
        assert "policy said NONE: actuation budget" in rec["decision"]["reason"]
        assert len(r1.requests) == 4 and r1.requests[-1].active == Active.ACTIVE
        m = c.get("/metrics").json()
        assert m["actuations_today"] == {"txcarrier0": 4} and m["actuation_budget_reached"] == 1 and m["actuation_alarm"] == 1
        assert sum("actuation ALARM" in r.getMessage() for r in caplog.records) == 1
        assert sum("actuation BUDGET reached" in r.getMessage() for r in caplog.records) == 1
        # the next sim day: the counter rolls over on the first PM of the day; a natural SLEEP is allowed again
        for mm in ("02:00", "02:30", "03:00"):
            _pm(c, active=Active.ACTIVE, prb=3.0, sim_time=f"2024-01-11T{mm}:00")
        # before the first cycle of the new day /metrics already reads the count as the engine will (0 for the new sim
        # day: no carrier at the alarm or the budget), while /policy still shows the raw stored pair of yesterday
        m = c.get("/metrics").json()
        assert m["actuations_today"] == {"txcarrier0": 0} and m["actuation_budget_reached"] == 0 and m["actuation_alarm"] == 0
        assert m["actuations_day"] == {"txcarrier0": DAY}
        pol = c.get("/policy").json()["txcarrier0"]
        assert (pol["actuations_day"], pol["actuations_today"]) == (DAY, 4)
        rec = _cycle(c)
        assert rec["decision"]["action"] == "SLEEP" and rec["decision"]["actuations_today"] == 0 and rec["decision"]["budget_reached"] is False, rec
        pol = c.get("/policy").json()["txcarrier0"]
        assert (pol["actuations_day"], pol["actuations_today"]) == ("2024-01-11", 1)
        m = c.get("/metrics").json()
        assert m["actuations_today"] == {"txcarrier0": 1} and m["actuation_budget_reached"] == 0 and m["actuation_alarm"] == 0
        assert m["actuation_alarm_carrier_days"] == 1 and m["actuation_budget_carrier_days"] == 1 and m["actuation_budget_refusals"] == 1


def test_failed_or_rejected_cm_writes_are_not_counted_and_zero_is_off(app_cfg, trained, monkeypatch):
    for r1 in (FakeR1(ok=False), FakeR1(fail=True)):
        with TestClient(create_app(app_cfg, model=trained, r1=r1)) as c:
            _pm(c, prb=3.0, sim_time=f"{DAY}T03:00:00")
            rec = _cycle(c)
            assert rec["decision"]["action"] == "SLEEP" and rec["cm"]["ok"] is False
            assert c.get("/metrics").json()["actuations_today"] == {"txcarrier0": 0}
            assert c.get("/policy").json()["txcarrier0"]["actuations_today"] == 0
    # a twin block stops the decision before any CM request: nothing counts either
    from ocp_auru.rapp.twin import TwinGuard

    class VetoTwin(TwinGuard):
        def evaluate_sleep(self, carrier):
            return False, {"enabled": True, "blocked": True, "coverage_rate_delta": -0.2}

    r1 = FakeR1()
    with TestClient(create_app(app_cfg, model=trained, r1=r1, twin=VetoTwin(app_cfg))) as c:
        _pm(c, prb=3.0, sim_time=f"{DAY}T03:00:00")
        assert "twin guardrail" in _cycle(c)["decision"]["reason"] and r1.requests == []
        assert c.get("/metrics").json()["actuations_today"] == {"txcarrier0": 0}
    # 0 = off: forced writes beyond any cap are never refused and never raise the alarm; the count is still exposed
    monkeypatch.setenv("RAPP_MAX_ACTUATIONS_PER_DAY", "0")
    monkeypatch.setenv("RAPP_ACTUATION_ALARM_PER_DAY", "0")
    off = Settings()
    r1 = FakeR1()
    with TestClient(create_app(off, model=trained, r1=r1)) as c:
        assert c.get("/health").json()["policy"]["max_actuations_per_day"] == 0
        _pm(c, prb=55.0, sim_time=f"{DAY}T12:00:00")
        for i in range(8):
            rec = _cycle(c, force="SLEEP" if i % 2 == 0 else "ACTIVE")
            assert rec["cm"]["ok"] is True and rec["decision"]["budget_reached"] is False and rec["decision"]["alarm"] is False
        m = c.get("/metrics").json()
        assert m["actuations_today"] == {"txcarrier0": 8} and m["actuation_alarm"] == 0 and m["actuation_budget_reached"] == 0
        assert m["actuation_alarm_carrier_days"] == 0 and m["actuation_budget_carrier_days"] == 0 and m["actuation_budget_refusals"] == 0


# ---- offline evaluation -----------------------------------------------------------------------------------------
TRAIN_DAYS, VALIDATE_DAYS, DAYS = 7, 3, 11


@pytest.fixture(scope="module")
def eval_cfg() -> Settings:
    """The primary run's settings: the shipped defaults (budget 6 / alarm 4, chatter control on) on the demo thresholds."""
    return replace(Settings(), rapp_sleep_threshold=0.7, rapp_wake_threshold=0.3, rapp_min_hold_s=20.0, rapp_history_len=64,
                   rapp_max_actuations_per_day=6, rapp_actuation_alarm_per_day=4)


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return synthetic.generate(n_cells=2, days=DAYS, seed=7)


@pytest.fixture(scope="module")
def params() -> ev.EvalParams:
    return ev.EvalParams(train_days=TRAIN_DAYS, validate_days=VALIDATE_DAYS, seed=7, sweep=False)


@pytest.fixture(scope="module")
def result(params, eval_cfg, table) -> ev.EvalResult:
    return ev.run_evaluation(params, eval_cfg, df=table)


DECISION_COLUMNS = ["cell", "ts", "entered_state", "state", "decision", "decision_reason", "p_low", "ruapp_wake", "demand_while_asleep",
                    "held", "variant_wh", "actuations_today_before", "actuations_today_after"]


def _actuations_per_cell_day(win: pd.DataFrame) -> pd.Series:
    dec = win["decision"].isin(["SLEEP", "ACTIVE"])
    return win.assign(n=dec.astype(int)).groupby(["cell", "day"])["n"].sum()


def test_defaults_leave_the_synthetic_decisions_unchanged(result, params, eval_cfg, table):
    """Budget 6 / alarm 4 (the shipped defaults) against the budget OFF on the same data and model: identical window
    frames for both variants (every decision, reason, state and charge), so the defaults change no decision."""
    off = ev.run_evaluation(params, replace(eval_cfg, rapp_max_actuations_per_day=0, rapp_actuation_alarm_per_day=0), df=table)
    for variant in ev.VARIANTS:
        a, b = result.replays[variant].windows, off.replays[variant].windows
        pd.testing.assert_frame_equal(a[DECISION_COLUMNS].reset_index(drop=True), b[DECISION_COLUMNS].reset_index(drop=True))
        assert not a["budget_held"].any() and not b["budget_held"].any() and not b["alarm_after"].any()
        # before/after: the count the decision saw (= the live Decision.actuations_today) and the count after the window's
        # applied write differ by exactly one on an actuation window and not at all otherwise; the undecided last window
        # of each cell has no 'before'
        decided = a.loc[a["decision"].notna()]
        assert decided["actuations_today_before"].notna().all() and a["actuations_today_before"].isna().sum() == a["cell"].nunique()
        applied = decided["decision"].isin(["SLEEP", "ACTIVE"])
        assert (decided.loc[applied, "actuations_today_after"] == decided.loc[applied, "actuations_today_before"] + 1).all()
        assert (decided.loc[~applied, "actuations_today_after"] == decided.loc[~applied, "actuations_today_before"]).all()
        assert "NONE:actuation-budget" not in result.replays[variant].decisions
        tot = result.summary["variants"][variant]["total"]
        assert tot["budget_reached_windows"] == 0 and tot["max_actuations_per_carrier_day"] <= 6
        assert tot["max_actuations_per_carrier_day"] == int(_actuations_per_cell_day(a).max())
    h = result.summary["headline"]
    assert h["budget_reached_windows_with_ruapp"] == 0 and h["budget_reached_windows_rapp_only"] == 0
    assert h["max_actuations_per_day"] == 6 and h["actuation_alarm_per_day"] == 4
    assert result.summary["params"]["rapp_max_actuations_per_day"] == 6 and result.summary["params"]["rapp_actuation_alarm_per_day"] == 4
    assert off.summary["headline"]["alarm_days_with_ruapp"] == 0 and off.summary["headline"]["max_actuations_per_day"] == 0
    lines = "\n".join(ev.headline_lines(result.summary))
    assert f"actuation budget     : {h['alarm_days_with_ruapp']} carrier-days hit the alarm, 0 windows held by the budget" in lines
    assert "max 6 per carrier-day, alarm at 4" in lines
    assert "max off per carrier-day, alarm at off" in "\n".join(ev.headline_lines(off.summary))


def test_budget_metric_identities_under_a_tight_budget(result, eval_cfg):
    """With a budget of 1 per cell-day the replay refuses actuations: the metrics obey their identities per variant /
    cell / day, the counter mirrors the app's (applied actuations per cell-day, reset at midnight), and after a refusal
    no further actuation is decided on that cell-day."""
    cell = "cell-0"
    sp, model = result.split, result.model
    thr = model.low_threshold
    tight = replace(eval_cfg, rapp_max_actuations_per_day=1, rapp_actuation_alarm_per_day=1, rapp_min_sleep_dwell_min=0.0)
    frames = {}
    for variant in ev.VARIANTS:
        rows = [ev.replay_cell(model, c, sp.train.loc[sp.train["cell_id"] == c], sp.validate.loc[sp.validate["cell_id"] == c], cfg=tight,
                               low_threshold=thr, demand_prb=thr, local_wake=(variant == "with_ruapp")) for c in ("cell-0", "cell-1")]
        frames[variant] = pd.concat([w for w, _ in rows], ignore_index=True)
        tally = sum((d for _, d in rows), start=__import__("collections").Counter())
        win = frames[variant]
        st = ev.variant_stats(win)
        held = win["decision_reason"] == "NONE:actuation-budget"
        assert st["budget_reached_windows"] == int(held.sum()) == tally["NONE:actuation-budget"]
        assert (win.loc[held, "budget_held"]).all() and (win.loc[~held, "budget_held"] == False).all()  # noqa: E712
        # the counter: applied actuations per cell-day == the day's peak count; never above the budget (no forced writes here)
        per_cd = _actuations_per_cell_day(win)
        peak = win.groupby(["cell", "day"])["actuations_today_after"].max()
        assert (per_cd == peak).all() and int(peak.max()) == st["max_actuations_per_carrier_day"] <= 1
        # alarm carrier-days = cell-days whose count reached the alarm threshold (1 here: every cell-day with an actuation)
        assert st["alarm_days"] == int((per_cd >= 1).sum()) == int(win.groupby(["cell", "day"])["alarm_after"].any().sum())
        # once refused, nothing more is actuated on that cell-day; the carrier stays in the state it was in
        for (c, day), g in win.groupby(["cell", "day"]):
            g = g.sort_values("ts")
            first_refusal = g.index[g["decision_reason"] == "NONE:actuation-budget"]
            if len(first_refusal):
                after = g.loc[g.index > first_refusal[0]]
                assert not after["decision"].isin(["SLEEP", "ACTIVE"]).any(), (c, day)
                assert (g.loc[first_refusal, "actuations_today_after"] == 1).all() and (g.loc[first_refusal, "actuations_today_before"] == 1).all()
        # the refused decisions exist on this data (the budget of 1 is below a clean night's 2)
        assert st["budget_reached_windows"] >= 1
        # a budget-refused PRE-EMPTIVE WAKE is a held wake: the next window is `held`, so a ruApp wake
        # (with ruApp) or the demand slept through (rApp-only) there is attributed to the governor; a budget-refused
        # SLEEP (carrier awake) holds nothing. No dwell / confirmation here, so the budget is the only holder.
        refused = win["decision_reason"] == "NONE:actuation-budget"
        held_prev = (refused & (win["state"] == "SLEEP")).groupby(win["cell"]).shift(1, fill_value=False)
        assert (win["held"] == held_prev).all() and int((refused & (win["state"] == "ACTIVE")).sum()) >= 1
        assert (win.loc[win["held"], "entered_state"] == "SLEEP").all() and st["held_windows"] == int(held_prev.sum()) >= 1
        assert st["ruapp_wakes_while_held"] == int((win["ruapp_wake"] & win["held"]).sum())
        assert st["demand_while_asleep_while_held"] == int((win["demand_while_asleep"] & win["held"]).sum())
        if variant == "with_ruapp":
            assert st["ruapp_wakes_while_held"] >= 1       # the ruApp served the demand the refused wake would have met
        else:
            assert st["demand_while_asleep_while_held"] >= 1
        assert ev.chatter_flags_on(tight)                  # the false-sleep rate label says it includes held wakes
    # the aggregation carries the metrics per cell / day / cell-day and they sum consistently
    agg = ev.aggregate({v: ev.ReplayResult(variant=v, windows=w) for v, w in frames.items()})
    for variant in ev.VARIANTS:
        tot, v = agg[variant]["total"], agg[variant]
        assert sum(c["budget_reached_windows"] for c in v["per_cell"].values()) == tot["budget_reached_windows"]
        assert sum(d["budget_reached_windows"] for d in v["per_day"]) == tot["budget_reached_windows"]
        assert sum(c["alarm_days"] for c in v["per_cell"].values()) == sum(d["alarm_days"] for d in v["per_day"]) == tot["alarm_days"]
        assert sum(cd["alarm_days"] for cd in v["per_cell_day"]) == tot["alarm_days"]
        assert max(cd["max_actuations_per_carrier_day"] for cd in v["per_cell_day"]) == tot["max_actuations_per_carrier_day"]
    # the budget only ever REMOVES actuations relative to the budget-off replay of the same cell
    base, _ = ev.replay_cell(model, cell, sp.train.loc[sp.train["cell_id"] == cell], sp.validate.loc[sp.validate["cell_id"] == cell],
                             cfg=replace(tight, rapp_max_actuations_per_day=0), low_threshold=thr, demand_prb=thr, local_wake=True)
    w0 = frames["with_ruapp"].loc[frames["with_ruapp"]["cell"] == cell]
    assert int(w0["decision"].isin(["SLEEP", "ACTIVE"]).sum()) < int(base["decision"].isin(["SLEEP", "ACTIVE"]).sum())
    assert not base["budget_held"].any() and not base["held"].any()   # budget off, no dwell: nothing holds a wake


def test_alarm_days_and_grid_sweep_rows_carry_the_budget(params, eval_cfg, table):
    """The alarm at 1 flags every cell-day with an actuation (budget off: nothing is refused); grid and sweep rows
    carry the three metrics; the shipped alarm 4 stays quiet on a clean synthetic night."""
    alarm1 = replace(eval_cfg, rapp_max_actuations_per_day=0, rapp_actuation_alarm_per_day=1)
    p = replace(params, grid=True, grid_dwell_min=(eval_cfg.rapp_min_sleep_dwell_min,), grid_confirm_windows=(1,), grid_wake_margins=(0.0,),
                grid_profile_scales=(eval_cfg.rapp_profile_scale,), sweep=True, sleep_thresholds=(0.7,), low_quantiles=(0.25,))
    r = ev.run_evaluation(p, alarm1, df=table)
    for variant in ev.VARIANTS:
        win = r.replays[variant].windows
        tot = r.summary["variants"][variant]["total"]
        assert tot["budget_reached_windows"] == 0 and tot["alarm_days"] == int((_actuations_per_cell_day(win) >= 1).sum()) >= 1
    h = r.summary["headline"]
    assert h["alarm_days_with_ruapp"] == r.summary["variants"]["with_ruapp"]["total"]["alarm_days"]
    assert h["alarm_days_per_cell_with_ruapp"] == {c: st["alarm_days"] for c, st in r.summary["variants"]["with_ruapp"]["per_cell"].items()}
    assert sum(h["alarm_days_per_cell_with_ruapp"].values()) == h["alarm_days_with_ruapp"]
    g = r.summary["grid"]
    assert len(g) == 1 and g[0]["primary"] is True
    for key in ("budget_reached_windows", "alarm_days", "max_actuations_per_carrier_day", "max_actuations_per_day", "actuation_alarm_per_day",
                "budget_reached_windows_rapp_only", "alarm_days_rapp_only"):
        assert key in g[0], key
    assert g[0]["alarm_days"] == h["alarm_days_with_ruapp"] and g[0]["max_actuations_per_day"] == 0 and g[0]["actuation_alarm_per_day"] == 1
    for c in g[0]["per_cell"].values():
        assert {"budget_reached_windows", "alarm_days", "max_actuations_per_carrier_day"} <= set(c)
    assert len(r.summary["sweep"]) == 1 and r.summary["sweep"][0]["alarm_days"] == h["alarm_days_with_ruapp"]
    assert r.summary["sweep"][0]["budget_reached_windows"] == 0 and "max_actuations_per_carrier_day" in r.summary["sweep"][0]
    out = ev.write_outputs(r, Path(r.cfg.var_dir) / "budget-test", plots=False)
    report, grid_md = (out / "report.md").read_text(), (out / "grid.md").read_text()
    assert "budget-held windows" in report and "alarm carrier-days" in report and "actuation budget (max applied" in report
    assert "budget-held windows" in grid_md and "alarm carrier-days" in grid_md and "not a grid axis" in grid_md
    assert "/Users/" not in report and "/Users/" not in grid_md
    lines = "\n".join(ev.headline_lines(r.summary))
    assert f"actuation budget     : {h['alarm_days_with_ruapp']} carrier-days hit the alarm, 0 windows held by the budget" in lines


def test_cli_budget_flags_reach_the_primary_run_and_are_validated(tmp_path):
    out = tmp_path / "eval"
    base = [sys.executable, str(ROOT / "scripts" / "eval_energy.py"), "--data-kind", "synthetic", "--days", str(DAYS), "--cells", "2",
            "--train-days", str(TRAIN_DAYS), "--validate-days", str(VALIDATE_DAYS), "--seed", "7", "--no-sweep", "--no-plots"]
    proc = subprocess.run(base + ["--max-actuations-per-day", "1", "--actuation-alarm-per-day", "1", "--min-sleep-dwell-min", "0",
                                  "--out", str(out)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    summary = json.loads((out / "summary.json").read_text())
    p, h = summary["params"], summary["headline"]
    assert p["rapp_max_actuations_per_day"] == 1 and p["rapp_actuation_alarm_per_day"] == 1
    assert h["budget_reached_windows_with_ruapp"] >= 1 and h["alarm_days_with_ruapp"] >= 1 and h["max_actuations_per_carrier_day_with_ruapp"] == 1
    assert (f"actuation budget     : {h['alarm_days_with_ruapp']} carrier-days hit the alarm, {h['budget_reached_windows_with_ruapp']} "
            f"windows held by the budget  (max 1 per carrier-day, alarm at 1") in proc.stdout
    assert "NONE:actuation-budget" in json.dumps(summary["variants"]["with_ruapp"]["decisions_by_reason"])
    report = (out / "report.md").read_text()
    assert "budget-held windows" in report and "/Users/" not in report
    for bad, flag in (("--max-actuations-per-day", "--max-actuations-per-day"), ("--actuation-alarm-per-day", "--actuation-alarm-per-day")):
        proc = subprocess.run(base + [bad, "-1", "--out", str(tmp_path / "bad")], cwd=ROOT, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 2 and flag in proc.stderr
