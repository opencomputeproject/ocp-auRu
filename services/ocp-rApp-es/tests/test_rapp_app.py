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
"""es-rApp HTTP service end to end in-process: PM delivery -> cycle -> CM request; R1 client."""
from __future__ import annotations

import json
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from ocp_auru.common.config import Settings
from ocp_auru.common.events import (Active, CarrierPm, CarrierState, CmCarrierRequest, CmCarrierResponse, PmReport, VesEnvelope,
                                WakeLatencyStats)
from ocp_auru.common.ves import measurement_event, notification_event
from ocp_auru.rapp import model as model_mod
from ocp_auru.rapp.app import RappState, create_app
from ocp_auru.rapp.data import synthetic
from ocp_auru.rapp.r1 import INFO_TYPE_ID, JOB_ID, SERVICE_NAME, R1Client
from ocp_auru.rapp.twin import TwinGuard


# joblib pickling under NumPy 2.5 emits a shape-assignment DeprecationWarning; not ours.
pytestmark = pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")


class FakeR1:
    def __init__(self, ok=True, fail=False, delay_s=0.0):
        self.requests: list[CmCarrierRequest] = []
        self.ok, self.fail, self.started, self.stopped, self.delay_s = ok, fail, False, False, delay_s

    def start_background(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def status(self):
        return {"registered": True, "job_created": True, "fake": True}

    def cm_request(self, req: CmCarrierRequest) -> CmCarrierResponse:
        self.requests.append(req)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail:
            raise httpx.ConnectError("smo down")
        return CmCarrierResponse(ok=self.ok, ru_id=req.ru_id, carrier=req.carrier, active=req.active, cm_rtt_ms=12.5, netconf_rtt_ms=4.25, detail="fake")


@pytest.fixture(scope="module")
def trained():
    return model_mod.train(synthetic.generate(n_cells=2, days=10, seed=11), seed=11)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("VAR_DIR", str(tmp_path))
    monkeypatch.setenv("RAPP_PORT", "28101")
    monkeypatch.setenv("SMO_PORT", "28103")
    monkeypatch.setenv("RAPP_AUTO_CYCLE", "0")
    monkeypatch.setenv("RAPP_MIN_HOLD_S", "0")
    monkeypatch.setenv("TWIN_URL", "")
    return Settings()


def pm_envelope(active=Active.ACTIVE, state=CarrierState.READY, prb=3.0, sim_time="2024-01-10T03:00:00", wake_events=0, lat=None):
    car = CarrierPm(name="txcarrier0", active=active, state=state, tx_power_dbm=24.0, prb_util_dl_pct=prb, prb_util_ul_pct=prb / 2,
                    rrc_conn=1, power_w_avg=255.0, energy_wh=0.35, prach_detections=1, wake_events=wake_events,
                    wake_latency_us=WakeLatencyStats(last=lat, p50=lat, max=lat, count=wake_events))
    rep = PmReport(ru_id="oru-test", interval_s=5.0, sim_time_iso=sim_time, carriers=[car], epoch_us=1_700_000_000_000_000)
    return measurement_event(rep).model_dump(mode="json")


def test_health_pm_cycle_policy_decisions_metrics(cfg, trained):
    r1 = FakeR1()
    with TestClient(create_app(cfg, model=trained, r1=r1, twin=TwinGuard(cfg))) as c:
        h = c.get("/health").json()
        assert h["ok"] is True and h["model"]["auc"] == trained.metrics["auc"] and h["r1"]["fake"] and h["profile"] == cfg.profile
        assert r1.started

        for i, mm in enumerate(("02:00", "02:30", "03:00")):
            r = c.post("/r1/pm-delivery", json=pm_envelope(prb=3.0 + i * 0.1, sim_time=f"2024-01-10T{mm}:00"))
            assert r.status_code == 200 and r.json()["carriers"] == 1
        pm = c.get("/pm").json()
        assert pm["ru_id"] == "oru-test" and pm["carriers"]["txcarrier0"]["history_len"] == 3
        assert pm["carriers"]["txcarrier0"]["latest"]["sim_time"] == "2024-01-10T03:00:00"

        r = c.post("/cycle", json={})
        assert r.status_code == 200
        d = r.json()["decisions"][0]["decision"]
        assert d["action"] == "SLEEP", d
        assert d["p_low"] >= cfg.rapp_sleep_threshold and d["forced"] is False and d["sim_time"] == "2024-01-10T03:00:00"
        assert d["envelope"]["wake_on_prach"] is True
        req = r1.requests[-1]
        assert req.active == Active.SLEEP and req.ruapp_policy.wake_on_prach is True and req.carrier == "txcarrier0" and req.ru_id == "oru-test"
        assert json.loads(req.model_dump_json())["active"] == "SLEEP"

        pol = c.get("/policy").json()["txcarrier0"]
        assert pol["active"] == "SLEEP" and pol["envelope"]["wake_on_prach"] is True and pol["last_cm"]["cm_rtt_ms"] == 12.5
        decs = c.get("/decisions").json()
        assert len(decs) == 1 and decs[0]["cm"]["ok"] is True and decs[0]["twin"] == {"enabled": False}

        # PM reports the RU woke the carrier locally: wake stats flow into /metrics
        c.post("/r1/pm-delivery", json=pm_envelope(active=Active.ACTIVE, prb=3.0, sim_time="2024-01-10T03:30:00", wake_events=1, lat=42.5))
        m = c.get("/metrics").json()
        assert m["last_cm_rtt_ms"] == 12.5 and m["last_netconf_rtt_ms"] == 4.25
        assert m["wake_latency_us"]["txcarrier0"]["last"] == 42.5 and m["wake_events"]["txcarrier0"] == 1
        assert m["decisions_count"] == 1 and m["sleep_windows"] == 1 and m["pm_reports"] == 4
    assert r1.stopped


def test_force_path_and_inactive_guard(cfg, trained):
    r1 = FakeR1()
    with TestClient(create_app(cfg, model=trained, r1=r1)) as c:
        c.post("/r1/pm-delivery", json=pm_envelope(prb=55.0, sim_time="2024-01-10T19:30:00"))
        auto = c.post("/cycle", json={}).json()["decisions"][0]["decision"]
        assert auto["action"] == "NONE" and r1.requests == []
        forced = c.post("/cycle", json={"force": "SLEEP"}).json()
        assert forced["forced"] == "SLEEP"
        d = forced["decisions"][0]["decision"]
        assert d["action"] == "SLEEP" and d["forced"] is True and "forced" in d["reason"]
        assert r1.requests[-1].active == Active.SLEEP
        forced = c.post("/cycle", json={"force": "ACTIVE", "carrier": "txcarrier0"}).json()
        assert forced["decisions"][0]["decision"]["action"] == "ACTIVE" and r1.requests[-1].active == Active.ACTIVE
        assert c.post("/cycle", json={"force": "INACTIVE"}).status_code == 403
        assert c.post("/cycle", json={"carrier": "nope"}).status_code == 404
        assert c.post("/cycle", json={"force": "BOGUS"}).status_code == 422


def test_twin_veto_and_cm_failure_are_recorded(cfg, trained, monkeypatch):
    class VetoTwin(TwinGuard):
        def evaluate_sleep(self, carrier):
            return False, {"enabled": True, "blocked": True, "coverage_rate_delta": -0.2}

    r1 = FakeR1()
    with TestClient(create_app(cfg, model=trained, r1=r1, twin=VetoTwin(cfg))) as c:
        c.post("/r1/pm-delivery", json=pm_envelope(prb=3.0))
        d = c.post("/cycle", json={}).json()["decisions"][0]
        assert d["decision"]["action"] == "NONE" and "twin guardrail" in d["decision"]["reason"] and d["twin"]["blocked"]
        assert r1.requests == []

    r1 = FakeR1(fail=True)
    with TestClient(create_app(cfg, model=trained, r1=r1)) as c:
        c.post("/r1/pm-delivery", json=pm_envelope(prb=3.0))
        d = c.post("/cycle", json={}).json()["decisions"][0]
        assert d["decision"]["action"] == "SLEEP" and d["cm"]["ok"] is False and "smo down" in d["cm"]["error"]
        assert c.get("/metrics").json()["sleep_windows"] == 0


def test_sleeping_carrier_is_woken_preemptively_from_the_hour_profile(cfg, trained):
    """HLD §3.7: SLEEP + load forecast to return -> ACTIVE. A sleeping carrier reports PRB 0, so the
    forecast must come from the expected load at that time of day, not from its own counters.
    Timestamps use the RU's simulated-clock format (tz-aware, seconds) throughout."""
    r1 = FakeR1()
    with TestClient(create_app(cfg, model=trained, r1=r1)) as c:
        # deep night: stay asleep
        c.post("/r1/pm-delivery", json=pm_envelope(active=Active.SLEEP, state=CarrierState.DISABLED, prb=0.0, sim_time="2026-01-01T02:17:15Z"))
        d = c.post("/cycle", json={}).json()["decisions"][0]["decision"]
        assert d["action"] == "NONE" and d["forecast_basis"] == "expected-hour-profile" and d["p_low"] >= cfg.rapp_wake_threshold, d
        assert r1.requests == []
        # morning ramp through midday (PM still says PRB 0): pre-emptive ACTIVE
        for hh in ("08:00", "10:00", "12:00"):
            c.post("/r1/pm-delivery", json=pm_envelope(active=Active.SLEEP, state=CarrierState.DISABLED, prb=0.0, sim_time=f"2026-01-01T{hh}:00Z"))
            d = c.post("/cycle", json={}).json()["decisions"][0]["decision"]
            assert d["action"] == "ACTIVE" and "pre-emptive" in d["reason"] and d["forecast_basis"] == "expected-hour-profile", (hh, d)
            assert r1.requests[-1].active == Active.ACTIVE
        # an ACTIVE carrier still uses its own PM window
        c.post("/r1/pm-delivery", json=pm_envelope(prb=3.0, sim_time="2026-01-02T03:00:00Z"))
        d = c.post("/cycle", json={}).json()["decisions"][0]["decision"]
        assert d["forecast_basis"] == "pm-window" and d["action"] == "SLEEP", d


def test_state_lock_is_released_during_the_cm_request(cfg, trained):
    """PM delivery and the read views must not wait for a slow SMO/RU hop (RAPP_CM_TIMEOUT_S=30 s);
    a second cycle for the same carrier while one is in flight is skipped, not actuated twice."""
    state = RappState(cfg)
    r1 = FakeR1(delay_s=0.8)
    twin = TwinGuard(cfg)
    state.ingest(VesEnvelope.model_validate(pm_envelope(prb=3.0, sim_time="2024-01-10T03:00:00")))
    results: dict = {}
    t = threading.Thread(target=lambda: results.setdefault("cycle", state.run_cycle(trained, r1, twin)))
    t.start()
    deadline = time.monotonic() + 2.0
    while not r1.requests and time.monotonic() < deadline:
        time.sleep(0.005)
    assert r1.requests, "cycle did not reach the CM request"
    t0 = time.perf_counter()
    ingest = state.ingest(VesEnvelope.model_validate(pm_envelope(prb=3.1, sim_time="2024-01-10T03:30:00")))
    pm, pol, met = state.pm_view(), state.policy_view(trained), state.metrics_view()
    elapsed = time.perf_counter() - t0
    assert ingest["accepted"] and pm["reports"] == 2 and elapsed < 0.3, f"reads blocked for {elapsed:.2f}s"
    assert met["in_flight"] == ["txcarrier0"] and pol["txcarrier0"]["last_cm"] is None
    second = state.run_cycle(trained, r1, twin)  # concurrent cycle: skipped, no second CM request
    assert "in flight" in second[0]["error"] and len(r1.requests) == 1
    t.join(5.0)
    rec = results["cycle"][0]
    assert rec["cm"]["ok"] is True and rec["decision"]["action"] == "SLEEP"
    assert state.metrics_view()["in_flight"] == [] and state.metrics_view()["sleep_windows"] == 1
    assert state.pm_latest["txcarrier0"]["active"] == "SLEEP"  # optimistic update committed after the call
    assert len(state.decisions) == 1


def test_notification_updates_carrier_state(cfg, trained):
    with TestClient(create_app(cfg, model=trained, r1=FakeR1())) as c:
        c.post("/r1/pm-delivery", json=pm_envelope(active=Active.SLEEP, state=CarrierState.DISABLED))
        n = notification_event("oru-test", "tx-array-carriers-state-change", {"carrier": "txcarrier0", "active": "ACTIVE", "state": "READY"}, 1)
        assert c.post("/r1/pm-delivery", json=n.model_dump(mode="json")).json()["domain"] == "notification"
        latest = c.get("/pm").json()["carriers"]["txcarrier0"]["latest"]
        assert latest["active"] == "ACTIVE" and latest["state"] == "READY"
        assert c.get("/metrics").json()["notifications"] == 1


# ---- R1 client against an in-memory platform (httpx.MockTransport) ---------------------------
def test_r1_client_registration_and_cm_request(cfg):
    seen: dict[str, list] = {"catalogue": [], "ics": [], "cm": []}
    ready = {"ok": False}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == f"/services/{SERVICE_NAME}" and request.method == "PUT":
            seen["catalogue"].append(json.loads(request.content))
            return httpx.Response(201 if ready["ok"] else 503, json={})
        if path == f"/data-consumer/v1/info-jobs/{JOB_ID}" and request.method == "PUT":
            seen["ics"].append(json.loads(request.content))
            return httpx.Response(200 if ready["ok"] else 404, json={})
        if path == "/o1/cm/carrier" and request.method == "POST":
            body = json.loads(request.content)
            seen["cm"].append(body)
            return httpx.Response(200, json={"ok": True, "ru_id": body["ru_id"], "carrier": body["carrier"], "active": body["active"],
                                             "cm_rtt_ms": 9.5, "netconf_rtt_ms": 3.0, "detail": "mock"})
        return httpx.Response(404)

    r1 = R1Client(cfg, transport=httpx.MockTransport(handler))
    assert r1.try_once() is False
    st = r1.status()
    assert st["registered"] is False and st["job_created"] is False and st["last_error"] and st["attempts"] == 1
    assert len(seen["catalogue"]) == 1 and len(seen["ics"]) == 1  # both platform services are tried every attempt
    ready["ok"] = True
    assert r1.try_once() is True
    st = r1.status()
    assert st["registered"] and st["job_created"] and st["attempts"] == 2 and st["last_error"] is None
    assert seen["catalogue"][-1] == {"version": "0.1.0", "display_name": "Energy Saving rApp", "description": seen["catalogue"][-1]["description"]}
    job = seen["ics"][-1]
    assert job["info_type_id"] == INFO_TYPE_ID and job["job_owner"] == SERVICE_NAME and job["job_definition"] == {}
    assert job["job_result_uri"] == f"{cfg.rapp_public_url}/r1/pm-delivery" and job["status_notification_uri"] == ""
    assert cfg.rapp_public_url.endswith(":28101")

    resp = r1.cm_request(CmCarrierRequest(ru_id="oru-test", carrier="txcarrier0", active=Active.SLEEP, reason="t"))
    assert isinstance(resp, CmCarrierResponse) and resp.ok and resp.active == Active.SLEEP and resp.cm_rtt_ms == 9.5
    assert seen["cm"][-1]["active"] == "SLEEP" and seen["cm"][-1]["ruapp_policy"]["wake_on_prach"] is True
    r1.stop()


def test_r1_background_retry_until_platform_up(cfg, monkeypatch):
    monkeypatch.setenv("RAPP_R1_RETRY_S", "0.05")
    cfg = Settings()
    state = {"up": False, "n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        return httpx.Response(200 if state["up"] else 503, json={})

    r1 = R1Client(cfg, transport=httpx.MockTransport(handler))
    r1.start_background()
    import time
    time.sleep(0.2)
    assert not r1.registered and state["n"] >= 2
    state["up"] = True
    deadline = time.time() + 3
    while time.time() < deadline and not (r1.registered and r1.job_created):
        time.sleep(0.02)
    assert r1.registered and r1.job_created
    r1.stop()


def test_twin_verdict_recorded_and_forced_sleep_overrides_a_block(cfg, trained, monkeypatch):
    """ADR-7: every SLEEP carries the twin verdict (cell, deltas, rung, latency) in /decisions[i].twin and
    /metrics counts checks/blocks; a forced SLEEP is an explicit operator override of a block (recorded)."""
    monkeypatch.setenv("TWIN_URL", "http://twin.invalid:1")   # enabled; the fake below never talks to it
    cfg = Settings()
    class RecordingTwin(TwinGuard):
        def __init__(self, cfg, cov):
            super().__init__(cfg)
            self.cov = cov

        def evaluate_sleep(self, carrier):
            self.checks += 1
            blocked = self.cov < self.cfg.twin_coverage_drop_max
            self.blocks += int(blocked)
            d = {"enabled": True, "cell_id": "g-s1-a-n78", "cell_source": "auto:/v1/cells[0]", "status": "ok", "latency_ms": 7.5,
                 "rung": "simulation, generic 3GPP defaults (uncalibrated)", "coverage_rate_delta": self.cov, "threshold": self.cfg.twin_coverage_drop_max,
                 "delta_vs_baseline": {"rsrp_p5": -0.3, "sinr_p5": -0.1, "outage_rate": 0.0, "coverage_rate": self.cov},
                 "blocked": blocked, "verdict": "blocked" if blocked else "allowed"}
            self.last_verdict = {"verdict": d["verdict"], "cell_id": d["cell_id"], "rung": d["rung"]}
            return (not blocked), d

    r1 = FakeR1()
    with TestClient(create_app(cfg, model=trained, r1=r1, twin=RecordingTwin(cfg, cov=-0.01))) as c:
        c.post("/r1/pm-delivery", json=pm_envelope(prb=3.0))
        rec = c.post("/cycle", json={}).json()["decisions"][0]
        assert rec["decision"]["action"] == "SLEEP" and rec["twin"]["verdict"] == "allowed" and rec["twin"]["cell_id"] == "g-s1-a-n78"
        assert "uncalibrated" in rec["twin"]["rung"] and rec["twin"]["delta_vs_baseline"]["coverage_rate"] == -0.01 and rec["twin"]["latency_ms"] == 7.5
        assert len(r1.requests) == 1
        m = c.get("/metrics").json()
        assert m["twin_checks"] == 1 and m["twin_blocks"] == 0 and m["last_twin_verdict"]["verdict"] == "allowed"
        assert c.get("/health").json()["twin"]["enabled"] is True

    r1 = FakeR1()
    with TestClient(create_app(cfg, model=trained, r1=r1, twin=RecordingTwin(cfg, cov=-0.2))) as c:
        c.post("/r1/pm-delivery", json=pm_envelope(prb=3.0))
        rec = c.post("/cycle", json={}).json()["decisions"][0]
        assert rec["decision"]["action"] == "NONE" and rec["twin"]["blocked"] is True and "g-s1-a-n78" in rec["decision"]["reason"]
        assert r1.requests == []
        rec = c.post("/cycle", json={"force": "SLEEP", "carrier": "txcarrier0"}).json()["decisions"][0]
        assert rec["decision"]["action"] == "SLEEP" and rec["decision"]["forced"] is True
        assert rec["twin"]["blocked"] is True and rec["twin"]["overridden"] is True and "overridden" in rec["decision"]["reason"]
        assert len(r1.requests) == 1 and r1.requests[0].active == Active.SLEEP
        m = c.get("/metrics").json()
        assert m["twin_checks"] == 2 and m["twin_blocks"] == 2


# ---- chatter control: PolicyState fields flow through the app and its views ----
def _policy_settings(cfg):
    return {"sleep_threshold": cfg.rapp_sleep_threshold, "wake_threshold": cfg.rapp_wake_threshold,
            "wake_margin": cfg.rapp_wake_margin, "wake_threshold_effective": cfg.rapp_wake_threshold_effective,
            "wake_confirm_windows": cfg.rapp_wake_confirm_windows, "min_sleep_dwell_min": cfg.rapp_min_sleep_dwell_min,
            "min_hold_s": cfg.rapp_min_hold_s, "allow_inactive": cfg.rapp_allow_inactive,
            "profile_scale": cfg.rapp_profile_scale, "post_local_wake_hold_min": cfg.rapp_post_local_wake_hold_min,
            "max_actuations_per_day": cfg.rapp_max_actuations_per_day, "actuation_alarm_per_day": cfg.rapp_actuation_alarm_per_day}


def test_policy_state_fields_flow_and_are_exposed(cfg, trained):
    """A SLEEP actuation starts the dwell clock at the PM simTime and resets the votes; inside the shipped dwell
    (RAPP_MIN_SLEEP_DWELL_MIN, on by default) a wake is held whatever the forecast; a pre-emptive
    ACTIVE past it clears the clock; /policy and /health expose the fields and the settings actually in use."""
    r1 = FakeR1()
    dwell = cfg.rapp_min_sleep_dwell_min
    with TestClient(create_app(cfg, model=trained, r1=r1)) as c:
        h = c.get("/health").json()
        assert h["policy"] == _policy_settings(cfg)
        # the exposed settings are the shipped defaults (pinned in test_rapp_policy): dwell on, one confirmation window
        assert h["policy"]["min_sleep_dwell_min"] == dwell > 0 and h["policy"]["wake_confirm_windows"] == cfg.rapp_wake_confirm_windows == 1
        c.post("/r1/pm-delivery", json=pm_envelope(prb=3.0, sim_time="2024-01-10T03:00:00"))
        pol = c.get("/policy").json()["txcarrier0"]
        assert pol["sleep_entered_sim"] is None and pol["wake_votes"] == 0 and pol["settings"] == h["policy"]
        d = c.post("/cycle", json={}).json()["decisions"][0]["decision"]
        assert d["action"] == "SLEEP" and d["wake_votes"] == 0
        pol = c.get("/policy").json()["txcarrier0"]
        assert pol["sleep_entered_sim"] == "2024-01-10T03:00:00" and pol["wake_votes"] == 0 and pol["last_action"] == "SLEEP"
        # 30 min of simulated time later the shipped dwell holds (no CM request); the clock and the votes are untouched
        c.post("/r1/pm-delivery", json=pm_envelope(active=Active.SLEEP, state=CarrierState.DISABLED, prb=0.0, sim_time="2024-01-10T03:30:00"))
        d = c.post("/cycle", json={}).json()["decisions"][0]["decision"]
        assert d["action"] == "NONE" and f"min sleep dwell: 30 of {dwell:g} min" in d["reason"] and d["wake_votes"] == 0, d
        assert len(r1.requests) == 1
        pol = c.get("/policy").json()["txcarrier0"]
        assert pol["sleep_entered_sim"] == "2024-01-10T03:00:00" and pol["last_action"] == "SLEEP"
        # asleep at the morning ramp, 7 h past the dwell: pre-emptive ACTIVE (1 window, no margin) clears the dwell clock
        c.post("/r1/pm-delivery", json=pm_envelope(active=Active.SLEEP, state=CarrierState.DISABLED, prb=0.0, sim_time="2024-01-10T10:00:00"))
        d = c.post("/cycle", json={}).json()["decisions"][0]["decision"]
        assert d["action"] == "ACTIVE" and d["wake_votes"] == 1 and r1.requests[-1].active == Active.ACTIVE
        pol = c.get("/policy").json()["txcarrier0"]
        assert pol["sleep_entered_sim"] is None and pol["wake_votes"] == 0 and pol["last_action"] == "ACTIVE"


def test_dwell_and_confirmation_windows_through_the_app(cfg, trained, monkeypatch):
    monkeypatch.setenv("RAPP_WAKE_CONFIRM_WINDOWS", "2")
    monkeypatch.setenv("RAPP_MIN_SLEEP_DWELL_MIN", "60")
    cfg2 = Settings()
    r1 = FakeR1()
    with TestClient(create_app(cfg2, model=trained, r1=r1)) as c:
        assert c.get("/health").json()["policy"]["min_sleep_dwell_min"] == 60.0
        c.post("/r1/pm-delivery", json=pm_envelope(prb=3.0, sim_time="2024-01-10T03:00:00"))
        assert c.post("/cycle", json={}).json()["decisions"][0]["decision"]["action"] == "SLEEP"
        assert c.get("/policy").json()["txcarrier0"]["sleep_entered_sim"] == "2024-01-10T03:00:00"
        # 30 min of simulated time later the dwell holds, whatever the forecast
        c.post("/r1/pm-delivery", json=pm_envelope(active=Active.SLEEP, state=CarrierState.DISABLED, prb=0.0, sim_time="2024-01-10T03:30:00"))
        d = c.post("/cycle", json={}).json()["decisions"][0]["decision"]
        assert d["action"] == "NONE" and "min sleep dwell: 30 of 60 min" in d["reason"] and d["wake_votes"] == 0
        # past the dwell at the morning ramp: first a pending vote (no CM request), then the confirmed wake
        c.post("/r1/pm-delivery", json=pm_envelope(active=Active.SLEEP, state=CarrierState.DISABLED, prb=0.0, sim_time="2024-01-10T10:00:00"))
        d = c.post("/cycle", json={}).json()["decisions"][0]["decision"]
        assert d["action"] == "NONE" and "wake vote 1 of 2" in d["reason"] and d["wake_votes"] == 1, d
        assert len(r1.requests) == 1 and c.get("/policy").json()["txcarrier0"]["wake_votes"] == 1
        c.post("/r1/pm-delivery", json=pm_envelope(active=Active.SLEEP, state=CarrierState.DISABLED, prb=0.0, sim_time="2024-01-10T10:30:00"))
        d = c.post("/cycle", json={}).json()["decisions"][0]["decision"]
        assert d["action"] == "ACTIVE" and "confirmed 2 of 2" in d["reason"], d
        assert r1.requests[-1].active == Active.ACTIVE and len(r1.requests) == 2
        pol = c.get("/policy").json()["txcarrier0"]
        assert pol["wake_votes"] == 0 and pol["sleep_entered_sim"] is None


# ---- one vote per PM window, clock hygiene, local-wake hold, both estimators ----
def _pol(c):
    return c.get("/policy").json()["txcarrier0"]


def _cycle(c, **body):
    return c.post("/cycle", json=body).json()["decisions"][0]["decision"]


def test_two_cycles_on_one_pm_window_cast_one_vote(cfg, trained, monkeypatch):
    """RAPP_WAKE_CONFIRM_WINDOWS counts distinct PM windows, not decisions: the auto cycle plus a POST /cycle
    (or a cycle shorter than the PM interval) on the same simTime must not confirm a wake on one window of evidence."""
    monkeypatch.setenv("RAPP_WAKE_CONFIRM_WINDOWS", "2")
    cfg2 = Settings()
    r1 = FakeR1()
    with TestClient(create_app(cfg2, model=trained, r1=r1)) as c:
        c.post("/r1/pm-delivery", json=pm_envelope(active=Active.SLEEP, state=CarrierState.DISABLED, prb=0.0, sim_time="2024-01-10T10:00:00"))
        d1 = _cycle(c)
        assert d1["action"] == "NONE" and "wake vote 1 of 2" in d1["reason"] and d1["wake_votes"] == 1 and d1["vote_sim"] == "2024-01-10T10:00:00"
        d2 = _cycle(c)                                                   # same PM window, no new PM
        assert d2["action"] == "NONE" and d2["wake_votes"] == 1 and "already counted on this PM window" in d2["reason"], d2
        d3 = _cycle(c)
        assert d3["action"] == "NONE" and d3["wake_votes"] == 1 and r1.requests == []
        pol = _pol(c)
        assert pol["wake_votes"] == 1 and pol["last_vote_sim"] == "2024-01-10T10:00:00"
        # the next PM window casts the second vote and confirms
        c.post("/r1/pm-delivery", json=pm_envelope(active=Active.SLEEP, state=CarrierState.DISABLED, prb=0.0, sim_time="2024-01-10T10:30:00"))
        d4 = _cycle(c)
        assert d4["action"] == "ACTIVE" and "confirmed 2 of 2 consecutive windows" in d4["reason"] and d4["vote_sim"] == "2024-01-10T10:30:00"
        assert [r.active for r in r1.requests] == [Active.ACTIVE]
        pol = _pol(c)
        assert pol["wake_votes"] == 0 and pol["last_vote_sim"] is None and pol["sleep_entered_sim"] is None


def test_failed_cm_write_does_not_start_the_dwell_clock(cfg, trained, monkeypatch):
    """A SLEEP the platform rejected (ok=False) or that raised never starts `sleep_entered_sim`: the carrier is
    still ACTIVE and a later SLEEP by someone else must not be measured from a stale stamp."""
    monkeypatch.setenv("RAPP_MIN_SLEEP_DWELL_MIN", "60")
    cfg2 = Settings()
    for r1 in (FakeR1(ok=False), FakeR1(fail=True)):
        with TestClient(create_app(cfg2, model=trained, r1=r1)) as c:
            c.post("/r1/pm-delivery", json=pm_envelope(prb=3.0, sim_time="2024-01-10T03:00:00"))
            rec = c.post("/cycle", json={}).json()["decisions"][0]
            assert rec["decision"]["action"] == "SLEEP" and rec["cm"]["ok"] is False
            pol = _pol(c)
            assert pol["sleep_entered_sim"] is None and pol["last_action"] == "SLEEP" and pol["active"] == "ACTIVE"
            # a SLEEP that the carrier later enters without an rApp write carries no dwell: the wake is not held
            c.post("/r1/pm-delivery", json=pm_envelope(active=Active.SLEEP, state=CarrierState.DISABLED, prb=0.0, sim_time="2024-01-10T10:00:00"))
            d = _cycle(c)
            assert d["action"] == "ACTIVE" and "pre-emptive" in d["reason"] and "dwell" not in d["reason"], d


def test_local_wake_clears_the_dwell_clock_and_holds_a_resleep(cfg, trained, monkeypatch):
    """PM showing SLEEP -> ACTIVE that the rApp did not write (and a grown wake_events counter) is a ruApp local
    wake: the dwell clock is cleared, `local_wake_sim` is stamped, and with RAPP_POST_LOCAL_WAKE_HOLD_MIN the rApp
    does not SLEEP the carrier again inside the hold (data time), whatever the forecast."""
    monkeypatch.setenv("RAPP_MIN_SLEEP_DWELL_MIN", "60")
    monkeypatch.setenv("RAPP_POST_LOCAL_WAKE_HOLD_MIN", "30")
    cfg2 = Settings()
    r1 = FakeR1()
    with TestClient(create_app(cfg2, model=trained, r1=r1)) as c:
        assert c.get("/health").json()["policy"]["post_local_wake_hold_min"] == 30.0
        c.post("/r1/pm-delivery", json=pm_envelope(prb=3.0, sim_time="2024-01-10T03:00:00"))
        assert _cycle(c)["action"] == "SLEEP" and _pol(c)["sleep_entered_sim"] == "2024-01-10T03:00:00"
        # the ruApp woke the carrier at 03:10 (PM active=ACTIVE, wake_events 0 -> 1); its load is low again
        c.post("/r1/pm-delivery", json=pm_envelope(active=Active.ACTIVE, prb=3.0, sim_time="2024-01-10T03:10:00", wake_events=1, lat=40.0))
        d = _cycle(c)
        assert d["action"] == "NONE" and d["reason"].startswith("post-local-wake hold: 0 of 30 min"), d
        pol = _pol(c)
        assert pol["sleep_entered_sim"] is None and pol["local_wake_sim"] == "2024-01-10T03:10:00" and pol["last_action"] == "SLEEP"
        assert len(r1.requests) == 1
        c.post("/r1/pm-delivery", json=pm_envelope(active=Active.ACTIVE, prb=3.0, sim_time="2024-01-10T03:20:00", wake_events=1, lat=40.0))
        assert "post-local-wake hold: 10 of 30 min" in _cycle(c)["reason"]
        # one window later the hold has elapsed and the normal SLEEP rule applies again
        c.post("/r1/pm-delivery", json=pm_envelope(active=Active.ACTIVE, prb=3.0, sim_time="2024-01-10T03:40:00", wake_events=1, lat=40.0))
        d = _cycle(c)
        assert d["action"] == "SLEEP" and [r.active for r in r1.requests] == [Active.SLEEP, Active.SLEEP]
        assert _pol(c)["sleep_entered_sim"] == "2024-01-10T03:40:00"
    # the transition alone (no wake_events change) is a local wake too when the rApp's last write was SLEEP
    r1 = FakeR1()
    with TestClient(create_app(cfg2, model=trained, r1=r1)) as c:
        c.post("/r1/pm-delivery", json=pm_envelope(prb=3.0, sim_time="2024-01-10T03:00:00"))
        assert _cycle(c)["action"] == "SLEEP"
        c.post("/r1/pm-delivery", json=pm_envelope(active=Active.ACTIVE, prb=3.0, sim_time="2024-01-10T03:10:00"))
        assert "post-local-wake hold" in _cycle(c)["reason"] and _pol(c)["local_wake_sim"] == "2024-01-10T03:10:00"
    # an rApp pre-emptive ACTIVE is NOT a local wake: no hold on the next SLEEP
    r1 = FakeR1()
    with TestClient(create_app(cfg2, model=trained, r1=r1)) as c:
        c.post("/r1/pm-delivery", json=pm_envelope(active=Active.SLEEP, state=CarrierState.DISABLED, prb=0.0, sim_time="2024-01-10T10:00:00"))
        assert _cycle(c)["action"] == "ACTIVE"
        c.post("/r1/pm-delivery", json=pm_envelope(active=Active.ACTIVE, prb=3.0, sim_time="2024-01-11T03:00:00"))
        d = _cycle(c)
        assert d["action"] == "SLEEP" and _pol(c)["local_wake_sim"] is None


def test_both_estimators_recorded_on_every_decision(cfg, trained):
    """Every decision carries p_low(pm-window) and p_low(expected-hour-profile) at the same instant, their
    difference and the divergence flag; /metrics counts divergent decisions; the profile scale defaults to 1.0."""
    r1 = FakeR1()
    with TestClient(create_app(cfg, model=trained, r1=r1)) as c:
        c.post("/r1/pm-delivery", json=pm_envelope(prb=3.0, sim_time="2024-01-10T03:00:00"))
        d = _cycle(c)
        assert d["forecast_basis"] == "pm-window" and d["p_low"] == d["p_low_pm_window"]
        assert d["p_low_profile"] is not None and d["forecast_divergence"] == pytest.approx(d["p_low_pm_window"] - d["p_low_profile"])
        assert d["forecast_divergent"] is (abs(d["forecast_divergence"]) > 0.5) and d["profile_scale"] == 1.0
        flagged = int(d["forecast_divergent"])
        # awake at the evening peak with a night-like PM window (a quiet carrier): own PM says "low", the estate
        # profile says "load" -> the divergence, flagged
        for mm in ("18:00", "18:30", "19:00"):
            c.post("/r1/pm-delivery", json=pm_envelope(active=Active.ACTIVE, prb=3.0, sim_time=f"2024-01-10T{mm}:00"))
        d = _cycle(c)
        assert d["forecast_basis"] == "pm-window" and d["p_low_pm_window"] > 0.5 > d["p_low_profile"]
        assert d["forecast_divergent"] is True and d["forecast_divergence"] > 0.5
        flagged += 1
        # asleep: both estimators are recorded (the own-PM one reads the zero counters) but NOT judged
        c.post("/r1/pm-delivery", json=pm_envelope(active=Active.SLEEP, state=CarrierState.DISABLED, prb=0.0, sim_time="2024-01-10T19:30:00"))
        d = _cycle(c)
        assert d["forecast_basis"] == "expected-hour-profile" and d["p_low"] == d["p_low_profile"] < cfg.rapp_wake_threshold
        assert d["p_low_pm_window"] is not None and d["forecast_divergence"] is not None and d["forecast_divergent"] is False
        assert c.get("/metrics").json()["forecast_divergent_decisions"] == flagged >= 1
        assert "forecast_divergent" in c.get("/decisions").json()[-1]["decision"]


def test_profile_scale_uses_the_carriers_own_awake_level(cfg, trained, monkeypatch):
    """RAPP_PROFILE_SCALE=1: the asleep forecast comes from the profile at the carrier's own level (ratio of its
    awake PM to the estate profile at the same bins); windows reported while asleep (PRB 0) are excluded."""
    monkeypatch.setenv("RAPP_PROFILE_SCALE", "1")
    cfg2 = Settings()
    r1 = FakeR1()
    with TestClient(create_app(cfg2, model=trained, r1=r1)) as c:
        assert c.get("/health").json()["policy"]["profile_scale"] == 1.0
        # a quiet carrier: 8 awake windows at ~0.2x the estate level around midday
        for k in range(8):
            hh, mm = 10 + k // 2, "00" if k % 2 == 0 else "30"
            c.post("/r1/pm-delivery", json=pm_envelope(prb=6.0, sim_time=f"2024-01-10T{hh:02d}:{mm}:00"))
        d = _cycle(c)
        assert d["profile_scale"] is not None and d["profile_scale"] < 0.6, d
        # asleep rows (PRB 0) do not pull the level further down
        c.post("/r1/pm-delivery", json=pm_envelope(active=Active.SLEEP, state=CarrierState.DISABLED, prb=0.0, sim_time="2024-01-10T14:00:00"))
        d2 = _cycle(c, force="ACTIVE") if False else _cycle(c)
        assert d2["profile_scale"] == pytest.approx(d["profile_scale"], rel=0.05)
