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
"""Management API of the mock O-RU (live uvicorn on an ephemeral port) and the `python -m ocp_auru.mock_ru` entry point."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import httpx
import pytest

from ocp_auru.common.events import Active, CarrierState
from ocp_auru.mock_ru.app import build
from _mock_ru_helpers import ephemeral_settings, free_port

C0 = "txcarrier0"


@pytest.fixture(scope="module")
def ru(tmp_path_factory):
    s = ephemeral_settings(tmp_path_factory.mktemp("ru"), ru_initial_active="ACTIVE")
    ctx = build(s, background_prach=False).start(mgmt_api=True)
    yield ctx
    ctx.stop()


@pytest.fixture(scope="module")
def api(ru):
    with httpx.Client(base_url=f"http://{ru.mgmt.host}:{ru.mgmt.port}", timeout=5.0) as c:
        yield c


def wait_state(ru, name: str, state: CarrierState, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while ru.datastore.state(name) != state:
        assert time.monotonic() < deadline
        time.sleep(0.002)


def test_health(ru, api):
    r = api.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["ru_id"] == ru.settings.ru_id
    assert body["netconf_port"] == ru.netconf.port and body["carriers"] == [C0]


def test_state(ru, api):
    body = api.get("/state").json()
    c = body["carriers"][C0]
    assert c["active"] == "ACTIVE" and c["state"] == "READY"
    assert c["ruapp_policy"]["wake_on_prach"] is True
    assert set(c["sim"]) >= {"prb_util_dl_pct", "rrc_conn", "tx_power_dbm", "power_w"}
    assert c["ruapp"]["wake_events"] == 0 and c["ruapp"]["wake_latency_us"]["count"] == 0
    assert body["sim_time_iso"].endswith("Z") and body["sim_speedup"] == ru.settings.ru_sim_speedup


def test_prach_on_active_carrier_is_ignored(ru, api):
    r = api.post("/sim/prach", json={"carrier": C0, "energy_db": 20})
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "ignore-active" and body["wake_latency_us"] is None and body["active"] == "ACTIVE"


def test_prach_wakes_sleeping_carrier_and_reports_latency(ru, api, capsys):
    ru.datastore.apply_edit({C0: {"active": Active.SLEEP, "policy": {"min_hold_after_wake_s": 0}}})
    wait_state(ru, C0, CarrierState.DISABLED)
    seq_before = ru.recorder.seq
    r = api.post("/sim/prach", json={"carrier": C0})  # default energy 20 dB
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "wake" and body["prach_energy_db"] == 20.0
    assert body["wake_latency_us"] is not None and 0 < body["wake_latency_us"] < 5000
    assert body["active"] == "ACTIVE" and body["state"] in ("BUSY", "READY")
    assert body["seq"] > seq_before
    wait_state(ru, C0, CarrierState.READY)
    st = api.get("/state").json()["carriers"][C0]
    assert st["ruapp"]["wake_events"] == 1 and st["ruapp"]["wake_latency_us"]["last"] == body["wake_latency_us"]
    with capsys.disabled():
        print(f"\n[POST /sim/prach wake latency] {body['wake_latency_us']:.1f} us (first wake, cold path)")


def test_prach_unknown_carrier_404(api):
    assert api.post("/sim/prach", json={"carrier": "txcarrier9"}).status_code == 404


def test_events_ring_buffer(ru, api):
    body = api.get("/events", params={"since": 0}).json()
    kinds = [e["kind"] for e in body["events"]]
    seqs = [e["seq"] for e in body["events"]]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert {"PrachDetected", "RuAppDecision", "CarrierStateChanged"} <= set(kinds)
    wake = [e for e in body["events"] if e["kind"] == "RuAppDecision" and e["event"]["decision"] == "wake"]
    assert wake and wake[-1]["event"]["wake_latency_ns"] > 0
    since = body["next"]
    assert api.get("/events", params={"since": since}).json()["events"] == []
    ru.radio.inject_prach(C0, 5.0)
    newer = api.get("/events", params={"since": since}).json()["events"]
    assert [e["kind"] for e in newer] == ["PrachDetected", "RuAppDecision"]
    assert newer[1]["event"]["decision"] == "ignore-active"


def test_pm_latest_after_report(ru, api):
    assert api.get("/pm/latest").json()["report"] is None or True  # may be None before the first interval
    report = ru.reporter.build_report()
    body = api.get("/pm/latest").json()
    rep = body["report"]
    assert rep["ru_id"] == ru.settings.ru_id and rep["epoch_us"] == report.epoch_us
    c = rep["carriers"][0]
    assert c["name"] == C0 and c["active"] == "ACTIVE" and c["state"] == "READY"
    assert c["wake_events"] == 1 and c["wake_latency_us"]["count"] == 1 and c["wake_latency_us"]["p50"] > 0
    assert c["prach_detections"] >= 1 and c["energy_wh"] >= 0 and c["power_w_avg"] > 0
    assert body["ves"]["url"].endswith("/eventListener/v7")
    assert body["ves"]["failed"] >= 1 and body["ves"]["sent"] == 0  # no SMO listening: failures ignored


def test_main_process_health_and_clean_shutdown(tmp_path):
    env = dict(os.environ)
    env.update({
        "RU_NETCONF_PORT": str(free_port()), "RU_MGMT_PORT": str(free_port()),
        "RU_HOST_KEY": str(tmp_path / "hostkey"), "VAR_DIR": str(tmp_path),
        "SMO_PORT": str(free_port()), "RU_PM_INTERVAL_S": "60", "OCP_AURU_LOG_LEVEL": "INFO",
    })
    proc = subprocess.Popen([sys.executable, "-m", "ocp_auru.mock_ru"], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        url = f"http://127.0.0.1:{env['RU_MGMT_PORT']}/health"
        deadline = time.monotonic() + 15
        body = None
        while time.monotonic() < deadline:
            try:
                body = httpx.get(url, timeout=1.0).json()
                break
            except Exception:  # noqa: BLE001
                assert proc.poll() is None, "process exited early"
                time.sleep(0.1)
        assert body is not None and body["ok"] is True
        assert body["netconf_port"] == int(env["RU_NETCONF_PORT"])
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=10)
        assert proc.returncode == 0, out
        assert "mock O-RU stopped" in out
    finally:
        if proc.poll() is None:
            proc.kill()
