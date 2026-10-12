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
"""End-to-end `lite` profile (HLD §4.3): the three processes (SMO, es-rApp, mock O-RU with its
ruApp) as subprocesses on ephemeral ports, plus the physics twin (generic mode) when its venv exists.

Narrative (same as scripts/run_demo.sh, driven with httpx): rApp cycle -> twin guardrail verdict
-> policy envelope active=SLEEP over R1 -> SMO -> M-Plane NETCONF edit-config -> RU DISABLED;
PRACH -> the ruApp wakes the carrier locally; one PM interval later the wake shows up at the SMO
(notification + measurement + twin ingest) and at the rApp (R1 job delivery). Asserts the two
measured numbers exist and are plausible, and that the SLEEP decision carries the twin's verdict
with the generic (uncalibrated) rung. No Docker, no external network; only 127.0.0.1.

Twin: `<twin_dir>/.venv` missing -> the twin assertions are skipped with that reason (the loop
still runs with TWIN_URL empty). Twin venv present but the twin at TWIN_DIR has no generic mode
(`TWIN_GENERIC=1` / `POST /v1/load-generic` -> 404) -> skipped with that reason, precisely.
"""
from __future__ import annotations

import os
import random
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx
import pytest

DEMO = Path(__file__).resolve().parents[1]
PROCS = ("smo", "rapp", "mock_ru")
TWIN_DIR = Path(os.environ.get("TWIN_DIR", DEMO.parent / "services" / "ocp-physics-digitalTwin"))
TWIN_PY = Path(os.environ.get("OCP_AURU_TWIN_VENV", str(TWIN_DIR / ".venv"))) / "bin" / "python"
GENERIC_RUNG_WORDS = ("generic", "uncalibrated")


def _free_ports(n: int) -> list[int]:
    """n distinct free ports in the 20000+ range (never the demo defaults)."""
    while True:
        base = 20000 + random.randint(0, 30000)
        ports = list(range(base, base + n))
        ok = True
        for p in ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", p))
                except OSError:
                    ok = False
                    break
        if ok:
            return ports


def _wait(desc: str, fn: Callable[[], Optional[Any]], timeout_s: float, every_s: float = 0.25) -> Any:
    t0 = time.monotonic()
    last: Optional[BaseException] = None
    while time.monotonic() - t0 < timeout_s:
        try:
            v = fn()
            if v:
                return v
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(every_s)
    raise TimeoutError(f"{desc} not satisfied within {timeout_s}s (last error: {last})")


def _tail(path: Path, n: int = 30) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return "(no log)"


def _start_twin(tmp_path: Path, var: Path, port: int, http: httpx.Client) -> tuple[Optional[subprocess.Popen], Optional[str], str]:
    """Start the physics twin in generic mode from its own venv, state under `var`. Returns
    `(proc, url, skip_reason)`; `url` is None (and the reason says why) when the twin cannot serve
    a generic estate — the loop then runs without the RF guardrail and the twin assertions skip."""
    if not TWIN_PY.exists():
        return None, None, f"twin venv missing at {TWIN_PY} (scripts/run_demo.sh creates it; twin assertions skipped)"
    (var / "twin").mkdir(parents=True, exist_ok=True)
    (var / "twin-data").mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TWIN_GENERIC": "1", "VAR_DIR": str(var / "twin"), "DATA_DIR": str(var / "twin-data"), "PYTHONPATH": str(TWIN_DIR)}
    log = (tmp_path / "twin.log").open("w")
    proc = subprocess.Popen([str(TWIN_PY), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
                            env=env, cwd=str(TWIN_DIR), stdout=log, stderr=subprocess.STDOUT, text=True)
    url = f"http://127.0.0.1:{port}"
    try:
        h = _wait("twin /health", lambda: http.get(f"{url}/health").json(), timeout_s=90)
    except TimeoutError as exc:
        return proc, None, f"twin did not become healthy: {exc}\n{_tail(tmp_path / 'twin.log')}"
    if not h.get("estate_loaded"):
        r = http.post(f"{url}/v1/load-generic", json={})
        if r.status_code == 404:
            return proc, None, "the physics twin at TWIN_DIR has no generic mode (POST /v1/load-generic -> 404); twin assertions skipped"
        assert r.status_code == 200, f"POST /v1/load-generic -> {r.status_code} {r.text[:300]}"
    return proc, url, ""


@pytest.mark.timeout(240)
def test_e2e_lite_sleep_wake_report(tmp_path: Path) -> None:
    ru_netconf, ru_mgmt, smo, rapp, twin_port = _free_ports(5)
    var = tmp_path / "var"
    var.mkdir()
    http = httpx.Client(timeout=httpx.Timeout(5.0, read=40.0))
    twin_proc, twin_url, twin_skip = _start_twin(tmp_path, var, twin_port, http)
    env = {
        **os.environ,
        "PROFILE": "lite",
        "VAR_DIR": str(var),
        "RU_HOST_KEY": str(var / "ru-host-key"),
        "TWIN_INGEST_PATH": str(var / "twin-ingest.jsonl"),
        "RU_NETCONF_PORT": str(ru_netconf),
        "RU_MGMT_PORT": str(ru_mgmt),
        "SMO_PORT": str(smo),
        "RAPP_PORT": str(rapp),
        "RU_PM_INTERVAL_S": "1",
        "RAPP_CYCLE_S": "2",
        "RAPP_AUTO_CYCLE": "0",
        "RAPP_R1_RETRY_S": "0.5",
        "RAPP_MIN_HOLD_S": "0",
        "TWIN_URL": twin_url or "",
        "TWIN_CELL_ID": "",
        "TWIN_AUTO_CELL": "1",
        "OCP_AURU_LOG_LEVEL": "INFO",
    }
    for k in ("SMO_URL", "RAPP_URL", "RU_MGMT_URL", "SMO_PUBLIC_URL", "RAPP_PUBLIC_URL", "RAPP_MODEL_PATH", "CM_ADAPTER",
              "RU_NETCONF_HOST", "RU_EXTERNAL", "RU_LOAD_PROFILE"):
        env.pop(k, None)  # derived from the ports above; a stale override would break isolation

    ru_url, smo_url, rapp_url = f"http://127.0.0.1:{ru_mgmt}", f"http://127.0.0.1:{smo}", f"http://127.0.0.1:{rapp}"
    procs: dict[str, subprocess.Popen] = {}
    logs: dict[str, Path] = {}
    if twin_proc is not None:
        procs["twin"] = twin_proc
        logs["twin"] = tmp_path / "twin.log"

    def get(url: str, **params: Any) -> Any:
        r = http.get(url, params=params or None)
        r.raise_for_status()
        return r.json()

    def post(url: str, body: Optional[dict] = None) -> Any:
        r = http.post(url, json=body if body is not None else {})
        r.raise_for_status()
        return r.json()

    def ru_state() -> dict[str, Any]:
        return get(f"{ru_url}/state")["carriers"]["txcarrier0"]

    def all_logs() -> str:
        return "\n".join(f"===== {n} =====\n{_tail(p)}" for n, p in logs.items())

    try:
        for name in PROCS:
            logs[name] = tmp_path / f"{name}.log"
            fh = logs[name].open("w")
            procs[name] = subprocess.Popen([sys.executable, "-m", f"ocp_auru.{name}"], env=env, cwd=str(DEMO),
                                           stdout=fh, stderr=subprocess.STDOUT, text=True)

        # ---- health of every process ---------------------------------------------------
        for name, url in (("smo", smo_url), ("rapp", rapp_url), ("mock_ru", ru_url)):
            try:
                body = _wait(f"{name} /health", lambda u=url: get(f"{u}/health"), timeout_s=60)
            except TimeoutError as exc:
                pytest.fail(f"{exc}\n{all_logs()}")
            assert body["ok"] is True, body
            assert procs[name].poll() is None, f"{name} exited early\n{all_logs()}"
        assert get(f"{ru_url}/health")["netconf_port"] == ru_netconf
        adapter = get(f"{smo_url}/health")["cm_adapter"]   # reference adapter: the SMO talks M-Plane to the O-RU itself
        assert adapter["name"] == "mplane-direct" and adapter["netconf_target"] == f"127.0.0.1:{ru_netconf}"

        # ---- initial state + R1 bring-up + first PM through the R1 job ---------------------
        st = ru_state()
        assert (st["active"], st["state"]) == ("ACTIVE", "READY")
        def r1_ready() -> Optional[dict]:
            x = get(f"{rapp_url}/health")
            return x if x["r1"]["registered"] and x["r1"]["job_created"] else None

        def rapp_pm(min_wake_events: int) -> Optional[dict]:
            x = get(f"{rapp_url}/pm")
            latest = x["carriers"].get("txcarrier0", {}).get("latest")
            return x if x["reports"] >= 1 and latest and (latest["wake_events"] or 0) >= min_wake_events else None

        h = _wait("rApp R1 registrations", r1_ready, timeout_s=30)
        assert h["model"]["auc"] is None or h["model"]["auc"] > 0.5
        assert get(f"{smo_url}/data-consumer/v1/info-jobs") == ["es-rapp-pm"]
        assert [s["name"] for s in get(f"{smo_url}/services")] == ["es-rapp"]
        pm = _wait("first PM at the rApp", lambda: rapp_pm(0), timeout_s=20)
        assert pm["carriers"]["txcarrier0"]["latest"]["active"] == "ACTIVE"

        # ---- slow loop: cycle -> SLEEP (natural if the model decides it, forced otherwise) ---
        def asleep() -> Optional[dict]:
            s = ru_state()
            return s if s["active"] == "SLEEP" and s["state"] == "DISABLED" else None

        forced = False
        first = post(f"{rapp_url}/cycle", {})["decisions"]
        assert first and first[0]["decision"]["carrier"] == "txcarrier0"
        deadline = time.monotonic() + 6.0
        while not asleep() and time.monotonic() < deadline:
            time.sleep(2.0)
            if not asleep():
                post(f"{rapp_url}/cycle", {})
        if not asleep():
            forced = True
            rec = post(f"{rapp_url}/cycle", {"force": "SLEEP", "carrier": "txcarrier0"})["decisions"][0]
            assert rec["decision"]["forced"] is True and rec["cm"]["ok"] is True, rec
            _wait("RU DISABLED after forced SLEEP", asleep, timeout_s=10)
        st = ru_state()
        assert (st["active"], st["state"]) == ("SLEEP", "DISABLED")
        assert st["ruapp_policy"]["wake_on_prach"] is True

        pol = get(f"{rapp_url}/policy")["txcarrier0"]
        cm = pol["last_cm"]
        assert cm["ok"] is True, cm
        assert cm["cm_rtt_ms"] > 0
        assert cm["netconf_rtt_ms"] is not None and cm["netconf_rtt_ms"] > 0
        assert cm["netconf_rtt_ms"] <= cm["cm_rtt_ms"]
        smo_last = get(f"{smo_url}/o1/cm/last")
        assert smo_last["ok"] is True and smo_last["active"] == "SLEEP"
        seen = get(f"{smo_url}/o1/cm/carriers")             # M-Plane <get> from the SMO: what the O-RU itself reports
        assert seen["ok"] is True and {"name": "txcarrier0", "active": "SLEEP", "state": "DISABLED"} in seen["carriers"], seen

        # ---- twin guardrail verdict on the SLEEP decision (ADR-7, generic rung) ---------------
        sleep_recs = [r for r in get(f"{rapp_url}/decisions") if r["decision"]["action"] == "SLEEP" and r.get("cm", {}).get("ok")]
        assert sleep_recs, "no successful SLEEP decision recorded"
        tw = sleep_recs[-1]["twin"]
        twin_view = get(f"{rapp_url}/health")["twin"]
        if twin_url:
            assert tw["enabled"] and tw["status"] == "ok", (tw, _tail(tmp_path / "twin.log"))
            assert tw["verdict"] in ("allowed", "blocked") and tw["cell_id"] and tw["cell_source"] == "auto:/v1/cells[0]", tw
            assert all(w in (tw["rung"] or "").lower() for w in GENERIC_RUNG_WORDS), f"rung must be labelled generic/uncalibrated: {tw['rung']!r}"
            assert set(tw["delta_vs_baseline"]) >= {"rsrp_p5", "sinr_p5", "outage_rate", "coverage_rate"}, tw
            assert tw["latency_ms"] > 0
            if tw["verdict"] == "blocked":   # a natural block only gets through as a forced (overridden) SLEEP
                assert sleep_recs[-1]["decision"]["forced"] and tw["overridden"] is True
            assert twin_view["status"] == "ok" and twin_view["cell_id"] == tw["cell_id"] and twin_view["checks"] >= 1
            assert twin_view["cells_listed"] >= 1 and "rung_warning" not in twin_view
            met_tw = get(f"{rapp_url}/metrics")
            assert met_tw["twin_checks"] >= 1 and met_tw["last_twin_verdict"]["cell_id"] == tw["cell_id"]
            assert (var / "twin").is_dir()   # the twin's state lives under the demo var dir, never in its source folder
        else:
            assert tw == {"enabled": False} and twin_view["status"] == "disabled"

        # ---- fast loop: PRACH -> local wake ---------------------------------------------------
        res = post(f"{ru_url}/sim/prach", {"carrier": "txcarrier0", "energy_db": 20})
        assert res["decision"] == "wake", res
        wake_us = res["wake_latency_us"]
        assert wake_us is not None and 0 < wake_us < 100_000, res
        assert res["active"] == "ACTIVE"
        def ready() -> Optional[dict]:
            s = ru_state()
            return s if s["state"] == "READY" else None

        st = _wait("carrier READY after wake", ready, timeout_s=5, every_s=0.02)
        assert st["active"] == "ACTIVE" and st["hold_remaining_s"] > 0
        # inside the hold the RU rejects a new SLEEP from the management plane (ADR-3 / §3.2)
        rej = post(f"{rapp_url}/cycle", {"force": "SLEEP", "carrier": "txcarrier0"})["decisions"][0]["cm"]
        assert rej["ok"] is False and "hold" in rej["detail"].lower(), rej
        assert ru_state()["active"] == "ACTIVE"

        # ---- reports: SMO notification + PM, rApp PM via R1, twin ingest --------------------------
        time.sleep(float(env["RU_PM_INTERVAL_S"]) + 1.0)

        def wake_notified() -> Optional[list]:
            hits = []
            for e in get(f"{smo_url}/o1/events", since=0)["events"]:
                nf = e["event"].get("notificationFields") or {}
                add = nf.get("additionalFields") or {}
                if nf.get("changeType") == "tx-array-carriers-state-change" and add.get("cause") == "ruapp-wake":
                    hits.append(e)
            return hits or None

        hits = _wait("SMO notification cause=ruapp-wake", wake_notified, timeout_s=5)
        assert hits[0]["event"]["notificationFields"]["additionalFields"]["active"] == "ACTIVE"
        latest = get(f"{smo_url}/o1/pm/latest")
        objs = next(iter(latest.values()))["event"]["measurementFields"]["additionalObjects"]
        assert objs[0]["name"] == "txcarrier0"
        rpm = _wait("rApp PM shows wake_events >= 1", lambda: rapp_pm(1), timeout_s=6)
        rl = rpm["carriers"]["txcarrier0"]["latest"]
        assert rl["wake_events"] >= 1 and rl["wake_latency_us"]["last"] > 0 and rl["active"] == "ACTIVE"
        met = get(f"{rapp_url}/metrics")
        assert met["last_cm_rtt_ms"] > 0 and met["wake_events"]["txcarrier0"] >= 1
        lines = (var / "twin-ingest.jsonl").read_text().splitlines()  # file first, stats second: the RU keeps
        twin = get(f"{smo_url}/twin/ingest/stats")                     # appending, so the counter may only be ahead
        assert twin["lines"] >= len(lines) > 0
        assert any('"kind":"notification"' in ln for ln in lines) and any('"kind":"measurement"' in ln for ln in lines)
        assert any('"powerModel":"illustrative"' in ln for ln in lines if '"kind":"measurement"' in ln)
        assert twin["path"] == "$VAR_DIR/twin-ingest.jsonl", twin  # display path: never the absolute checkout path
        print(f"\n[e2e lite] sleep decision {'FORCED' if forced else 'natural'}; wake_latency_us={wake_us}; "
              f"cm_rtt_ms={cm['cm_rtt_ms']} netconf_rtt_ms={cm['netconf_rtt_ms']}; twin lines={twin['lines']}; "
              f"twin guardrail: {tw.get('verdict') or 'disabled'} cell={tw.get('cell_id')} [{tw.get('rung')}]")
        if twin_skip:
            pytest.skip(twin_skip)   # after the loop assertions: the loop itself passed, only the twin part is skipped
    finally:
        http.close()
        for name, p in procs.items():
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        for name, p in procs.items():
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)
