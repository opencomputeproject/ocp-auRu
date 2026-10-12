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
"""Shared fixtures for the SMO tests: isolated Settings, VES sample events, a threaded HTTP receiver."""
from __future__ import annotations

import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ocp_auru.common import ves
from ocp_auru.common.config import Settings
from ocp_auru.common.events import Active, CarrierPm, CarrierState, PmReport, WakeLatencyStats


def make_settings(monkeypatch, tmp_path: Path, **env: str) -> Settings:
    base = 20000 + random.randint(0, 20000)
    monkeypatch.setenv("PROFILE", "lite")
    monkeypatch.setenv("VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("TWIN_INGEST_PATH", str(tmp_path / "var" / "twin-ingest.jsonl"))
    monkeypatch.setenv("SMO_PORT", str(base))
    monkeypatch.setenv("RU_NETCONF_PORT", str(base + 2))
    monkeypatch.setenv("RAPP_PORT", str(base + 3))
    monkeypatch.setenv("MPLANE_TIMEOUT_S", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return Settings()


def measurement(ru_id: str = "oru-test", active: Active = Active.ACTIVE) -> dict[str, Any]:
    pm = CarrierPm(name="txcarrier0", active=active, state=CarrierState.READY, tx_power_dbm=40.0, prb_util_dl_pct=12.5,
                   prb_util_ul_pct=3.0, rrc_conn=7, power_w_avg=268.0, energy_wh=0.37, prach_detections=3, wake_events=1,
                   wake_latency_us=WakeLatencyStats(last=42.0, p50=42.0, max=42.0, count=1))
    report = PmReport(ru_id=ru_id, interval_s=5.0, sim_time_iso="2026-01-01T02:00:00", carriers=[pm], epoch_us=1_700_000_000_000_000)
    return ves.measurement_event(report).model_dump(mode="json")


def notification(ru_id: str = "oru-test", state: str = "DISABLED") -> dict[str, Any]:
    return ves.notification_event(ru_id, "tx-array-carriers-state-change",
                                  {"carrier": "txcarrier0", "state": state, "active": "SLEEP"}, 1_700_000_000_000_000).model_dump(mode="json")


class Receiver:
    """Threaded HTTP server on 127.0.0.1:0 that records every request (method, path, JSON body)."""

    def __init__(self, status: int = 200):
        self.requests: list[dict[str, Any]] = []
        self.status = status
        self._lock = threading.Lock()
        outer = self

        class H(BaseHTTPRequestHandler):
            def _handle(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                body = json.loads(raw) if raw else None
                with outer._lock:
                    outer.requests.append({"method": self.command, "path": self.path, "body": body})
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

            do_POST = do_PUT = do_GET = do_DELETE = _handle

            def log_message(self, *a):  # quiet
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> "Receiver":
        self._thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def posts(self, path_prefix: str = "") -> list[dict[str, Any]]:
        with self._lock:
            return [r for r in self.requests if r["method"] == "POST" and r["path"].startswith(path_prefix)]
