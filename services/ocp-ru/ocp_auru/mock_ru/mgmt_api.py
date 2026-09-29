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
"""Management/inspection HTTP API of the mock O-RU (not an O-RAN interface: demo tooling).

`GET /health`, `GET /state`, `POST /sim/prach` (the demo trigger: inject a PRACH burst and
return the ruApp's decision incl. `wake_latency_us`), `GET /events?since=`, `GET /pm/latest`.
Runs uvicorn in a background thread bound to `settings.ru_netconf_host:settings.ru_mgmt_port`.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ocp_auru.common.events import RuAppDecision

if TYPE_CHECKING:  # pragma: no cover
    from .app import RuContext

log = logging.getLogger(__name__)


class PrachRequest(BaseModel):
    carrier: Optional[str] = None
    energy_db: float = 20.0


def create_app(ctx: "RuContext") -> FastAPI:
    app = FastAPI(title="ocp_auru mock O-RU management API", version="0.1.0")
    s = ctx.settings

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "ru_id": s.ru_id, "netconf_port": ctx.netconf.port, "netconf_host": ctx.netconf.host,
                "carriers": list(ctx.datastore.carrier_names()), "sim_time_iso": ctx.radio.clock.sim_time_iso()}

    @app.get("/state")
    def state() -> dict:
        metrics = ctx.radio.latest_all()
        out = {}
        for name, st in ctx.datastore.statuses().items():
            out[name] = {
                **st.model_dump(mode="json"),
                "hold_remaining_s": round(ctx.datastore.hold_remaining_s(name), 2),
                "sim": metrics.get(name, {}),
                "ruapp": ctx.agent.snapshot(name),
            }
        return {"ru_id": s.ru_id, "sim_time_iso": ctx.radio.clock.sim_time_iso(),
                "sim_speedup": s.ru_sim_speedup, "carriers": out,
                "interfaces": ctx.datastore.interfaces(), "netconf_subscribers": ctx.netconf.methods.subscriber_count()}

    @app.post("/sim/prach")
    def sim_prach(req: PrachRequest) -> dict:
        carrier = req.carrier or ctx.datastore.carrier_names()[0]
        if carrier not in ctx.datastore.carrier_names():
            raise HTTPException(status_code=404, detail=f"unknown carrier {carrier}")
        seq0 = ctx.recorder.seq
        ev = ctx.radio.inject_prach(carrier, req.energy_db)
        rec = ctx.recorder.wait_for(
            lambda r: r.kind == "RuAppDecision" and r.event.carrier == carrier, timeout=0.05, after_seq=seq0)
        if rec is None:
            raise HTTPException(status_code=504, detail="ruApp produced no decision within 50 ms")
        d: RuAppDecision = rec.event
        st = ctx.datastore.status(carrier)
        return {
            "carrier": carrier, "prach": ev.model_dump(mode="json"),
            "decision": d.decision, "prach_energy_db": d.prach_energy_db,
            "wake_latency_ns": d.wake_latency_ns,
            "wake_latency_us": None if d.wake_latency_ns is None else round(d.wake_latency_ns / 1000.0, 3),
            "active": st.active.value, "state": st.state.value, "policy": st.ruapp_policy.model_dump(),
            "seq": rec.seq,
        }

    @app.get("/events")
    def events(since: int = 0) -> dict:
        items = [r.as_json() for r in ctx.recorder.since(since)]
        return {"since": since, "next": items[-1]["seq"] if items else since, "events": items}

    @app.get("/pm/latest")
    def pm_latest() -> dict:
        rep = ctx.reporter.latest
        return {"report": rep.model_dump(mode="json") if rep else None,
                "ves": {"url": ctx.reporter.url, "sent": ctx.reporter.sent, "failed": ctx.reporter.failed}}

    return app


class MgmtServer:
    """uvicorn in a thread; `port` is the bound port (useful with port 0 in tests)."""

    def __init__(self, app: FastAPI, host: str, port: int) -> None:
        self._config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
        self._server = uvicorn.Server(self._config)
        self._thread = threading.Thread(target=self._server.run, name="ru-mgmt-api", daemon=True)
        self.host = host

    @property
    def port(self) -> int:
        try:
            return self._server.servers[0].sockets[0].getsockname()[1]
        except (AttributeError, IndexError):
            return self._config.port

    def start(self, wait_s: float = 5.0) -> "MgmtServer":
        self._thread.start()
        deadline = time.monotonic() + wait_s
        while not self._server.started and time.monotonic() < deadline:
            if not self._thread.is_alive():
                raise RuntimeError("management API failed to start")
            time.sleep(0.01)
        log.info("management API listening on http://%s:%d", self.host, self.port)
        return self

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5.0)
