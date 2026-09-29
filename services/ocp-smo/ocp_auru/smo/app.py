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
"""SMO FastAPI app (HLD §3.6): VES collector, CM routes + adapter, R1 DME producer, twin ingest.

`create_app(cfg)` builds an isolated instance (tests pass their own `Settings`); the module
level `app` uses the process settings. Routes:

- `POST /eventListener/v7` — VES-style collector for the O-RU's north-bound push (hybrid
  M-Plane, `o-ran-ves-subscribed-notifications` pattern); 202. Measurements -> R1 job delivery.
- `GET /o1/pm/latest[?ru_id=]`, `GET /o1/events?since=<seq>` — inspection.
- `POST /o1/cm/carrier`, `GET /o1/cm/carriers`, `GET /o1/cm/last`: configuration management
  through the configured CM adapter (see `cm.py`, `o1_adapter.py`; reference: direct M-Plane).
- `GET /health`, `GET /twin/ingest/stats`.
- `lite` only: ICS-/catalogue-compatible routes (see `dme.py`); both: `/r1/producer/*`.

No authentication on any route (demo; bound to 127.0.0.1 by default). As in real ICS, an
R1 job's `job_result_uri` is taken as given: whoever can register a job makes the SMO POST
PM data there. Keep the SMO on loopback or put it behind an authenticating proxy.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, Query

from ocp_auru.common.config import Settings, settings
from ocp_auru.common.events import VesEnvelope
from ocp_auru.smo import dme, o1_adapter
from ocp_auru.smo.cm import cm_router
from ocp_auru.smo.store import EventStore
from ocp_auru.smo.twin_sink import TwinSink

log = logging.getLogger(__name__)


def create_app(cfg: Optional[Settings] = None, registrar_retry_s: float = dme.ICS_REGISTER_RETRY_S,
               adapter: Optional[o1_adapter.CmAdapter] = None) -> FastAPI:
    cfg = cfg or settings
    adapter = adapter or o1_adapter.create(cfg)
    store = EventStore()
    sink = TwinSink(cfg.twin_ingest_path, cfg.var_dir)
    registry = dme.JobRegistry()
    deliverer = dme.Deliverer(registry)
    registrar = dme.Registrar(cfg, registry, retry_s=registrar_retry_s)
    last_cm: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        registrar.start()
        log.info("SMO up: profile=%s, ICS=%s, catalogue=%s, CM adapter=%s, twin ingest=%s", cfg.profile, cfg.ics_url, cfg.catalogue_url,
                 adapter.name, sink.display_path)
        try:
            yield
        finally:
            registrar.stop()

    app = FastAPI(title="OCP Autonomous AI-RU SMO", lifespan=lifespan)
    app.state.cfg, app.state.store, app.state.sink, app.state.adapter = cfg, store, sink, adapter
    app.state.registry, app.state.deliverer, app.state.registrar = registry, deliverer, registrar
    if cfg.profile != "full":
        registrar.register_local()  # eager so lite routes work even before lifespan (TestClient without `with`)

    # -- O-RU north-bound PM/notifications (VES-style collector) --------------------------
    @app.post("/eventListener/v7", status_code=202)
    def ves_listener(envelope: VesEnvelope, background: BackgroundTasks):
        data = envelope.model_dump(mode="json")
        domain, source, seq = store.ingest(data)
        sink.append(domain, source, data["event"])
        if domain == "measurement":
            background.add_task(deliverer.deliver, data)
        else:
            hdr = envelope.event.commonEventHeader
            log.info("FM/notification #%s from %s: %s %s", seq, source, hdr.eventName,
                     (envelope.event.notificationFields or {}).get("additionalFields", {}))
        return {"ok": True, "domain": domain, "seq": seq}

    @app.get("/o1/pm/latest")
    def pm_latest(ru_id: Optional[str] = None):
        return store.latest_pm(ru_id)

    @app.get("/o1/events")
    def events(since: int = Query(0, ge=0), limit: int = Query(1000, ge=1, le=1000)):
        items = store.events_since(since, limit)
        return {"since": since, "last_seq": store.last_seq, "events": items}

    # -- CM routes (adapter seam), R1 producer callbacks, lite ICS/catalogue --------------
    app.include_router(cm_router(cfg, adapter, last_cm))
    app.include_router(dme.producer_router(registry))
    if cfg.profile != "full":
        app.include_router(dme.ics_lite_router(registry))
        app.include_router(dme.catalogue_lite_router(registry))

    # -- health / twin --------------------------------------------------------------------
    @app.get("/health")
    def health():
        return {
            "ok": True,
            "component": "smo",
            "profile": cfg.profile,
            "r1": {"registration": dict(registrar.state), "jobs": registry.job_ids(), "delivery": dict(deliverer.stats)},
            "o1": {"pm_sources": list(store.latest_pm()), "events_last_seq": store.last_seq, "counts": dict(store.counts)},
            "cm_adapter": adapter.describe(),
            "twin_ingest": sink.stats(),
        }

    @app.get("/twin/ingest/stats")
    def twin_stats():
        return {"ok": True, **sink.stats()}

    return app


app = create_app()
