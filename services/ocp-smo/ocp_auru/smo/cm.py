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
"""Configuration-management routes of the SMO: CM request from the rApp -> CM adapter -> radio.

`POST /o1/cm/carrier` hands a `CmCarrierRequest` to the configured adapter (`o1_adapter.py`)
and measures `cm_rtt_ms` around the whole SMO-side round trip (adapter call, which for the
reference adapter is NETCONF session setup + edit-config + session close). The adapter's
own `netconf_rtt_ms` (edit-config only) is passed through.

Error contract (documented choice): the endpoint always answers HTTP 200 with a
`CmCarrierResponse`; a failed or unreachable radio yields `ok=false` with the reason in
`detail`, so the rApp reads one shape and still gets the round-trip time.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter

from ocp_auru.common.config import Settings
from ocp_auru.common.events import AdapterResult, CmCarrierRequest, CmCarrierResponse
from ocp_auru.smo.o1_adapter import CmAdapter

log = logging.getLogger(__name__)


def cm_router(cfg: Settings, adapter: CmAdapter, last: dict[str, Any]) -> APIRouter:
    r = APIRouter(prefix="/o1/cm", tags=["cm"])

    @r.post("/carrier", response_model=CmCarrierResponse)
    def cm_carrier(req: CmCarrierRequest) -> CmCarrierResponse:
        log.info("CM: %s carrier %s -> %s (%s) via adapter %s", req.ru_id, req.carrier, req.active.value,
                 req.reason or "no reason given", adapter.name)
        t0 = time.perf_counter()
        try:
            result = adapter.set_carrier_state(req, include_policy=cfg.cm_include_ruapp_policy)
        except Exception as exc:  # noqa: BLE001 - an adapter must not raise; keep the one-shape contract if it does
            log.exception("CM: adapter %s raised", adapter.name)
            result = AdapterResult(ok=False, detail=f"adapter {adapter.name} failed: {type(exc).__name__}: {exc}")
        cm_rtt = round((time.perf_counter() - t0) * 1000.0, 3)
        if not result.ok:
            log.error("CM: actuation failed: %s", result.detail)
        else:
            log.info("CM: applied; cm_rtt_ms=%.3f netconf_rtt_ms=%s", cm_rtt, result.netconf_rtt_ms)
        resp = CmCarrierResponse(ok=result.ok, ru_id=req.ru_id, carrier=req.carrier, active=req.active,
                                 cm_rtt_ms=cm_rtt, netconf_rtt_ms=result.netconf_rtt_ms, detail=result.detail)
        last.update(resp.model_dump(mode="json"))
        return resp

    @r.get("/carriers")
    def cm_carriers():
        """Inspection: the carriers' `active`/`state` as the adapter reads them (M-Plane `<get>`)."""
        try:
            return adapter.get_carriers()
        except Exception as exc:  # noqa: BLE001
            log.exception("CM: adapter %s raised", adapter.name)
            return {"ok": False, "detail": f"adapter {adapter.name} failed: {type(exc).__name__}: {exc}", "carriers": []}

    @r.get("/last")
    def cm_last():
        return last

    return r
