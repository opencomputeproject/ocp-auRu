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
"""R1 client — the rApp side of the Non-RT RIC platform interface (HLD ADR-2).

R1 service management: `PUT /services/es-rapp` on the rApp Catalogue (1.0.x shape).
R1 data management & exposure: `PUT /data-consumer/v1/info-jobs/es-rapp-pm` on ICS
(1.2.x shape) so O1 PM is delivered to our `job_result_uri`. Both registrations retry in
the background until the platform is up. The CM request goes to the SMO's CM route
(`POST /o1/cm/carrier`); the rApp never talks M-Plane itself. Identical across the
`lite` (SMO-served routes) and `full` (real NONRTRIC) profiles via `settings.*_url`.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import httpx

from ocp_auru.common.config import Settings, settings
from ocp_auru.common.events import CmCarrierRequest, CmCarrierResponse

log = logging.getLogger(__name__)

SERVICE_NAME = "es-rapp"
SERVICE_VERSION = "0.1.0"
JOB_ID = "es-rapp-pm"
INFO_TYPE_ID = "ocp-auru-ru-pm"


class R1Client:
    def __init__(self, cfg: Settings = settings, transport: Optional[httpx.BaseTransport] = None):
        self.cfg = cfg
        self._http = httpx.Client(timeout=cfg.rapp_http_timeout_s, transport=transport)
        self._cm_http = httpx.Client(timeout=cfg.rapp_cm_timeout_s, transport=transport)
        self.registered = False
        self.job_created = False
        self.attempts = 0
        self.last_error: Optional[str] = None
        self.registered_at: Optional[float] = None
        self.job_created_at: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- R1 service management (rApp Catalogue) --------------------------------------------
    def service_payload(self) -> dict[str, Any]:
        return {"version": SERVICE_VERSION, "display_name": "Energy Saving rApp",
                "description": "Forecasts low-load windows from O1 PM and sets tx-array-carriers/active=SLEEP "
                               "with a wake-on-PRACH ruApp envelope via R1 -> SMO -> M-Plane (NETCONF) -> O-RU"}

    def register_service(self) -> bool:
        url = f"{self.cfg.catalogue_url}/services/{SERVICE_NAME}"
        try:
            r = self._http.put(url, json=self.service_payload())
            ok = r.status_code in (200, 201, 204)
            if ok and not self.registered:
                log.info("R1 catalogue: registered %s at %s (%d)", SERVICE_NAME, url, r.status_code)
            self.registered = ok
            if not ok:
                self.last_error = f"catalogue {r.status_code}: {r.text[:200]}"
        except Exception as exc:
            self.registered = False
            self.last_error = f"catalogue: {exc}"
        return self.registered

    # ---- R1 data management (ICS info job) --------------------------------------------------
    def job_payload(self) -> dict[str, Any]:
        return {"info_type_id": INFO_TYPE_ID, "job_owner": SERVICE_NAME, "job_definition": {},
                "job_result_uri": f"{self.cfg.rapp_public_url}/r1/pm-delivery", "status_notification_uri": ""}

    def ensure_pm_job(self) -> bool:
        url = f"{self.cfg.ics_url}/data-consumer/v1/info-jobs/{JOB_ID}"
        try:
            r = self._http.put(url, json=self.job_payload())
            ok = r.status_code in (200, 201, 204)
            if ok and not self.job_created:
                log.info("R1 ICS: info job %s (type %s) at %s (%d)", JOB_ID, INFO_TYPE_ID, url, r.status_code)
            self.job_created = ok
            if not ok:
                self.last_error = f"ics {r.status_code}: {r.text[:200]}"
        except Exception as exc:
            self.job_created = False
            self.last_error = f"ics: {exc}"
        return self.job_created

    # ---- background bring-up -----------------------------------------------------------------
    def try_once(self) -> bool:
        self.attempts += 1
        if not self.registered and self.register_service():
            self.registered_at = time.time()
        if not self.job_created and self.ensure_pm_job():
            self.job_created_at = time.time()
        if self.registered and self.job_created:
            self.last_error = None
        return self.registered and self.job_created

    def start_background(self) -> None:
        """Retry both registrations every `RAPP_R1_RETRY_S` until they succeed."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def _loop() -> None:
            while not self._stop.is_set():
                if self.try_once():
                    return
                if self.attempts in (1, 5) or self.attempts % 30 == 0:
                    log.warning("R1 platform not ready (attempt %d): %s", self.attempts, self.last_error)
                self._stop.wait(self.cfg.rapp_r1_retry_s)

        self._thread = threading.Thread(target=_loop, name="r1-register", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._http.close()
        self._cm_http.close()

    def status(self) -> dict[str, Any]:
        return {"service": SERVICE_NAME, "registered": self.registered, "job_created": self.job_created,
                "job_id": JOB_ID, "info_type_id": INFO_TYPE_ID, "attempts": self.attempts,
                "catalogue_url": self.cfg.catalogue_url, "ics_url": self.cfg.ics_url,
                "job_result_uri": self.job_payload()["job_result_uri"], "last_error": self.last_error}

    # ---- R1 CM request -> SMO CM route ------------------------------------------------------
    def cm_request(self, req: CmCarrierRequest) -> CmCarrierResponse:
        url = f"{self.cfg.smo_url}/o1/cm/carrier"
        r = self._cm_http.post(url, json=req.model_dump(mode="json"))
        r.raise_for_status()
        return CmCarrierResponse.model_validate(r.json())
