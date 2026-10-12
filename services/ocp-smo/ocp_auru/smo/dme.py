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
"""R1 Data Management & Exposure — the SMO as ICS *producer* of info-type `ocp-auru-ru-pm`.

One job registry and one delivery path in both profiles (ADR-1):
- `lite`: this process also serves the ICS 1.2.x-compatible subset (`/data-producer/v1`,
  `/data-consumer/v1`) and the rApp Catalogue 1.0.x-compatible subset (`/services`), so the
  consumer's `PUT /data-consumer/v1/info-jobs/{id}` lands in the local `JobRegistry`.
- `full`: the SMO registers type + producer in the real ICS (`settings.ics_url_full`) with
  retry; ICS then pushes each job to `POST {smo_public_url}/r1/producer/job[/{jobId}]`, which
  lands in the same `JobRegistry`.
Delivery (`Deliverer`) = POST the measurement VES envelope to every active job's target URI.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from ocp_auru.common.config import Settings

log = logging.getLogger(__name__)

PM_INFO_TYPE = "ocp-auru-ru-pm"
PRODUCER_ID = "ocp-auru-smo-pm"
DELIVERY_TIMEOUT_S = 2.0  # per HLD: R1 push must never stall the collector
ICS_REGISTER_RETRY_S = 5.0

PM_JOB_DATA_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ocp-auru-ru-pm job definition",
    "type": "object",
    "properties": {"ru_id": {"type": "string", "description": "optional filter: only this RU's reports"}},
    "additionalProperties": True,
}


# --------------------------------------------------------------------------------------
# Registry (shared by both profiles)
# --------------------------------------------------------------------------------------
class Job(BaseModel):
    job_id: str
    info_type_id: str
    owner: str = ""
    target_uri: str
    job_definition: dict[str, Any] = Field(default_factory=dict)
    status_notification_uri: str = ""
    last_updated: str = ""


class JobRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self.info_types: dict[str, dict[str, Any]] = {}
        self.producers: dict[str, dict[str, Any]] = {}
        self.jobs: dict[str, Job] = {}
        self.services: dict[str, dict[str, Any]] = {}

    # -- jobs -------------------------------------------------------------------------
    def put_job(self, job: Job) -> bool:
        """Insert or replace; returns True when created."""
        job.last_updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            created = job.job_id not in self.jobs
            self.jobs[job.job_id] = job
        return created

    def delete_job(self, job_id: str) -> bool:
        with self._lock:
            return self.jobs.pop(job_id, None) is not None

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self.jobs.get(job_id)

    def jobs_for_type(self, info_type_id: str) -> list[Job]:
        with self._lock:
            return [j for j in self.jobs.values() if j.info_type_id == info_type_id]

    def job_ids(self, info_type_id: Optional[str] = None, owner: Optional[str] = None) -> list[str]:
        with self._lock:
            return [j.job_id for j in self.jobs.values()
                    if (info_type_id is None or j.info_type_id == info_type_id) and (owner is None or j.owner == owner)]

    # -- types / producers / services (lite ICS + catalogue emulation) ------------------
    def put_info_type(self, type_id: str, body: dict[str, Any]) -> bool:
        with self._lock:
            created = type_id not in self.info_types
            self.info_types[type_id] = body
        return created

    def put_producer(self, producer_id: str, body: dict[str, Any]) -> bool:
        with self._lock:
            created = producer_id not in self.producers
            self.producers[producer_id] = body
        return created

    def producers_for_type(self, type_id: str) -> list[str]:
        with self._lock:
            return [pid for pid, p in self.producers.items() if type_id in p.get("supported_info_types", [])]

    def put_service(self, name: str, body: dict[str, Any]) -> bool:
        with self._lock:
            created = name not in self.services
            existing = self.services.get(name, {})
            self.services[name] = {**body, "name": name,
                                   "registrationDate": existing.get("registrationDate") or datetime.now(timezone.utc).date().isoformat()}
        return created


# --------------------------------------------------------------------------------------
# Delivery (one code path)
# --------------------------------------------------------------------------------------
class Deliverer:
    def __init__(self, registry: JobRegistry, timeout_s: float = DELIVERY_TIMEOUT_S):
        self.registry = registry
        self.timeout_s = timeout_s
        self.stats: dict[str, Any] = {"delivered": 0, "failed": 0, "last_error": None, "last_target": None}

    def deliver(self, envelope: dict[str, Any]) -> int:
        """POST a measurement envelope to every `ocp-auru-ru-pm` job; returns the number of successful posts."""
        ru_id = envelope.get("event", {}).get("commonEventHeader", {}).get("sourceName")
        ok = 0
        for job in self.registry.jobs_for_type(PM_INFO_TYPE):
            wanted = job.job_definition.get("ru_id")
            if wanted and wanted != ru_id:
                continue
            try:
                r = httpx.post(job.target_uri, json=envelope, timeout=self.timeout_s)
                r.raise_for_status()
                ok += 1
                self.stats["delivered"] += 1
                self.stats["last_target"] = job.target_uri
            except Exception as exc:
                self.stats["failed"] += 1
                self.stats["last_error"] = f"{job.job_id} -> {job.target_uri}: {type(exc).__name__}: {exc}"
                log.warning("R1 delivery failed for job %s -> %s: %s", job.job_id, job.target_uri, exc)
        return ok


# --------------------------------------------------------------------------------------
# Producer-side routes (both profiles): what the real ICS calls in `full`
# --------------------------------------------------------------------------------------
def producer_router(registry: JobRegistry) -> APIRouter:
    r = APIRouter(prefix="/r1/producer", tags=["r1-producer"])

    def _job_from_ics(body: dict[str, Any], job_id: Optional[str]) -> Job:
        # ICS 1.2.x ProducerJobInfo: info_job_identity, info_type_identity, target_uri, owner, info_job_data, last_updated
        jid = job_id or body.get("info_job_identity") or body.get("ei_job_identity")
        if not jid:
            raise HTTPException(400, "info_job_identity missing")
        target = body.get("target_uri")
        if not target:
            raise HTTPException(400, "target_uri missing")
        return Job(job_id=jid, info_type_id=body.get("info_type_identity") or body.get("ei_type_identity") or PM_INFO_TYPE,
                   owner=body.get("owner", ""), target_uri=target, job_definition=body.get("info_job_data") or {})

    @r.post("/job")
    def job_create(body: dict[str, Any]):
        job = _job_from_ics(body, None)
        registry.put_job(job)
        log.info("R1 job (from ICS) %s type=%s -> %s", job.job_id, job.info_type_id, job.target_uri)
        return {"ok": True, "job_id": job.job_id}

    @r.post("/job/{job_id}")
    def job_create_by_id(job_id: str, body: dict[str, Any]):
        job = _job_from_ics(body, job_id)
        registry.put_job(job)
        log.info("R1 job (from ICS) %s type=%s -> %s", job.job_id, job.info_type_id, job.target_uri)
        return {"ok": True, "job_id": job.job_id}

    @r.delete("/job/{job_id}")
    def job_delete(job_id: str):
        registry.delete_job(job_id)
        return {"ok": True}

    @r.get("/supervision")
    def supervision():
        return {"ok": True, "producer": PRODUCER_ID}

    return r


# --------------------------------------------------------------------------------------
# `lite`: ICS-compatible subset (NONRTRIC ICS 1.2.x shapes) served in-process
# --------------------------------------------------------------------------------------
def ics_lite_router(registry: JobRegistry) -> APIRouter:
    r = APIRouter(tags=["ics-lite"])

    # -- producer API ---------------------------------------------------------------
    @r.put("/data-producer/v1/info-types/{type_id}")
    def put_type(type_id: str, body: dict[str, Any], response: Response):
        if "info_job_data_schema" not in body:
            raise HTTPException(400, "info_job_data_schema missing")
        response.status_code = 201 if registry.put_info_type(type_id, body) else 200
        return {}

    @r.get("/data-producer/v1/info-types/{type_id}")
    def get_type(type_id: str):
        if type_id not in registry.info_types:
            raise HTTPException(404, f"info type {type_id} not found")
        return registry.info_types[type_id]

    @r.delete("/data-producer/v1/info-types/{type_id}", status_code=204)
    def delete_type(type_id: str):
        if registry.info_types.pop(type_id, None) is None:
            raise HTTPException(404, f"info type {type_id} not found")
        return Response(status_code=204)

    @r.get("/data-producer/v1/info-types")
    def list_types_producer():
        return list(registry.info_types)

    @r.put("/data-producer/v1/info-producers/{producer_id}")
    def put_producer(producer_id: str, body: dict[str, Any], response: Response):
        for k in ("supported_info_types", "info_job_callback_url"):
            if k not in body:
                raise HTTPException(400, f"{k} missing")
        unknown = [t for t in body["supported_info_types"] if t not in registry.info_types]
        if unknown:
            raise HTTPException(404, f"info type(s) not found: {unknown}")
        response.status_code = 201 if registry.put_producer(producer_id, body) else 200
        return {}

    @r.get("/data-producer/v1/info-producers/{producer_id}")
    def get_producer(producer_id: str):
        if producer_id not in registry.producers:
            raise HTTPException(404, f"producer {producer_id} not found")
        return registry.producers[producer_id]

    @r.delete("/data-producer/v1/info-producers/{producer_id}", status_code=204)
    def delete_producer(producer_id: str):
        if registry.producers.pop(producer_id, None) is None:
            raise HTTPException(404, f"producer {producer_id} not found")
        return Response(status_code=204)

    @r.get("/data-producer/v1/info-producers/{producer_id}/status")
    def producer_status(producer_id: str):
        if producer_id not in registry.producers:
            raise HTTPException(404, f"producer {producer_id} not found")
        return {"operational_state": "ENABLED"}

    @r.get("/data-producer/v1/info-producers")
    def list_producers(infoTypeId: Optional[str] = None):
        return registry.producers_for_type(infoTypeId) if infoTypeId else list(registry.producers)

    # -- consumer API ---------------------------------------------------------------
    @r.put("/data-consumer/v1/info-jobs/{job_id}")
    def put_job(job_id: str, body: dict[str, Any], response: Response):
        for k in ("info_type_id", "job_result_uri"):
            if not body.get(k):
                raise HTTPException(400, f"{k} missing")
        if body["info_type_id"] not in registry.info_types:
            raise HTTPException(404, f"info type {body['info_type_id']} not found")
        job = Job(job_id=job_id, info_type_id=body["info_type_id"], owner=body.get("job_owner", ""),
                  target_uri=body["job_result_uri"], job_definition=body.get("job_definition") or {},
                  status_notification_uri=body.get("status_notification_uri", ""))
        created = registry.put_job(job)
        log.info("R1 job %s type=%s owner=%s -> %s", job_id, job.info_type_id, job.owner, job.target_uri)
        response.status_code = 201 if created else 200
        return {}

    @r.get("/data-consumer/v1/info-jobs/{job_id}")
    def get_job(job_id: str):
        job = registry.get_job(job_id)
        if job is None:
            raise HTTPException(404, f"job {job_id} not found")
        return {"info_type_id": job.info_type_id, "job_owner": job.owner, "job_definition": job.job_definition,
                "job_result_uri": job.target_uri, "status_notification_uri": job.status_notification_uri}

    @r.delete("/data-consumer/v1/info-jobs/{job_id}", status_code=204)
    def delete_job(job_id: str):
        if not registry.delete_job(job_id):
            raise HTTPException(404, f"job {job_id} not found")
        return Response(status_code=204)

    @r.get("/data-consumer/v1/info-jobs/{job_id}/status")
    def job_status(job_id: str):
        job = registry.get_job(job_id)
        if job is None:
            raise HTTPException(404, f"job {job_id} not found")
        return {"info_job_status": "ENABLED", "producers": registry.producers_for_type(job.info_type_id)}

    @r.get("/data-consumer/v1/info-jobs")
    def list_jobs(infoTypeId: Optional[str] = None, owner: Optional[str] = None):
        return registry.job_ids(infoTypeId, owner)

    @r.get("/data-consumer/v1/info-types")
    def list_types_consumer():
        return list(registry.info_types)

    @r.get("/data-consumer/v1/info-types/{type_id}")
    def get_type_consumer(type_id: str):
        if type_id not in registry.info_types:
            raise HTTPException(404, f"info type {type_id} not found")
        return {"job_data_schema": registry.info_types[type_id].get("info_job_data_schema", {})}

    return r


# --------------------------------------------------------------------------------------
# `lite`: rApp-Catalogue-compatible subset (NONRTRIC rApp Catalogue 1.0.x shapes)
# --------------------------------------------------------------------------------------
def catalogue_lite_router(registry: JobRegistry) -> APIRouter:
    r = APIRouter(prefix="/services", tags=["catalogue-lite"])

    @r.put("/{service_name}")
    def put_service(service_name: str, body: dict[str, Any], response: Response):
        if not body.get("version"):
            raise HTTPException(400, "version missing")
        created = registry.put_service(service_name, {k: body.get(k, "") for k in ("version", "display_name", "description")})
        log.info("R1 catalogue: service %s v%s registered", service_name, body["version"])
        response.status_code = 201 if created else 200
        return {}

    @r.get("/{service_name}")
    def get_service(service_name: str):
        if service_name not in registry.services:
            raise HTTPException(404, f"service {service_name} not found")
        return registry.services[service_name]

    @r.delete("/{service_name}", status_code=204)
    def delete_service(service_name: str):
        if registry.services.pop(service_name, None) is None:
            raise HTTPException(404, f"service {service_name} not found")
        return Response(status_code=204)

    @r.get("")
    def list_services():
        return list(registry.services.values())

    return r


# --------------------------------------------------------------------------------------
# Producer registration: local (`lite`) or real ICS with background retry (`full`)
# --------------------------------------------------------------------------------------
def producer_registration(cfg: Settings) -> tuple[dict[str, Any], dict[str, Any]]:
    """(info-type body, producer body) as sent to ICS (`full`) or stored locally (`lite`)."""
    type_body = {"info_job_data_schema": PM_JOB_DATA_SCHEMA}
    producer_body = {
        "supported_info_types": [PM_INFO_TYPE],
        "info_job_callback_url": f"{cfg.smo_public_url}/r1/producer/job",
        "info_producer_supervision_callback_url": f"{cfg.smo_public_url}/r1/producer/supervision",
    }
    return type_body, producer_body


class Registrar:
    """Keeps `state` for /health: {'profile', 'state': pending|registered|failed|local, 'ics_url', 'attempts', 'last_error'}."""

    def __init__(self, cfg: Settings, registry: JobRegistry, retry_s: float = ICS_REGISTER_RETRY_S):
        self.cfg, self.registry, self.retry_s = cfg, registry, retry_s
        self.state: dict[str, Any] = {"profile": cfg.profile, "state": "pending", "ics_url": cfg.ics_url,
                                      "info_type": PM_INFO_TYPE, "producer": PRODUCER_ID, "attempts": 0, "last_error": None}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def register_local(self) -> None:
        type_body, producer_body = producer_registration(self.cfg)
        self.registry.put_info_type(PM_INFO_TYPE, type_body)
        self.registry.put_producer(PRODUCER_ID, producer_body)
        self.state.update(state="local", attempts=1)

    def register_remote_once(self) -> bool:
        type_body, producer_body = producer_registration(self.cfg)
        base = self.cfg.ics_url_full.rstrip("/")
        self.state["attempts"] += 1
        try:
            with httpx.Client(timeout=DELIVERY_TIMEOUT_S) as c:
                r = c.put(f"{base}/data-producer/v1/info-types/{PM_INFO_TYPE}", json=type_body)
                r.raise_for_status()
                r = c.put(f"{base}/data-producer/v1/info-producers/{PRODUCER_ID}", json=producer_body)
                r.raise_for_status()
        except Exception as exc:
            self.state.update(state="pending", last_error=f"{type(exc).__name__}: {exc}")
            return False
        self.state.update(state="registered", last_error=None)
        log.info("R1 DME: registered info-type %s and producer %s at %s", PM_INFO_TYPE, PRODUCER_ID, base)
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self.register_remote_once():
                return
            log.warning("R1 DME: ICS registration at %s failed (attempt %d): %s — retrying in %.0fs",
                        self.cfg.ics_url_full, self.state["attempts"], self.state["last_error"], self.retry_s)
            self._stop.wait(self.retry_s)

    def start(self) -> None:
        if self.cfg.profile == "full":
            self._thread = threading.Thread(target=self._loop, name="ics-registrar", daemon=True)
            self._thread.start()
        else:
            self.register_local()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(2)

    def wait(self, timeout_s: float) -> bool:
        """Block until registered (tests); returns the registered flag."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and self.state["state"] not in ("registered", "local"):
            time.sleep(0.02)
        return self.state["state"] in ("registered", "local")
