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
"""SMO R1 DME: lite ICS/catalogue-compatible routes, one delivery path, `full` registration with the real ICS mocked."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from ocp_auru.smo import dme
from ocp_auru.smo.app import create_app
from _smo_helpers import Receiver, make_settings, measurement


@pytest.fixture
def receiver():
    r = Receiver(status=200).start()
    yield r
    r.stop()


def _wait(pred, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def test_lite_ics_job_lifecycle_and_delivery(monkeypatch, tmp_path, receiver):
    cfg = make_settings(monkeypatch, tmp_path)
    with TestClient(create_app(cfg)) as c:
        # the SMO self-registers as producer of ocp-auru-ru-pm in its own (lite) ICS
        assert dme.PM_INFO_TYPE in c.get("/data-consumer/v1/info-types").json()
        assert c.get(f"/data-producer/v1/info-types/{dme.PM_INFO_TYPE}").json()["info_job_data_schema"]["type"] == "object"
        assert c.get(f"/data-producer/v1/info-producers/{dme.PRODUCER_ID}/status").json() == {"operational_state": "ENABLED"}
        prod = c.get(f"/data-producer/v1/info-producers/{dme.PRODUCER_ID}").json()
        assert prod["supported_info_types"] == [dme.PM_INFO_TYPE] and prod["info_job_callback_url"].endswith("/r1/producer/job")

        # consumer (the rApp) creates its job, exactly as against NONRTRIC ICS 1.2.x
        job = {"info_type_id": dme.PM_INFO_TYPE, "job_owner": "es-rapp", "job_definition": {},
               "job_result_uri": f"{receiver.url}/r1/pm-delivery", "status_notification_uri": ""}
        assert c.put("/data-consumer/v1/info-jobs/es-rapp-pm", json=job).status_code == 201
        assert c.put("/data-consumer/v1/info-jobs/es-rapp-pm", json=job).status_code == 200  # update
        assert c.get("/data-consumer/v1/info-jobs").json() == ["es-rapp-pm"]
        assert c.get("/data-consumer/v1/info-jobs", params={"infoTypeId": "other"}).json() == []
        assert c.get("/data-consumer/v1/info-jobs/es-rapp-pm").json()["job_result_uri"] == job["job_result_uri"]
        st = c.get("/data-consumer/v1/info-jobs/es-rapp-pm/status").json()
        assert st["info_job_status"] == "ENABLED" and st["producers"] == [dme.PRODUCER_ID]
        assert c.put("/data-consumer/v1/info-jobs/bad", json={**job, "info_type_id": "nope"}).status_code == 404

        # a measurement arriving at the VES collector is delivered to the job's result URI
        env = measurement("oru-a")
        assert c.post("/eventListener/v7", json=env).status_code == 202
        assert _wait(lambda: len(receiver.posts("/r1/pm-delivery")) == 1)
        assert receiver.posts("/r1/pm-delivery")[0]["body"] == env
        assert c.get("/health").json()["r1"]["delivery"]["delivered"] == 1

        # job_definition.ru_id filters deliveries
        c.put("/data-consumer/v1/info-jobs/es-rapp-pm", json={**job, "job_definition": {"ru_id": "oru-b"}})
        c.post("/eventListener/v7", json=measurement("oru-a"))
        c.post("/eventListener/v7", json=measurement("oru-b"))
        assert _wait(lambda: len(receiver.posts("/r1/pm-delivery")) == 2)
        assert receiver.posts("/r1/pm-delivery")[1]["body"]["event"]["commonEventHeader"]["sourceName"] == "oru-b"

        assert c.delete("/data-consumer/v1/info-jobs/es-rapp-pm").status_code == 204
        assert c.get("/data-consumer/v1/info-jobs/es-rapp-pm").status_code == 404
        c.post("/eventListener/v7", json=measurement("oru-b"))
        time.sleep(0.1)
        assert len(receiver.posts("/r1/pm-delivery")) == 2


def test_lite_delivery_failure_is_logged_not_raised(monkeypatch, tmp_path):
    cfg = make_settings(monkeypatch, tmp_path)
    with TestClient(create_app(cfg)) as c:
        job = {"info_type_id": dme.PM_INFO_TYPE, "job_owner": "x", "job_definition": {},
               "job_result_uri": f"http://127.0.0.1:{cfg.rapp_port}/nowhere", "status_notification_uri": ""}
        assert c.put("/data-consumer/v1/info-jobs/j1", json=job).status_code == 201
        assert c.post("/eventListener/v7", json=measurement()).status_code == 202
        h = c.get("/health").json()
        assert h["r1"]["delivery"]["failed"] == 1 and "j1" in h["r1"]["delivery"]["last_error"]


def test_lite_external_producer_and_type_registration(monkeypatch, tmp_path):
    cfg = make_settings(monkeypatch, tmp_path)
    with TestClient(create_app(cfg)) as c:
        assert c.put("/data-producer/v1/info-types/t1", json={"info_job_data_schema": {"type": "object"}}).status_code == 201
        assert c.put("/data-producer/v1/info-types/t1", json={"info_job_data_schema": {"type": "object"}}).status_code == 200
        assert c.put("/data-producer/v1/info-types/t2", json={}).status_code == 400
        p = {"supported_info_types": ["t1"], "info_job_callback_url": "http://x/job", "info_producer_supervision_callback_url": "http://x/sup"}
        assert c.put("/data-producer/v1/info-producers/p1", json=p).status_code == 201
        assert c.put("/data-producer/v1/info-producers/p2", json={**p, "supported_info_types": ["missing"]}).status_code == 404
        assert c.get("/data-producer/v1/info-producers", params={"infoTypeId": "t1"}).json() == ["p1"]
        assert sorted(c.get("/data-producer/v1/info-types").json()) == sorted([dme.PM_INFO_TYPE, "t1"])
        assert c.delete("/data-producer/v1/info-producers/p1").status_code == 204
        assert c.delete("/data-producer/v1/info-types/t1").status_code == 204
        assert c.get("/data-producer/v1/info-types/t1").status_code == 404


def test_lite_catalogue_put_get_list_delete(monkeypatch, tmp_path):
    cfg = make_settings(monkeypatch, tmp_path)
    with TestClient(create_app(cfg)) as c:
        body = {"version": "0.1.0", "display_name": "Energy-saving rApp", "description": "carrier sleep policy"}
        assert c.put("/services/es-rapp", json=body).status_code == 201
        assert c.put("/services/es-rapp", json=body).status_code == 200
        got = c.get("/services/es-rapp").json()
        assert {k: got[k] for k in body} == body and got["name"] == "es-rapp"
        assert len(got["registrationDate"]) == 10  # YYYY-MM-DD like rApp Catalogue 1.0.x
        assert [s["name"] for s in c.get("/services").json()] == ["es-rapp"]
        assert c.put("/services/bad", json={"display_name": "no version"}).status_code == 400
        assert c.delete("/services/es-rapp").status_code == 204
        assert c.get("/services/es-rapp").status_code == 404
        assert c.get("/services").json() == []


def test_full_profile_registers_with_ics_and_accepts_pushed_job(monkeypatch, tmp_path, receiver):
    ics = Receiver(status=201).start()
    try:
        cfg = make_settings(monkeypatch, tmp_path, PROFILE="full", ICS_URL=ics.url, SMO_PUBLIC_URL="http://smo.test:1")
        app = create_app(cfg, registrar_retry_s=0.05)
        with TestClient(app) as c:
            assert app.state.registrar.wait(5.0)
            h = c.get("/health").json()
            assert h["profile"] == "full" and h["r1"]["registration"]["state"] == "registered"
            puts = [r for r in ics.requests if r["method"] == "PUT"]
            assert [r["path"] for r in puts] == [f"/data-producer/v1/info-types/{dme.PM_INFO_TYPE}",
                                                 f"/data-producer/v1/info-producers/{dme.PRODUCER_ID}"]
            assert puts[1]["body"]["info_job_callback_url"] == "http://smo.test:1/r1/producer/job"
            assert puts[1]["body"]["info_producer_supervision_callback_url"] == "http://smo.test:1/r1/producer/supervision"
            # lite-only routes are not served in `full`
            assert c.get("/data-consumer/v1/info-jobs").status_code == 404
            assert c.get("/r1/producer/supervision").status_code == 200

            # ICS pushes the consumer's job to the producer callback (ICS 1.2.x ProducerJobInfo shape)
            pushed = {"info_job_identity": "es-rapp-pm", "info_type_identity": dme.PM_INFO_TYPE, "owner": "es-rapp",
                      "target_uri": f"{receiver.url}/r1/pm-delivery", "info_job_data": {}, "last_updated": "2026-01-01T00:00:00Z"}
            assert c.post("/r1/producer/job", json=pushed).status_code == 200
            assert c.post("/r1/producer/job/es-rapp-pm", json=pushed).status_code == 200
            assert c.get("/health").json()["r1"]["jobs"] == ["es-rapp-pm"]
            env = measurement("oru-a")
            c.post("/eventListener/v7", json=env)
            assert _wait(lambda: len(receiver.posts("/r1/pm-delivery")) == 1)
            assert receiver.posts("/r1/pm-delivery")[0]["body"] == env
            assert c.delete("/r1/producer/job/es-rapp-pm").status_code == 200
            assert c.get("/health").json()["r1"]["jobs"] == []
    finally:
        ics.stop()


def test_full_profile_retries_until_ics_answers(monkeypatch, tmp_path):
    ics = Receiver(status=500)  # ICS "down": answers 5xx until flipped
    ics.start()
    try:
        cfg = make_settings(monkeypatch, tmp_path, PROFILE="full", ICS_URL=ics.url)
        app = create_app(cfg, registrar_retry_s=0.05)
        with TestClient(app) as c:
            assert _wait(lambda: app.state.registrar.state["attempts"] >= 2)
            h = c.get("/health").json()["r1"]["registration"]
            assert h["state"] == "pending" and "500" in (h["last_error"] or "")
            ics.status = 201
            assert app.state.registrar.wait(5.0)
            assert c.get("/health").json()["r1"]["registration"]["state"] == "registered"
    finally:
        ics.stop()
