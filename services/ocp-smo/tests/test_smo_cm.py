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
"""SMO CM routes: CM request -> CM adapter (a recording adapter, then the reference adapter with a fake NETCONF client)."""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from lxml import etree

from ocp_auru.common import yang
from ocp_auru.common.events import Active, AdapterResult, NetconfResult
from ocp_auru.smo import o1_adapter
from ocp_auru.smo.app import create_app
from ocp_auru.smo.mplane_client import MplaneError
from _smo_helpers import make_settings

REQ = {"ru_id": "oru-a", "carrier": "txcarrier0", "active": "SLEEP",
       "ruapp_policy": {"wake_on_prach": True, "prach_energy_threshold_db": 10, "min_hold_after_wake_s": 30, "report_interval_s": 5},
       "reason": "p_low=0.91"}


class RecordingAdapter:
    """A CmAdapter that records what the routes hand it and answers with a canned result."""

    name = "recording"

    def __init__(self, result: AdapterResult, delay_s: float = 0.0):
        self.result, self.delay_s = result, delay_s
        self.calls: list[tuple] = []

    def describe(self):
        return {"name": self.name}

    def set_carrier_state(self, req, include_policy=True):
        self.calls.append((req, include_policy))
        time.sleep(self.delay_s)
        return self.result

    def get_carriers(self):
        return {"ok": True, "carriers": [{"name": "txcarrier0", "active": "SLEEP", "state": "DISABLED"}], "netconf": {"ok": True, "rtt_ms": 0.2}}


class FakeMplaneClient:
    """Stands in for `MplaneClient`: records the `<config>`, fails when the carrier is named 'bad'."""

    target = "fake:0"

    def __init__(self):
        self.configs: list[str] = []

    def edit_config(self, config_xml: str) -> NetconfResult:
        self.configs.append(config_xml)
        if "<name>bad</name>" in config_xml:
            raise MplaneError("rpc-error: carrier 'bad' does not exist")
        time.sleep(0.002)  # the session setup the SMO-side round trip includes
        return NetconfResult(rtt_ms=0.5, connect_ms=1.5, target=self.target)

    def get_carrier_status(self):
        return {"carriers": [{"name": "txcarrier0", "active": "SLEEP", "state": "DISABLED"}], "netconf": {"ok": True, "rtt_ms": 0.2}}


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    return make_settings(monkeypatch, tmp_path)


def test_cm_carrier_success_measures_rtt(cfg):
    adapter = RecordingAdapter(AdapterResult(ok=True, netconf_rtt_ms=12.3, detail="edit-config on 127.0.0.1:1"), delay_s=0.01)
    with TestClient(create_app(cfg, adapter=adapter)) as c:
        r = c.post("/o1/cm/carrier", json=REQ)
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True and body["netconf_rtt_ms"] == 12.3 and body["cm_rtt_ms"] >= 10.0
        assert body["ru_id"] == "oru-a" and body["carrier"] == "txcarrier0" and body["active"] == "SLEEP"
        assert set(body) == {"ok", "ru_id", "carrier", "active", "cm_rtt_ms", "netconf_rtt_ms", "detail"}
        (req, include_policy), = adapter.calls
        assert req.carrier == "txcarrier0" and req.active == Active.SLEEP and include_policy is True
        assert req.ruapp_policy.model_dump() == REQ["ruapp_policy"]
        assert c.get("/o1/cm/last").json()["cm_rtt_ms"] == body["cm_rtt_ms"]
        assert c.get("/health").json()["cm_adapter"] == {"name": "recording"}


def test_cm_carrier_adapter_error_is_ok_false_with_rtt(cfg):
    adapter = RecordingAdapter(AdapterResult(ok=False, detail="NETCONF edit-config failed: rpc-error: no such carrier"))
    with TestClient(create_app(cfg, adapter=adapter)) as c:
        r = c.post("/o1/cm/carrier", json=REQ)
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False and "no such carrier" in body["detail"] and body["cm_rtt_ms"] >= 0.0 and body["netconf_rtt_ms"] is None


def test_cm_carrier_raising_adapter_keeps_the_one_shape_contract(cfg):
    class Raising(RecordingAdapter):
        def set_carrier_state(self, req, include_policy=True):
            raise RuntimeError("boom")

    with TestClient(create_app(cfg, adapter=Raising(AdapterResult(ok=True)))) as c:
        body = c.post("/o1/cm/carrier", json=REQ).json()
        assert body["ok"] is False and "RuntimeError: boom" in body["detail"] and body["cm_rtt_ms"] >= 0.0


def test_cm_carrier_ru_unreachable(cfg):
    # default adapter (mplane-direct); nothing listens on cfg.ru_netconf_port (ephemeral 20000+ pick)
    with TestClient(create_app(cfg)) as c:
        r = c.post("/o1/cm/carrier", json=REQ)
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False and body["detail"].startswith("NETCONF edit-config failed") and body["cm_rtt_ms"] >= 0.0
        st = c.get("/o1/cm/carriers").json()
        assert st["ok"] is False and st["detail"].startswith("NETCONF get failed") and st["carriers"] == []
        h = c.get("/health").json()["cm_adapter"]
        assert h["name"] == "mplane-direct" and h["netconf_target"] == f"127.0.0.1:{cfg.ru_netconf_port}"


def test_cm_carrier_rejects_bad_request(cfg):
    with TestClient(create_app(cfg, adapter=RecordingAdapter(AdapterResult(ok=True)))) as c:
        assert c.post("/o1/cm/carrier", json={**REQ, "active": "OFF"}).status_code == 422


def test_cm_carrier_through_reference_adapter_with_fake_netconf(cfg):
    fake = FakeMplaneClient()
    adapter = o1_adapter.MplaneDirectAdapter(cfg, client=fake)
    with TestClient(create_app(cfg, adapter=adapter)) as c:
        body = c.post("/o1/cm/carrier", json=REQ).json()
        assert body["ok"] is True and body["netconf_rtt_ms"] == 0.5 and body["cm_rtt_ms"] > body["netconf_rtt_ms"]
        assert body["detail"] == "edit-config on fake:0"
        edits = yang.parse_carrier_edits(etree.fromstring(fake.configs[0]))
        assert edits == {"txcarrier0": {"active": Active.SLEEP, "policy": REQ["ruapp_policy"]}}
        st = c.get("/o1/cm/carriers").json()
        assert st["ok"] is True and st["carriers"][0]["state"] == "DISABLED"


def test_cm_ruapp_policy_can_be_left_out_of_the_edit(monkeypatch, tmp_path):
    cfg = make_settings(monkeypatch, tmp_path, CM_RUAPP_POLICY="0")
    fake = FakeMplaneClient()
    with TestClient(create_app(cfg, adapter=o1_adapter.MplaneDirectAdapter(cfg, client=fake))) as c:
        assert c.post("/o1/cm/carrier", json={**REQ, "active": "ACTIVE"}).json()["ok"] is True
    # the standard `active` leaf only: no extension namespace in the payload
    assert yang.parse_carrier_edits(etree.fromstring(fake.configs[0])) == {"txcarrier0": {"active": Active.ACTIVE, "policy": None}}
    assert yang.NS_RUAPP not in fake.configs[0]


def test_netconf_rpc_error_surfaces_as_ok_false(cfg):
    fake = FakeMplaneClient()
    with TestClient(create_app(cfg, adapter=o1_adapter.MplaneDirectAdapter(cfg, client=fake))) as c:
        body = c.post("/o1/cm/carrier", json={**REQ, "carrier": "bad"}).json()
    assert body["ok"] is False and "rpc-error" in body["detail"] and "does not exist" in body["detail"]
    assert body["netconf_rtt_ms"] is None and len(fake.configs) == 1
