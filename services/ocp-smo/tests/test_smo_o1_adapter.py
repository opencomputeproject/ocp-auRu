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
"""CM adapter seam: registry, selection by config, and the interface the CM routes rely on."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ocp_auru.common.events import AdapterResult
from ocp_auru.smo import o1_adapter
from ocp_auru.smo.app import create_app
from _smo_helpers import make_settings

REQ = {"ru_id": "oru-a", "carrier": "txcarrier0", "active": "SLEEP"}


def test_reference_adapter_is_the_default(monkeypatch, tmp_path):
    cfg = make_settings(monkeypatch, tmp_path)
    assert cfg.cm_adapter == "mplane-direct" and "mplane-direct" in o1_adapter.available()
    adapter = o1_adapter.create(cfg)
    assert isinstance(adapter, o1_adapter.MplaneDirectAdapter) and isinstance(adapter, o1_adapter.CmAdapter)
    assert adapter.describe()["netconf_target"] == cfg.ru_netconf_target


def test_unknown_adapter_name_fails_at_start(monkeypatch, tmp_path):
    cfg = make_settings(monkeypatch, tmp_path, CM_ADAPTER="no-such-adapter")
    with pytest.raises(ValueError, match="no-such-adapter.*mplane-direct"):
        create_app(cfg)


def test_a_registered_adapter_is_selected_by_config(monkeypatch, tmp_path):
    """The seam a hierarchical (through an O-DU) path would use: register a factory, select it by name."""
    seen = []

    class ViaDu:
        name = "test-via-du"

        def __init__(self, cfg):
            self.cfg = cfg

        def describe(self):
            return {"name": self.name}

        def set_carrier_state(self, req, include_policy=True):
            seen.append((req.carrier, req.active.value, include_policy))
            return AdapterResult(ok=True, netconf_rtt_ms=None, detail="applied by the test adapter")

        def get_carriers(self):
            return {"ok": True, "carriers": []}

    monkeypatch.setitem(o1_adapter._REGISTRY, ViaDu.name, ViaDu)
    cfg = make_settings(monkeypatch, tmp_path, CM_ADAPTER=ViaDu.name)
    with TestClient(create_app(cfg)) as c:
        body = c.post("/o1/cm/carrier", json=REQ).json()
        assert body["ok"] is True and body["netconf_rtt_ms"] is None and body["detail"] == "applied by the test adapter"
        assert c.get("/health").json()["cm_adapter"] == {"name": "test-via-du"}
        assert c.get("/o1/cm/carriers").json() == {"ok": True, "carriers": []}
    assert seen == [("txcarrier0", "SLEEP", True)]
