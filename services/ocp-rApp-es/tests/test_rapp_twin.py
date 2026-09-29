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
"""es-rApp twin guardrail (ADR-7): veto on coverage loss, advisory on any error, auto-cell
discovery via GET /v1/cells, and the verdict recorded with cell/deltas/rung/latency."""
from __future__ import annotations

import httpx
import pytest

from ocp_auru.common.config import Settings
from ocp_auru.rapp import twin as twin_mod

GENERIC_RUNG = "simulation, generic 3GPP defaults (uncalibrated)"


class _Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)

    def json(self):
        return self._payload


def _evaluate_reply(cov=-0.01, rung=GENERIC_RUNG):
    return {"guardrail_kpis": {"rsrp_p5": -101.2, "sinr_p5": -3.1, "outage_rate": 0.04, "coverage_rate": 0.93},
            "baseline_kpis": {"rsrp_p5": -100.9, "sinr_p5": -2.9, "outage_rate": 0.03, "coverage_rate": 0.93 - cov},
            "delta_vs_baseline": {"rsrp_p5": -0.3, "sinr_p5": -0.2, "outage_rate": 0.01, "coverage_rate": cov},
            "serving_share": {"g-s1-b-n78": 0.5, "g-s1-c-n78": 0.5}, "mode": "device_mean", "rung": rung}


def _cells_reply():
    return {"cells": [{"cell_id": "g-s1-a-n78", "freq_mhz": 3500.0}, {"cell_id": "g-s1-b-n78", "freq_mhz": 3500.0}], "count": 2}


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setenv("TWIN_URL", "http://twin.invalid:1/")
    monkeypatch.setenv("TWIN_CELL_ID", "")
    return Settings()


@pytest.fixture
def no_get(monkeypatch):
    """Cell discovery unreachable (no GET route): the guard falls back to the carrier name."""
    def boom(url, timeout=None):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(twin_mod.httpx, "get", boom)


def test_disabled_without_url(monkeypatch):
    monkeypatch.setenv("TWIN_URL", "")
    g = twin_mod.TwinGuard(Settings())
    assert not g.enabled and g.evaluate_sleep("txcarrier0") == (True, {"enabled": False})
    assert g.health()["status"] == "disabled" and g.metrics()["twin_checks"] == 0


def test_veto_on_coverage_drop(cfg, monkeypatch, no_get):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json, timeout))
        return _Resp(_evaluate_reply(cov=-0.08))

    monkeypatch.setattr(twin_mod.httpx, "post", fake_post)
    g = twin_mod.TwinGuard(cfg)
    allowed, detail = g.evaluate_sleep("txcarrier0")
    assert allowed is False and detail["blocked"] is True and detail["coverage_rate_delta"] == -0.08
    assert detail["verdict"] == "blocked" and detail["status"] == "ok" and detail["rung"] == GENERIC_RUNG
    assert detail["delta_vs_baseline"]["coverage_rate"] == -0.08 and detail["latency_ms"] >= 0
    assert calls[0][0] == "http://twin.invalid:1/v1/evaluate"
    assert calls[0][1] == {"cell_configs": [{"cell_id": "txcarrier0", "on_off": False}]}  # discovery failed -> carrier name
    assert calls[0][2] == cfg.twin_timeout_s
    assert detail["cell_source"].startswith("unresolved") or "carrier" in detail["cell_source"]
    m = g.metrics()
    assert m["twin_checks"] == 1 and m["twin_blocks"] == 1 and m["last_twin_verdict"]["verdict"] == "blocked"


def test_allow_on_small_drop_and_uses_twin_cell_id(cfg, monkeypatch):
    seen = {}
    gets = []
    monkeypatch.setenv("TWIN_CELL_ID", "cell-7")
    monkeypatch.setattr(twin_mod.httpx, "get", lambda url, timeout=None: gets.append(url) or _Resp(_cells_reply()))
    monkeypatch.setattr(twin_mod.httpx, "post", lambda url, json=None, timeout=None: seen.update(json) or _Resp(_evaluate_reply(cov=-0.01)))
    g = twin_mod.TwinGuard(Settings())
    allowed, detail = g.evaluate_sleep("txcarrier0")
    assert allowed is True and detail["blocked"] is False and detail["verdict"] == "allowed"
    assert seen["cell_configs"][0]["cell_id"] == "cell-7" and detail["cell_id"] == "cell-7" and detail["cell_source"] == "TWIN_CELL_ID"
    assert gets == []  # an explicit TWIN_CELL_ID never triggers discovery
    assert g.health()["cell_id"] == "cell-7" and g.health()["status"] == "ok"


def test_auto_cell_discovery_picks_first_cell_once(cfg, monkeypatch):
    gets, posts = [], []
    monkeypatch.setattr(twin_mod.httpx, "get", lambda url, timeout=None: gets.append(url) or _Resp(_cells_reply()))
    monkeypatch.setattr(twin_mod.httpx, "post", lambda url, json=None, timeout=None: posts.append(json) or _Resp(_evaluate_reply()))
    g = twin_mod.TwinGuard(cfg)
    h0 = g.health()
    assert h0["cell_id"] is None and h0["status"] == "unknown" and h0["cell_source"] == "unresolved"
    allowed, detail = g.evaluate_sleep("txcarrier0")
    assert allowed and detail["cell_id"] == "g-s1-a-n78" and detail["cell_source"] == "auto:/v1/cells[0]"
    assert posts[0] == {"cell_configs": [{"cell_id": "g-s1-a-n78", "on_off": False}]}
    g.evaluate_sleep("txcarrier0")
    assert gets == ["http://twin.invalid:1/v1/cells"]  # cached after the first success
    h = g.health()
    assert h["cell_id"] == "g-s1-a-n78" and h["cells_listed"] == 2 and h["status"] == "ok" and h["rung"] == GENERIC_RUNG
    assert h["checks"] == 2 and h["blocks"] == 0 and "rung_warning" not in h
    assert h["expected_rung"] == "generic 3GPP defaults (uncalibrated)"


def test_discovery_retried_until_the_twin_lists_cells(cfg, monkeypatch):
    state = {"up": False}

    def get(url, timeout=None):
        if not state["up"]:
            raise httpx.ConnectError("refused")
        return _Resp(_cells_reply())

    posts = []
    monkeypatch.setattr(twin_mod.httpx, "get", get)
    monkeypatch.setattr(twin_mod.httpx, "post", lambda url, json=None, timeout=None: posts.append(json) or _Resp(_evaluate_reply()))
    g = twin_mod.TwinGuard(cfg)
    assert g.discover_cell() is None and g.health()["status"] == "unreachable" and "cell discovery" in g.health()["last_error"]
    g.evaluate_sleep("txcarrier0")
    assert posts[-1]["cell_configs"][0]["cell_id"] == "txcarrier0"
    state["up"] = True
    g.evaluate_sleep("txcarrier0")
    assert posts[-1]["cell_configs"][0]["cell_id"] == "g-s1-a-n78" and g.health()["status"] == "ok"


def test_probe_reads_rung_and_warns_when_not_generic(cfg, monkeypatch):
    monkeypatch.setattr(twin_mod.httpx, "get", lambda url, timeout=None: _Resp({"status": "ok", "estate_loaded": True, "rung": "simulation, calibrated to supplied measurement data"}
                                                                                if url.endswith("/health") else _cells_reply()))
    g = twin_mod.TwinGuard(cfg)
    h = g.probe()
    assert h["estate_loaded"] is True
    view = g.health()
    assert view["status"] == "ok" and view["cell_id"] == "g-s1-a-n78" and "calibrated" in view["rung"]
    assert "rung_warning" in view  # TWIN_GENERIC=1 (default) expects the generic rung


def test_probe_ignores_the_rung_of_a_twin_with_no_model(cfg, monkeypatch):
    """`/health` of an empty twin (`model: null`) carries no model rung: whatever string sits in
    the field, the rApp must not present it as the guardrail's rung."""
    monkeypatch.setattr(twin_mod.httpx, "get", lambda url, timeout=None: _Resp({"status": "ok", "estate_loaded": False, "model": None,
                                                                                "rung": "simulation, calibrated to supplied measurement data"}
                                                                               if url.endswith("/health") else _cells_reply()))
    g = twin_mod.TwinGuard(cfg)
    g.probe()
    view = g.health()
    assert view["status"] == "ok" and view["rung"] is None and "rung_warning" not in view
    # ... and a loaded generic model's rung is adopted as before
    monkeypatch.setattr(twin_mod.httpx, "get", lambda url, timeout=None: _Resp({"status": "ok", "estate_loaded": True, "model": "generic",
                                                                                "rung": GENERIC_RUNG}
                                                                               if url.endswith("/health") else _cells_reply()))
    g.probe()
    assert g.health()["rung"] == GENERIC_RUNG


def test_rung_warning_when_generic_is_not_expected(monkeypatch):
    """TWIN_GENERIC=0 says the operator expects a calibrated twin; a generic rung is then a mismatch
    worth saying out loud, symmetric to the default case."""
    monkeypatch.setenv("TWIN_URL", "http://twin.invalid:1")
    monkeypatch.setenv("TWIN_GENERIC", "0")
    cfg0 = Settings()
    monkeypatch.setattr(twin_mod.httpx, "get", lambda url, timeout=None: _Resp({"status": "ok", "estate_loaded": True, "model": "generic",
                                                                                "rung": GENERIC_RUNG}
                                                                               if url.endswith("/health") else _cells_reply()))
    g = twin_mod.TwinGuard(cfg0)
    g.probe()
    view = g.health()
    assert view["expected_rung"] == "calibrated (locally supplied data)" and view["rung"] == GENERIC_RUNG
    assert "rung_warning" in view and "TWIN_GENERIC=0" in view["rung_warning"]


def test_errors_are_advisory(cfg, monkeypatch, no_get):
    def boom(url, json=None, timeout=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(twin_mod.httpx, "post", boom)
    g = twin_mod.TwinGuard(cfg)
    allowed, detail = g.evaluate_sleep("txcarrier0")
    assert allowed is True and "error" in detail and detail["status"] == "unreachable" and detail["verdict"] == "allowed (twin unreachable)"
    assert g.health()["status"] == "unreachable" and g.metrics()["twin_errors"] == 1 and g.metrics()["twin_status"] == "unreachable"
    monkeypatch.setattr(twin_mod.httpx, "post", lambda url, json=None, timeout=None: _Resp({"unexpected": 1}))
    allowed, detail = g.evaluate_sleep("txcarrier0")
    assert allowed is True and detail["status"] == "ok" and detail["coverage_rate_delta"] is None and "note" in detail
    monkeypatch.setattr(twin_mod.httpx, "post", lambda url, json=None, timeout=None: _Resp({}, status=500))
    allowed, detail = g.evaluate_sleep("txcarrier0")
    assert allowed is True and detail["status"] == "error"
    assert g.metrics()["twin_checks"] == 3 and g.metrics()["twin_blocks"] == 0
