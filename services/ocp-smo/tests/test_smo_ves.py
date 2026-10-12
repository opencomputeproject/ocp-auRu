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
"""SMO VES collector: ingestion, latest PM, notification ring with seq numbers, twin ingest sink, health."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ocp_auru.smo.app import create_app
from _smo_helpers import make_settings, measurement, notification


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    return make_settings(monkeypatch, tmp_path)


def test_ves_ingest_pm_events_and_twin_sink(cfg):
    with TestClient(create_app(cfg)) as c:
        r = c.post("/eventListener/v7", json=measurement("oru-a"))
        assert r.status_code == 202 and r.json()["ok"] is True and r.json()["domain"] == "measurement"
        r = c.post("/eventListener/v7", json=notification("oru-a", "DISABLED"))
        assert r.status_code == 202 and r.json()["seq"] == 1
        r = c.post("/eventListener/v7", json=notification("oru-a", "READY"))
        assert r.json()["seq"] == 2

        latest = c.get("/o1/pm/latest").json()
        assert list(latest) == ["oru-a"]
        carriers = latest["oru-a"]["event"]["measurementFields"]["additionalObjects"]
        assert carriers[0]["name"] == "txcarrier0" and carriers[0]["wake_latency_us"]["last"] == 42.0
        assert c.get("/o1/pm/latest", params={"ru_id": "nope"}).json() == {}

        ev = c.get("/o1/events", params={"since": 0}).json()
        assert ev["last_seq"] == 2 and [e["seq"] for e in ev["events"]] == [1, 2]
        assert ev["events"][1]["event"]["notificationFields"]["additionalFields"]["state"] == "READY"
        assert [e["seq"] for e in c.get("/o1/events", params={"since": 1}).json()["events"]] == [2]

        stats = c.get("/twin/ingest/stats").json()
        # `path` is a display path (relative to the demo dir / $VAR_DIR), never the absolute checkout path
        assert stats["ok"] is True and stats["lines"] == 3 and stats["path"] == "$VAR_DIR/twin-ingest.jsonl"
        lines = [json.loads(l) for l in cfg.twin_ingest_path.read_text().splitlines()]
        assert [l["kind"] for l in lines] == ["measurement", "notification", "notification"]
        assert all(l["ru_id"] == "oru-a" and "ts" in l and "commonEventHeader" in l["payload"] for l in lines)

        h = c.get("/health").json()
        assert h["ok"] is True and h["component"] == "smo" and h["profile"] == "lite"
        assert h["o1"]["pm_sources"] == ["oru-a"] and h["o1"]["counts"] == {"measurement": 1, "notification": 2, "heartbeat": 0}
        assert h["r1"]["registration"]["state"] == "local"
        assert h["twin_ingest"]["lines"] == 3


def test_ves_rejects_malformed_envelope(cfg):
    with TestClient(create_app(cfg)) as c:
        assert c.post("/eventListener/v7", json={"event": {"commonEventHeader": {"domain": "bogus"}}}).status_code == 422


def test_notification_ring_is_bounded(cfg):
    from ocp_auru.smo.store import EventStore
    s = EventStore(ring=5)
    for i in range(8):
        s.ingest(notification("oru-a", f"S{i}"))
    got = s.events_since(0)
    assert [e["seq"] for e in got] == [4, 5, 6, 7, 8]
    assert s.events_since(7)[0]["seq"] == 8


def test_twin_sink_counts_existing_lines(cfg, tmp_path):
    from ocp_auru.smo.twin_sink import TwinSink
    p = tmp_path / "var" / "pre.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"a":1}\n{"b":2}\n')
    sink = TwinSink(p)  # outside the demo dir and no var_dir given: only the file name is shown
    sink.append("measurement", "oru-a", {"x": 1})
    assert sink.stats() == {"path": "pre.jsonl", "lines": 3}
    assert TwinSink(p, tmp_path).stats()["path"] == "$VAR_DIR/var/pre.jsonl"
