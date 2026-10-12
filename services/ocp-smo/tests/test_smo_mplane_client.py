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
"""M-Plane client of the SMO: payload builders, `<get>` parsing, timing and error mapping (fake ncclient manager)."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from lxml import etree

from ocp_auru.common import yang
from ocp_auru.common.events import Active, RuAppPolicy
from ocp_auru.smo import mplane_client
from ocp_auru.smo.mplane_client import MplaneClient, MplaneError
from _smo_helpers import make_settings

GET_DATA = (
    f'<data xmlns="{yang.NS_NC}"><user-plane-configuration xmlns="{yang.NS_UPLANE}">'
    '<tx-array-carriers><name>txcarrier0</name><active>SLEEP</active><state>DISABLED</state></tx-array-carriers>'
    '<rx-array-carriers><name>rxcarrier0</name><active>SLEEP</active></rx-array-carriers>'
    '</user-plane-configuration></data>')


class FakeManager:
    """The slice of an ncclient manager the client uses."""

    def __init__(self, ok: bool = True, raises: Exception | None = None):
        self.ok, self.raises = ok, raises
        self.edits: list[dict] = []
        self.closed = 0
        self.timeout = None

    def edit_config(self, target, config):
        self.edits.append({"target": target, "config": config})
        if self.raises:
            raise self.raises
        time.sleep(0.002)
        return SimpleNamespace(ok=self.ok, error="carrier does not exist")

    def get(self, filter):  # noqa: A002 - ncclient's keyword
        self.filter = filter
        return SimpleNamespace(data_ele=etree.fromstring(GET_DATA))

    def close_session(self):
        self.closed += 1


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    return make_settings(monkeypatch, tmp_path)


def _client(cfg, monkeypatch, fake: FakeManager) -> MplaneClient:
    seen = {}

    def connect(**kw):
        seen.update(kw)
        return fake

    monkeypatch.setattr(mplane_client.manager, "connect", connect)
    c = MplaneClient(cfg)
    c.connect_args = seen
    return c


def test_carrier_edit_payload_round_trips_through_the_parser():
    pol = RuAppPolicy(wake_on_prach=True, prach_energy_threshold_db=12, min_hold_after_wake_s=20, report_interval_s=5)
    edits = yang.parse_carrier_edits(etree.fromstring(yang.build_carrier_edit("txcarrier0", Active.SLEEP, pol)))
    assert edits == {"txcarrier0": {"active": Active.SLEEP, "policy": pol.model_dump()}}
    edits = yang.parse_carrier_edits(etree.fromstring(yang.build_carrier_edit("txcarrier0", Active.ACTIVE, None)))
    assert edits == {"txcarrier0": {"active": Active.ACTIVE, "policy": None}}


def test_interface_edit_payload_round_trips_through_the_parser():
    xml = etree.fromstring(yang.build_interface_edit("eth0", "hello"))
    assert yang.parse_interface_edits(xml) == {"eth0": "hello"}
    assert yang.parse_carrier_edits(xml) == {}


def test_parse_carriers_from_get_data():
    assert mplane_client.parse_carriers(etree.fromstring(GET_DATA)) == [{"name": "txcarrier0", "active": "SLEEP", "state": "DISABLED"}]


def test_edit_config_runs_on_running_times_the_rpc_and_closes_the_session(cfg, monkeypatch):
    fake = FakeManager()
    c = _client(cfg, monkeypatch, fake)
    res = c.edit_config("<config/>")
    assert fake.edits == [{"target": "running", "config": "<config/>"}] and fake.closed == 1
    assert res.ok and res.operation == "edit-config" and res.rtt_ms >= 2.0 and res.connect_ms >= 0.0
    assert res.target == f"127.0.0.1:{cfg.ru_netconf_port}" == c.target
    assert c.connect_args["host"] == "127.0.0.1" and c.connect_args["port"] == cfg.ru_netconf_port
    assert c.connect_args["hostkey_verify"] is False and c.connect_args["timeout"] == cfg.mplane_timeout_s == 1.0
    assert fake.timeout == cfg.mplane_timeout_s


def test_hostkey_verification_follows_the_setting(monkeypatch, tmp_path):
    cfg = make_settings(monkeypatch, tmp_path, MPLANE_HOSTKEY_VERIFY="1")
    c = _client(cfg, monkeypatch, FakeManager())
    c.edit_config("<config/>")
    assert c.connect_args["hostkey_verify"] is True


def test_not_ok_reply_and_transport_failure_become_mplane_error(cfg, monkeypatch):
    fake = FakeManager(ok=False)
    with pytest.raises(MplaneError, match="rpc-error: carrier does not exist"):
        _client(cfg, monkeypatch, fake).edit_config("<config/>")
    assert fake.closed == 1  # the session is closed on the failure path too
    fake = FakeManager(raises=TimeoutError("no reply"))
    with pytest.raises(MplaneError, match="TimeoutError: no reply"):
        _client(cfg, monkeypatch, fake).edit_config("<config/>")
    assert fake.closed == 1


def test_get_carrier_status_uses_the_uplane_subtree_filter(cfg, monkeypatch):
    fake = FakeManager()
    out = _client(cfg, monkeypatch, fake).get_carrier_status()
    assert fake.filter == ("subtree", yang.build_uplane_filter()) and fake.closed == 1
    assert out["carriers"] == [{"name": "txcarrier0", "active": "SLEEP", "state": "DISABLED"}]
    assert out["netconf"]["ok"] is True and out["netconf"]["operation"] == "get"


def test_unreachable_o_ru_is_an_mplane_error(cfg):
    # nothing listens on cfg.ru_netconf_port; the real ncclient connect fails
    with pytest.raises(MplaneError):
        MplaneClient(cfg).edit_config("<config/>")
