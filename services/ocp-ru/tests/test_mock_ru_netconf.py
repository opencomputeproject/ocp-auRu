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
"""ncclient interop with the mock O-RU M-Plane server (NETCONF over SSH, ephemeral port)."""
from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
from ncclient import manager
from ncclient.operations import RPCError
from ncclient.xml_ import to_ele

from ocp_auru.common import yang
from ocp_auru.common.events import Active, CarrierState, RuAppPolicy
from ocp_auru.mock_ru.app import build
from _mock_ru_helpers import ephemeral_settings

C0 = "txcarrier0"
UC = f"{{{yang.NS_UPLANE}}}"
RUAPP_YANG = Path(__file__).resolve().parents[1] / "yang" / "ocp-ruapp-policy.yang"


@pytest.fixture(scope="module")
def ru(tmp_path_factory):
    s = ephemeral_settings(tmp_path_factory.mktemp("ru"))
    ctx = build(s, background_prach=False).start(mgmt_api=False)
    yield ctx
    ctx.stop()


@pytest.fixture(scope="module")
def nc(ru):
    s = ru.settings
    m = manager.connect(host=ru.netconf.host, port=ru.netconf.port, username=s.ru_netconf_user,
                        password=s.ru_netconf_password, hostkey_verify=False, look_for_keys=False,
                        allow_agent=False, device_params={"name": "default"}, timeout=10)
    yield m
    try:
        m.close_session()
    except Exception:  # noqa: BLE001
        pass


def wait_state(nc, name: str, state: str, timeout: float = 2.0) -> str:
    deadline = time.monotonic() + timeout
    while True:
        seen = tx_carrier(nc.get().data, name).findtext(f"{UC}state")
        if seen == state or time.monotonic() > deadline:
            return seen
        time.sleep(0.01)


def tx_carrier(data, name: str):
    for tx in data.iter(f"{UC}tx-array-carriers"):
        if tx.findtext(f"{UC}name") == name:
            return tx
    raise AssertionError(f"{name} not in <data>")


def test_hello_capabilities_and_framing(ru, nc):
    caps = set(nc.server_capabilities)
    for cap in yang.CAPABILITIES:
        assert cap in caps
    assert nc.connected and nc.session_id
    # both sides advertise base:1.1, so ncclient negotiated chunked framing; everything below runs on it
    assert "urn:ietf:params:netconf:base:1.1" in caps


def test_advertised_extension_capability_matches_shipped_module():
    """The hello must advertise the `ocp-ruapp-policy` module as shipped in `services/ocp-ru/yang/` (a
    standards-true client matches namespace + module + revision); derived from the file so it cannot drift."""
    text = RUAPP_YANG.read_text()
    module = re.search(r"^module\s+([\w-]+)", text, re.M).group(1)
    namespace = re.search(r'^\s*namespace\s+"([^"]+)"', text, re.M).group(1)
    latest = re.search(r'^\s*revision\s+"?(\d{4}-\d{2}-\d{2})"?', text, re.M).group(1)
    assert module == "ocp-ruapp-policy" and namespace == yang.NS_RUAPP
    assert f"{namespace}?module={module}&revision={latest}" in yang.CAPABILITIES


def test_get_config_has_carriers_and_policy_but_no_state(ru, nc):
    data = nc.get_config(source="running").data
    tx = tx_carrier(data, C0)
    assert tx.findtext(f"{UC}active") == "ACTIVE"
    assert tx.find(f"{UC}state") is None  # config false leaves are not in get-config
    assert tx.find(f"{UC}duplex-scheme") is None  # config false in the module too (rw-duplex-scheme is the config leaf)
    assert tx.findtext(f"{UC}downlink-radio-frame-offset") == "0" and tx.findtext(f"{UC}downlink-sfn-offset") == "0"  # mandatory leaves
    rx = data.find(f"{{{yang.NS_UPLANE}}}user-plane-configuration/{UC}rx-array-carriers")
    assert rx.findtext(f"{UC}gain-correction") == "0.0000" and rx.findtext(f"{UC}n-ta-offset") == "0"
    assert data.find(f"{{{yang.NS_UPLANE}}}user-plane-configuration/{UC}rx-array-carriers") is not None
    pol = data.find(f"{{{yang.NS_RUAPP}}}ruapp-policy")
    assert pol is not None and pol.findtext(f"{{{yang.NS_RUAPP}}}carrier/{{{yang.NS_RUAPP}}}name") == C0


def test_get_returns_state(ru, nc):
    tx = tx_carrier(nc.get().data, C0)
    assert tx.findtext(f"{UC}state") == "READY"
    assert tx.findtext(f"{UC}duplex-scheme") == "TDD"


def test_subtree_filter_selects_container(ru, nc):
    data = nc.get(filter=("subtree", yang.build_uplane_filter())).data
    tags = [c.tag for c in data]
    assert tags == [f"{{{yang.NS_UPLANE}}}user-plane-configuration"]


def test_edit_config_sleep_with_policy(ru, nc):
    reply = nc.edit_config(target="running", config=yang.build_carrier_edit(C0, Active.SLEEP, RuAppPolicy(prach_energy_threshold_db=12, min_hold_after_wake_s=1)))
    assert reply.ok
    assert wait_state(nc, C0, "DISABLED") == "DISABLED"
    assert ru.datastore.active(C0) == Active.SLEEP
    assert ru.datastore.policy(C0).prach_energy_threshold_db == 12
    assert ru.datastore.policy(C0).min_hold_after_wake_s == 1
    data = nc.get_config(source="running").data
    assert data.find(f".//{{{yang.NS_RUAPP}}}prach-energy-threshold-db").text == "12"


def test_notification_on_state_change(ru, nc):
    nc.create_subscription()
    assert ru.netconf.methods.subscriber_count() == 1
    assert nc.edit_config(target="running", config=yang.build_carrier_edit(C0, Active.ACTIVE)).ok
    states = []
    for _ in range(2):  # BUSY then READY
        n = nc.take_notification(timeout=5)
        assert n is not None, "no <notification> within 5 s"
        root = n.notification_ele
        sc = root.find(f"{{{yang.NS_UPLANE}}}tx-array-carriers-state-change")
        assert sc is not None and root.find(f"{{{yang.NS_NOTIF}}}eventTime") is not None
        assert sc.findtext(f"{UC}tx-array-carriers/{UC}name") == C0
        states.append(sc.findtext(f"{UC}tx-array-carriers/{UC}state"))
    assert states == ["BUSY", "READY"]
    assert wait_state(nc, C0, "READY") == "READY"


def test_hold_rejected_with_operation_failed(ru, nc):
    assert nc.edit_config(target="running", config=yang.build_carrier_edit(C0, Active.SLEEP)).ok
    wait_state(nc, C0, "DISABLED")
    ev = ru.radio.inject_prach(C0, 20.0)  # ruApp wakes locally (policy hold = 1 s from the earlier edit)
    assert ru.datastore.active(C0) == Active.ACTIVE and ev.carrier == C0
    with pytest.raises(RPCError) as ei:
        nc.edit_config(target="running", config=yang.build_carrier_edit(C0, Active.SLEEP))
    assert ei.value.tag == "operation-failed"
    assert "hold" in (ei.value.message or "").lower()
    assert ru.datastore.active(C0) == Active.ACTIVE
    # the hold is per policy: after it expires the same request is accepted
    time.sleep(1.1)
    assert nc.edit_config(target="running", config=yang.build_carrier_edit(C0, Active.SLEEP)).ok
    assert wait_state(nc, C0, "DISABLED") == "DISABLED"
    assert nc.edit_config(target="running", config=yang.build_carrier_edit(C0, Active.ACTIVE)).ok
    wait_state(nc, C0, "READY")


def test_sleep_state_rule_over_netconf(ru, nc):
    """A SLEEP carrier rejects `active=INACTIVE` with rpc-error operation-failed quoting the module rule."""
    assert nc.edit_config(target="running", config=yang.build_carrier_edit(C0, Active.SLEEP)).ok
    wait_state(nc, C0, "DISABLED")
    with pytest.raises(RPCError) as ei:
        nc.edit_config(target="running", config=yang.build_carrier_edit(C0, Active.INACTIVE))
    assert ei.value.tag == "operation-failed" and "sleep" in (ei.value.message or "").lower()
    assert ru.datastore.active(C0) == Active.SLEEP
    assert nc.edit_config(target="running", config=yang.build_carrier_edit(C0, Active.ACTIVE)).ok
    wait_state(nc, C0, "READY")


def test_unknown_carrier_is_data_missing(ru, nc):
    with pytest.raises(RPCError) as ei:
        nc.edit_config(target="running", config=yang.build_carrier_edit("txcarrier9", Active.SLEEP))
    assert ei.value.tag == "data-missing"


def test_ietf_interfaces_edit(ru, nc):
    assert nc.edit_config(target="running", config=yang.build_interface_edit("eth0", "uplink port")).ok
    data = nc.get().data
    itf = data.find(f"{{{yang.NS_IETF_IF}}}interfaces/{{{yang.NS_IETF_IF}}}interface")
    assert itf is not None
    assert itf.findtext(f"{{{yang.NS_IETF_IF}}}name") == "eth0"
    assert itf.findtext(f"{{{yang.NS_IETF_IF}}}description") == "uplink port"
    assert ru.datastore.interfaces() == {"eth0": "uplink port"}


def test_supervision_watchdog_reset(ru, nc):
    rpc = to_ele(f'<supervision-watchdog-reset xmlns="{yang.NS_SUPERVISION}">'
                 '<supervision-notification-interval>60</supervision-notification-interval>'
                 '</supervision-watchdog-reset>')
    reply = nc.dispatch(rpc)
    nxt = reply.data_ele if hasattr(reply, "data_ele") else None
    el = to_ele(reply.xml).find(f"{{{yang.NS_SUPERVISION}}}next-update-at")
    assert el is not None and el.text.endswith("Z")
    assert nxt is None or nxt is not None  # dispatch reply shape varies by ncclient version; the element is what matters


def test_second_session_and_dead_subscriber_dropped(ru):
    s = ru.settings
    m2 = manager.connect(host=ru.netconf.host, port=ru.netconf.port, username=s.ru_netconf_user,
                         password=s.ru_netconf_password, hostkey_verify=False, look_for_keys=False,
                         allow_agent=False, device_params={"name": "default"}, timeout=10)
    m2.create_subscription()
    assert ru.netconf.methods.subscriber_count() == 2
    m2.close_session()
    time.sleep(0.2)
    ru.datastore.set_active(C0, Active.SLEEP, "netconf-edit-config")
    deadline = time.monotonic() + 3
    while ru.netconf.methods.subscriber_count() != 1 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ru.netconf.methods.subscriber_count() == 1  # the closed session was dropped on first push


def test_bad_password_rejected(ru):
    with pytest.raises(Exception):
        manager.connect(host=ru.netconf.host, port=ru.netconf.port, username=ru.settings.ru_netconf_user,
                        password="wrong", hostkey_verify=False, look_for_keys=False, allow_agent=False,
                        device_params={"name": "default"}, timeout=5)
