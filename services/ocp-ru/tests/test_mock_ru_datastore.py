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
"""o-ran-uplane-conf running datastore: active/state transitions, settle, ruApp hold."""
from __future__ import annotations

import time

import pytest

from ocp_auru.common.events import Active, CarrierState, CarrierStateChanged
from ocp_auru.mock_ru.bus import Bus
from ocp_auru.mock_ru.datastore import Datastore, HoldViolation, SleepStateViolation, UnknownCarrier
from _mock_ru_helpers import ephemeral_settings

C0, C1 = "txcarrier0", "txcarrier1"


def wait_state(ds: Datastore, name: str, state: CarrierState, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while ds.state(name) != state:
        if time.monotonic() > deadline:
            raise AssertionError(f"{name} did not reach {state}, is {ds.state(name)}")
        time.sleep(0.002)


@pytest.fixture
def rig(tmp_path):
    s = ephemeral_settings(tmp_path, ru_carriers=(C0, C1), ru_settle_ms=5.0)
    bus = Bus()
    events: list[CarrierStateChanged] = []
    bus.subscribe(CarrierStateChanged, events.append)
    ds = Datastore(s, bus)
    yield ds, events, s
    ds.close()
    bus.close()


def test_initial_state_active_ready(rig):
    ds, _, _ = rig
    st = ds.status(C0)
    assert st.active == Active.ACTIVE and st.state == CarrierState.READY
    assert ds.policy(C0).wake_on_prach is True


def test_initial_state_sleep_disabled(tmp_path):
    s = ephemeral_settings(tmp_path, ru_initial_active="SLEEP")
    bus = Bus()
    ds = Datastore(s, bus)
    assert ds.active(C0) == Active.SLEEP and ds.state(C0) == CarrierState.DISABLED
    ds.close(); bus.close()


def test_edit_sleep_goes_busy_then_disabled(rig):
    ds, events, _ = rig
    out = ds.apply_edit({C0: {"active": Active.SLEEP, "policy": None}})
    assert len(out) == 1 and out[0].state == CarrierState.BUSY and out[0].cause == "netconf-edit-config"
    assert ds.state(C0) == CarrierState.BUSY and ds.active(C0) == Active.SLEEP
    wait_state(ds, C0, CarrierState.DISABLED)
    assert [(e.state, e.cause) for e in events if e.carrier == C0] == [
        (CarrierState.BUSY, "netconf-edit-config"), (CarrierState.DISABLED, "settle")]
    assert ds.state(C1) == CarrierState.READY  # untouched


def test_edit_inactive_then_active_settles_ready(rig):
    ds, events, _ = rig
    ds.apply_edit({C0: {"active": Active.INACTIVE, "policy": None}})
    wait_state(ds, C0, CarrierState.DISABLED)
    ds.apply_edit({C0: {"active": Active.ACTIVE, "policy": None}})
    assert ds.state(C0) == CarrierState.BUSY
    wait_state(ds, C0, CarrierState.READY)
    assert events[-1].active == Active.ACTIVE and events[-1].state == CarrierState.READY


def test_settle_timing_respects_ru_settle_ms(tmp_path):
    s = ephemeral_settings(tmp_path, ru_settle_ms=40.0)
    bus = Bus(); ds = Datastore(s, bus)
    ds.set_active(C0, Active.SLEEP, "netconf-edit-config")
    time.sleep(0.015)
    assert ds.state(C0) == CarrierState.BUSY  # still settling
    wait_state(ds, C0, CarrierState.DISABLED, timeout=0.5)
    ds.close(); bus.close()


def test_same_value_is_a_noop(rig):
    ds, events, _ = rig
    assert ds.apply_edit({C0: {"active": Active.ACTIVE, "policy": None}}) == []
    assert ds.state(C0) == CarrierState.READY and events == []


def test_policy_merge_is_partial(rig):
    ds, _, _ = rig
    ds.apply_edit({C0: {"active": None, "policy": {"prach_energy_threshold_db": 15, "min_hold_after_wake_s": 3}}})
    p = ds.policy(C0)
    assert p.prach_energy_threshold_db == 15 and p.min_hold_after_wake_s == 3
    assert p.wake_on_prach is True and p.report_interval_s == 5  # defaults kept
    assert ds.policy(C1).prach_energy_threshold_db == 10
    assert ds.state(C0) == CarrierState.READY  # a policy-only edit does not touch state


def test_hold_after_local_wake_rejects_sleep_and_inactive(rig):
    ds, events, _ = rig
    ds.apply_edit({C0: {"active": Active.SLEEP, "policy": {"min_hold_after_wake_s": 30}}})
    wait_state(ds, C0, CarrierState.DISABLED)
    ev = ds.compare_and_set_active(C0, Active.SLEEP, Active.ACTIVE, cause="ruapp-wake")
    assert ev is not None and ev.cause == "ruapp-wake"
    assert 29.0 < ds.hold_remaining_s(C0) <= 30.0
    with pytest.raises(HoldViolation) as ei:
        ds.apply_edit({C0: {"active": Active.SLEEP, "policy": {"prach_energy_threshold_db": 99}}})
    assert "hold" in str(ei.value).lower() and ei.value.carrier == C0
    assert ds.policy(C0).prach_energy_threshold_db == 10  # rejected edit changed nothing (atomic)
    with pytest.raises(HoldViolation):
        ds.apply_edit({C0: {"active": Active.INACTIVE, "policy": None}})
    ds.apply_edit({C0: {"active": Active.ACTIVE, "policy": None}})  # ACTIVE is always allowed
    ds.apply_edit({C1: {"active": Active.SLEEP, "policy": None}})    # other carrier not held
    assert ds.active(C0) == Active.ACTIVE and ds.active(C1) == Active.SLEEP


def test_hold_zero_allows_immediate_sleep(rig):
    ds, _, _ = rig
    ds.apply_edit({C0: {"active": Active.SLEEP, "policy": {"min_hold_after_wake_s": 0}}})
    ds.compare_and_set_active(C0, Active.SLEEP, Active.ACTIVE, cause="ruapp-wake")
    ds.apply_edit({C0: {"active": Active.SLEEP, "policy": None}})
    assert ds.active(C0) == Active.SLEEP


def test_hold_expires(tmp_path):
    s = ephemeral_settings(tmp_path)
    bus = Bus(); ds = Datastore(s, bus)
    ds.apply_edit({C0: {"active": Active.SLEEP, "policy": {"min_hold_after_wake_s": 1}}})
    ds.compare_and_set_active(C0, Active.SLEEP, Active.ACTIVE, cause="ruapp-wake")
    with pytest.raises(HoldViolation):
        ds.apply_edit({C0: {"active": Active.SLEEP, "policy": None}})
    time.sleep(1.05)
    ds.apply_edit({C0: {"active": Active.SLEEP, "policy": None}})
    assert ds.active(C0) == Active.SLEEP
    ds.close(); bus.close()


def test_client_edits_do_not_start_a_hold(rig):
    ds, _, _ = rig
    ds.apply_edit({C0: {"active": Active.SLEEP, "policy": None}})
    ds.apply_edit({C0: {"active": Active.ACTIVE, "policy": None}})  # rApp wake, not a local wake
    assert ds.hold_remaining_s(C0) == 0.0
    ds.apply_edit({C0: {"active": Active.SLEEP, "policy": None}})
    assert ds.active(C0) == Active.SLEEP


def test_compare_and_set_precondition(rig):
    ds, _, _ = rig
    assert ds.compare_and_set_active(C0, Active.SLEEP, Active.ACTIVE, "ruapp-wake") is None  # it is ACTIVE
    assert ds.active(C0) == Active.ACTIVE and ds.hold_remaining_s(C0) == 0.0


def test_superseded_settle_is_dropped(rig):
    ds, events, _ = rig
    ds.set_active(C0, Active.SLEEP, "netconf-edit-config")
    ds.set_active(C0, Active.ACTIVE, "netconf-edit-config")
    wait_state(ds, C0, CarrierState.READY)
    time.sleep(0.03)
    settled = [e for e in events if e.cause == "settle"]
    assert [e.state for e in settled] == [CarrierState.READY]  # the stale DISABLED settle never fired


def test_unknown_carrier_rejected_atomically(rig):
    ds, _, _ = rig
    with pytest.raises(UnknownCarrier):
        ds.apply_edit({C0: {"active": Active.SLEEP, "policy": None}, "nope": {"active": Active.SLEEP, "policy": None}})
    assert ds.active(C0) == Active.ACTIVE


def test_interfaces_table(rig):
    ds, _, _ = rig
    assert ds.interfaces() == {}
    ds.set_interface_description("eth0", "uplink port")
    assert ds.interfaces() == {"eth0": "uplink port"}


def test_publish_initial_emits_init_events(rig):
    ds, events, _ = rig
    ds.publish_initial()
    assert sorted((e.carrier, e.cause, e.state) for e in events) == [
        (C0, "init", CarrierState.READY), (C1, "init", CarrierState.READY)]


def test_sleep_state_rule_rejects_inactive_but_allows_active_and_policy(rig):
    """o-ran-uplane-conf `active`: in SLEEP the server rejects every write to tx-array-carriers
    except the change to ACTIVE; the separate ruapp-policy container is not covered by the rule."""
    ds, events, _ = rig
    ds.apply_edit({C0: {"active": Active.SLEEP, "policy": None}})
    wait_state(ds, C0, CarrierState.DISABLED)
    with pytest.raises(SleepStateViolation) as ei:
        ds.apply_edit({C0: {"active": Active.INACTIVE, "policy": {"prach_energy_threshold_db": 99}}})
    assert ei.value.carrier == C0 and ei.value.requested == Active.INACTIVE and "SLEEP" in str(ei.value)
    assert ds.active(C0) == Active.SLEEP and ds.policy(C0).prach_energy_threshold_db == 10  # atomic: nothing changed
    assert ds.apply_edit({C0: {"active": Active.SLEEP, "policy": None}}) == []  # same value: no-op, not an error
    ds.apply_edit({C0: {"active": None, "policy": {"prach_energy_threshold_db": 15}}})  # policy-only edit allowed in SLEEP
    assert ds.policy(C0).prach_energy_threshold_db == 15 and ds.active(C0) == Active.SLEEP
    ds.apply_edit({C0: {"active": Active.ACTIVE, "policy": None}})  # the one permitted transition
    wait_state(ds, C0, CarrierState.READY)
    ds.apply_edit({C0: {"active": Active.INACTIVE, "policy": None}})  # ACTIVE -> INACTIVE is not restricted
    assert ds.active(C0) == Active.INACTIVE
