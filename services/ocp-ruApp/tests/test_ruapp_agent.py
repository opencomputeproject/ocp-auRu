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
"""ruApp: SLEEP->ACTIVE wake on PRACH with measured latency; INACTIVE never wakes; policy gates."""
from __future__ import annotations

import statistics
import time

import pytest

from ocp_auru.common.events import Active, CarrierState, CarrierStateChanged, PrachDetected, RuAppDecision
from ocp_auru.mock_ru.bus import Bus
from ocp_auru.mock_ru.datastore import Datastore
from ocp_auru.ruapp.agent import RuApp
from _mock_ru_helpers import ephemeral_settings

C0 = "txcarrier0"


def prach(energy_db: float = 20.0, carrier: str = C0) -> PrachDetected:
    return PrachDetected(carrier=carrier, energy_db=energy_db, t_ns=time.perf_counter_ns(), wall_us=time.time_ns() // 1000)


def wait_state(ds: Datastore, name: str, state: CarrierState, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while ds.state(name) != state:
        assert time.monotonic() < deadline, f"{name} stuck in {ds.state(name)}"
        time.sleep(0.002)


@pytest.fixture
def rig(tmp_path):
    s = ephemeral_settings(tmp_path, ru_initial_active="SLEEP")
    bus = Bus()
    decisions: list[RuAppDecision] = []
    bus.subscribe(RuAppDecision, decisions.append)
    ds = Datastore(s, bus)
    agent = RuApp(ds, bus)
    agent.start()
    yield ds, bus, agent, decisions
    agent.stop(); ds.close(); bus.close()


def test_wake_from_sleep_measures_latency(rig):
    ds, bus, agent, decisions = rig
    assert ds.active(C0) == Active.SLEEP
    bus.publish(prach(20.0))
    assert len(decisions) == 1
    d = decisions[0]
    assert d.decision == "wake" and d.carrier == C0 and d.prach_energy_db == 20.0
    assert d.wake_latency_ns is not None and 0 < d.wake_latency_ns < 5_000_000  # < 5 ms even on a loaded CI box
    assert ds.active(C0) == Active.ACTIVE
    wait_state(ds, C0, CarrierState.READY)
    assert agent.wake_events(C0) == 1
    st = agent.stats(C0)
    assert st.count == 1 and st.last == pytest.approx(d.wake_latency_ns / 1000.0) and st.p50 == st.last == st.max
    assert ds.hold_remaining_s(C0) > 0  # the local wake armed the hold


def test_wake_latency_ends_at_the_datastore_write_not_after_fanout(rig):
    """`wake_latency_ns` = CarrierStateChanged.t_ns - PrachDetected.t_ns: the event stamp taken at the
    write, before the bus fans out (a slow subscriber must not inflate the number)."""
    ds, bus, agent, decisions = rig
    seen: list = []
    bus.subscribe(CarrierStateChanged, lambda ev: (seen.append(ev), time.sleep(0.02)))  # slow subscriber after the write
    ev = prach(20.0)
    bus.publish(ev)
    d = decisions[-1]
    assert d.decision == "wake" and seen and seen[0].cause == "ruapp-wake"
    assert d.wake_latency_ns == seen[0].t_ns - ev.t_ns
    assert d.wake_latency_ns < 5_000_000  # the 20 ms subscriber sleep is not in the number


def test_inactive_never_wakes(rig):
    ds, bus, agent, decisions = rig
    ds.set_active(C0, Active.INACTIVE, "netconf-edit-config")
    for _ in range(3):
        bus.publish(prach(60.0))
    assert [d.decision for d in decisions] == ["ignore-inactive"] * 3
    assert all(d.wake_latency_ns is None for d in decisions)
    assert ds.active(C0) == Active.INACTIVE and agent.wake_events(C0) == 0
    wait_state(ds, C0, CarrierState.DISABLED)


def test_energy_below_threshold_is_ignored(rig):
    ds, bus, agent, decisions = rig
    bus.publish(prach(9.9))  # default threshold 10 dB
    assert decisions[-1].decision == "ignore-threshold" and ds.active(C0) == Active.SLEEP
    bus.publish(prach(10.0))  # >= threshold wakes
    assert decisions[-1].decision == "wake" and ds.active(C0) == Active.ACTIVE


def test_threshold_from_policy(rig):
    ds, bus, _, decisions = rig
    ds.apply_edit({C0: {"active": None, "policy": {"prach_energy_threshold_db": 25}}})
    bus.publish(prach(20.0))
    assert decisions[-1].decision == "ignore-threshold" and ds.active(C0) == Active.SLEEP
    bus.publish(prach(25.0))
    assert decisions[-1].decision == "wake"


def test_wake_on_prach_false(rig):
    ds, bus, agent, decisions = rig
    ds.apply_edit({C0: {"active": None, "policy": {"wake_on_prach": False}}})
    bus.publish(prach(50.0))
    assert decisions[-1].decision == "ignore-disabled-policy"
    assert ds.active(C0) == Active.SLEEP and agent.wake_events(C0) == 0


def test_active_carrier_ignored(rig):
    ds, bus, _, decisions = rig
    ds.set_active(C0, Active.ACTIVE, "netconf-edit-config")
    bus.publish(prach(50.0))
    assert decisions[-1].decision == "ignore-active" and decisions[-1].wake_latency_ns is None


def test_unknown_carrier_in_prach_is_ignored_safely(rig):
    ds, bus, _, decisions = rig
    bus.publish(prach(50.0, carrier="txcarrier9"))  # handler error is logged, publisher unaffected
    assert decisions == []
    assert ds.active(C0) == Active.SLEEP


def test_latency_stats_over_many_wakes(rig, capsys):
    ds, bus, agent, decisions = rig
    ds.apply_edit({C0: {"active": None, "policy": {"min_hold_after_wake_s": 0}}})
    n = 50
    for _ in range(n):
        ds.apply_edit({C0: {"active": Active.SLEEP, "policy": None}})
        bus.publish(prach(20.0))
        assert decisions[-1].decision == "wake"
    st = agent.stats(C0)
    assert st.count == n and agent.wake_events(C0) == n
    lat_us = [d.wake_latency_ns / 1000.0 for d in decisions if d.wake_latency_ns is not None]
    assert st.p50 == pytest.approx(statistics.median(lat_us)) and st.max == pytest.approx(max(lat_us))
    assert st.last == pytest.approx(lat_us[-1])
    assert st.p50 < 2000.0  # sub-2 ms in-process wake
    with capsys.disabled():
        print(f"\n[ruApp wake latency over {n} wakes] p50={st.p50:.1f} us  max={st.max:.1f} us  min={min(lat_us):.1f} us")
