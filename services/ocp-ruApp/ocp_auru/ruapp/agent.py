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
"""The ruApp decision loop: PRACH energy -> local SLEEP->ACTIVE wake inside the policy envelope.

O-RAN concept (ADR-3): the rApp's policy envelope is the standard `o-ran-uplane-conf`
`tx-array-carriers/active` leaf. `SLEEP` = the ruApp MAY wake the carrier locally when it
detects demand (PRACH energy >= `prach-energy-threshold-db`, `wake-on-prach` true).
`INACTIVE` = hard off, never woken. `ACTIVE` = nothing to do. Every decision is emitted
as `RuAppDecision` and folded into PM counters (`wake_events`, `wake_latency_us
{last,p50,max}`) so the SMO sees every transition.

`wake_latency_ns = CarrierStateChanged.t_ns - PrachDetected.t_ns`: the datastore stamps the
event with `perf_counter_ns()` inside its lock right after the `active` write, so the number
ends at the write and excludes the bus fan-out to the recorder/reporter/NETCONF subscribers.
The handler runs inline in the detector's thread (bus `inline=True`): no queue, no hop.
"""
from __future__ import annotations

import logging
import statistics
import threading
import time
from collections import deque
from typing import Optional

from ocp_auru.common.events import Active, PrachDetected, RuAppDecision, WakeLatencyStats
from ocp_auru.mock_ru.bus import Bus, Subscription
from ocp_auru.mock_ru.datastore import Datastore

log = logging.getLogger(__name__)


class RuApp:
    def __init__(self, datastore: Datastore, bus: Bus, history: int = 1000) -> None:
        self._ds = datastore
        self._bus = bus
        self._lock = threading.Lock()
        self._lat_ns: dict[str, deque[int]] = {n: deque(maxlen=history) for n in datastore.carrier_names()}
        self._wake_events: dict[str, int] = {n: 0 for n in datastore.carrier_names()}
        self._decisions: dict[str, int] = {n: 0 for n in datastore.carrier_names()}
        self._max_ns: dict[str, int] = {n: 0 for n in datastore.carrier_names()}
        self._count: dict[str, int] = {n: 0 for n in datastore.carrier_names()}
        self._sub: Optional[Subscription] = None

    def start(self) -> None:
        self._sub = self._bus.subscribe(PrachDetected, self.on_prach, inline=True, name="ruapp")

    def stop(self) -> None:
        if self._sub is not None:
            self._bus.unsubscribe(self._sub)
            self._sub = None

    # ---- the decision -------------------------------------------------------------------
    def on_prach(self, ev: PrachDetected) -> RuAppDecision:
        carrier = ev.carrier
        active = self._ds.active(carrier)
        policy = self._ds.policy(carrier)
        latency_ns: Optional[int] = None
        if active == Active.INACTIVE:
            decision = "ignore-inactive"
        elif active == Active.ACTIVE:
            decision = "ignore-active"
        elif not policy.wake_on_prach:
            decision = "ignore-disabled-policy"
        elif ev.energy_db < policy.prach_energy_threshold_db:
            decision = "ignore-threshold"
        else:
            applied = self._ds.compare_and_set_active(carrier, Active.SLEEP, Active.ACTIVE, cause="ruapp-wake")
            if applied is None:
                decision = "ignore-active"  # lost a race with a concurrent wake/edit
            else:
                decision = "wake"
                latency_ns = applied.t_ns - ev.t_ns  # stamped by the datastore at the write, before fan-out
        out = RuAppDecision(carrier=carrier, decision=decision, prach_energy_db=ev.energy_db,
                            wake_latency_ns=latency_ns, wall_us=time.time_ns() // 1000)
        with self._lock:
            self._decisions[carrier] = self._decisions.get(carrier, 0) + 1
            if latency_ns is not None:
                self._wake_events[carrier] = self._wake_events.get(carrier, 0) + 1
                self._lat_ns.setdefault(carrier, deque(maxlen=1000)).append(latency_ns)
                self._max_ns[carrier] = max(self._max_ns.get(carrier, 0), latency_ns)
                self._count[carrier] = self._count.get(carrier, 0) + 1
        if decision == "wake":
            log.info("ruApp: %s PRACH %.1f dB >= %d dB -> ACTIVE (wake latency %.1f us)",
                     carrier, ev.energy_db, policy.prach_energy_threshold_db, latency_ns / 1000.0)
        else:
            log.debug("ruApp: %s PRACH %.1f dB -> %s", carrier, ev.energy_db, decision)
        self._bus.publish(out)
        return out

    # ---- PM exposure ---------------------------------------------------------------------
    def stats(self, carrier: str) -> WakeLatencyStats:
        """Wake latency statistics in microseconds (last/p50/max over the retained history, count total)."""
        with self._lock:
            hist = self._lat_ns.get(carrier)
            if not hist:
                return WakeLatencyStats(count=self._count.get(carrier, 0))
            return WakeLatencyStats(
                last=hist[-1] / 1000.0,
                p50=statistics.median(hist) / 1000.0,
                max=self._max_ns[carrier] / 1000.0,
                count=self._count[carrier],
            )

    def wake_events(self, carrier: str) -> int:
        with self._lock:
            return self._wake_events.get(carrier, 0)

    def decisions(self, carrier: str) -> int:
        with self._lock:
            return self._decisions.get(carrier, 0)

    def snapshot(self, carrier: str) -> dict:
        s = self.stats(carrier)
        return {"wake_events": self.wake_events(carrier), "decisions": self.decisions(carrier),
                "wake_latency_us": s.model_dump()}
