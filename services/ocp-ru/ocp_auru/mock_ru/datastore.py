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
"""Thread-safe `running` datastore for the `o-ran-uplane-conf` subset (HLD §3.1).

O-RAN concept: `tx-array-carriers/active` (config, written by the M-Plane client) drives
`tx-array-carriers/state` (config false, owned by the RU):

    write active  ->  state=BUSY  --settle-->  READY   (active == ACTIVE)
                                            DISABLED (active == SLEEP | INACTIVE)

The ruApp policy envelope (ADR-3): SLEEP allows a *local* SLEEP->ACTIVE wake; INACTIVE
forbids it. After a local wake the RU enforces `min-hold-after-wake-s`: a SLEEP/INACTIVE
request inside the hold window is rejected (the NETCONF server turns `HoldViolation`
into an `operation-failed` rpc-error). The module's own SLEEP-state rule is enforced too
(`active` leaf description, o-ran-uplane-conf v07.01): while a carrier is in SLEEP the
server rejects every other write to that tx-array-carriers object except the change to
ACTIVE (`SleepStateViolation`); the separate `ocp-ruapp-policy` container is not
part of that object, so policy edits stay allowed in SLEEP. Also holds the ruApp policy
tunables per carrier and a tiny `ietf-interfaces` table (interface descriptions).

Every state change is published on the bus as `CarrierStateChanged`. Settle timers run
on one scheduler thread so the wake path never creates a thread.
"""
from __future__ import annotations

import heapq
import logging
import threading
import time
from typing import Callable, Optional

from ocp_auru.common.config import Settings
from ocp_auru.common.events import Active, CarrierConfig, CarrierState, CarrierStateChanged, CarrierStatus, RuAppPolicy

from .bus import Bus

log = logging.getLogger(__name__)


class HoldViolation(Exception):
    """A SLEEP/INACTIVE request arrived inside the ruApp's post-wake hold window."""

    def __init__(self, carrier: str, requested: Active, remaining_s: float, hold_s: int) -> None:
        self.carrier, self.requested, self.remaining_s, self.hold_s = carrier, requested, remaining_s, hold_s
        super().__init__(
            f"carrier {carrier}: {requested.value} rejected - ruApp min-hold-after-wake "
            f"({hold_s} s) still active for {remaining_s:.1f} s after a local wake"
        )


class SleepStateViolation(Exception):
    """o-ran-uplane-conf: a carrier in SLEEP accepts no write to tx-array-carriers except `active=ACTIVE`."""

    def __init__(self, carrier: str, requested: Active) -> None:
        self.carrier, self.requested = carrier, requested
        super().__init__(
            f"carrier {carrier}: active={requested.value} rejected - array carrier is in SLEEP status; "
            "o-ran-uplane-conf: the Netconf server rejects all other operation requests to the tx-array-carriers "
            "object except a request to change from sleep to active status"
        )


class UnknownCarrier(KeyError):
    pass


class _Scheduler(threading.Thread):
    """One timer thread for all settle transitions (no per-transition thread creation)."""

    def __init__(self) -> None:
        super().__init__(name="ru-settle-scheduler", daemon=True)
        self._cond = threading.Condition()
        self._heap: list[tuple[float, int, Callable[[], None]]] = []
        self._n = 0
        self._stop = False

    def schedule(self, delay_s: float, fn: Callable[[], None]) -> None:
        with self._cond:
            self._n += 1
            heapq.heappush(self._heap, (time.monotonic() + delay_s, self._n, fn))
            self._cond.notify()

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()

    def run(self) -> None:
        while True:
            with self._cond:
                while not self._stop and (not self._heap or self._heap[0][0] > time.monotonic()):
                    timeout = None if not self._heap else max(0.0, self._heap[0][0] - time.monotonic())
                    self._cond.wait(timeout)
                if self._stop:
                    return
                _, _, fn = heapq.heappop(self._heap)
            try:
                fn()
            except Exception:  # noqa: BLE001
                log.exception("settle callback failed")


class Datastore:
    def __init__(self, settings: Settings, bus: Bus) -> None:
        self._settings = settings
        self._bus = bus
        self._lock = threading.RLock()
        initial = Active(settings.ru_initial_active)
        self._carriers: dict[str, CarrierConfig] = {n: CarrierConfig(name=n, active=initial) for n in settings.ru_carriers}
        self._state: dict[str, CarrierState] = {
            n: (CarrierState.READY if initial == Active.ACTIVE else CarrierState.DISABLED) for n in settings.ru_carriers
        }
        self._gen: dict[str, int] = {n: 0 for n in settings.ru_carriers}  # invalidates stale settle callbacks
        self._hold_until: dict[str, float] = {n: 0.0 for n in settings.ru_carriers}  # time.monotonic()
        self._interfaces: dict[str, str] = {}
        self._sched = _Scheduler()
        self._sched.start()

    # ---- read side ------------------------------------------------------------------
    def carriers(self) -> list[CarrierConfig]:
        with self._lock:
            return [c.model_copy(deep=True) for c in self._carriers.values()]

    def carrier_names(self) -> list[str]:
        return list(self._carriers.keys())

    def _check(self, name: str) -> None:
        if name not in self._carriers:
            raise UnknownCarrier(name)

    def status(self, name: str) -> CarrierStatus:
        with self._lock:
            self._check(name)
            c = self._carriers[name]
            return CarrierStatus(name=name, active=c.active, state=self._state[name], ruapp_policy=c.ruapp_policy.model_copy())

    def statuses(self) -> dict[str, CarrierStatus]:
        with self._lock:
            return {n: self.status(n) for n in self._carriers}

    def active(self, name: str) -> Active:
        with self._lock:
            self._check(name)
            return self._carriers[name].active

    def state(self, name: str) -> CarrierState:
        with self._lock:
            self._check(name)
            return self._state[name]

    def policy(self, name: str) -> RuAppPolicy:
        with self._lock:
            self._check(name)
            return self._carriers[name].ruapp_policy

    def policies(self) -> dict[str, RuAppPolicy]:
        with self._lock:
            return {n: c.ruapp_policy.model_copy() for n, c in self._carriers.items()}

    def hold_remaining_s(self, name: str) -> float:
        with self._lock:
            self._check(name)
            return max(0.0, self._hold_until[name] - time.monotonic())

    def interfaces(self) -> dict[str, str]:
        with self._lock:
            return dict(self._interfaces)

    # ---- write side -----------------------------------------------------------------
    def set_interface_description(self, name: str, description: str) -> None:
        with self._lock:
            self._interfaces[name] = description

    def apply_edit(self, carrier_edits: dict[str, dict], cause: str = "netconf-edit-config") -> list[CarrierStateChanged]:
        """Merge an edit-config (as parsed by `yang.parse_carrier_edits`) into `running`.

        Validates everything first (unknown carrier, SLEEP-state rule, hold window) so a
        rejected edit changes nothing; then merges policies, then writes `active`.
        """
        events: list[CarrierStateChanged] = []
        with self._lock:
            now = time.monotonic()
            for name, edit in carrier_edits.items():
                self._check(name)
                requested = edit.get("active")
                if requested is not None and self._carriers[name].active == Active.SLEEP and requested not in (Active.ACTIVE, Active.SLEEP):
                    raise SleepStateViolation(name, requested)
                if requested in (Active.SLEEP, Active.INACTIVE) and self._hold_until[name] > now:
                    raise HoldViolation(name, requested, self._hold_until[name] - now,
                                        self._carriers[name].ruapp_policy.min_hold_after_wake_s)
            for name, edit in carrier_edits.items():
                pol = edit.get("policy")
                if pol:
                    c = self._carriers[name]
                    c.ruapp_policy = c.ruapp_policy.model_copy(update=pol)
                    log.info("carrier %s ruapp-policy merged: %s", name, c.ruapp_policy.model_dump())
            for name, edit in carrier_edits.items():
                requested = edit.get("active")
                if requested is not None:
                    ev = self._write_active(name, requested, cause)
                    if ev is not None:
                        events.append(ev)
        for ev in events:  # publish outside the lock
            self._bus.publish(ev)
        return events

    def set_active(self, name: str, active: Active, cause: str) -> Optional[CarrierStateChanged]:
        """Write `active` (no hold check: the hold only guards SLEEP/INACTIVE requests from the client)."""
        with self._lock:
            self._check(name)
            ev = self._write_active(name, active, cause)
        if ev is not None:
            self._bus.publish(ev)
        return ev

    def compare_and_set_active(self, name: str, expected: Active, active: Active, cause: str) -> Optional[CarrierStateChanged]:
        """Atomic `active` write guarded by a precondition; returns None if the precondition failed."""
        with self._lock:
            self._check(name)
            if self._carriers[name].active != expected:
                return None
            ev = self._write_active(name, active, cause)
        if ev is not None:
            self._bus.publish(ev)
        return ev

    def _write_active(self, name: str, active: Active, cause: str) -> Optional[CarrierStateChanged]:
        """Caller holds the lock. Returns the BUSY event (or None when it is a no-op)."""
        c = self._carriers[name]
        if c.active == active and self._state[name] != CarrierState.BUSY:
            return None  # same value already applied: a real RU would not re-run the activation
        c.active = active
        if cause == "ruapp-wake":
            self._hold_until[name] = time.monotonic() + c.ruapp_policy.min_hold_after_wake_s
        self._gen[name] += 1
        gen = self._gen[name]
        self._state[name] = CarrierState.BUSY
        ev = self._event(name, cause)
        self._sched.schedule(self._settings.ru_settle_ms / 1000.0, lambda: self._settle(name, gen, cause))
        # DEBUG on purpose: stdout I/O must stay off the measured wake path; the settle log is the INFO line.
        log.debug("carrier %s active=%s -> state=BUSY (cause=%s)", name, active.value, cause)
        return ev

    def _settle(self, name: str, gen: int, cause: str) -> None:
        with self._lock:
            if self._gen[name] != gen:
                return  # superseded by a newer write
            active = self._carriers[name].active
            target = CarrierState.READY if active == Active.ACTIVE else CarrierState.DISABLED
            self._state[name] = target
            ev = self._event(name, "settle")
        log.info("carrier %s active=%s -> state=%s (cause=%s, settled after %.1f ms)",
                 name, active.value, target.value, cause, self._settings.ru_settle_ms)
        self._bus.publish(ev)

    def _event(self, name: str, cause: str) -> CarrierStateChanged:
        return CarrierStateChanged(
            carrier=name, active=self._carriers[name].active, state=self._state[name],
            t_ns=time.perf_counter_ns(), wall_us=time.time_ns() // 1000, cause=cause,
        )

    def publish_initial(self) -> None:
        """Announce the initial state (cause='init') once subscribers are wired."""
        with self._lock:
            evs = [self._event(n, "init") for n in self._carriers]
        for ev in evs:
            self._bus.publish(ev)

    def close(self) -> None:
        self._sched.stop()
