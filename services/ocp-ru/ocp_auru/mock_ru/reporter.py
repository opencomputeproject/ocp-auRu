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
"""North-bound PM/notification push from the O-RU to the SMO, VES-shaped (HLD §3.4).

O-RAN concept: O1 terminates on O-CU/O-DU/Near-RT RIC managed elements; an O-RU is managed
over the WG4 Open Fronthaul M-Plane, and in the hybrid model the SMO talks to the O-RU
directly. This is that north-bound leg: the RU pushes VES-style events
(`o-ran-ves-subscribed-notifications` pattern) straight to the SMO collector:

* every `settings.ru_pm_interval_s`: one `measurement` event carrying `CarrierPm` per
  carrier (radio counters, illustrative power/energy, PRACH detections, ruApp wake stats);
* immediately: one `notification` event per `CarrierStateChanged`
  (`tx-array-carriers-state-change`) and per `RuAppDecision` (`ruapp-decision`).

Delivery is best-effort: `POST $SMO_URL/eventListener/v7`, 2 s timeout, failures logged at
WARNING (rate-limited) and dropped - the RU keeps running without an SMO.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Optional

import httpx

from ocp_auru.common import ves
from ocp_auru.common.config import Settings
from ocp_auru.common.events import CarrierPm, CarrierStateChanged, PmReport, RuAppDecision, VesEnvelope
from ocp_auru.ruapp.agent import RuApp

from .bus import Bus
from .datastore import Datastore
from .radio import RadioSim

log = logging.getLogger(__name__)

_WARN_EVERY = 12  # log every Nth consecutive failure at WARNING (others at DEBUG)


class Reporter(threading.Thread):
    def __init__(self, settings: Settings, datastore: Datastore, radio: RadioSim, agent: RuApp, bus: Bus,
                 client: Optional[httpx.Client] = None) -> None:
        super().__init__(name="ru-reporter", daemon=True)
        self._settings = settings
        self._ds = datastore
        self._radio = radio
        self._agent = agent
        self._bus = bus
        self._client = client or httpx.Client(timeout=2.0)
        self._q: queue.Queue[Any] = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.latest: Optional[PmReport] = None
        self.sent = 0
        self.failed = 0
        self._consecutive_failures = 0
        self._subs: list = []

    @property
    def url(self) -> str:
        return f"{self._settings.smo_url}/eventListener/v7"

    # ---- lifecycle ------------------------------------------------------------------------
    def start(self) -> None:  # type: ignore[override]
        self._subs.append(self._bus.subscribe(CarrierStateChanged, self._q.put, inline=True, name="reporter-state"))
        self._subs.append(self._bus.subscribe(RuAppDecision, self._q.put, inline=True, name="reporter-decision"))
        super().start()

    def stop(self) -> None:
        for s in self._subs:
            self._bus.unsubscribe(s)
        self._subs.clear()
        self._stop.set()
        self._q.put(None)

    def run(self) -> None:
        interval = max(0.1, float(self._settings.ru_pm_interval_s))
        next_pm = time.monotonic() + interval
        while not self._stop.is_set():
            timeout = max(0.0, next_pm - time.monotonic())
            try:
                item = self._q.get(timeout=timeout)
            except queue.Empty:
                item = None
                report = self.build_report(interval)
                self._post(ves.measurement_event(report))
                next_pm = time.monotonic() + interval
                continue
            if item is None:
                continue
            self._post(self.mirror(item))

    # ---- builders -------------------------------------------------------------------------
    def build_report(self, interval_s: Optional[float] = None) -> PmReport:
        interval_s = float(interval_s if interval_s is not None else self._settings.ru_pm_interval_s)
        carriers: list[CarrierPm] = []
        for st in self._ds.statuses().values():
            m = self._radio.latest(st.name)
            iv = self._radio.take_interval(st.name)
            carriers.append(CarrierPm(
                name=st.name, active=st.active, state=st.state, tx_power_dbm=m.tx_power_dbm,
                prb_util_dl_pct=round(iv.prb_util_dl_pct, 2), prb_util_ul_pct=round(iv.prb_util_ul_pct, 2),
                rrc_conn=iv.rrc_conn, power_w_avg=round(iv.power_w_avg, 2), energy_wh=round(iv.energy_wh, 3),
                prach_detections=iv.prach_detections, wake_events=self._agent.wake_events(st.name),
                wake_latency_us=self._agent.stats(st.name),
            ))
        report = PmReport(ru_id=self._settings.ru_id, interval_s=interval_s, sim_time_iso=self._radio.clock.sim_time_iso(),
                          carriers=carriers, epoch_us=time.time_ns() // 1000)
        with self._lock:
            self.latest = report
        return report

    def mirror(self, ev: Any) -> VesEnvelope:
        ru = self._settings.ru_id
        if isinstance(ev, CarrierStateChanged):
            fields = {"carrier": ev.carrier, "active": ev.active.value, "state": ev.state.value, "cause": ev.cause, "wall_us": ev.wall_us}
            return ves.notification_event(ru, "tx-array-carriers-state-change", fields, ev.wall_us)
        if isinstance(ev, RuAppDecision):
            fields = ev.model_dump(mode="json")
            fields["wake_latency_us"] = None if ev.wake_latency_ns is None else round(ev.wake_latency_ns / 1000.0, 3)
            return ves.notification_event(ru, "ruapp-decision", fields, ev.wall_us)
        raise TypeError(type(ev))

    # ---- transport ------------------------------------------------------------------------
    def _post(self, envelope: VesEnvelope) -> bool:
        try:
            r = self._client.post(self.url, json=envelope.model_dump(mode="json"))
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - the RU must run without an SMO
            with self._lock:
                self.failed += 1
                self._consecutive_failures += 1
                n = self._consecutive_failures
            lvl = logging.WARNING if n == 1 or n % _WARN_EVERY == 0 else logging.DEBUG
            log.log(lvl, "VES push to %s failed (%d consecutive): %s", self.url, n, exc)
            return False
        with self._lock:
            self.sent += 1
            self._consecutive_failures = 0
        log.debug("VES %s sent", envelope.event.commonEventHeader.eventName)
        return True
