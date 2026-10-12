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
"""Wiring of the mock O-RU process: datastore + bus + radio + ruApp + M-Plane + PM push + mgmt API."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from ocp_auru.common.config import Settings
from ocp_auru.common.events import CarrierStateChanged, PrachDetected, RuAppDecision
from ocp_auru.ruapp.agent import RuApp

from .bus import Bus
from .datastore import Datastore
from .mgmt_api import MgmtServer, create_app
from .netconf_server import RuNetconfServer
from .radio import RadioSim
from .recorder import EventRecorder
from .reporter import Reporter

log = logging.getLogger(__name__)


@dataclass
class RuContext:
    settings: Settings
    bus: Bus
    datastore: Datastore
    recorder: EventRecorder
    radio: RadioSim
    agent: RuApp
    netconf: RuNetconfServer
    reporter: Reporter
    mgmt: Optional[MgmtServer] = None
    started: bool = field(default=False)

    def start(self, mgmt_api: bool = True) -> "RuContext":
        self.agent.start()
        self.netconf.start()
        self.reporter.start()
        self.radio.start()
        if mgmt_api:
            self.mgmt = MgmtServer(create_app(self), self.settings.ru_netconf_host, self.settings.ru_mgmt_port).start()
        self.datastore.publish_initial()
        self.started = True
        log.info("mock O-RU %s up: carriers=%s initial=%s", self.settings.ru_id, list(self.settings.ru_carriers), self.settings.ru_initial_active)
        return self

    def stop(self) -> None:
        if self.mgmt is not None:
            self.mgmt.stop()
        self.radio.stop()
        self.reporter.stop()
        self.netconf.stop()
        self.agent.stop()
        self.datastore.close()
        self.bus.close()
        self.started = False


def build(settings: Settings, *, background_prach: bool = True, seed: Optional[int] = None) -> RuContext:
    bus = Bus()
    datastore = Datastore(settings, bus)
    recorder = EventRecorder(bus, maxlen=500, kinds=(CarrierStateChanged, RuAppDecision, PrachDetected))
    radio = RadioSim(settings, datastore, bus, seed=seed, background_prach=background_prach)
    agent = RuApp(datastore, bus)
    netconf = RuNetconfServer(settings, datastore, bus)
    reporter = Reporter(settings, datastore, radio, agent, bus)
    return RuContext(settings=settings, bus=bus, datastore=datastore, recorder=recorder, radio=radio,
                     agent=agent, netconf=netconf, reporter=reporter)
