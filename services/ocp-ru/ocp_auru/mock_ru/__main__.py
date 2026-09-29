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
"""`python -m ocp_auru.mock_ru` - run the mock O-RU (M-Plane NETCONF/SSH + ruApp + mgmt API)."""
from __future__ import annotations

import signal
import threading

from ocp_auru.common.config import settings
from ocp_auru.common.log import setup_logging

from .app import build


def main() -> int:
    log = setup_logging("mock_ru")
    ctx = build(settings)
    ctx.start()
    log.info("NETCONF ssh://%s@%s:%d  mgmt http://%s:%d  SMO %s",
             settings.ru_netconf_user, ctx.netconf.host, ctx.netconf.port,
             settings.ru_netconf_host, ctx.mgmt.port if ctx.mgmt else settings.ru_mgmt_port, settings.smo_url)
    done = threading.Event()

    def _stop(signum, _frame):
        log.info("signal %s: shutting down", signal.Signals(signum).name)
        done.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    while not done.wait(0.5):
        pass
    ctx.stop()
    log.info("mock O-RU stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
