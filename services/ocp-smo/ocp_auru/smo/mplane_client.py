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
"""M-Plane client of the SMO: NETCONF over SSH to the O-RU (O-RAN WG4 M-Plane, ncclient).

One operation per session: open a NETCONF session, run `edit-config` on `running` (or a
`<get>` with a subtree filter), close. Session setup and the operation are timed separately:
`rtt_ms` is the operation's own round trip, `connect_ms` the session setup. `rtt_ms` of an
`edit-config` is the `netconf_rtt_ms` surfaced to the rApp.

Security note: the demo connects with password auth and host-key verification off
(`MPLANE_HOSTKEY_VERIFY=0`, the mock O-RU generates its key on first start). A real O-RU
deployment must pin the RU host key (`MPLANE_HOSTKEY_VERIFY=1` and a known_hosts entry)
and use per-device credentials.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from lxml import etree
from ncclient import manager
from ncclient.operations import RPCError

from ocp_auru.common import yang
from ocp_auru.common.config import Settings, settings
from ocp_auru.common.events import NetconfResult

log = logging.getLogger(__name__)


class MplaneError(RuntimeError):
    """Any failure between the SMO and the O-RU (transport, auth, or an `<rpc-error>`)."""


def _describe(exc: Exception) -> str:
    if isinstance(exc, RPCError):
        return f"rpc-error: {exc.message or exc}".strip()
    return f"{type(exc).__name__}: {exc}"


def parse_carriers(data_el: etree._Element) -> list[dict[str, Any]]:
    """`tx-array-carriers` entries (`name`, `active`, `state`) from a `<data>`/`<user-plane-configuration>` element."""
    out: list[dict[str, Any]] = []
    for tx in data_el.iter(f"{{{yang.NS_UPLANE}}}tx-array-carriers"):
        name = tx.findtext(f"{{{yang.NS_UPLANE}}}name")
        if not name:
            continue
        out.append({
            "name": name,
            "active": tx.findtext(f"{{{yang.NS_UPLANE}}}active"),
            "state": tx.findtext(f"{{{yang.NS_UPLANE}}}state"),
        })
    return out


class MplaneClient:
    """NETCONF client for one O-RU, configured from `Settings` (`RU_NETCONF_*`, `MPLANE_*`)."""

    def __init__(self, cfg: Optional[Settings] = None):
        self.cfg = cfg or settings

    @property
    def target(self) -> str:
        return self.cfg.ru_netconf_target

    def _connect(self):
        return manager.connect(
            host=self.cfg.ru_netconf_host,
            port=self.cfg.ru_netconf_port,
            username=self.cfg.ru_netconf_user,
            password=self.cfg.ru_netconf_password,
            hostkey_verify=self.cfg.mplane_hostkey_verify,
            look_for_keys=False,
            allow_agent=False,
            device_params={"name": "default"},
            timeout=self.cfg.mplane_timeout_s,
        )

    @staticmethod
    def _close(m) -> None:
        try:
            m.close_session()
        except Exception as exc:  # the O-RU may drop the transport first; not an error for the operation
            log.debug("close-session ignored: %s", exc)

    def edit_config(self, config_xml: str) -> NetconfResult:
        """`edit-config target=running` with the given `<config>` payload. Raises `MplaneError` on any failure."""
        try:
            t0 = time.perf_counter()
            m = self._connect()
            connect_ms = (time.perf_counter() - t0) * 1000.0
            try:
                m.timeout = self.cfg.mplane_timeout_s
                t1 = time.perf_counter()
                reply = m.edit_config(target="running", config=config_xml)
                rtt_ms = (time.perf_counter() - t1) * 1000.0
                if not reply.ok:
                    raise MplaneError(f"rpc-error: {reply.error}")
            finally:
                self._close(m)
        except MplaneError:
            raise
        except Exception as exc:
            raise MplaneError(_describe(exc)) from exc
        return NetconfResult(rtt_ms=round(rtt_ms, 3), connect_ms=round(connect_ms, 3), target=self.target, operation="edit-config")

    def get_carrier_status(self) -> dict[str, Any]:
        """`<get>` with a `user-plane-configuration` subtree filter: `{"carriers": [...], "netconf": {...}}`."""
        try:
            t0 = time.perf_counter()
            m = self._connect()
            connect_ms = (time.perf_counter() - t0) * 1000.0
            try:
                m.timeout = self.cfg.mplane_timeout_s
                t1 = time.perf_counter()
                reply = m.get(filter=("subtree", yang.build_uplane_filter()))
                rtt_ms = (time.perf_counter() - t1) * 1000.0
                carriers = parse_carriers(reply.data_ele)
            finally:
                self._close(m)
        except Exception as exc:
            raise MplaneError(_describe(exc)) from exc
        result = NetconfResult(rtt_ms=round(rtt_ms, 3), connect_ms=round(connect_ms, 3), target=self.target, operation="get")
        return {"carriers": carriers, "netconf": result.model_dump()}
