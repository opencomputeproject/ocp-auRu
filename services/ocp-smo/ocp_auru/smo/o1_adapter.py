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
"""CM adapter seam of the SMO: what the CM routes (`cm.py`) call to reach the radio.

The routes know nothing about the transport. They call a `CmAdapter`:

- `set_carrier_state(req, include_policy)` writes the policy envelope (`active`) of one
  carrier, plus the ruApp policy tunables when `include_policy` is true;
- `get_carriers()` reads the carriers' `active` / `state` for inspection;
- `describe()` names the adapter and its target for `/health`.

Adapters are selected by name through `Settings.cm_adapter` (`CM_ADAPTER`) from a registry.
`register()` adds one; `create()` builds the configured one.

Reference adapter, `mplane-direct`: the SMO performs the NETCONF `edit-config` on the O-RU's
`o-ran-uplane-conf` itself. This is the O-RAN hybrid M-Plane model, in which the SMO manages
the O-RU directly.

A hierarchical path, in which the SMO asks an O-DU over O1 and the O-DU runs the M-Plane
session towards its O-RU, plugs in through the same seam: implement `CmAdapter` for that
O-DU's management interface, `register()` it under a name, and select it with `CM_ADAPTER`.
The CM routes, the rApp and the response contract stay as they are.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Protocol, runtime_checkable

from ocp_auru.common import yang
from ocp_auru.common.config import Settings
from ocp_auru.common.events import AdapterResult, CmCarrierRequest
from ocp_auru.smo.mplane_client import MplaneClient, MplaneError

log = logging.getLogger(__name__)


@runtime_checkable
class CmAdapter(Protocol):
    """South-bound interface of the SMO's CM routes."""

    name: str

    def describe(self) -> dict[str, Any]:
        """Adapter name and target, shown in the SMO's `/health`."""

    def set_carrier_state(self, req: CmCarrierRequest, include_policy: bool = True) -> AdapterResult:
        """Write `active` (+ the ruApp policy) for `req.carrier`. Never raises: failures are `ok=False` with `detail`."""

    def get_carriers(self) -> dict[str, Any]:
        """`{"ok": bool, "carriers": [{"name", "active", "state"}], "netconf": {...}}` or `{"ok": False, "detail", "carriers": []}`."""


class MplaneDirectAdapter:
    """Reference adapter: NETCONF `edit-config` from the SMO straight to the O-RU (hybrid M-Plane)."""

    name = "mplane-direct"

    def __init__(self, cfg: Settings, client: MplaneClient | None = None):
        self.cfg = cfg
        self.client = client or MplaneClient(cfg)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "model": "hybrid M-Plane (SMO -> O-RU)", "netconf_target": self.client.target,
                "netconf_timeout_s": self.cfg.mplane_timeout_s}

    def set_carrier_state(self, req: CmCarrierRequest, include_policy: bool = True) -> AdapterResult:
        config_xml = yang.build_carrier_edit(req.carrier, req.active, req.ruapp_policy if include_policy else None)
        log.info("M-Plane: edit tx-array-carriers/%s/active -> %s%s via NETCONF at %s", req.carrier, req.active.value,
                 " + ruapp-policy" if include_policy else "", self.client.target)
        try:
            result = self.client.edit_config(config_xml)
        except MplaneError as exc:
            log.error("M-Plane: edit-config failed: %s", exc)
            return AdapterResult(ok=False, detail=f"NETCONF edit-config failed: {exc}")
        log.info("M-Plane: edit-config applied (rtt %.3f ms, session setup %.3f ms)", result.rtt_ms, result.connect_ms)
        return AdapterResult(ok=True, netconf_rtt_ms=result.rtt_ms, detail=f"edit-config on {result.target}")

    def get_carriers(self) -> dict[str, Any]:
        try:
            status = self.client.get_carrier_status()
        except MplaneError as exc:
            log.error("M-Plane: get failed: %s", exc)
            return {"ok": False, "detail": f"NETCONF get failed: {exc}", "carriers": []}
        return {"ok": True, **status}


AdapterFactory = Callable[[Settings], CmAdapter]
_REGISTRY: dict[str, AdapterFactory] = {}


def register(name: str, factory: AdapterFactory) -> None:
    """Make an adapter selectable as `CM_ADAPTER=<name>`."""
    _REGISTRY[name] = factory


def available() -> list[str]:
    return sorted(_REGISTRY)


def create(cfg: Settings) -> CmAdapter:
    """Build the adapter named by `cfg.cm_adapter`; `ValueError` for an unknown name."""
    try:
        factory = _REGISTRY[cfg.cm_adapter]
    except KeyError:
        raise ValueError(f"unknown CM adapter {cfg.cm_adapter!r} (CM_ADAPTER); available: {', '.join(available())}") from None
    return factory(cfg)


register(MplaneDirectAdapter.name, MplaneDirectAdapter)
