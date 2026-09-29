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
"""XML helpers for the M-Plane subset used in the demo.

Models: O-RAN WG4 `o-ran-uplane-conf` (v07.01 namespace `urn:o-ran:uplane-conf:1.0`),
`ietf-interfaces`, and the demo extension `ocp-ruapp-policy`
(`urn:ocp:auru:ruapp-policy:1.0`, see `services/ocp-ru/yang/`). Builders produce the *content* of
`<config>`/`<data>`; callers wrap it in the NETCONF operation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional

from lxml import etree

from .events import Active, CarrierConfig, CarrierState, CarrierStatus, RuAppPolicy

NS_NC = "urn:ietf:params:xml:ns:netconf:base:1.0"
NS_NOTIF = "urn:ietf:params:xml:ns:netconf:notification:1.0"
NS_UPLANE = "urn:o-ran:uplane-conf:1.0"
NS_IETF_IF = "urn:ietf:params:xml:ns:yang:ietf-interfaces"
NS_RUAPP = "urn:ocp:auru:ruapp-policy:1.0"
NS_SUPERVISION = "urn:o-ran:supervision:1.0"

NSMAP = {"nc": NS_NC, "uc": NS_UPLANE, "if": NS_IETF_IF, "orp": NS_RUAPP, "sv": NS_SUPERVISION, "ncn": NS_NOTIF}

CAPABILITIES = [
    "urn:ietf:params:netconf:base:1.0",
    "urn:ietf:params:netconf:base:1.1",
    "urn:ietf:params:netconf:capability:notification:1.0",
    "urn:ietf:params:netconf:capability:interleave:1.0",
    # Revisions are the latest `revision` statements of the O-RAN WG4 M-Plane v07.01 model pack
    # (o-ran-uplane-conf 7.0.0, o-ran-supervision 5.1.0).
    f"{NS_UPLANE}?module=o-ran-uplane-conf&revision=2021-07-26",
    f"{NS_SUPERVISION}?module=o-ran-supervision&revision=2021-03-22",
    f"{NS_IETF_IF}?module=ietf-interfaces&revision=2018-02-20",
    f"{NS_RUAPP}?module=ocp-ruapp-policy&revision=2026-08-25",
]


def _sub(parent: etree._Element, tag: str, text: Optional[object] = None, ns: str = NS_UPLANE) -> etree._Element:
    el = etree.SubElement(parent, f"{{{ns}}}{tag}")
    if text is not None:
        el.text = str(text)
    return el


# --------------------------------------------------------------------------------------
# Builders (client side)
# --------------------------------------------------------------------------------------
def build_carrier_edit(name: str, active: Active, policy: Optional[RuAppPolicy] = None) -> str:
    """`<config>` payload: set `tx-array-carriers[name]/active` (+ ruApp policy)."""
    config = etree.Element(f"{{{NS_NC}}}config", nsmap={None: NS_NC})
    upc = etree.SubElement(config, f"{{{NS_UPLANE}}}user-plane-configuration", nsmap={None: NS_UPLANE})
    car = _sub(upc, "tx-array-carriers")
    _sub(car, "name", name)
    _sub(car, "active", active.value)
    if policy is not None:
        rp = etree.SubElement(config, f"{{{NS_RUAPP}}}ruapp-policy", nsmap={None: NS_RUAPP})
        c = _sub(rp, "carrier", ns=NS_RUAPP)
        _sub(c, "name", name, ns=NS_RUAPP)
        _sub(c, "wake-on-prach", "true" if policy.wake_on_prach else "false", ns=NS_RUAPP)
        _sub(c, "prach-energy-threshold-db", policy.prach_energy_threshold_db, ns=NS_RUAPP)
        _sub(c, "min-hold-after-wake-s", policy.min_hold_after_wake_s, ns=NS_RUAPP)
        _sub(c, "report-interval-s", policy.report_interval_s, ns=NS_RUAPP)
    return etree.tostring(config, encoding="unicode")


def build_interface_edit(interface: str, description: str) -> str:
    """`<config>` payload: set `ietf-interfaces/interface[name]/description`."""
    config = etree.Element(f"{{{NS_NC}}}config", nsmap={None: NS_NC})
    ifs = etree.SubElement(config, f"{{{NS_IETF_IF}}}interfaces", nsmap={None: NS_IETF_IF})
    itf = _sub(ifs, "interface", ns=NS_IETF_IF)
    _sub(itf, "name", interface, ns=NS_IETF_IF)
    _sub(itf, "description", description, ns=NS_IETF_IF)
    return etree.tostring(config, encoding="unicode")


def build_uplane_filter() -> str:
    return f'<user-plane-configuration xmlns="{NS_UPLANE}"/>'


# --------------------------------------------------------------------------------------
# Serialisers (server side)
# --------------------------------------------------------------------------------------
# Illustrative constants for the leaves o-ran-uplane-conf marks `mandatory true` on
# tx-/rx-array-carriers (frame/SFN offsets, rx gain correction, n-ta-offset); the demo does not
# model timing, so they are fixed. The served subtree is a *subset* and is not schema-validated.
_DL_RADIO_FRAME_OFFSET = 0
_DL_SFN_OFFSET = 0
_RX_GAIN_CORRECTION = "0.0000"
_RX_N_TA_OFFSET = 0


def carriers_to_xml(carriers: Iterable[CarrierConfig], status: dict[str, CarrierStatus], include_state: bool = True) -> etree._Element:
    """`<user-plane-configuration>` element with tx/rx array carriers (+ `config false` leaves
    `state` and `duplex-scheme` when `include_state`, i.e. for `<get>` but not `<get-config>`)."""
    upc = etree.Element(f"{{{NS_UPLANE}}}user-plane-configuration", nsmap={None: NS_UPLANE})
    for c in carriers:
        tx = _sub(upc, "tx-array-carriers")
        _sub(tx, "name", c.name)
        # `absolute-frequency-center` is `status deprecated` since 5.0.0 (kept for older clients);
        # `center-of-channel-bandwidth` is the current leaf.
        _sub(tx, "absolute-frequency-center", c.absolute_frequency_center)
        _sub(tx, "center-of-channel-bandwidth", c.center_of_channel_bandwidth)
        _sub(tx, "channel-bandwidth", c.channel_bandwidth)
        _sub(tx, "active", c.active.value)
        if include_state:
            _sub(tx, "state", status[c.name].state.value)
        _sub(tx, "type", c.type)
        if include_state:
            _sub(tx, "duplex-scheme", c.duplex_scheme)  # config false in the module (rw-duplex-scheme is the config leaf)
        _sub(tx, "gain", f"{c.gain:.4f}")
        _sub(tx, "downlink-radio-frame-offset", _DL_RADIO_FRAME_OFFSET)
        _sub(tx, "downlink-sfn-offset", _DL_SFN_OFFSET)
        rx = _sub(upc, "rx-array-carriers")
        _sub(rx, "name", c.name.replace("tx", "rx", 1) if c.name.startswith("tx") else f"rx-{c.name}")
        _sub(rx, "absolute-frequency-center", c.absolute_frequency_center)
        _sub(rx, "center-of-channel-bandwidth", c.center_of_channel_bandwidth)
        _sub(rx, "channel-bandwidth", c.channel_bandwidth)
        _sub(rx, "active", c.active.value)
        if include_state:
            _sub(rx, "state", status[c.name].state.value)
        _sub(rx, "type", c.type)
        if include_state:
            _sub(rx, "duplex-scheme", c.duplex_scheme)
        _sub(rx, "downlink-radio-frame-offset", _DL_RADIO_FRAME_OFFSET)
        _sub(rx, "downlink-sfn-offset", _DL_SFN_OFFSET)
        _sub(rx, "gain-correction", _RX_GAIN_CORRECTION)
        _sub(rx, "n-ta-offset", _RX_N_TA_OFFSET)
    return upc


def ruapp_policy_to_xml(policies: dict[str, RuAppPolicy]) -> etree._Element:
    rp = etree.Element(f"{{{NS_RUAPP}}}ruapp-policy", nsmap={None: NS_RUAPP})
    for name, p in policies.items():
        c = _sub(rp, "carrier", ns=NS_RUAPP)
        _sub(c, "name", name, ns=NS_RUAPP)
        _sub(c, "wake-on-prach", "true" if p.wake_on_prach else "false", ns=NS_RUAPP)
        _sub(c, "prach-energy-threshold-db", p.prach_energy_threshold_db, ns=NS_RUAPP)
        _sub(c, "min-hold-after-wake-s", p.min_hold_after_wake_s, ns=NS_RUAPP)
        _sub(c, "report-interval-s", p.report_interval_s, ns=NS_RUAPP)
    return rp


def state_change_notification(carrier: str, state: CarrierState, when: Optional[datetime] = None) -> str:
    """RFC 5277 `<notification>` carrying `o-ran-uplane-conf:tx-array-carriers-state-change`."""
    when = when or datetime.now(timezone.utc)
    n = etree.Element(f"{{{NS_NOTIF}}}notification", nsmap={None: NS_NOTIF})
    et = etree.SubElement(n, f"{{{NS_NOTIF}}}eventTime")
    et.text = when.isoformat(timespec="microseconds").replace("+00:00", "Z")
    sc = etree.SubElement(n, f"{{{NS_UPLANE}}}tx-array-carriers-state-change", nsmap={None: NS_UPLANE})
    car = _sub(sc, "tx-array-carriers")
    _sub(car, "name", carrier)
    _sub(car, "state", state.value)
    return etree.tostring(n, encoding="unicode")


# --------------------------------------------------------------------------------------
# Parsers (server side: what did the client ask to change?)
# --------------------------------------------------------------------------------------
def parse_carrier_edits(config_el: etree._Element) -> dict[str, dict]:
    """Return `{carrier_name: {"active": Active|None, "policy": {...}|None}}` from a `<config>` element."""
    out: dict[str, dict] = {}
    for tx in config_el.iter(f"{{{NS_UPLANE}}}tx-array-carriers"):
        name = tx.findtext(f"{{{NS_UPLANE}}}name")
        if not name:
            continue
        entry = out.setdefault(name, {"active": None, "policy": None})
        active = tx.findtext(f"{{{NS_UPLANE}}}active")
        if active:
            entry["active"] = Active(active)
    for c in config_el.iter(f"{{{NS_RUAPP}}}carrier"):
        name = c.findtext(f"{{{NS_RUAPP}}}name")
        if not name:
            continue
        entry = out.setdefault(name, {"active": None, "policy": None})
        pol: dict = {}
        v = c.findtext(f"{{{NS_RUAPP}}}wake-on-prach")
        if v is not None:
            pol["wake_on_prach"] = v.strip().lower() == "true"
        for tag, key in (
            ("prach-energy-threshold-db", "prach_energy_threshold_db"),
            ("min-hold-after-wake-s", "min_hold_after_wake_s"),
            ("report-interval-s", "report_interval_s"),
        ):
            v = c.findtext(f"{{{NS_RUAPP}}}{tag}")
            if v is not None:
                pol[key] = int(v)
        entry["policy"] = pol
    return out


def parse_interface_edits(config_el: etree._Element) -> dict[str, str]:
    out: dict[str, str] = {}
    for itf in config_el.iter(f"{{{NS_IETF_IF}}}interface"):
        name = itf.findtext(f"{{{NS_IETF_IF}}}name")
        desc = itf.findtext(f"{{{NS_IETF_IF}}}description")
        if name and desc is not None:
            out[name] = desc
    return out
