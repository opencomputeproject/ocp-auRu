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
"""Typed contracts shared across processes (pydantic v2).

Naming follows O-RAN WG4 `o-ran-uplane-conf` for carrier semantics and the VES event
envelope shape for north-bound events. Anything here is part of a cross-process contract;
change it with the HLD (`artifacts/design/HLD_closed-loop.md` §3).
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------------------
# o-ran-uplane-conf carrier semantics
# --------------------------------------------------------------------------------------
class Active(str, Enum):
    """`tx-array-carriers/active` (config): the policy envelope written by the rApp."""

    INACTIVE = "INACTIVE"  # transmission disabled; the ruApp MUST NOT wake it
    SLEEP = "SLEEP"        # energy-saving; the ruApp MAY wake it locally (SLEEP -> ACTIVE only)
    ACTIVE = "ACTIVE"      # normal service


class CarrierState(str, Enum):
    """`tx-array-carriers/state` (config false): the operational state reported by the RU."""

    DISABLED = "DISABLED"
    BUSY = "BUSY"
    READY = "READY"


class RuAppPolicy(BaseModel):
    """`ocp-ruapp-policy` tunables carried in the same edit-config as `active`."""

    wake_on_prach: bool = True
    prach_energy_threshold_db: int = 10
    min_hold_after_wake_s: int = 30
    report_interval_s: int = 5


class CarrierConfig(BaseModel):
    name: str
    absolute_frequency_center: int = 650000   # NR-ARFCN (illustrative n77-ish value)
    center_of_channel_bandwidth: int = 3750000000
    channel_bandwidth: int = 100000000
    active: Active = Active.ACTIVE
    type: Literal["NR", "LTE"] = "NR"
    duplex_scheme: Literal["TDD", "FDD"] = "TDD"
    gain: float = 20.0
    ruapp_policy: RuAppPolicy = Field(default_factory=RuAppPolicy)


class CarrierStatus(BaseModel):
    name: str
    active: Active
    state: CarrierState
    ruapp_policy: RuAppPolicy


# --------------------------------------------------------------------------------------
# RU-internal events (in-process bus between radio sim, datastore, ruApp, reporters)
# --------------------------------------------------------------------------------------
class PrachDetected(BaseModel):
    carrier: str
    energy_db: float
    t_ns: int  # time.perf_counter_ns() at detection
    wall_us: int  # epoch microseconds


class CarrierStateChanged(BaseModel):
    carrier: str
    active: Active
    state: CarrierState
    t_ns: int
    wall_us: int
    cause: str  # "netconf-edit-config" | "ruapp-wake" | "settle" | "init"


class RuAppDecision(BaseModel):
    carrier: str
    decision: Literal["wake", "ignore-inactive", "ignore-disabled-policy", "ignore-threshold", "ignore-active"]
    prach_energy_db: float
    wake_latency_ns: Optional[int] = None  # detection -> ACTIVE applied
    wall_us: int


# --------------------------------------------------------------------------------------
# PM (measurement) payloads
# --------------------------------------------------------------------------------------
class WakeLatencyStats(BaseModel):
    last: Optional[float] = None
    p50: Optional[float] = None
    max: Optional[float] = None
    count: int = 0


class CarrierPm(BaseModel):
    name: str
    active: Active
    state: CarrierState
    tx_power_dbm: float
    prb_util_dl_pct: float
    prb_util_ul_pct: float
    rrc_conn: int
    power_w_avg: float
    energy_wh: float
    prach_detections: int
    wake_events: int
    wake_latency_us: WakeLatencyStats = Field(default_factory=WakeLatencyStats)


class PmReport(BaseModel):
    ru_id: str
    interval_s: float
    sim_time_iso: str  # simulated wall clock (accelerated diurnal)
    carriers: list[CarrierPm]
    epoch_us: int


# --------------------------------------------------------------------------------------
# VES-shaped north-bound envelope (RU -> SMO collector)
# Only `CommonEventHeader` follows VES 7.2.1 field types; the measurement/notification
# bodies are project-specific dicts (see ocp_auru/common/ves.py).
# --------------------------------------------------------------------------------------
def _now_us() -> int:
    return time.time_ns() // 1000


class CommonEventHeader(BaseModel):
    domain: Literal["measurement", "notification", "heartbeat"]
    eventName: str
    eventId: str
    sourceName: str
    reportingEntityName: str
    priority: Literal["High", "Medium", "Normal", "Low"] = "Normal"
    sequence: int = 0
    startEpochMicrosec: int = Field(default_factory=_now_us)
    lastEpochMicrosec: int = Field(default_factory=_now_us)
    version: str = "4.1"
    vesEventListenerVersion: str = "7.2.1"


class VesEvent(BaseModel):
    commonEventHeader: CommonEventHeader
    measurementFields: Optional[dict[str, Any]] = None
    notificationFields: Optional[dict[str, Any]] = None


class VesEnvelope(BaseModel):
    event: VesEvent


# --------------------------------------------------------------------------------------
# CM request (rApp -> SMO) and what the SMO's south-bound adapter reports back
# --------------------------------------------------------------------------------------
class CmCarrierRequest(BaseModel):
    ru_id: str
    carrier: str
    active: Active
    ruapp_policy: RuAppPolicy = Field(default_factory=RuAppPolicy)
    reason: str = ""


class NetconfResult(BaseModel):
    """One NETCONF operation on the O-RU as timed by the M-Plane client."""

    ok: bool = True
    rtt_ms: float                 # the operation's own round trip (rpc sent -> rpc-reply)
    connect_ms: float             # NETCONF session setup (SSH + hello)
    target: str                   # "host:port" of the O-RU's M-Plane server
    operation: Literal["edit-config", "get"] = "edit-config"


class AdapterResult(BaseModel):
    """What a CM adapter (`ocp_auru/smo/o1_adapter.py`) returns for one carrier write."""

    ok: bool
    netconf_rtt_ms: Optional[float] = None
    detail: str = ""


class CmCarrierResponse(BaseModel):
    ok: bool
    ru_id: str
    carrier: str
    active: Active
    cm_rtt_ms: float          # whole SMO-side round trip (adapter call incl. session setup), measured at the SMO
    netconf_rtt_ms: Optional[float] = None  # edit-config only, measured by the M-Plane client
    detail: str = ""


# --------------------------------------------------------------------------------------
# Canonical KPI row used by the rApp's data layer (30-minute granularity)
# --------------------------------------------------------------------------------------
CANONICAL_KPI_COLUMNS = [
    "ts",               # ISO-8601 (naive, local-to-report) or tz-aware
    "cell_id",          # opaque, stable, non-identifying label
    "prb_util_dl_pct",
    "prb_util_ul_pct",
    "thr_dl_mbps",
    "thr_ul_mbps",
    "vol_dl_mib",
    "active_users",
]
