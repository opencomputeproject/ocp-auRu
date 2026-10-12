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
"""Radio simulation inside the mock O-RU (HLD §3.3): accelerated diurnal load, PRACH, energy.

O-RAN concept: the O-RU's PHY sees random-access preambles (PRACH) on the receive path
even when the transmit carrier is in `SLEEP` - that is the demo premise for a local
wake-on-demand. This module produces the per-carrier counters the RU reports as O1 PM
(`prb_util_dl_pct`, `prb_util_ul_pct`, `rrc_conn`, `tx_power_dbm`, power/energy) and
raises `PrachDetected` events for the ruApp.

Clock: simulated seconds = wall seconds x `RU_SIM_SPEEDUP` (720 = one simulated hour per
5 s), starting at `RU_SIM_START_HOUR` on a fixed, generic date.

ALL CONSTANTS BELOW ARE ILLUSTRATIVE (not measurements of any product or network):
power model ACTIVE P = 250 W + 1.5 W x prb_util_dl_pct, SLEEP 60 W, INACTIVE 20 W;
tx power 43 dBm when transmitting; PRACH rate and energy follow the load curve.
Energy is integrated over *simulated* time, so one PM interval at 720x reports roughly
one simulated hour of consumption.
"""
from __future__ import annotations

import json
import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from ocp_auru.common.config import Settings
from ocp_auru.common.events import Active, CarrierState, PrachDetected

from .bus import Bus
from .datastore import Datastore

log = logging.getLogger(__name__)

SIM_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)  # fixed, generic date for the simulated clock

# --- illustrative constants (see module docstring) ----------------------------------------
P_ACTIVE_BASE_W = 250.0
P_ACTIVE_PER_PRB_PCT_W = 1.5
P_SLEEP_W = 60.0
P_INACTIVE_W = 20.0
TX_POWER_ACTIVE_DBM = 43.0
TX_POWER_OFF_DBM = 0.0  # stands in for -inf / "not transmitting"
PRB_UTIL_AT_FULL_LOAD_PCT = 95.0
UL_TO_DL_PRB_RATIO = 0.4
RRC_CONN_AT_FULL_LOAD = 180
PRACH_RATE_MIN_HZ = 0.02      # background arrivals per wall second at zero load
PRACH_RATE_PEAK_HZ = 0.30     # additional arrivals per wall second at full load
PRACH_ENERGY_FLOOR_DB = 3.0   # typical detected energy above noise at zero load
PRACH_ENERGY_PER_LOAD_DB = 12.0
PRACH_ENERGY_SIGMA_DB = 2.0

# Generic diurnal profile: (hour, normalised load). Low 02:00-06:00, peaks ~12:00 and ~19:00.
_DIURNAL_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.0, 0.25), (2.0, 0.10), (6.0, 0.10), (9.0, 0.55), (12.0, 0.90), (14.0, 0.70),
    (17.0, 0.75), (19.0, 0.95), (22.0, 0.50), (24.0, 0.25),
)


def diurnal_load(hour_of_day: float) -> float:
    """Piecewise-linear normalised load in [0, 1] for an hour-of-day in [0, 24)."""
    h = hour_of_day % 24.0
    for (h0, l0), (h1, l1) in zip(_DIURNAL_ANCHORS, _DIURNAL_ANCHORS[1:]):
        if h0 <= h <= h1:
            return l0 + (l1 - l0) * (h - h0) / (h1 - h0)
    return _DIURNAL_ANCHORS[-1][1]


def load_prb_profile(path: str) -> Optional[tuple[float, ...]]:
    """Optional replay profile, a JSON file with the mean DL PRB utilisation of one day in equal slots:
    `{"prb_util_dl_pct_by_hour": [24 numbers]}` or `{"prb_util_dl_pct_by_half_hour": [48 numbers]}` (for example one
    day of a KPI history). Returns None when `path` is empty."""
    if not path:
        return None
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, n in (("prb_util_dl_pct_by_half_hour", 48), ("prb_util_dl_pct_by_hour", 24)):
        if key in doc:
            values = doc[key]
            if len(values) != n:
                raise ValueError(f"{path}: {key} needs {n} values, got {len(values)}")
            return tuple(min(100.0, max(0.0, float(v))) for v in values)
    raise ValueError(f"{path}: expected prb_util_dl_pct_by_hour or prb_util_dl_pct_by_half_hour")


def profile_load(profile: tuple[float, ...], hour_of_day: float) -> float:
    """Normalised load in [0, 1] from a day profile; each value sits at the middle of its slot, linear in between."""
    n = len(profile)
    x = (hour_of_day * n / 24.0 - 0.5) % n
    i = int(x)
    prb = profile[i] + (profile[(i + 1) % n] - profile[i]) * (x - i)
    return prb / PRB_UTIL_AT_FULL_LOAD_PCT


class SimClock:
    def __init__(self, speedup: float, start_hour: float) -> None:
        self.speedup = speedup
        self._start_s = start_hour * 3600.0
        self._t0 = time.monotonic()

    def sim_seconds(self) -> float:
        return self._start_s + (time.monotonic() - self._t0) * self.speedup

    def sim_datetime(self) -> datetime:
        return SIM_EPOCH + timedelta(seconds=self.sim_seconds())

    def sim_time_iso(self) -> str:
        return self.sim_datetime().isoformat(timespec="seconds").replace("+00:00", "Z")

    def hour_of_day(self) -> float:
        return (self.sim_seconds() / 3600.0) % 24.0


@dataclass
class CarrierMetrics:
    """Latest instantaneous values for one carrier."""

    load: float = 0.0
    prb_util_dl_pct: float = 0.0
    prb_util_ul_pct: float = 0.0
    rrc_conn: int = 0
    tx_power_dbm: float = TX_POWER_OFF_DBM
    power_w: float = P_INACTIVE_W

    def as_dict(self) -> dict:
        return {
            "load": round(self.load, 3), "prb_util_dl_pct": round(self.prb_util_dl_pct, 2),
            "prb_util_ul_pct": round(self.prb_util_ul_pct, 2), "rrc_conn": self.rrc_conn,
            "tx_power_dbm": self.tx_power_dbm, "power_w": round(self.power_w, 1),
        }


@dataclass
class _IntervalAcc:
    """Accumulators reset at every PM report."""

    energy_wh: float = 0.0
    power_w_sum: float = 0.0
    samples: int = 0
    prach_detections: int = 0
    prb_dl_sum: float = 0.0
    prb_ul_sum: float = 0.0
    rrc_sum: int = 0


@dataclass
class IntervalSummary:
    power_w_avg: float
    energy_wh: float
    prach_detections: int
    prb_util_dl_pct: float
    prb_util_ul_pct: float
    rrc_conn: int


def power_w(active: Active, prb_util_dl_pct: float) -> float:
    if active == Active.ACTIVE:
        return P_ACTIVE_BASE_W + P_ACTIVE_PER_PRB_PCT_W * prb_util_dl_pct
    if active == Active.SLEEP:
        return P_SLEEP_W
    return P_INACTIVE_W


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth's algorithm; lam is small here (< ~1 per tick)."""
    if lam <= 0:
        return 0
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


class RadioSim(threading.Thread):
    def __init__(self, settings: Settings, datastore: Datastore, bus: Bus, tick_s: float = 0.05,
                 seed: Optional[int] = None, background_prach: bool = True) -> None:
        super().__init__(name="ru-radio-sim", daemon=True)
        self._settings = settings
        self._ds = datastore
        self._bus = bus
        self._tick_s = tick_s
        self._rng = random.Random(seed)
        self._background_prach = background_prach
        self._prb_profile = load_prb_profile(settings.ru_load_profile)   # None = generic diurnal curve
        self._prb_sigma = settings.ru_sim_prb_sigma
        self.clock = SimClock(settings.ru_sim_speedup, settings.ru_sim_start_hour)
        self._lock = threading.Lock()
        self._metrics: dict[str, CarrierMetrics] = {n: CarrierMetrics() for n in settings.ru_carriers}
        self._acc: dict[str, _IntervalAcc] = {n: _IntervalAcc() for n in settings.ru_carriers}
        self._stop = threading.Event()

    # ---- public API ---------------------------------------------------------------------
    def latest(self, carrier: str) -> CarrierMetrics:
        with self._lock:
            m = self._metrics[carrier]
            return CarrierMetrics(**vars(m))

    def latest_all(self) -> dict[str, dict]:
        with self._lock:
            return {n: m.as_dict() for n, m in self._metrics.items()}

    def take_interval(self, carrier: str) -> IntervalSummary:
        """Return and reset the PM-interval accumulators for a carrier."""
        with self._lock:
            a = self._acc[carrier]
            m = self._metrics[carrier]
            n = max(a.samples, 1)
            out = IntervalSummary(
                power_w_avg=a.power_w_sum / n if a.samples else m.power_w,
                energy_wh=a.energy_wh, prach_detections=a.prach_detections,
                prb_util_dl_pct=a.prb_dl_sum / n if a.samples else m.prb_util_dl_pct,
                prb_util_ul_pct=a.prb_ul_sum / n if a.samples else m.prb_util_ul_pct,
                rrc_conn=int(round(a.rrc_sum / n)) if a.samples else m.rrc_conn,
            )
            self._acc[carrier] = _IntervalAcc()
            return out

    def inject_prach(self, carrier: str, energy_db: float) -> PrachDetected:
        """Demo trigger: a PRACH burst is detected *now*; published synchronously to the ruApp."""
        if carrier not in self._metrics:
            raise KeyError(carrier)
        ev = PrachDetected(carrier=carrier, energy_db=float(energy_db), t_ns=time.perf_counter_ns(), wall_us=time.time_ns() // 1000)
        with self._lock:
            self._acc[carrier].prach_detections += 1
        self._bus.publish(ev)
        return ev

    def stop(self) -> None:
        self._stop.set()

    # ---- simulation loop ----------------------------------------------------------------
    def run(self) -> None:
        last = time.monotonic()
        while not self._stop.wait(self._tick_s):
            now = time.monotonic()
            dt_wall = now - last
            last = now
            self._tick(dt_wall)

    def _tick(self, dt_wall: float) -> None:
        hour = self.clock.hour_of_day()
        load = profile_load(self._prb_profile, hour) if self._prb_profile else diurnal_load(hour)
        dt_sim_h = dt_wall * self.clock.speedup / 3600.0
        prach_events: list[PrachDetected] = []
        for name in list(self._metrics.keys()):
            active = self._ds.active(name)
            state = self._ds.state(name)
            m = CarrierMetrics(load=load)
            if active == Active.ACTIVE:
                m.prb_util_dl_pct = min(100.0, max(0.0, load * PRB_UTIL_AT_FULL_LOAD_PCT + self._rng.gauss(0, self._prb_sigma)))
                m.prb_util_ul_pct = min(100.0, max(0.0, m.prb_util_dl_pct * UL_TO_DL_PRB_RATIO + self._rng.gauss(0, 1.5)))
                m.rrc_conn = max(0, int(round(load * RRC_CONN_AT_FULL_LOAD + self._rng.gauss(0, 8.0))))
            m.tx_power_dbm = TX_POWER_ACTIVE_DBM if (active == Active.ACTIVE and state == CarrierState.READY) else TX_POWER_OFF_DBM
            m.power_w = power_w(active, m.prb_util_dl_pct)
            n_prach = 0
            if self._background_prach and active != Active.INACTIVE:
                n_prach = _poisson(self._rng, (PRACH_RATE_MIN_HZ + PRACH_RATE_PEAK_HZ * load) * dt_wall)
            with self._lock:
                self._metrics[name] = m
                a = self._acc[name]
                a.energy_wh += m.power_w * dt_sim_h
                a.power_w_sum += m.power_w
                a.prb_dl_sum += m.prb_util_dl_pct
                a.prb_ul_sum += m.prb_util_ul_pct
                a.rrc_sum += m.rrc_conn
                a.samples += 1
                a.prach_detections += n_prach
            for _ in range(n_prach):
                energy = PRACH_ENERGY_FLOOR_DB + PRACH_ENERGY_PER_LOAD_DB * load + self._rng.gauss(0, PRACH_ENERGY_SIGMA_DB)
                prach_events.append(PrachDetected(carrier=name, energy_db=round(energy, 2), t_ns=time.perf_counter_ns(), wall_us=time.time_ns() // 1000))
        for ev in prach_events:
            self._bus.publish(ev)
