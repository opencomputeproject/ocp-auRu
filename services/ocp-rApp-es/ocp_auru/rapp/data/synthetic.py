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
"""Synthetic canonical KPI data with a *generic* diurnal profile (HLD ADR-6, shipped default).

No input data is encoded here: the profile is the textbook shape of mobile traffic —
night trough (02–05 h), morning ramp, midday plateau, evening peak (~19–21 h), damped and
later-rising weekends — with per-cell scale and multiplicative noise. Ranges are chosen to
look like a mid-loaded NR cell (PRB 1–60 %, 20–200 Mbps, 0–40 users); volume is derived.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from ocp_auru.common.events import CANONICAL_KPI_COLUMNS

from .canonical import GRANULARITY, to_canonical

# Hour-of-day anchors for np.interp (0 and 24 both present -> periodic).
_HOURS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24], dtype=float)
_WEEKDAY = np.array([.30, .20, .10, .08, .08, .10, .18, .32, .48, .58, .62, .62, .62, .64, .70, .82, .95, 1.0, .95, .75, .50, .30])
_WEEKEND = np.array([.35, .25, .12, .09, .09, .10, .14, .20, .30, .42, .52, .60, .62, .64, .68, .78, .90, .95, .90, .75, .55, .35]) * 0.85

PRB_DL_RANGE = (1.0, 60.0)
PRB_UL_RANGE = (1.0, 40.0)
THR_DL_RANGE = (20.0, 200.0)
THR_UL_RANGE = (2.0, 60.0)
USERS_RANGE = (0, 40)


def diurnal_load(hour: np.ndarray, weekend: np.ndarray) -> np.ndarray:
    """Normalised load 0..1 for fractional hour-of-day and a weekend flag."""
    wd = np.interp(hour, _HOURS, _WEEKDAY)
    we = np.interp(hour, _HOURS, _WEEKEND)
    return np.where(weekend, we, wd)


def generate(n_cells: int = 3, days: int = 14, seed: int = 42, start: Optional[datetime] = None) -> pd.DataFrame:
    """Canonical table with `n_cells * days * 48` rows; deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    start = start or datetime(2024, 1, 1)  # a Monday, naive local time
    n = days * 48
    ts = pd.date_range(start, periods=n, freq=GRANULARITY)
    hour = np.asarray(ts.hour + ts.minute / 60.0, dtype=float)
    weekend = np.asarray(ts.dayofweek >= 5)
    base = diurnal_load(hour, weekend)

    frames = []
    for i in range(n_cells):
        scale = rng.uniform(0.75, 1.15)
        load = np.clip(base * scale * rng.lognormal(0.0, 0.12, n), 0.0, 1.2)
        prb_dl = np.clip(PRB_DL_RANGE[0] + (PRB_DL_RANGE[1] - PRB_DL_RANGE[0]) * load, *PRB_DL_RANGE)
        prb_ul = np.clip(PRB_UL_RANGE[0] + 0.45 * (prb_dl - PRB_DL_RANGE[0]) * rng.lognormal(0.0, 0.10, n), *PRB_UL_RANGE)
        thr_dl = np.clip(THR_DL_RANGE[0] + (THR_DL_RANGE[1] - THR_DL_RANGE[0]) * load * rng.lognormal(0.0, 0.10, n), *THR_DL_RANGE)
        thr_ul = np.clip(0.25 * thr_dl, *THR_UL_RANGE)
        users = np.clip(rng.poisson(USERS_RANGE[1] * np.clip(load, 0, 1)), *USERS_RANGE)
        # Volume derived: busy-time throughput integrated over the 30-min window, MB -> MiB.
        vol_dl = thr_dl * (prb_dl / 100.0) * GRANULARITY.total_seconds() / 8.0 / 1.048576
        frames.append(pd.DataFrame({
            "ts": ts,
            "cell_id": f"cell-{i}",
            "prb_util_dl_pct": np.round(prb_dl, 2),
            "prb_util_ul_pct": np.round(prb_ul, 2),
            "thr_dl_mbps": np.round(thr_dl, 2),
            "thr_ul_mbps": np.round(thr_ul, 2),
            "vol_dl_mib": np.round(vol_dl, 1),
            "active_users": users.astype(float),
        }, columns=CANONICAL_KPI_COLUMNS))
    return to_canonical(pd.concat(frames, ignore_index=True))
