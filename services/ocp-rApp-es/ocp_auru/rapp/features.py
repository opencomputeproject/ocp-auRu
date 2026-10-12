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
"""Feature engineering for the low-load forecaster (HLD §3.7).

One feature row per 30-minute window: time-of-day (sin/cos), day-of-week + weekend flag,
the current PRB/throughput/user counters, two lags and a rolling-3 mean of DL PRB
utilisation. The same builder serves training (canonical table) and inference (the short
PM history the rApp keeps per carrier), so the two paths cannot drift apart.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ocp_auru.common.events import CANONICAL_KPI_COLUMNS, CarrierPm

FEATURE_COLUMNS = [
    "hour_sin", "hour_cos", "dow", "is_weekend",
    "prb_util_dl_pct", "prb_util_ul_pct", "thr_dl_mbps", "active_users",
    "prb_dl_lag1", "prb_dl_lag2", "prb_dl_roll3",
]
_RAW = ["prb_util_dl_pct", "prb_util_ul_pct", "thr_dl_mbps", "active_users"]


def estimate_thr_dl_mbps(prb_util_dl_pct: float) -> float:
    """Proxy DL throughput when a PM report has no throughput counter (the mock RU's
    `CarrierPm` does not). Linear in PRB utilisation with the same slope the synthetic
    generator uses (~3 Mbps per PRB-%), clipped to its 20–200 Mbps range."""
    return float(np.clip(20.0 + 3.0 * max(float(prb_util_dl_pct) - 1.0, 0.0), 20.0, 200.0))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features for every row of a canonical table (sorted by cell_id, ts)."""
    ts = pd.to_datetime(df["ts"])
    hour = (ts.dt.hour + ts.dt.minute / 60.0).astype(float)
    out = pd.DataFrame(index=df.index)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow"] = ts.dt.dayofweek.astype(float)
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(float)
    for c in _RAW:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    g = out.groupby(df["cell_id"].astype(str), sort=False)["prb_util_dl_pct"]
    out["prb_dl_lag1"] = g.shift(1).fillna(out["prb_util_dl_pct"])
    out["prb_dl_lag2"] = g.shift(2).fillna(out["prb_dl_lag1"])
    out["prb_dl_roll3"] = g.transform(lambda s: s.rolling(3, min_periods=1).mean())
    return out.loc[:, FEATURE_COLUMNS]


def build_labels(df: pd.DataFrame, low_threshold: float) -> pd.Series:
    """`1.0` if the *next* window's DL PRB utilisation is below `low_threshold`, `0.0`
    otherwise, NaN for the last window of each cell (no next window)."""
    nxt = df.groupby(df["cell_id"].astype(str), sort=False)["prb_util_dl_pct"].shift(-1)
    return (nxt < low_threshold).astype(float).where(nxt.notna())


def pm_to_kpi_row(pm: CarrierPm | Mapping[str, Any], sim_time: str) -> dict:
    """Map one carrier's VES measurement object to a canonical KPI row (ts = simulated clock)."""
    d = pm.model_dump() if isinstance(pm, CarrierPm) else dict(pm)
    prb_dl = float(d.get("prb_util_dl_pct", np.nan))
    thr = d.get("thr_dl_mbps")
    return {
        "ts": sim_time,
        "cell_id": str(d.get("name", d.get("cell_id", "carrier"))),
        "prb_util_dl_pct": prb_dl,
        "prb_util_ul_pct": float(d.get("prb_util_ul_pct", np.nan)),
        "thr_dl_mbps": float(thr) if thr is not None else estimate_thr_dl_mbps(prb_dl),
        "thr_ul_mbps": float(d.get("thr_ul_mbps", np.nan)),
        "vol_dl_mib": float(d.get("vol_dl_mib", np.nan)),
        "active_users": float(d.get("active_users", d.get("rrc_conn", np.nan))),
    }


def features_from_window(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """One feature row for the latest of `rows` (oldest first), lags from the earlier ones."""
    if not rows:
        raise ValueError("features_from_window needs at least one PM row")
    df = pd.DataFrame([dict(r) for r in rows])
    for c in CANONICAL_KPI_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df["cell_id"] = "window"
    missing_thr = df["thr_dl_mbps"].isna()
    if missing_thr.any():
        df.loc[missing_thr, "thr_dl_mbps"] = df.loc[missing_thr, "prb_util_dl_pct"].map(estimate_thr_dl_mbps)
    feats = build_features(df.reset_index(drop=True))
    return feats.iloc[[-1]].reset_index(drop=True)
