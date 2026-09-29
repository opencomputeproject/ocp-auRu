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
"""Canonical per-cell 30-minute KPI table (HLD ADR-6).

This is the O1 PM stream in tabular form — one row per (cell, 30-minute window) with the
columns of `events.CANONICAL_KPI_COLUMNS`. Every loader returns it; the feature builder
and the model consume nothing else. Cell ids are opaque labels (`cell-0`, ...), never
site identifiers.
"""
from __future__ import annotations

import logging

import pandas as pd

from ocp_auru.common.events import CANONICAL_KPI_COLUMNS

log = logging.getLogger(__name__)

GRANULARITY = pd.Timedelta(minutes=30)
KEY_COLUMNS = ["cell_id", "ts"]  # sort order of the canonical table
NUMERIC_COLUMNS = [c for c in CANONICAL_KPI_COLUMNS if c not in KEY_COLUMNS]


class CanonicalError(ValueError):
    """Raised when a frame cannot be brought to the canonical schema."""


def empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64" if c in NUMERIC_COLUMNS else "object") for c in CANONICAL_KPI_COLUMNS})


def to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, drop unparsable timestamps, sort by (cell_id, ts), de-duplicate keys."""
    missing = [c for c in CANONICAL_KPI_COLUMNS if c not in df.columns]
    if missing:
        raise CanonicalError(f"missing canonical columns {missing}; expected {CANONICAL_KPI_COLUMNS}")
    out = df.loc[:, CANONICAL_KPI_COLUMNS].copy()
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    bad = out["ts"].isna()
    if bad.any():
        log.warning("dropping %d rows with unparsable ts", int(bad.sum()))
        out = out.loc[~bad]
    out["cell_id"] = out["cell_id"].astype(str)
    for c in NUMERIC_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out = (
        out.sort_values(KEY_COLUMNS, kind="mergesort")
        .drop_duplicates(KEY_COLUMNS, keep="last")
        .reset_index(drop=True)
    )
    return out


def granularity_report(df: pd.DataFrame) -> dict:
    """Share of consecutive per-cell gaps that equal 30 minutes (1.0 = perfectly regular)."""
    if df.empty:
        return {"n_rows": 0, "n_cells": 0, "share_30min": 0.0}
    gaps = df.groupby("cell_id", sort=False)["ts"].diff().dropna()
    share = float((gaps == GRANULARITY).mean()) if len(gaps) else 1.0
    return {"n_rows": int(len(df)), "n_cells": int(df["cell_id"].nunique()), "share_30min": round(share, 4)}


def validate(df: pd.DataFrame, *, min_rows: int = 48) -> dict:
    """Raise `CanonicalError` on an unusable table; return the granularity report otherwise."""
    if list(df.columns) != CANONICAL_KPI_COLUMNS:
        raise CanonicalError(f"columns {list(df.columns)} != {CANONICAL_KPI_COLUMNS}; call to_canonical() first")
    if len(df) < min_rows:
        raise CanonicalError(f"{len(df)} rows < {min_rows}: not enough 30-minute windows to train on")
    if df["prb_util_dl_pct"].isna().all():
        raise CanonicalError("prb_util_dl_pct is entirely NaN; the label column is unusable")
    rep = granularity_report(df)
    if rep["share_30min"] < 0.9:
        log.warning("only %.0f%% of windows are 30-minute spaced; check the report granularity", 100 * rep["share_30min"])
    return rep
