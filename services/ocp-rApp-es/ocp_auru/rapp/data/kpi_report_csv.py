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
"""Vendor-neutral adapter for 30-minute KPI report CSV exports (HLD ADR-6).

Reads every `*.csv` in a directory (sorted by name) and maps columns to the canonical
table via `kpi_report_map.yaml`. Publication rules: the cell label is `cell-<i>` by file
order and is NEVER derived from the filename or the content (filenames may carry site
identifiers); errors and logs name files by index only.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from ocp_auru.common.events import CANONICAL_KPI_COLUMNS

from .canonical import to_canonical

log = logging.getLogger(__name__)

DEFAULT_MAP_PATH = Path(__file__).with_name("kpi_report_map.yaml")


class MappingError(ValueError):
    """Raised when the CSV headers do not match the column map."""


def load_map(map_path: Optional[Path] = None) -> dict:
    with open(map_path or DEFAULT_MAP_PATH, encoding="utf-8") as fh:
        m = yaml.safe_load(fh) or {}
    if not isinstance(m.get("columns"), dict) or not m["columns"]:
        raise MappingError(f"column map {map_path or DEFAULT_MAP_PATH} has no `columns:` mapping")
    m.setdefault("ts_format", "%Y-%m-%d %H:%M:%S")
    m.setdefault("drop_columns_containing", ["NULL"])
    m.setdefault("required", ["ts", "prb_util_dl_pct"])
    unknown = sorted(set(m["columns"].values()) - set(CANONICAL_KPI_COLUMNS))
    if unknown:
        raise MappingError(f"column map targets unknown canonical columns {unknown}; allowed: {CANONICAL_KPI_COLUMNS}")
    return m


def _resolve_map_path(data_dir: Path, map_path: Optional[Path]) -> Path:
    if map_path is not None:
        return Path(map_path)
    local = data_dir / "kpi_report_map.yaml"
    return local if local.exists() else DEFAULT_MAP_PATH


def read_one(path: Path, cell_id: str, cmap: dict, index: int = 0) -> pd.DataFrame:
    """One CSV -> canonical-shaped frame labelled `cell_id`. Blank cells become NaN."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, skipinitialspace=True)
    raw.columns = [str(c).strip() for c in raw.columns]
    tokens = [t for t in cmap["drop_columns_containing"] if t]
    keep = [c for c in raw.columns if not any(t in c for t in tokens)]
    raw = raw.loc[:, keep].replace("", np.nan)

    colmap = {str(k).strip(): v for k, v in cmap["columns"].items()}
    present = {src: dst for src, dst in colmap.items() if src in raw.columns}
    missing_required = [dst for dst in cmap["required"] if dst not in present.values()]
    if missing_required:
        expected = [f"{src!r} -> {dst}" for src, dst in colmap.items()]
        raise MappingError(
            f"CSV #{index}: required canonical columns {missing_required} not found. "
            f"Expected source headers: {expected}. Found headers: {list(raw.columns)}"
        )
    df = raw.loc[:, list(present)].rename(columns=present)
    for dst in CANONICAL_KPI_COLUMNS:
        if dst not in df.columns and dst != "cell_id":
            log.warning("CSV #%d: no source column for %s; filled with NaN", index, dst)
            df[dst] = np.nan
    df["ts"] = pd.to_datetime(df["ts"], format=cmap["ts_format"], errors="coerce")
    df["cell_id"] = cell_id
    return df.loc[:, CANONICAL_KPI_COLUMNS]


def load(data_dir: Path, map_path: Optional[Path] = None) -> pd.DataFrame:
    """All `*.csv` in `data_dir` -> one canonical table; `cell-<i>` by sorted file order."""
    data_dir = Path(data_dir)
    files = sorted(p for p in data_dir.glob("*.csv") if p.is_file())
    cmap_path = _resolve_map_path(data_dir, map_path)
    cmap = load_map(cmap_path)
    if not files:
        expected = [f"{src!r} -> {dst}" for src, dst in cmap["columns"].items()]
        raise MappingError(f"no *.csv files in {data_dir}; each file is one cell with headers {expected}")
    frames = []
    for i, path in enumerate(files):
        log.debug("reading CSV #%d (%d bytes)", i, path.stat().st_size)
        frames.append(read_one(path, f"cell-{i}", cmap, index=i))
    df = to_canonical(pd.concat(frames, ignore_index=True))
    log.info("kpi_report_csv: %d files -> %d rows, %d cells", len(files), len(df), df["cell_id"].nunique())
    return df
