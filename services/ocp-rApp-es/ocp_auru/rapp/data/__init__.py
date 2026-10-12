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
"""Data loaders for the es-rApp (HLD ADR-6).

`load(kind, data_dir)` is the single entry point: `synthetic` (generic diurnal profile,
shipped) or `kpi_report_csv` (vendor-neutral 30-minute KPI report adapter reading the
gitignored private directory). Both return the canonical KPI table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ocp_auru.common.config import Settings, settings

from . import canonical, kpi_report_csv, synthetic

KINDS = ("synthetic", "kpi_report_csv")


def load(kind: str, data_dir: Optional[Path] = None, cfg: Settings = settings) -> pd.DataFrame:
    """Return the canonical KPI table for `kind`; raises with the expected schema on failure."""
    if kind == "synthetic":
        return synthetic.generate(n_cells=cfg.rapp_synthetic_cells, days=cfg.rapp_synthetic_days, seed=cfg.rapp_seed)
    if kind == "kpi_report_csv":
        return kpi_report_csv.load(Path(data_dir) if data_dir is not None else cfg.rapp_data_dir)
    raise ValueError(f"unknown RAPP_DATA kind {kind!r}; expected one of {KINDS}")


__all__ = ["load", "KINDS", "canonical", "synthetic", "kpi_report_csv"]
