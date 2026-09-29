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
"""es-rApp data layer: synthetic generator and the vendor-neutral KPI report CSV adapter."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ocp_auru.common.events import CANONICAL_KPI_COLUMNS
from ocp_auru.rapp.data import canonical, kpi_report_csv, load, synthetic

MAP = {
    "ts": "period_start",
    "prb_util_dl_pct": "dl_prb_utilization_pct",
    "prb_util_ul_pct": "ul_prb_utilization_pct",
    "thr_dl_mbps": "dl_throughput_mbps",
    "thr_ul_mbps": "ul_throughput_mbps",
    "vol_dl_mib": "dl_volume_mib",
    "active_users": "active_users",
}


def _write_report(path, n=8, blank_at=(), start="2024-03-04 00:00:00", drop=()):
    ts = pd.date_range(start, periods=n, freq="30min")
    cols = {
        MAP["ts"]: [t.strftime("%Y-%m-%d %H:%M:%S") for t in ts],
        "SOME_NULL_COUNTER": ["7"] * n,
        MAP["prb_util_dl_pct"]: [f"{5 + i}" for i in range(n)],
        MAP["prb_util_ul_pct"]: [f"{2 + i}" for i in range(n)],
        MAP["thr_dl_mbps"]: [f"{40 + i}" for i in range(n)],
        MAP["thr_ul_mbps"]: [f"{10 + i}" for i in range(n)],
        MAP["vol_dl_mib"]: [f"{100 + i}" for i in range(n)],
        MAP["active_users"]: [f"{i}" for i in range(n)],
        "UNRELATED": ["x"] * n,
    }
    for c in drop:
        cols.pop(c)
    df = pd.DataFrame(cols)
    for (col, i) in blank_at:
        df.loc[i, MAP[col]] = ""
    df.to_csv(path, index=False)


# ---- synthetic ------------------------------------------------------------------------------
def test_synthetic_shape_ranges_and_order():
    df = synthetic.generate(n_cells=3, days=14, seed=1)
    assert list(df.columns) == CANONICAL_KPI_COLUMNS
    assert len(df) == 3 * 14 * 48
    assert sorted(df["cell_id"].unique()) == ["cell-0", "cell-1", "cell-2"]
    assert df.equals(df.sort_values(["cell_id", "ts"], kind="mergesort").reset_index(drop=True))
    assert df["prb_util_dl_pct"].between(1, 60).all()
    assert df["thr_dl_mbps"].between(20, 200).all()
    assert df["active_users"].between(0, 40).all()
    assert (df["vol_dl_mib"] >= 0).all()
    assert canonical.granularity_report(df)["share_30min"] == 1.0


def test_synthetic_is_diurnal_weekend_damped_and_deterministic():
    df = synthetic.generate(n_cells=2, days=14, seed=7)
    hour = df["ts"].dt.hour
    trough = df.loc[hour.between(2, 4), "prb_util_dl_pct"].mean()
    peak = df.loc[hour.between(19, 21), "prb_util_dl_pct"].mean()
    midday = df.loc[hour.between(11, 15), "prb_util_dl_pct"].mean()
    assert trough < midday < peak
    assert peak > 3 * trough
    weekend = df["ts"].dt.dayofweek >= 5
    assert df.loc[weekend, "prb_util_dl_pct"].mean() < df.loc[~weekend, "prb_util_dl_pct"].mean()
    assert synthetic.generate(n_cells=2, days=2, seed=7).equals(synthetic.generate(n_cells=2, days=2, seed=7))
    assert not synthetic.generate(n_cells=2, days=2, seed=8).equals(synthetic.generate(n_cells=2, days=2, seed=7))


def test_canonical_validation_errors():
    with pytest.raises(canonical.CanonicalError, match="missing canonical columns"):
        canonical.to_canonical(pd.DataFrame({"ts": ["2024-01-01"], "cell_id": ["cell-0"]}))
    with pytest.raises(canonical.CanonicalError, match="not enough"):
        canonical.validate(synthetic.generate(n_cells=1, days=1, seed=0).head(10))


# ---- kpi_report_csv adapter -------------------------------------------------------------------
def test_csv_adapter_maps_columns_and_labels_cells_by_file_order(tmp_path):
    # Filenames deliberately look like they carry identifiers: the label must NOT come from them.
    _write_report(tmp_path / "report-b-99887.csv", n=6, blank_at=[("thr_dl_mbps", 2), ("prb_util_dl_pct", 3)])
    _write_report(tmp_path / "report-a-11223.csv", n=6, start="2024-03-05 00:00:00")
    df = kpi_report_csv.load(tmp_path)
    assert list(df.columns) == CANONICAL_KPI_COLUMNS
    assert sorted(df["cell_id"].unique()) == ["cell-0", "cell-1"]
    assert "NULL" not in " ".join(df.columns)
    # sorted file order: report-a -> cell-0, report-b -> cell-1
    a = df[df["cell_id"] == "cell-0"]
    b = df[df["cell_id"] == "cell-1"]
    assert a["ts"].min() == pd.Timestamp("2024-03-05 00:00:00")
    assert b["ts"].min() == pd.Timestamp("2024-03-04 00:00:00")
    assert pd.api.types.is_datetime64_any_dtype(df["ts"])
    assert b["thr_dl_mbps"].isna().sum() == 1 and b["prb_util_dl_pct"].isna().sum() == 1
    assert a["prb_util_dl_pct"].tolist() == [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert a["active_users"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert canonical.granularity_report(df)["share_30min"] == 1.0


def test_csv_adapter_error_lists_expected_columns(tmp_path):
    _write_report(tmp_path / "x.csv", n=4, drop=[MAP["prb_util_dl_pct"]])
    with pytest.raises(kpi_report_csv.MappingError) as ei:
        kpi_report_csv.load(tmp_path)
    msg = str(ei.value)
    assert "prb_util_dl_pct" in msg and "period_start" in msg and "Expected source headers" in msg
    assert "x.csv" not in msg  # files are named by index only


def test_csv_adapter_optional_column_becomes_nan(tmp_path):
    _write_report(tmp_path / "x.csv", n=4, drop=[MAP["vol_dl_mib"]])
    df = kpi_report_csv.load(tmp_path)
    assert df["vol_dl_mib"].isna().all()
    assert df["prb_util_dl_pct"].notna().all()


def test_csv_adapter_empty_dir_error(tmp_path):
    with pytest.raises(kpi_report_csv.MappingError, match="no \\*.csv files"):
        kpi_report_csv.load(tmp_path)


def test_loader_factory(tmp_path, monkeypatch):
    monkeypatch.setenv("RAPP_SYNTHETIC_DAYS", "2")
    monkeypatch.setenv("RAPP_SYNTHETIC_CELLS", "1")
    from ocp_auru.common.config import Settings

    cfg = Settings()
    df = load("synthetic", cfg=cfg)
    assert len(df) == 96
    _write_report(tmp_path / "r.csv", n=4)
    assert len(load("kpi_report_csv", tmp_path, cfg=cfg)) == 4
    with pytest.raises(ValueError, match="unknown RAPP_DATA kind"):
        load("bogus", cfg=cfg)
