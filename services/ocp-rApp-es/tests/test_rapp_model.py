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
"""es-rApp model: training on synthetic data, holdout quality, persistence round-trip."""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from ocp_auru.common.config import Settings
from ocp_auru.rapp import model as model_mod
from ocp_auru.rapp.data import synthetic
from ocp_auru.rapp.features import FEATURE_COLUMNS, build_features, build_labels, features_from_window, pm_to_kpi_row


# joblib pickling under NumPy 2.5 emits a shape-assignment DeprecationWarning; not ours.
pytestmark = pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")


@pytest.fixture(scope="module")
def trained():
    df = synthetic.generate(n_cells=3, days=14, seed=42)
    t0 = time.perf_counter()
    m = model_mod.train(df, seed=42)
    m.metrics["_wall"] = time.perf_counter() - t0
    return m


def test_features_and_labels_shape():
    df = synthetic.generate(n_cells=2, days=2, seed=3)
    X = build_features(df)
    assert list(X.columns) == FEATURE_COLUMNS and len(X) == len(df)
    assert not X.isna().any().any()
    # lags are per cell: first row of each cell falls back to its own value
    first_rows = df.groupby("cell_id").head(1).index
    assert (X.loc[first_rows, "prb_dl_lag1"] == X.loc[first_rows, "prb_util_dl_pct"]).all()
    y = build_labels(df, df["prb_util_dl_pct"].quantile(0.25))
    assert y.isna().sum() == 2  # last window per cell has no "next"
    assert set(y.dropna().unique()) <= {0.0, 1.0}


def test_train_auc_and_metrics(trained):
    m = trained
    assert m.metrics["auc"] is not None and m.metrics["auc"] > 0.7, m.metrics
    assert 0 < m.metrics["accuracy"] <= 1
    assert m.metrics["n_train"] > m.metrics["n_test"] > 0
    assert m.metrics["low_threshold"] > 0
    assert m.metrics["_wall"] < 10.0


def test_predict_night_vs_evening(trained):
    night = [{"ts": f"2024-01-10T0{h}:{mm}:00", "prb_util_dl_pct": 4.0, "prb_util_ul_pct": 2.0, "active_users": 1}
             for h, mm in ((2, "00"), (2, "30"), (3, "00"))]
    evening = [{"ts": f"2024-01-10T{h}:{mm}:00", "prb_util_dl_pct": 55.0, "prb_util_ul_pct": 20.0, "active_users": 35}
               for h, mm in ((19, "00"), (19, "30"), (20, "00"))]
    p_night = trained.predict_low_load_probability(night)
    p_evening = trained.predict_low_load_probability(evening)
    assert 0.0 <= p_evening < 0.5 < p_night <= 1.0


def test_features_from_window_uses_last_row_and_proxy_throughput():
    rows = [pm_to_kpi_row({"name": "txcarrier0", "prb_util_dl_pct": 3.0, "prb_util_ul_pct": 1.0, "rrc_conn": 2}, "2024-01-10T03:00:00")]
    X = features_from_window(rows)
    assert len(X) == 1 and X.loc[0, "thr_dl_mbps"] == pytest.approx(26.0)
    assert X.loc[0, "active_users"] == 2.0
    with pytest.raises(ValueError):
        features_from_window([])


def test_persistence_round_trip(tmp_path, trained, monkeypatch):
    path = tmp_path / "es-rapp-model.joblib"
    trained.save(path)
    assert path.exists() and path.with_suffix(".metrics.json").exists()
    loaded = model_mod.LowLoadModel.load(path)
    window = [{"ts": "2024-01-10T03:00:00", "prb_util_dl_pct": 4.0, "prb_util_ul_pct": 2.0, "active_users": 1}]
    assert loaded.predict_low_load_probability(window) == pytest.approx(trained.predict_low_load_probability(window))
    assert loaded.low_threshold == trained.low_threshold
    assert json.loads(path.with_suffix(".metrics.json").read_text())["auc"] == trained.metrics["auc"]

    # load_or_train: reuse the persisted model; retrain when RAPP_RETRAIN=1 or data_kind differs
    monkeypatch.setenv("VAR_DIR", str(tmp_path))
    monkeypatch.setenv("RAPP_SYNTHETIC_DAYS", "3")
    calls = []

    def loader():
        calls.append(1)
        return synthetic.generate(n_cells=1, days=3, seed=1)

    m1 = model_mod.load_or_train(loader, Settings())
    assert calls == [] and m1.low_threshold == trained.low_threshold
    monkeypatch.setenv("RAPP_RETRAIN", "1")
    m2 = model_mod.load_or_train(loader, Settings())
    assert calls == [1] and m2.metrics["n_cells"] == 1
    monkeypatch.setenv("RAPP_RETRAIN", "0")
    monkeypatch.setenv("RAPP_DATA", "kpi_report_csv")
    m3 = model_mod.load_or_train(lambda: synthetic.generate(n_cells=2, days=3, seed=2), Settings())
    assert m3.metrics["data_kind"] == "kpi_report_csv" and m3.metrics["n_cells"] == 2


def test_hourly_profile_and_asleep_forecast(trained):
    prof = trained.hourly_profile
    assert prof is not None and len(prof) == 96 and not prof.isna().any().any()
    assert prof.loc[16, "prb_util_dl_pct"] > prof.loc[6, "prb_util_dl_pct"]  # 08:00 busier than 03:00 (weekday)
    win = trained.expected_window("2024-01-10T08:00:00")
    assert [r["ts"] for r in win] == ["2024-01-10T07:00:00", "2024-01-10T07:30:00", "2024-01-10T08:00:00"]
    assert trained.predict_low_load_probability_asleep("2024-01-10T03:00:00") > 0.7
    assert trained.predict_low_load_probability_asleep("2024-01-10T08:00:00") < 0.3
    assert trained.predict_low_load_probability_asleep("2026-01-01T02:17:15Z") > 0.7  # RU clock format
    # by contrast the sleeping carrier's own PM (PRB 0) reads as "certainly low" - the bug the profile fixes
    zero = [{"ts": "2024-01-10T08:00:00", "prb_util_dl_pct": 0.0, "prb_util_ul_pct": 0.0, "active_users": 0}]
    assert trained.predict_low_load_probability(zero) > 0.7


def test_model_without_profile_is_retrained(tmp_path, trained, monkeypatch):
    import joblib
    path = tmp_path / "es-rapp-model.joblib"
    trained.save(path)
    blob = joblib.load(path)
    blob.pop("hourly_profile")
    joblib.dump(blob, path)  # an older model file
    assert model_mod.LowLoadModel.load(path).hourly_profile is None
    monkeypatch.setenv("VAR_DIR", str(tmp_path))
    calls = []

    def loader():
        calls.append(1)
        return synthetic.generate(n_cells=1, days=3, seed=1)

    m = model_mod.load_or_train(loader, Settings())
    assert calls == [1] and m.hourly_profile is not None


def test_configured_low_threshold():
    df = synthetic.generate(n_cells=1, days=4, seed=5)
    m = model_mod.train(df, low_threshold=12.5)
    assert m.low_threshold == 12.5 and m.metrics["low_threshold_source"] == "configured"


def test_profile_level_ratio_and_scaled_asleep_forecast(trained):
    """Root cause of the policy chatter: the profile is estate-pooled. `profile_level_ratio`
    measures a carrier's own level against the profile at the same bins from its AWAKE history; `asleep_forecast`
    scales the profile by it with weight RAPP_PROFILE_SCALE (0 = off: the estate profile, the shipped default)."""
    from ocp_auru.rapp.model import PROFILE_COLUMNS, PROFILE_SCALE_MIN_ROWS, PROFILE_SCALE_RATIO_RANGE
    df = synthetic.generate(n_cells=3, days=14, seed=42)
    own = df.loc[df["cell_id"] == "cell-0"].tail(64)
    rows = own.assign(ts=own["ts"].map(lambda t: t.isoformat())).to_dict("records")
    ratio, n = trained.profile_level_ratio(rows)
    assert n == 64 and set(ratio) == set(PROFILE_COLUMNS) and all(0.7 < r < 1.5 for r in ratio.values())
    # a carrier at 0.4x the estate: ratios ~0.4 for every column, clipped to the allowed range
    quiet = [dict(r, **{c: r[c] * 0.4 for c in PROFILE_COLUMNS}) for r in rows]
    rq, _ = trained.profile_level_ratio(quiet)
    assert all(0.3 < r < 0.5 for r in rq.values())
    tiny = [dict(r, **{c: r[c] * 0.01 for c in PROFILE_COLUMNS}) for r in rows]
    assert trained.profile_level_ratio(tiny)[0]["prb_util_dl_pct"] == PROFILE_SCALE_RATIO_RANGE[0]
    # asleep rows (the live rApp tags them) are skipped; too few awake rows -> neutral
    tagged = [dict(r, prb_util_dl_pct=0.0, active="SLEEP") for r in rows[:40]] + [dict(r, active="ACTIVE") for r in rows[40:]]
    rt, nt = trained.profile_level_ratio(tagged)
    assert nt == 24 and 0.7 < rt["prb_util_dl_pct"] < 1.5
    neutral = {c: 1.0 for c in PROFILE_COLUMNS}
    assert trained.profile_level_ratio(rows[:PROFILE_SCALE_MIN_ROWS - 1]) == (neutral, 0)
    assert trained.profile_level_ratio([]) == (neutral, 0)
    assert trained.profile_level_ratio([dict(r, active="SLEEP") for r in rows]) == (neutral, 0)
    # weight 0 = the estate profile (identical to the plain asleep forecast); weight 1 = the profile at the own level
    t = "2024-01-13T07:30:00"
    p0, i0 = trained.asleep_forecast(t, quiet, 0.0)
    assert p0 == trained.predict_low_load_probability_asleep(t) and i0["profile_scale"] == 1.0 and i0["profile_scale_rows"] == 0
    p1, i1 = trained.asleep_forecast(t, quiet, 1.0)
    assert i1["profile_scale"] == pytest.approx(rq["prb_util_dl_pct"]) and i1["profile_scale_rows"] == 64 and i1["profile_scale_weight"] == 1.0
    assert p1 == trained.predict_low_load_probability_asleep(t, quiet, 1.0)
    # the scaled profile IS the expected window times the factors
    win0 = trained.expected_window(t)
    win1 = trained.expected_window(t, scale={c: 1.0 + 1.0 * (rq[c] - 1.0) for c in PROFILE_COLUMNS})
    for a, b in zip(win0, win1):
        assert b["prb_util_dl_pct"] == pytest.approx(a["prb_util_dl_pct"] * rq["prb_util_dl_pct"])
    # at the morning ramp a quiet carrier's scaled profile is still "low" while the estate profile already says "load"
    ramp = next(tt for tt in ("2024-01-10T07:00:00", "2024-01-10T07:30:00", "2024-01-10T08:00:00")
                if trained.predict_low_load_probability_asleep(tt) < 0.5)
    assert trained.asleep_forecast(ramp, quiet, 1.0)[0] > trained.asleep_forecast(ramp, quiet, 0.0)[0]
    # half weight halves the level shift
    assert trained.asleep_forecast(t, quiet, 0.5)[1]["profile_scale"] == pytest.approx(1.0 + 0.5 * (rq["prb_util_dl_pct"] - 1.0))
