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
"""Offline energy-saving evaluation (ocp_auru.rapp.eval): split, identities, real code paths, CLI.

Synthetic data only (never the private directory); short horizons keep the module < 40 s.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ocp_auru.common.config import Settings
from ocp_auru.mock_ru.radio import P_ACTIVE_BASE_W, P_SLEEP_W
from ocp_auru.rapp import eval as ev
from ocp_auru.rapp import policy_engine
from ocp_auru.rapp.data import synthetic

pytestmark = pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")

TRAIN_DAYS, VALIDATE_DAYS, DAYS = 7, 3, 11  # one extra day must be dropped by the split
ROOT = Path(__file__).resolve().parents[3] / "demo"   # the CLI (scripts/eval_energy.py) lives with the demo


@pytest.fixture(scope="module")
def cfg() -> Settings:
    """The settings in use for the primary run: the shipped chatter-control defaults (pinned in test_rapp_policy) on the demo thresholds."""
    return replace(Settings(), rapp_sleep_threshold=0.7, rapp_wake_threshold=0.3, rapp_min_hold_s=20.0, rapp_history_len=64)


# The baseline without chatter control: governor off (no dwell, one confirmation window, no margin), the estate-pooled
# profile as-is, no post-local-wake hold, and no actuation budget / alarm (the shipped 6/4 cap binds on chatter-prone
# data and would silently move a 'legacy' row). Tests that prove the legacy behaviour pin it explicitly through this.
LEGACY = dict(rapp_profile_scale=0.0, rapp_min_sleep_dwell_min=0.0, rapp_post_local_wake_hold_min=0.0,
              rapp_wake_confirm_windows=1, rapp_wake_margin=0.0, rapp_max_actuations_per_day=0, rapp_actuation_alarm_per_day=0)


@pytest.fixture(scope="module")
def legacy_cfg(cfg) -> Settings:
    return replace(cfg, **LEGACY)


def _primary_label(c: Settings) -> str:
    """The grid label of the settings a run actually uses: the row the grid must flag `primary`."""
    return ev.grid_label(float(c.rapp_min_sleep_dwell_min), int(c.rapp_wake_confirm_windows), float(c.rapp_wake_margin),
                         float(c.rapp_profile_scale))


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return synthetic.generate(n_cells=2, days=DAYS, seed=7)


@pytest.fixture(scope="module")
def params() -> ev.EvalParams:
    return ev.EvalParams(train_days=TRAIN_DAYS, validate_days=VALIDATE_DAYS, seed=7, sweep=False)


@pytest.fixture(scope="module")
def result(params, cfg, table) -> ev.EvalResult:
    return ev.run_evaluation(params, cfg, df=table)


# ---- split ----------------------------------------------------------------------------------
def test_split_boundaries_and_no_leakage(table):
    sp = ev.split_by_time(table, TRAIN_DAYS, VALIDATE_DAYS)
    assert sp.train_end < sp.validate_start
    assert sp.train["ts"].max() < sp.validate["ts"].min()
    assert sp.train["ts"].dt.normalize().nunique() == TRAIN_DAYS
    assert sp.validate["ts"].dt.normalize().nunique() == VALIDATE_DAYS
    assert sp.validate_end < table["ts"].max()  # the 11th day is dropped
    for cell, b in sp.per_cell.items():
        assert pd.Timestamp(b["train_end"]) < pd.Timestamp(b["validate_start"])
        assert b["n_train"] == TRAIN_DAYS * 48 and b["n_validate"] == VALIDATE_DAYS * 48
        assert b["validate_windows_per_day_min"] == 48
    d = sp.as_dict()
    assert d["cells"] == ["cell-0", "cell-1"] and d["train_end"] < d["validate_start"]


def test_split_rejects_short_history(table):
    with pytest.raises(ValueError, match="distinct dates"):
        ev.split_by_time(table, 20, 5)
    short = table.loc[~((table["cell_id"] == "cell-1") & (table["ts"] < "2024-01-03"))]
    with pytest.raises(ValueError, match="cell-1: no windows"):
        ev.split_by_time(short, TRAIN_DAYS, VALIDATE_DAYS)


def test_model_and_labels_come_from_train_frame_only(result):
    thr_train = float(result.split.train["prb_util_dl_pct"].quantile(ev.LOW_QUANTILE))
    assert result.model.low_threshold == pytest.approx(thr_train)
    assert result.model.metrics["low_threshold_source"] == "data-quantile-0.25"
    clf = result.classifier
    n_cells = result.split.validate["cell_id"].nunique()
    assert clf["n"] == len(result.split.validate) - n_cells  # last validation window per cell has no label
    assert clf["auc"] is not None and clf["auc"] > 0.7
    assert 0 <= clf["precision"] <= 1 and 0 <= clf["recall"] <= 1


# ---- energy identities -----------------------------------------------------------------------
def test_energy_identities(result):
    v = result.summary["variants"]
    for variant in ("with_ruapp", "rapp_only"):
        tot = v[variant]["total"]
        assert tot["baseline_kwh"] >= tot["variant_kwh"] >= 0
        win = result.replays[variant].windows
        assert tot["savings_kwh"] == pytest.approx((win["baseline_wh"] - win["variant_wh"]).sum() / 1000.0)
        assert sum(d["variant_kwh"] for d in v[variant]["per_day"]) == pytest.approx(tot["variant_kwh"])
        assert sum(d["baseline_kwh"] for d in v[variant]["per_day"]) == pytest.approx(tot["baseline_kwh"])
        assert sum(c["variant_kwh"] for c in v[variant]["per_cell"].values()) == pytest.approx(tot["variant_kwh"])
        # per-window energy is the radio's power model x 0.5 h
        exp_base = (P_ACTIVE_BASE_W + 1.5 * win["prb_util_dl_pct"]) * 0.5
        assert np.allclose(win["baseline_wh"], exp_base)
        sleep_rows = win["state"] == "SLEEP"
        assert np.allclose(win.loc[sleep_rows, "variant_wh"], P_SLEEP_W * 0.5)
        assert np.allclose(win.loc[~sleep_rows, "variant_wh"], win.loc[~sleep_rows, "baseline_wh"])
    with_ru = v["with_ruapp"]["total"]
    assert with_ru["oracle_savings_kwh"] >= with_ru["savings_kwh"]  # oracle bounds the no-impact variant
    assert with_ru["oracle_savings_kwh"] > 0 and with_ru["savings_kwh"] > 0
    assert with_ru["sleep_decisions"] > 0


# ---- real code paths are exercised ---------------------------------------------------------------
def test_replay_calls_decide_and_asleep_forecast(monkeypatch, result, cfg, legacy_cfg):
    """Every decided window goes through the real decide() and the real asleep forecast (`LowLoadModel.asleep_forecast`,
    memoised per simulated time). Under the shipped RAPP_PROFILE_SCALE 1 the forecaster calls it directly with the
    cfg's weight (level scaling from the carrier's own history); under the pinned legacy scale 0 it goes through the
    `predict_low_load_probability_asleep` wrapper and the factor is 1.0 everywhere (the estate profile as-is)."""
    calls = {"decide": 0, "asleep": 0, "wrapper": 0}
    expect = {"cfg": cfg}
    real_decide = policy_engine.decide

    def spy_decide(status, latest_pm, p_low, now, state, **kw):
        calls["decide"] += 1
        assert status.state.value == "READY" and status.active.value in ("ACTIVE", "SLEEP")
        assert "prb_util_dl_pct" in latest_pm and kw["cfg"] is expect["cfg"]
        return real_decide(status, latest_pm, p_low, now, state, **kw)

    model_cls = type(result.model)
    real_asleep, real_wrapper = model_cls.asleep_forecast, model_cls.predict_low_load_probability_asleep
    asleep_times: list[str] = []
    weights: set[float] = set()

    def spy_asleep(self, sim_time, history_rows=None, scale_weight=0.0):
        calls["asleep"] += 1
        asleep_times.append(sim_time)
        weights.add(float(scale_weight))
        return real_asleep(self, sim_time, history_rows, scale_weight)

    def spy_wrapper(self, sim_time, history_rows=None, scale_weight=0.0):
        calls["wrapper"] += 1
        return real_wrapper(self, sim_time, history_rows, scale_weight)

    monkeypatch.setattr(policy_engine, "decide", spy_decide)
    monkeypatch.setattr(model_cls, "asleep_forecast", spy_asleep)
    monkeypatch.setattr(model_cls, "predict_low_load_probability_asleep", spy_wrapper)
    cell = "cell-0"
    tr = result.split.train.loc[result.split.train["cell_id"] == cell]
    va = result.split.validate.loc[result.split.validate["cell_id"] == cell]
    kw = dict(low_threshold=result.model.low_threshold, demand_prb=result.model.low_threshold, local_wake=True)
    win, dec = ev.replay_cell(result.model, cell, tr, va, cfg=cfg, **kw)
    assert calls["decide"] == len(va) - 1 == win["decision"].notna().sum()
    asleep_windows = win.loc[win["forecast_basis"] == "expected-hour-profile"]
    assert len(asleep_windows) > 0 and (asleep_windows["state"] == "SLEEP").all()
    assert (win.loc[win["forecast_basis"] == "pm-window", "state"] == "ACTIVE").all()
    # BOTH estimators run on every decided window (side-by-side record), memoised per
    # simulated time: one real profile call per decided window, whatever the carrier's state
    decided = win.loc[win["decision"].notna()]
    assert calls["asleep"] == decided["ts"].nunique() == len(va) - 1
    # level scaling (the shipped default): the forecaster calls asleep_forecast itself with the cfg's weight, never the wrapper
    assert cfg.rapp_profile_scale > 0 and weights == {float(cfg.rapp_profile_scale)} and calls["wrapper"] == 0
    assert decided["p_low_pm_window"].notna().all() and decided["p_low_profile"].notna().all()
    assert np.allclose(decided["forecast_divergence"], decided["p_low_pm_window"] - decided["p_low_profile"])
    assert (decided["forecast_divergent"] == (decided["forecast_divergence"].abs() > ev.FORECAST_DIVERGENCE_FLAG)).all()
    # the decision's p_low is the estimator the carrier's state calls for
    assert np.allclose(asleep_windows["p_low"], asleep_windows["p_low_profile"])
    awake = win.loc[win["forecast_basis"] == "pm-window"]
    assert np.allclose(awake["p_low"], awake["p_low_pm_window"])
    # the recorded factor is the one the forecast applied: the carrier's own level (clipped to [0.2, 5]) once enough
    # awake windows are in the history, so it departs from the estate's 1.0 on this replay
    assert (decided["profile_scale"] > 0).all() and (decided["profile_scale"] != 1.0).any()
    assert sum(dec.values()) == calls["decide"]
    # app.py:_plan convention: the asleep forecast's sim_time is the ts of the latest PM row, the same instant
    # the ACTIVE branch's latest history row carries (not the next window's stamp)
    assert sorted(asleep_times) == sorted(pd.Timestamp(t).isoformat() for t in decided["ts"])
    assert {pd.Timestamp(t) for t in asleep_times} <= set(va["ts"])
    # legacy baseline (RAPP_PROFILE_SCALE 0, pinned): the wrapper path, still one real call per decided window,
    # weight 0 and the estate profile as-is (factor 1.0 on every window)
    calls.update(decide=0, asleep=0, wrapper=0)
    asleep_times.clear()
    weights.clear()
    expect["cfg"] = legacy_cfg
    win0, dec0 = ev.replay_cell(result.model, cell, tr, va, cfg=legacy_cfg, **kw)
    decided0 = win0.loc[win0["decision"].notna()]
    assert calls["decide"] == len(va) - 1 == sum(dec0.values())
    assert calls["wrapper"] == calls["asleep"] == decided0["ts"].nunique() == len(va) - 1
    assert weights == {0.0} and (decided0["profile_scale"] == 1.0).all()
    assert sorted(asleep_times) == sorted(pd.Timestamp(t).isoformat() for t in decided0["ts"])


def test_ruapp_wake_exactly_when_sleep_meets_demand(result):
    thr = result.model.low_threshold
    win = result.replays["with_ruapp"].windows
    expect = (win["entered_state"] == "SLEEP") & ~(win["prb_util_dl_pct"] < thr)
    assert (win["ruapp_wake"] == expect).all()
    assert win["ruapp_wake"].sum() == result.summary["headline"]["ruapp_wakes"]
    assert (win.loc[win["ruapp_wake"], "state"] == "ACTIVE").all()   # woken window charged ACTIVE
    assert not win["demand_while_asleep"].any()
    # rApp-only never wakes locally: it sleeps through the demand window
    ro = result.replays["rapp_only"].windows
    assert not ro["ruapp_wake"].any()
    expect_ro = (ro["entered_state"] == "SLEEP") & ~(ro["prb_util_dl_pct"] < thr)
    assert (ro["demand_while_asleep"] == expect_ro).all()
    assert (ro["state"] == ro["entered_state"]).all()
    assert result.summary["headline"]["demand_while_asleep_rapp_only"] == int(expect_ro.sum())
    # after a demand-while-asleep window the rApp wakes it at a later cycle through the normal rule
    if expect_ro.any():
        assert (ro["decision"] == "ACTIVE").sum() > 0


def test_demand_prb_override(result, cfg):
    cell = "cell-0"
    tr = result.split.train.loc[result.split.train["cell_id"] == cell]
    va = result.split.validate.loc[result.split.validate["cell_id"] == cell]
    hi, _ = ev.replay_cell(result.model, cell, tr, va, cfg=cfg, low_threshold=result.model.low_threshold,
                           demand_prb=1000.0, local_wake=True)
    assert not hi["ruapp_wake"].any()  # nothing ever counts as demand


def test_state_machine_never_inactive_and_starts_active(result):
    for rr in result.replays.values():
        win = rr.windows
        assert set(win["state"]) <= {"ACTIVE", "SLEEP"} and set(win["entered_state"]) <= {"ACTIVE", "SLEEP"}
        assert (win.groupby("cell")["entered_state"].first() == "ACTIVE").all()
        assert set(win["decision"].dropna()) <= {"SLEEP", "ACTIVE", "NONE"}


# ---- determinism and sweep ------------------------------------------------------------------
def _comparable(summary: dict) -> dict:
    s = json.loads(json.dumps(ev._jsonable(summary)))
    for k in ("generated_at", "timing_s", "outputs"):
        s.pop(k, None)
    s["model"].pop("train_seconds", None)
    return s


def test_deterministic_same_seed(params, cfg, table):
    a = ev.run_evaluation(params, cfg, df=table)
    b = ev.run_evaluation(params, cfg, df=table)
    assert _comparable(a.summary) == _comparable(b.summary)


def test_sweep_rows(cfg, table):
    p = ev.EvalParams(train_days=TRAIN_DAYS, validate_days=VALIDATE_DAYS, seed=7, sweep=True,
                      sleep_thresholds=(0.6, 0.8), low_quantiles=(0.25, 0.5))
    r = ev.run_evaluation(p, cfg, df=table)
    rows = r.summary["sweep"]
    assert [(x["sleep_threshold"], x["low_quantile"]) for x in rows] == [(0.6, 0.25), (0.8, 0.25), (0.6, 0.5), (0.8, 0.5)]
    for x in rows:
        assert 0 <= x["savings_pct_with_ruapp"] <= 100 and x["low_threshold"] > 0
    thr25 = [x["low_threshold"] for x in rows if x["low_quantile"] == 0.25][0]
    thr50 = [x["low_threshold"] for x in rows if x["low_quantile"] == 0.5][0]
    assert thr25 == pytest.approx(r.model.low_threshold) and thr50 > thr25


# ---- CLI smoke --------------------------------------------------------------------------------
def test_cli_smoke_writes_report_and_plots(tmp_path):
    out = tmp_path / "eval"
    cmd = [sys.executable, str(ROOT / "scripts" / "eval_energy.py"), "--data-kind", "synthetic", "--days", str(DAYS),
           "--cells", "2", "--train-days", str(TRAIN_DAYS), "--validate-days", str(VALIDATE_DAYS), "--seed", "7",
           "--no-sweep", "--out", str(out)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "savings with ruApp" in proc.stdout and "validation AUC" in proc.stdout and "ILLUSTRATIVE" in proc.stdout
    for name in ("report.md", "summary.json", "timeline_cell-0.png", "timeline_cell-1.png", "energy_daily.png", "impact.png"):
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    assert not (out / "sweep.png").exists()  # --no-sweep: no placeholder plot
    assert "4 plots" in proc.stdout
    report = (out / "report.md").read_text()
    assert "ILLUSTRATIVE" in report and "per SITE" in report and "Assertion max(train ts) < min(validate ts) holds" in report
    summary = json.loads((out / "summary.json").read_text())
    assert summary["power_model"]["label"] == "ILLUSTRATIVE" and summary["split"]["train_end"] < summary["split"]["validate_start"]
    assert summary["headline"]["savings_pct_oracle"] >= summary["headline"]["savings_pct_with_ruapp"]
    assert "/Users/" not in report and "/Users/" not in json.dumps(summary)  # no checkout paths in shared outputs
    assert "BY CONSTRUCTION" in report and summary["headline"]["with_ruapp_impact_free_by_construction"] is True
    assert "sleep h per cell-day" in report and "sleep_hours_per_cell_day_with_ruapp" in summary["headline"]
    assert "do not select a threshold" in report  # sweep caveat is present even when the sweep is disabled
    assert summary["params"]["data_dir"] is None


# ---- regressions --------------------------------------------------------------------
def test_params_validate_quantiles_and_name_the_flag():
    with pytest.raises(ValueError, match="--low-quantile"):
        ev.EvalParams(low_quantile=1.5)
    with pytest.raises(ValueError, match="--sweep-low-quantiles"):
        ev.EvalParams(low_quantiles=(0.25, 0.0))
    with pytest.raises(ValueError, match="--sweep-sleep-thresholds"):
        ev.EvalParams(sleep_thresholds=(0.5, 1.2))
    with pytest.raises(ValueError, match="demand_margins"):
        ev.EvalParams(demand_margins=(1.0,))
    ev.EvalParams(low_quantile=0.5, sleep_thresholds=(0.0, 1.0))  # closed interval for thresholds


def test_offloaded_energy_identity_and_labels(result):
    for variant in ("with_ruapp", "rapp_only"):
        win = result.replays[variant].windows
        tot = result.summary["variants"][variant]["total"]
        slept = win["state"] == "SLEEP"
        exp = (1.5 * win.loc[slept, "prb_util_dl_pct"] * 0.5).sum() / 1000.0
        assert tot["offloaded_kwh"] == pytest.approx(exp)
        assert (win.loc[~slept, "offloaded_wh"] == 0).all()
        assert tot["savings_kwh_net_of_offload"] == pytest.approx(tot["savings_kwh"] - tot["offloaded_kwh"])
        exp_orc = (1.5 * win.loc[win["actually_low"], "prb_util_dl_pct"] * 0.5).sum() / 1000.0
        assert tot["oracle_offloaded_kwh"] == pytest.approx(exp_orc)
        # sleep hours: per cell-day and summed over cells
        n_cells = win["cell"].nunique()
        assert tot["sleep_hours_per_day_all_cells"] == pytest.approx(tot["sleep_hours_per_cell_day"] * n_cells)
        assert "sleep_hours_per_day" not in tot
    h = result.summary["headline"]
    assert h["offloaded_kwh_with_ruapp"] > 0 and h["savings_pct_with_ruapp_net_of_offload"] < h["savings_pct_with_ruapp"]
    lines = "\n".join(ev.headline_lines(result.summary))
    assert "sleep h per cell-day" in lines and "service impact:" in lines and "BY CONSTRUCTION" in lines
    assert "not subtracted" in lines


def test_demand_sensitivity_rows(result):
    ds = result.summary["demand_sensitivity"]
    h = result.summary["headline"]
    assert ds["default_savings_pct"] == pytest.approx(h["savings_pct_with_ruapp"])
    fr = [b["burst_fraction"] for b in ds["burst_fractions"]]
    assert fr == list(ev.BURST_FRACTIONS)
    # all savings come from slept windows, so charging a share f of them ACTIVE scales the savings by (1 - f)
    for b in ds["burst_fractions"]:
        assert b["savings_pct"] == pytest.approx((1 - b["burst_fraction"]) * h["savings_pct_with_ruapp"])
    assert [m["demand_margin"] for m in ds["demand_margins"]] == list(ev.DEMAND_MARGINS)
    for m in ds["demand_margins"]:
        assert m["demand_prb"] == pytest.approx(m["demand_margin"] * result.model.low_threshold)
        assert m["savings_pct"] <= h["savings_pct_with_ruapp"] + 1e-9  # a stricter demand definition never saves more
        assert m["ruapp_wakes"] >= h["ruapp_wakes"]
    assert "by construction" in ds["note"]


def test_missing_kpi_windows_are_not_wakes_or_demand(params, cfg, table):
    """NaN prb while asleep: charged ACTIVE (conservative) but never a false sleep / demand while asleep."""
    base = ev.run_evaluation(params, cfg, df=table)
    win0 = base.replays["with_ruapp"].windows
    slept = win0.loc[(win0["cell"] == "cell-0") & (win0["state"] == "SLEEP"), "ts"]
    assert len(slept) >= 3
    holes = list(slept.iloc[[0, len(slept) // 2, -1]])  # three slept windows lose their KPI sample
    t2 = table.copy()
    hit = (t2["cell_id"] == "cell-0") & t2["ts"].isin(holes)
    assert int(hit.sum()) == 3
    t2.loc[hit, "prb_util_dl_pct"] = np.nan
    res = ev.run_evaluation(params, cfg, df=t2)
    for variant in ("with_ruapp", "rapp_only"):
        win = res.replays[variant].windows
        miss = win.loc[win["prb_missing"]]
        assert len(miss) == 3 and (miss["cell"] == "cell-0").all()
        assert not miss["ruapp_wake"].any() and not miss["demand_while_asleep"].any()
        assert np.allclose(miss["baseline_wh"], P_ACTIVE_BASE_W * 0.5)  # PRB 0 in the power model
        tot = res.summary["variants"][variant]["total"]
        assert tot["missing_kpi_windows"] == 3
        entered = miss["entered_state"] == "SLEEP"
        assert tot["missing_kpi_asleep"] == int(entered.sum()) == int(miss["missing_kpi_asleep"].sum())
        if variant == "with_ruapp":
            assert (miss.loc[entered, "state"] == "ACTIVE").all()   # charged ACTIVE, conservative
            sleep_entered = int((win["entered_state"] == "SLEEP").sum())
            assert tot["false_sleep_rate"] == pytest.approx(tot["ruapp_wakes"] / (sleep_entered - tot["missing_kpi_asleep"]))
        else:
            assert (miss.loc[entered, "state"] == "SLEEP").all()
    assert res.summary["notes"] and "no prb_util_dl_pct sample" in res.summary["notes"][-1]
    json.loads(json.dumps(ev._jsonable(res.summary)))  # strict JSON


def test_demand_prb_override_adds_oracle_note(cfg, table):
    p = ev.EvalParams(train_days=TRAIN_DAYS, validate_days=VALIDATE_DAYS, seed=7, sweep=False, demand_prb=1000.0,
                      demand_margins=(), burst_fractions=())
    r = ev.run_evaluation(p, cfg, df=table)
    assert r.summary["headline"]["with_ruapp_impact_free_by_construction"] is False
    assert r.summary["params"]["demand_prb_is_low_threshold"] is False
    note = "\n".join(r.summary["notes"])
    assert "NO LONGER a bound" in note and note in "\n".join(r.summary["caveats"])
    assert any("note" in line and "NO LONGER a bound" in line for line in ev.headline_lines(r.summary))
    assert r.summary["demand_sensitivity"]["burst_fractions"] == [] and r.summary["demand_sensitivity"]["demand_margins"] == []


def test_data_dir_never_written_to_outputs(params, cfg, table, tmp_path):
    secret = tmp_path / "acme-operator-kpi-export"
    p = replace(params, data_kind="kpi_report_csv", data_dir=secret)
    r = ev.run_evaluation(p, cfg, df=table)   # df given: the loader is bypassed, the param is still recorded
    out = ev.write_outputs(r, tmp_path / "out", plots=False)
    blob = (out / "report.md").read_text() + (out / "summary.json").read_text()
    assert "acme-operator" not in blob and str(tmp_path) not in blob
    assert r.summary["params"]["data_dir"] == ev.DATA_DIR_PLACEHOLDER


def _export_kpi_csvs(table: pd.DataFrame, d: Path) -> None:
    """The synthetic table as per-cell CSVs in the shipped column map's format."""
    from ocp_auru.rapp.data import kpi_report_csv
    inv = {dst: src for src, dst in kpi_report_csv.load_map()["columns"].items()}
    d.mkdir(parents=True)
    for i, (_, g) in enumerate(table.groupby("cell_id", sort=True)):
        cols = {inv["ts"]: g["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")}
        cols.update({inv[c]: g[c] for c in inv if c != "ts"})
        pd.DataFrame(cols).to_csv(d / f"site-export-{i}.csv", index=False)


def test_cli_private_path_never_names_the_data_dir(tmp_path, table):
    data = tmp_path / "acme-operator-kpi-export"
    _export_kpi_csvs(table, data)
    out = tmp_path / "eval"
    cmd = [sys.executable, str(ROOT / "scripts" / "eval_energy.py"), "--data-kind", "kpi_report_csv", "--data-dir", str(data),
           "--train-days", str(TRAIN_DAYS), "--validate-days", str(VALIDATE_DAYS), "--seed", "7", "--no-sweep", "--no-plots",
           "--out", str(out)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    blob = (out / "report.md").read_text() + (out / "summary.json").read_text() + proc.stdout
    assert "acme-operator" not in blob and str(tmp_path) not in blob
    summary = json.loads((out / "summary.json").read_text())
    assert summary["params"]["data_kind"] == "kpi_report_csv" and summary["params"]["data_dir"] == ev.DATA_DIR_PLACEHOLDER
    assert summary["split"]["cells"] == ["cell-0", "cell-1"]
    assert summary["headline"]["savings_pct_oracle"] >= summary["headline"]["savings_pct_with_ruapp"]


def test_cli_rejects_bad_quantile_naming_the_flag(tmp_path):
    cmd = [sys.executable, str(ROOT / "scripts" / "eval_energy.py"), "--data-kind", "synthetic", "--days", str(DAYS), "--cells", "1",
           "--train-days", str(TRAIN_DAYS), "--validate-days", str(VALIDATE_DAYS), "--low-quantile", "1.5", "--no-sweep",
           "--no-plots", "--out", str(tmp_path / "x")]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2 and "--low-quantile" in proc.stderr and "percentiles" not in proc.stderr


# ---- policy chatter metrics and the chatter-control grid -------------
def test_sleep_episodes_helper_and_chatter_stats():
    ts = pd.date_range("2024-01-01", periods=8, freq="30min")
    win = pd.DataFrame({"cell": ["a"] * 8, "ts": ts, "state": ["SLEEP", "SLEEP", "ACTIVE", "SLEEP", "ACTIVE", "ACTIVE", "SLEEP", "SLEEP"]})
    assert ev.sleep_episodes(win) == [2, 1, 2]
    assert ev.sleep_episodes(win.iloc[::-1]) == [2, 1, 2]                      # time order, not row order
    two = pd.concat([win, win.assign(cell="b")], ignore_index=True)
    assert sorted(ev.sleep_episodes(two)) == [1, 1, 2, 2, 2, 2]                 # never merged across cells
    assert ev.sleep_episodes(win.assign(state="ACTIVE")) == []
    st = ev.chatter_stats(win.assign(day="2024-01-01", decision=["SLEEP", None, "ACTIVE", "SLEEP", "NONE", "ACTIVE", "SLEEP", None]))
    assert st["sleep_episodes"] == 3 and st["short_episodes"] == 3 and st["mean_episode_h"] == pytest.approx(5 * 0.5 / 3)
    assert st["actuations"] == 5 and st["actuations_per_day"] == 5.0 and st["actuations_per_cell_day"] == 5.0
    empty = ev.chatter_stats(win.assign(state="ACTIVE", day="2024-01-01", decision=None))
    assert empty["sleep_episodes"] == 0 and empty["mean_episode_h"] is None and empty["actuations"] == 0


def test_chatter_metric_identities(result, cfg):
    """The chatter / governor / divergence figures obey their identities whatever the settings in use (here the
    shipped defaults: the dwell is on, so held windows exist on the primary run)."""
    for variant in ("with_ruapp", "rapp_only"):
        win = result.replays[variant].windows
        v = result.summary["variants"][variant]
        tot = v["total"]
        dec = win["decision"].dropna()
        assert tot["actuations"] == int((dec == "SLEEP").sum() + (dec == "ACTIVE").sum()) == tot["sleep_decisions"] + tot["preemptive_active_decisions"]
        assert tot["actuations_per_day"] == pytest.approx(tot["actuations"] / win["day"].nunique())
        assert tot["actuations_per_cell_day"] == pytest.approx(tot["actuations"] / win.groupby(["cell", "day"]).ngroups)
        assert (win["state"] == "SLEEP").any() and tot["sleep_episodes"] >= 1
        assert 0 <= tot["short_episodes"] <= tot["sleep_episodes"]
        lengths = ev.sleep_episodes(win)
        assert len(lengths) == tot["sleep_episodes"] and sum(lengths) == tot["sleep_windows"]   # episodes partition the slept windows
        assert tot["mean_episode_h"] == pytest.approx(tot["sleep_windows"] * 0.5 / tot["sleep_episodes"])
        assert tot["sleep_episodes_per_day"] == pytest.approx(tot["sleep_episodes"] / win["day"].nunique())
        # an episode starts wherever a SLEEP-charged window follows a non-SLEEP one (per cell)
        starts = 0
        for _, g in win.groupby("cell"):
            s = g.sort_values("ts")["state"].eq("SLEEP").to_numpy()
            starts += int(s[0]) + int((s[1:] & ~s[:-1]).sum())
        assert starts == tot["sleep_episodes"]
        # per-cell figures sum to the total; per-day episodes are cut at midnight so they can only exceed it
        assert sum(c["sleep_episodes"] for c in v["per_cell"].values()) == tot["sleep_episodes"]
        assert sum(c["actuations"] for c in v["per_cell"].values()) == tot["actuations"]
        assert sum(d["actuations"] for d in v["per_day"]) == tot["actuations"]
        assert sum(d["sleep_episodes"] for d in v["per_day"]) >= tot["sleep_episodes"]
    h = result.summary["headline"]
    w = result.summary["variants"]["with_ruapp"]["total"]
    assert h["actuations_with_ruapp"] == w["actuations"] and h["sleep_episodes_with_ruapp"] == w["sleep_episodes"]
    assert h["short_episodes_with_ruapp"] == w["short_episodes"] and h["mean_episode_h_with_ruapp"] == w["mean_episode_h"]
    assert h["actuations_per_day_with_ruapp"] == pytest.approx(w["actuations_per_day"])
    # the governor's cost mirrors the with-ruApp totals and sums over the cells; a held wake can only exist while a
    # rate limiter (dwell / confirmation / margin) is on, and the false-sleep label says so
    assert h["held_windows_with_ruapp"] == w["held_windows"] and h["ruapp_wakes_while_held"] == w["ruapp_wakes_while_held"]
    assert h["preemptive_wake_delay_windows_mean"] == w["preemptive_wake_delay_windows_mean"]
    assert h["false_sleep_rate_includes_held_wakes"] is ev.chatter_flags_on(cfg)
    assert w["ruapp_wakes_while_held"] <= min(w["ruapp_wakes"], w["held_windows"])
    if not ev.chatter_flags_on(cfg):
        assert w["held_windows"] == 0 and w["preemptive_wakes_delayed"] == 0 and w["preemptive_wake_delay_windows_mean"] is None
    per_cell = result.summary["variants"]["with_ruapp"]["per_cell"]
    assert sum(c["held_windows"] for c in per_cell.values()) == w["held_windows"]
    assert sum(c["ruapp_wakes_while_held"] for c in per_cell.values()) == w["ruapp_wakes_while_held"]
    assert h["short_episodes_per_cell_with_ruapp"] == {c: st["short_episodes"] for c, st in per_cell.items()}
    assert h["forecast_divergent_windows_per_cell"] == {c: st["forecast_divergent_windows"] for c, st in per_cell.items()}
    assert h["forecast_divergent_windows"] == sum(h["forecast_divergent_windows_per_cell"].values()) == w["forecast_divergent_windows"]
    lines = "\n".join(ev.headline_lines(result.summary))
    assert "sleep episodes" in lines and "actuations" in lines and "chatter control" in lines and "/day" in lines
    assert "chatter per cell" in lines and "governor cost" in lines and "forecast divergence" in lines
    # the recorded params are the settings actually in use (the shipped defaults here, pinned in test_rapp_policy)
    p = result.summary["params"]
    for name in ("rapp_min_sleep_dwell_min", "rapp_wake_confirm_windows", "rapp_wake_margin", "rapp_profile_scale",
                 "rapp_post_local_wake_hold_min", "rapp_wake_threshold_effective"):
        assert p[name] == getattr(cfg, name), name
    assert p["rapp_wake_threshold_effective"] == pytest.approx(max(0.0, p["rapp_wake_threshold"] - p["rapp_wake_margin"]))
    assert result.summary["grid"] == [] and "grid" in result.summary["timing_s"]


def test_replay_mirrors_the_app_state_update_with_chatter_control(result, cfg, legacy_cfg):
    """Dwell / confirmation / margin change decisions only through decide(); the replay feeds sim_now (the
    window ts) and keeps PolicyState exactly as app.py does (votes persisted after every decision, reset and
    dwell clock set on actuation). Each knob is switched on alone against the pinned legacy baseline."""
    cell = "cell-0"
    tr = result.split.train.loc[result.split.train["cell_id"] == cell]
    va = result.split.validate.loc[result.split.validate["cell_id"] == cell]
    thr = result.model.low_threshold
    kw = dict(low_threshold=thr, demand_prb=thr, local_wake=True)
    base, dec0 = ev.replay_cell(result.model, cell, tr, va, cfg=legacy_cfg, **kw)
    assert dec0["ACTIVE:pre-emptive-wake"] >= 1 and "NONE:wake-vote-pending" not in dec0 and "NONE:min-sleep-dwell" not in dec0
    assert not base["held"].any()
    # the shipped defaults (dwell 60 min, one confirmation window) hold at least one wake back on this data: the
    # replay tallies the rule under the reason key the app logs and marks the windows entered while held
    prim, decp = ev.replay_cell(result.model, cell, tr, va, cfg=cfg, **kw)
    assert decp["NONE:min-sleep-dwell"] >= 1 and "NONE:wake-vote-pending" not in decp and prim["held"].sum() >= 1
    # confirmation windows: every pre-emptive ACTIVE is immediately preceded by one pending vote of the same cell
    c2 = replace(legacy_cfg, rapp_wake_confirm_windows=2)
    win2, dec2 = ev.replay_cell(result.model, cell, tr, va, cfg=c2, **kw)
    assert dec2["NONE:wake-vote-pending"] >= 1
    reasons = list(win2["decision_reason"])
    pe = [i for i, r in enumerate(reasons) if r == "ACTIVE:pre-emptive-wake"]
    assert pe and all(reasons[i - 1] == "NONE:wake-vote-pending" for i in pe)
    assert (win2.loc[win2["decision_reason"] == "NONE:wake-vote-pending", "state"] == "SLEEP").all()
    # dwell longer than any night: no pre-emptive wake at all (the ruApp still wakes on demand), the rule is tallied
    cd = replace(legacy_cfg, rapp_min_sleep_dwell_min=24 * 60.0)
    wind, decd = ev.replay_cell(result.model, cell, tr, va, cfg=cd, **kw)
    assert (wind["decision"] != "ACTIVE").all() and decd["NONE:min-sleep-dwell"] >= 1 and wind["ruapp_wake"].sum() >= 1
    assert (wind["state"] == "SLEEP").sum() >= (base["state"] == "SLEEP").sum()
    # a margin >= the wake threshold disables the pre-emptive wake too (effective threshold 0)
    cm = replace(legacy_cfg, rapp_wake_margin=1.0)
    winm, decm = ev.replay_cell(result.model, cell, tr, va, cfg=cm, **kw)
    assert (winm["decision"] != "ACTIVE").all() and decm["NONE:stay-sleep"] >= 1
    # the chatter metrics of any variant obey the identities
    for w in (prim, win2, wind, winm):
        st = ev.variant_stats(w)
        assert st["short_episodes"] <= st["sleep_episodes"] and st["actuations"] == st["sleep_decisions"] + st["preemptive_active_decisions"]


def test_grid_rows_and_params_validation(cfg, table):
    p = ev.EvalParams(train_days=TRAIN_DAYS, validate_days=VALIDATE_DAYS, seed=7, sweep=False, grid=True,
                      grid_dwell_min=(0.0, 60.0), grid_confirm_windows=(1, 2), grid_wake_margins=(0.0,), grid_profile_scales=(0.0, 1.0))
    r = ev.run_evaluation(p, cfg, df=table)
    g = r.summary["grid"]
    labels = [x["label"] for x in g]
    assert labels == ["s0/d0/c1/m0", "s0/d0/c2/m0", "s0/d60/c1/m0", "s0/d60/c2/m0",
                      "s1/d0/c1/m0", "s1/d0/c2/m0", "s1/d60/c1/m0", "s1/d60/c2/m0"]
    # the row flagged primary is the one whose settings are the run's own (the shipped defaults lie on these axes,
    # so no off-grid row is appended) and it reproduces the headline
    prim_label = _primary_label(cfg)
    assert prim_label in labels
    assert [x["primary"] for x in g] == [lab == prim_label for lab in labels] and not any(x["off_grid"] for x in g)
    prim = next(x for x in g if x["primary"])
    h = r.summary["headline"]
    assert prim["savings_pct_with_ruapp"] == pytest.approx(h["savings_pct_with_ruapp"]) and prim["actuations"] == h["actuations_with_ruapp"]
    assert prim["sleep_episodes"] == h["sleep_episodes_with_ruapp"]
    assert prim["forecast_divergent_windows"] == h["forecast_divergent_windows"]
    assert prim["held_windows"] == h["held_windows_with_ruapp"] and prim["ruapp_wakes_while_held"] == h["ruapp_wakes_while_held"]
    cells = sorted(r.split.validate["cell_id"].unique())
    for x in g:
        assert x["short_episodes"] <= x["sleep_episodes"] and x["actuations"] == x["sleep_decisions"] + x["preemptive_active_decisions"]
        assert x["actuations_per_day"] == pytest.approx(x["actuations"] / VALIDATE_DAYS)
        assert 0 <= x["savings_pct_with_ruapp"] <= x["oracle_savings_pct"] + 1e-9
        # per-cell attribution: the per-cell figures sum to the row, the worst cell is one of them
        assert sorted(x["per_cell"]) == cells
        assert sum(c["short_episodes"] for c in x["per_cell"].values()) == x["short_episodes"]
        assert sum(c["actuations"] for c in x["per_cell"].values()) == x["actuations"]
        assert sum(c["ruapp_wakes_while_held"] for c in x["per_cell"].values()) == x["ruapp_wakes_while_held"]
        assert x["max_cell_short_episodes"] == max(c["short_episodes"] for c in x["per_cell"].values())
        assert x["max_cell_actuations_per_day"] == pytest.approx(max(c["actuations_per_day"] for c in x["per_cell"].values()))
        assert x["worst_cell_short_episodes"] in cells and x["worst_cell_actuations_per_day"] in cells
        # a held wake exists only when a rate limiter or the budget is on; the false-sleep label follows
        flags_on = x["min_sleep_dwell_min"] > 0 or x["wake_confirm_windows"] > 1 or x["wake_margin"] > 0 or x["max_actuations_per_day"] > 0
        assert x["false_sleep_rate_includes_held_wakes"] is flags_on
        if not flags_on:
            assert x["held_windows"] == 0 and x["ruapp_wakes_while_held"] == 0 and x["preemptive_wakes_delayed"] == 0
        assert x["ruapp_wakes_while_held"] <= min(x["ruapp_wakes"], x["held_windows"])
        # the profile scale is the row's own setting; scale 0 rows use the estate profile (factor 1.0 everywhere)
        assert x["profile_scale"] in (0.0, 1.0)
        # the post-local-wake hold is not a grid axis: every row keeps the run's own
        assert x["post_local_wake_hold_min"] == cfg.rapp_post_local_wake_hold_min
        # the actuation budget is not a grid axis either, but the other way round: only the primary row carries the
        # run's cap, every other row replays with it OFF (the alarm threshold, a pure metric, is kept)
        assert x["max_actuations_per_day"] == (cfg.rapp_max_actuations_per_day if x["primary"] else 0)
        assert x["actuation_alarm_per_day"] == cfg.rapp_actuation_alarm_per_day
        if not x["primary"]:
            assert x["budget_reached_windows"] == 0 and x["budget_reached_windows_rapp_only"] == 0
    assert cfg.rapp_max_actuations_per_day > 0 and prim["max_actuations_per_day"] == cfg.rapp_max_actuations_per_day
    # with 2 confirmation windows and no dwell every pre-emptive wake was held exactly one window
    c2 = next(x for x in g if x["label"] == "s0/d0/c2/m0")
    assert c2["preemptive_wakes_delayed"] == c2["preemptive_active_decisions"] > 0
    assert c2["preemptive_wake_delay_windows_mean"] == pytest.approx(1.0) and c2["held_windows"] >= c2["preemptive_wakes_delayed"]
    assert r.summary["timing_s"]["grid"] < 60
    with pytest.raises(ValueError, match="--grid-confirm-windows"):
        ev.EvalParams(grid_confirm_windows=(0,))
    with pytest.raises(ValueError, match="--grid-dwell-min"):
        ev.EvalParams(grid_dwell_min=(-1.0,))
    with pytest.raises(ValueError, match="--grid-wake-margins"):
        ev.EvalParams(grid_wake_margins=())
    with pytest.raises(ValueError, match="--grid-profile-scales"):
        ev.EvalParams(grid_profile_scales=(0.0, 1.5))


def test_grid_appends_the_primary_row_when_off_axis(cfg, table):
    """A primary setting on none of the axes (dwell 45 against a (0, 60) dwell axis, the other primary settings
    on theirs) gets its own '(primary, off-grid)' row so the headline is always comparable with the matrix."""
    c45 = replace(cfg, rapp_min_sleep_dwell_min=45.0)
    p = ev.EvalParams(train_days=TRAIN_DAYS, validate_days=VALIDATE_DAYS, seed=7, sweep=False, grid=True,
                      grid_dwell_min=(0.0, 60.0), grid_confirm_windows=(cfg.rapp_wake_confirm_windows,),
                      grid_wake_margins=(cfg.rapp_wake_margin,), grid_profile_scales=(cfg.rapp_profile_scale,))
    r = ev.run_evaluation(p, c45, df=table)
    g = r.summary["grid"]
    on_axis = [_primary_label(replace(c45, rapp_min_sleep_dwell_min=d)) for d in (0.0, 60.0)]
    off = _primary_label(c45)
    assert off not in on_axis and [x["label"] for x in g] == on_axis + [off]
    assert [x["primary"] for x in g] == [False, False, True] and [x["off_grid"] for x in g] == [False, False, True]
    h = r.summary["headline"]
    assert g[-1]["savings_pct_with_ruapp"] == pytest.approx(h["savings_pct_with_ruapp"]) and g[-1]["actuations"] == h["actuations_with_ruapp"]
    assert g[-1]["min_sleep_dwell_min"] == 45.0 and g[-1]["profile_scale"] == cfg.rapp_profile_scale
    lines = "\n".join(ev.headline_lines(r.summary))
    assert re.search(r"^\s+" + re.escape(off) + r"\(off-grid\)\s+\* saved", lines, re.M), lines
    out = ev.write_outputs(r, Path(r.cfg.var_dir) / "off-grid-test", plots=False)
    grid_md = (out / "grid.md").read_text()
    assert f"| {off} (primary, off-grid) |" in grid_md and grid_md.count("(primary") >= 1


def test_held_windows_delay_and_resleep_attribution(result, cfg, legacy_cfg):
    """Windows entered while a dwell / pending vote held a wake back are marked `held`; ruApp wakes and
    rApp-only demand in them are attributed to the governor; the delay from the first suppressed wake to the
    ACTIVE is recorded on the ACTIVE's window; a SLEEP taken in the ruApp's wake window is a re-sleep. Each
    knob is switched on alone against the pinned legacy baseline (governor off: nothing is ever held)."""
    cell = "cell-0"
    tr = result.split.train.loc[result.split.train["cell_id"] == cell]
    va = result.split.validate.loc[result.split.validate["cell_id"] == cell]
    thr = result.model.low_threshold
    kw = dict(low_threshold=thr, demand_prb=thr)
    base, _ = ev.replay_cell(result.model, cell, tr, va, cfg=legacy_cfg, local_wake=True, **kw)
    assert not base["held"].any() and base["preemptive_wake_delay_windows"].isna().all()
    assert (base["resleep_after_local_wake"] == (base["ruapp_wake"] & (base["decision"] == "SLEEP"))).all()
    c2 = replace(legacy_cfg, rapp_wake_confirm_windows=2)
    win, _ = ev.replay_cell(result.model, cell, tr, va, cfg=c2, local_wake=True, **kw)
    held_prev = win["decision_reason"].shift(1).isin(list(ev.HELD_REASONS))
    assert (win["held"] == held_prev).all()                                 # held = entered after a held decision
    assert (win.loc[win["held"], "entered_state"] == "SLEEP").all()          # a held wake means the carrier was asleep
    st = ev.variant_stats(win)
    assert st["held_windows"] == int(win["held"].sum()) >= 1
    assert st["ruapp_wakes_while_held"] == int((win["ruapp_wake"] & win["held"]).sum())
    delays = win["preemptive_wake_delay_windows"].dropna()
    assert len(delays) == st["preemptive_wakes_delayed"] == int((win["decision"] == "ACTIVE").sum()) >= 1
    assert (delays == 1).all() and st["preemptive_wake_delay_windows_mean"] == 1.0 and st["preemptive_wake_delay_windows_max"] == 1
    assert (win.loc[delays.index, "decision"] == "ACTIVE").all()
    ro, _ = ev.replay_cell(result.model, cell, tr, va, cfg=c2, local_wake=False, **kw)
    sro = ev.variant_stats(ro)
    assert sro["demand_while_asleep_while_held"] == int((ro["demand_while_asleep"] & ro["held"]).sum())
    assert sro["ruapp_wakes_while_held"] == 0
    # a dwell longer than the night: every suppressed wake is held until the ruApp serves the demand
    cd = replace(legacy_cfg, rapp_min_sleep_dwell_min=24 * 60.0)
    wd, _ = ev.replay_cell(result.model, cell, tr, va, cfg=cd, local_wake=True, **kw)
    sd = ev.variant_stats(wd)
    assert sd["preemptive_wakes_delayed"] == 0 and sd["ruapp_wakes_while_held"] >= 1 and sd["ruapp_wakes_while_held"] <= sd["ruapp_wakes"]
    assert ev.chatter_flags_on(cd) and ev.chatter_flags_on(c2) and not ev.chatter_flags_on(legacy_cfg)
    assert ev.chatter_flags_on(cfg)   # the shipped dwell is a rate limiter: the primary run's false-sleep rate includes held wakes


def test_post_local_wake_hold_mirrors_the_app(result, cfg):
    """RAPP_POST_LOCAL_WAKE_HOLD_MIN: after a ruApp wake the replay stamps `local_wake_sim` as app.py:_observe does and
    decide() refuses a SLEEP in that window (30 min = one window, the shipped default); 0 = off.
    Both values are pinned explicitly here."""
    thr = result.model.low_threshold
    frames = {}
    for hold in (0.0, 30.0):
        c = replace(cfg, rapp_post_local_wake_hold_min=hold)
        rows = [ev.replay_cell(result.model, cell, result.split.train.loc[result.split.train["cell_id"] == cell],
                               result.split.validate.loc[result.split.validate["cell_id"] == cell], cfg=c, low_threshold=thr,
                               demand_prb=thr, local_wake=True) for cell in ("cell-0", "cell-1")]
        frames[hold] = (pd.concat([w for w, _ in rows], ignore_index=True), sum((d for _, d in rows), start=__import__("collections").Counter()))
    w0, d0 = frames[0.0]
    w30, d30 = frames[30.0]
    assert ev.variant_stats(w30)["resleep_after_local_wake"] == 0
    assert "NONE:post-local-wake-hold" not in d0
    # the hold names itself exactly in the wake windows where a SLEEP would otherwise have been decided
    wake_windows = w0.loc[w0["ruapp_wake"] & (w0["decision"] == "SLEEP")]
    assert d30.get("NONE:post-local-wake-hold", 0) == len(wake_windows)
    if len(wake_windows):
        assert (w30.loc[wake_windows.index, "decision_reason"] == "NONE:post-local-wake-hold").all()


def _quiet_estate(factor: float, days: int = 20) -> pd.DataFrame:
    """Three synthetic cells, the third scaled to `factor` of its load: a quiet site."""
    t = synthetic.generate(n_cells=3, days=days, seed=42)
    q = t["cell_id"] == "cell-2"
    for c in ("prb_util_dl_pct", "prb_util_ul_pct", "thr_dl_mbps", "active_users"):
        t.loc[q, c] = (t.loc[q, c] * factor).round(2)
    return t


def test_profile_scale_removes_the_quiet_cell_chatter_where_the_governor_only_rate_limits(cfg):
    """Root cause : the estate-pooled profile forecasts a quiet cell from the
    estate mean, so it flips at the trough edge. Dwell + confirmation halve the writes but move the wakes onto
    the ruApp; the level-scaled profile (RAPP_PROFILE_SCALE=1) removes the short episodes, cuts the divergent
    forecasts and raises the savings, without any governor."""
    table = _quiet_estate(0.35)
    p = ev.EvalParams(train_days=15, validate_days=5, seed=42, sweep=False, grid=True, grid_dwell_min=(0.0, 60.0),
                      grid_confirm_windows=(1, 2), grid_wake_margins=(0.0,), grid_profile_scales=(0.0, 1.0), burst_fractions=(), demand_margins=())
    r = ev.run_evaluation(p, cfg, df=table)
    g = {x["label"]: x for x in r.summary["grid"]}
    est, gov, fix = g["s0/d0/c1/m0"], g["s0/d60/c2/m0"], g["s1/d0/c1/m0"]
    quiet = "cell-2"
    # the estate profile chatters on the quiet cell only (per cell, not on totals)
    assert est["per_cell"][quiet]["short_episodes"] >= 5 and est["per_cell"][quiet]["actuations_per_day"] >= 4
    assert all(est["per_cell"][c]["short_episodes"] == 0 for c in ("cell-0", "cell-1"))
    assert est["per_cell"][quiet]["forecast_divergent_windows"] >= 2 * max(est["per_cell"][c]["forecast_divergent_windows"] for c in ("cell-0", "cell-1"))
    # the governor rate-limits: fewer writes, but the quiet cell still flips and wakes move onto the ruApp
    assert gov["actuations"] < est["actuations"] and gov["per_cell"][quiet]["short_episodes"] >= 1
    assert gov["ruapp_wakes_while_held"] >= 1 and gov["forecast_divergent_windows"] == est["forecast_divergent_windows"]
    # the level-scaled profile removes the chatter and the divergence, and saves more
    assert fix["per_cell"][quiet]["short_episodes"] == 0 and fix["per_cell"][quiet]["actuations_per_day"] <= 2.5
    assert fix["forecast_divergent_windows"] < est["forecast_divergent_windows"] / 3
    assert fix["savings_pct_with_ruapp"] > est["savings_pct_with_ruapp"] and fix["held_windows"] == 0
    assert fix["max_cell_short_episodes"] < est["max_cell_short_episodes"]
    # the headline is the primary run - the row flagged primary, i.e. the settings actually in use (the shipped
    # defaults lie on these axes) - and reports the chatter and the divergence per cell
    prim = next(x for x in g.values() if x["primary"])
    assert prim["label"] == _primary_label(cfg)
    h = r.summary["headline"]
    assert h["forecast_divergent_windows_per_cell"] == {c: st["forecast_divergent_windows"] for c, st in prim["per_cell"].items()}
    assert h["short_episodes_per_cell_with_ruapp"] == {c: st["short_episodes"] for c, st in prim["per_cell"].items()}
    lines = "\n".join(ev.headline_lines(r.summary))
    assert "chatter per cell" in lines and "governor cost" in lines and "forecast divergence" in lines and "root cause" in lines


def test_grid_rows_off_the_primary_are_not_capped_by_the_budget(cfg):
    """Regression: the shipped budget (6/4) binds on the quiet estate's 'estate profile, no guards'
    row (~8 writes/day on the quiet cell: peak 8 > cap 6), so replaying every grid row under the run's cap silently
    moved the EVAL 7 'before' row (43 -> 39 actuations, 11 -> 9 short episodes, 13.16 -> 13.06 %). The grid replays
    its off-primary rows with the budget OFF: the s0/d0/c1/m0 row equals a direct budget-off replay of those settings,
    a budget-on replay of the same settings differs, and only the primary row carries the cap."""
    table = _quiet_estate(0.35)
    p = ev.EvalParams(train_days=15, validate_days=5, seed=42, sweep=False, grid=True, grid_dwell_min=(0.0, 60.0),
                      grid_confirm_windows=(1,), grid_wake_margins=(0.0,), grid_profile_scales=(0.0, 1.0), burst_fractions=(), demand_margins=())
    assert cfg.rapp_max_actuations_per_day == 6 and cfg.rapp_actuation_alarm_per_day == 4   # the shipped budget is what is tested
    r = ev.run_evaluation(p, cfg, df=table)
    g = {x["label"]: x for x in r.summary["grid"]}
    est = g["s0/d0/c1/m0"]
    assert not est["primary"] and est["max_actuations_per_day"] == 0 and est["actuation_alarm_per_day"] == 4
    assert est["budget_reached_windows"] == 0 and est["budget_reached_windows_rapp_only"] == 0
    # the cap WOULD have bound on this row: the quiet cell's peak carrier-day exceeds it (and the shipped alarm is crossed)
    assert est["max_actuations_per_carrier_day"] > cfg.rapp_max_actuations_per_day and est["alarm_days"] >= 1
    assert est["per_cell"]["cell-2"]["actuations_per_day"] >= 6
    keys = ("actuations", "sleep_episodes", "short_episodes", "ruapp_wakes", "held_windows", "savings_pct_with_ruapp", "savings_pct_rapp_only",
            "max_actuations_per_carrier_day", "alarm_days")
    fc = ev.Forecaster(r.model)
    # the row's settings exactly: the axes at s0/d0/c1/m0, the run's own post-local-wake hold and alarm, the cap OFF
    legacy = replace(cfg, rapp_profile_scale=0.0, rapp_min_sleep_dwell_min=0.0, rapp_wake_confirm_windows=1, rapp_wake_margin=0.0,
                     rapp_max_actuations_per_day=0)
    off = ev._grid_result_row(cfg, legacy, "off", False, ev.aggregate(ev.replay(r.model, r.split, cfg=legacy, forecaster=fc)))
    assert {k: est[k] for k in keys} == {k: off[k] for k in keys}
    capped = replace(legacy, rapp_max_actuations_per_day=cfg.rapp_max_actuations_per_day)
    on = ev._grid_result_row(cfg, capped, "on", False, ev.aggregate(ev.replay(r.model, r.split, cfg=capped, forecaster=fc)))
    assert on["budget_reached_windows"] >= 1 and on["actuations"] < est["actuations"] and on["max_actuations_per_carrier_day"] == 6
    assert est["held_windows"] == 0      # no guard and no cap: nothing holds a wake on the uncapped row (budget-held wakes: test_rapp_actuation_budget)
    # the primary row keeps the run's cap and reproduces the headline; the other rows say 'cap 0' and hold nothing
    prim = next(x for x in g.values() if x["primary"])
    assert prim["max_actuations_per_day"] == 6 and prim["actuations"] == r.summary["headline"]["actuations_with_ruapp"]
    assert all(x["max_actuations_per_day"] == 0 and x["budget_reached_windows"] == 0 for x in g.values() if not x["primary"])
    assert r.summary["params"]["rapp_max_actuations_per_day"] == 6
    out = ev.write_outputs(r, Path(r.cfg.var_dir) / "grid-budget-test", plots=False)
    grid_md = (out / "grid.md").read_text()
    assert "not a grid axis" in grid_md and "budget OFF" in grid_md and "only the primary row" in grid_md
    lines = "\n".join(ev.headline_lines(r.summary))
    assert "budget in force on the primary row only, off elsewhere" in lines


def test_cli_grid_writes_grid_md_with_expected_rows(tmp_path):
    """The CLI's chatter flags reach the primary run; the flags not given (profile scale, post-local-wake hold) fall
    back to config.py's shipped defaults, so the starred / '(primary)' row follows the settings actually in use."""
    shipped = Settings()          # the subprocess reads the same environment
    scale, hold = shipped.rapp_profile_scale, shipped.rapp_post_local_wake_hold_min
    out = tmp_path / "eval"
    cmd = [sys.executable, str(ROOT / "scripts" / "eval_energy.py"), "--data-kind", "synthetic", "--days", str(DAYS),
           "--cells", "2", "--train-days", str(TRAIN_DAYS), "--validate-days", str(VALIDATE_DAYS), "--seed", "7",
           "--no-sweep", "--grid", "--grid-dwell-min", "0,60", "--grid-confirm-windows", "1,2", "--grid-wake-margins", "0,0.1",
           "--min-sleep-dwell-min", "60", "--wake-confirm-windows", "2", "--wake-margin", "0.1", "--out", str(out)]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "chatter grid         : 16 configs" in proc.stdout                      # default profile-scale axis (0, 1) doubles the 8
    prim_label = ev.grid_label(60.0, 2, 0.1, scale)
    assert re.search(r"^\s+" + re.escape(prim_label) + r"\s+\* saved", proc.stdout, re.M), proc.stdout   # the primary row is starred
    assert len(re.findall(r"^\s+s[\d.]+/d\d+/c\d/m[\d.]+\s+[* ] saved", proc.stdout, re.M)) == 16
    assert (f"chatter control      : profile scale {scale:g}, dwell 60 min, 2 confirmation window(s), wake margin 0.1 "
            f"(effective wake threshold 0.20), post-local-wake hold {hold:g} min") in proc.stdout
    assert "sleep episodes" in proc.stdout and "actuations" in proc.stdout and "grid.md" in proc.stdout and "5 plots" in proc.stdout
    assert "while held" in proc.stdout and "divergent" in proc.stdout and "worst cell" in proc.stdout
    for name in ("grid.md", "grid.png", "report.md", "summary.json", "energy_daily.png", "impact.png"):
        assert (out / name).exists() and (out / name).stat().st_size > 0, name
    labels = [f"s{s_}/d{d}/c{c}/m{m}" for s_ in ("0", "1") for d in ("0", "60") for c in ("1", "2") for m in ("0", "0.1")]
    assert prim_label in labels
    idx = labels.index(prim_label)
    grid_md = (out / "grid.md").read_text()
    main, per_cell = grid_md.split("## Per cell")
    rows = [line for line in main.splitlines() if line.startswith("| s")]
    assert [r.split(" | ")[0].lstrip("| ").replace(" (primary)", "") for r in rows] == labels
    assert rows[idx].startswith(f"| {prim_label} (primary) |") and sum("(primary)" in r for r in rows) == 1
    cell_rows = [line for line in per_cell.splitlines() if line.startswith("| s")]
    assert len(cell_rows) == 16 * 2 and cell_rows[0].split(" | ")[1] == "cell-0"
    assert "does not select a setting" in grid_md and "ILLUSTRATIVE" in grid_md and "/Users/" not in grid_md
    assert "RATE LIMITING" in grid_md and "root cause" in grid_md and "incl. held wakes" in grid_md
    summary = json.loads((out / "summary.json").read_text())
    g = summary["grid"]
    assert [x["label"] for x in g] == labels and [x["primary"] for x in g] == [lab == prim_label for lab in labels]
    prim = g[idx]
    h = summary["headline"]
    assert prim["savings_pct_with_ruapp"] == pytest.approx(h["savings_pct_with_ruapp"]) and prim["actuations"] == h["actuations_with_ruapp"]
    assert prim["false_sleep_rate_includes_held_wakes"] is True and g[0]["false_sleep_rate_includes_held_wakes"] is False
    assert h["false_sleep_rate_includes_held_wakes"] is True
    report = (out / "report.md").read_text()
    assert "ruApp wakes while held" in report and "divergent forecasts" in report and "PER CELL" in report
    p = summary["params"]
    assert p["rapp_min_sleep_dwell_min"] == 60 and p["rapp_wake_confirm_windows"] == 2 and p["rapp_wake_margin"] == pytest.approx(0.1)
    assert p["rapp_wake_threshold_effective"] == pytest.approx(0.2)
    assert p["rapp_profile_scale"] == scale and p["rapp_post_local_wake_hold_min"] == hold   # the CLI defaults are config.py's
    assert summary["outputs"]["grid"] == "grid.md" and "grid.png" in summary["outputs"]["plots"]
    report = (out / "report.md").read_text()
    assert "grid.md" in report and "sleep episodes" in report and "actuations/day" in report and "/Users/" not in report
    # the primary run with confirmation windows tallies pending votes
    assert "NONE:wake-vote-pending" in json.dumps(summary["variants"]["with_ruapp"]["decisions_by_reason"])


def test_cli_rejects_bad_chatter_flags(tmp_path):
    cmd = [sys.executable, str(ROOT / "scripts" / "eval_energy.py"), "--data-kind", "synthetic", "--days", str(DAYS), "--cells", "1",
           "--train-days", str(TRAIN_DAYS), "--validate-days", str(VALIDATE_DAYS), "--wake-confirm-windows", "0", "--no-sweep",
           "--no-plots", "--out", str(tmp_path / "x")]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2 and "--wake-confirm-windows" in proc.stderr


# ---- priced service impact: accounting only, decisions never change -----------
# the totals keys the pricing knobs are allowed to move; every other key must be byte-for-byte the default run's
_PRICED_KEYS = {"ruapp_wake_service_cost_min", "ruapp_wake_energy_wh", "service_impact_min", "service_impact_min_from_wakes",
                "saved_kwh_per_impact_min", "variant_kwh", "ruapp_wake_kwh", "savings_kwh", "savings_pct",
                "savings_kwh_net_of_offload", "savings_pct_net_of_offload"}


def test_wake_pricing_identities_and_defaults_unchanged(params, cfg, table, result):
    """Impact minutes = counts x cost (with ruApp: wakes x cost; rApp-only: demand windows x 30); the energy adds
    exactly wakes x Wh to the with-ruApp kWh; the replayed windows (the decisions) and every unpriced figure are
    identical to the default run's, and the default run is byte-for-byte an explicit cost-0 run."""
    cost, wh = 5.0, 20.0
    priced = ev.run_evaluation(replace(params, ruapp_wake_service_cost_min=cost, ruapp_wake_energy_wh=wh), cfg, df=table)
    for variant in ("with_ruapp", "rapp_only"):
        pd.testing.assert_frame_equal(priced.replays[variant].windows, result.replays[variant].windows)   # decisions never change
        v0, v1 = result.summary["variants"][variant], priced.summary["variants"][variant]
        t0, t1 = v0["total"], v1["total"]
        wakes, asleep = t1["ruapp_wakes"], t1["demand_while_asleep"]
        assert (wakes, asleep) == (t0["ruapp_wakes"], t0["demand_while_asleep"])
        assert t1["service_impact_min"] == pytest.approx(wakes * cost + asleep * ev.WINDOW_MIN)
        assert t1["service_impact_min_from_wakes"] == pytest.approx(wakes * cost)
        assert t1["service_impact_min_from_demand_asleep"] == pytest.approx(asleep * ev.WINDOW_MIN)
        assert t0["service_impact_min"] == pytest.approx(asleep * ev.WINDOW_MIN) and t0["ruapp_wake_kwh"] == 0
        assert t1["service_impact_events"] == t0["service_impact_events"] == wakes + asleep
        assert t1["ruapp_wake_kwh"] == pytest.approx(wakes * wh / 1000.0)
        assert t1["variant_kwh"] == pytest.approx(t0["variant_kwh"] + wakes * wh / 1000.0)
        assert t1["variant_kwh_power_model"] == pytest.approx(t0["variant_kwh"]) == pytest.approx(t0["variant_kwh_power_model"])
        assert t1["savings_kwh"] == pytest.approx(t0["savings_kwh"] - wakes * wh / 1000.0)
        assert t1["savings_kwh_net_of_offload"] == pytest.approx(t1["savings_kwh"] - t1["offloaded_kwh"])
        if t1["service_impact_min"] > 0:
            assert t1["saved_kwh_per_impact_min"] == pytest.approx(t1["savings_kwh"] / t1["service_impact_min"])
        else:
            assert t1["saved_kwh_per_impact_min"] is None
        # per-cell and per-day figures sum to the total
        assert sum(c["service_impact_min"] for c in v1["per_cell"].values()) == pytest.approx(t1["service_impact_min"])
        assert sum(d["service_impact_min"] for d in v1["per_day"]) == pytest.approx(t1["service_impact_min"])
        assert sum(d["ruapp_wake_kwh"] for d in v1["per_day"]) == pytest.approx(t1["ruapp_wake_kwh"])
        for k, val in t0.items():
            if k not in _PRICED_KEYS:
                assert t1[k] == val, k
    w, r = priced.summary["variants"]["with_ruapp"]["total"], priced.summary["variants"]["rapp_only"]["total"]
    assert w["ruapp_wakes"] > 0 and w["demand_while_asleep"] == 0 and r["ruapp_wakes"] == 0 and r["demand_while_asleep"] > 0
    assert r["service_impact_min"] == pytest.approx(result.summary["variants"]["rapp_only"]["total"]["service_impact_min"])
    h0, h1 = result.summary["headline"], priced.summary["headline"]
    assert h0["with_ruapp_impact_free_by_construction"] is True and h0["with_ruapp_impact_priced"] is False
    assert h1["with_ruapp_impact_free_by_construction"] is False and h1["with_ruapp_impact_priced"] is True
    assert h0["service_impact_min_with_ruapp"] == 0 and h0["saved_kwh_per_impact_min_with_ruapp"] is None
    assert h0["saved_kwh_per_impact_min_rapp_only"] == pytest.approx(r["savings_kwh"] / r["service_impact_min"])
    assert h1["service_impact_min_with_ruapp"] == pytest.approx(w["ruapp_wakes"] * cost)
    assert h1["service_impact_min_rapp_only"] == h0["service_impact_min_rapp_only"] == pytest.approx(r["demand_while_asleep"] * ev.WINDOW_MIN)
    assert h1["ruapp_wake_kwh_with_ruapp"] == pytest.approx(w["ruapp_wakes"] * wh / 1000.0)
    assert (h1["ruapp_wake_service_cost_min"], h1["ruapp_wake_energy_wh"]) == (cost, wh) and (h0["ruapp_wake_service_cost_min"], h0["ruapp_wake_energy_wh"]) == (0.0, 0.0)
    assert priced.summary["params"]["ruapp_wake_service_cost_min"] == cost and priced.summary["params"]["ruapp_wake_energy_wh"] == wh
    # the default run is byte-for-byte an explicit cost-0 run
    explicit = ev.run_evaluation(replace(params, ruapp_wake_service_cost_min=0.0, ruapp_wake_energy_wh=0.0), cfg, df=table)
    assert _comparable(explicit.summary) == _comparable(result.summary)
    # a cost without energy moves no kWh figure and no headline line other than the priced ones
    only_cost = ev.run_evaluation(replace(params, ruapp_wake_service_cost_min=cost), cfg, df=table)
    hc = only_cost.summary["headline"]
    assert hc["savings_pct_with_ruapp"] == h0["savings_pct_with_ruapp"] and hc["savings_kwh_with_ruapp"] == h0["savings_kwh_with_ruapp"]
    priced_prefixes = ("service impact", "wake-cost sens.", "savings with ruApp", "note")
    strip = lambda lines: [ln for ln in lines if not ln.lstrip().startswith(priced_prefixes)]  # noqa: E731
    assert strip(ev.headline_lines(only_cost.summary)) == strip(ev.headline_lines(result.summary))
    assert only_cost.summary["variants"]["with_ruapp"]["total"]["ruapp_wake_kwh"] == 0
    # headline wording: 'with ruApp X min (N wakes x cost), rApp-only Y min (M windows x 30)'
    lines = "\n".join(ev.headline_lines(priced.summary))
    assert (f"service impact       : with ruApp {w['ruapp_wakes'] * cost:.1f} min ({w['ruapp_wakes']} wakes x 5 min), rApp-only "
            f"{r['demand_while_asleep'] * ev.WINDOW_MIN:.1f} min ({r['demand_while_asleep']} windows x 30 min)") in lines
    assert "PRICED at 5 min per ruApp wake" in lines and "re-activation energy" in lines and "BY CONSTRUCTION" not in lines
    lines0 = "\n".join(ev.headline_lines(result.summary))
    assert "service impact       : with ruApp 0.0 min" in lines0 and "with ruApp n/a" in lines0 and "BY CONSTRUCTION" in lines0
    assert any("PRICED" in n for n in priced.summary["notes"]) and not any("PRICED" in n for n in result.summary["notes"])
    assert any("PRICED" in c for c in result.summary["caveats"])   # the static caveat is always there


def test_wake_cost_sensitivity_rows_monotone_and_breakeven(result):
    """Accounting-only rows on the same replay: with-ruApp impact rises linearly with the cost, its score falls;
    rApp-only stays at demand windows x 30 min whatever the cost; the break-even is c* = 30 M S_w / (N S_r) and
    the 'rApp-only scores higher' flag flips exactly there."""
    ds = result.summary["demand_sensitivity"]
    rows = ds["wake_costs"]
    assert [x["wake_cost_min"] for x in rows] == list(ev.WAKE_COST_SENSITIVITY_MIN) == [0.0, 1.0, 5.0, 30.0]
    w, r = result.summary["variants"]["with_ruapp"]["total"], result.summary["variants"]["rapp_only"]["total"]
    n_w, n_r, s_w, s_r = w["ruapp_wakes"], r["demand_while_asleep"], w["savings_kwh"], r["savings_kwh"]
    assert n_w > 0 and n_r > 0 and s_w > 0 and s_r > 0
    for x in rows:
        assert x["service_impact_min_with_ruapp"] == pytest.approx(n_w * x["wake_cost_min"])
        assert x["service_impact_min_rapp_only"] == pytest.approx(n_r * ev.WINDOW_MIN)
        assert x["savings_kwh_with_ruapp"] == pytest.approx(s_w) and x["savings_kwh_rapp_only"] == pytest.approx(s_r)
        assert x["saved_kwh_per_impact_min_rapp_only"] == pytest.approx(r["saved_kwh_per_impact_min"])
    impacts = [x["service_impact_min_with_ruapp"] for x in rows]
    assert impacts == sorted(impacts) and impacts[0] == 0 and rows[0]["saved_kwh_per_impact_min_with_ruapp"] is None
    scores = [x["saved_kwh_per_impact_min_with_ruapp"] for x in rows[1:]]
    assert all(s is not None for s in scores) and scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(s_w / n_w)                              # cost 1 min: kWh per wake-minute
    assert rows[0]["rapp_only_scores_higher"] is False                        # an impact-free variant cannot be beaten
    be = ds["wake_cost_breakeven_min"]
    assert be == pytest.approx(ev.WINDOW_MIN * n_r * s_w / (n_w * s_r))
    assert 0 < be < ev.WINDOW_MIN                                             # rApp-only saves more per demand window slept through
    for x in rows[1:]:
        assert x["rapp_only_scores_higher"] is (x["wake_cost_min"] > be)
    assert rows[-1]["rapp_only_scores_higher"] is True                        # at 30 min a wake costs the whole window: rApp-only wins
    at_be = ev.impact_stats(n_w, 0, s_w, wake_service_cost_min=be)
    assert at_be["saved_kwh_per_impact_min"] == pytest.approx(r["saved_kwh_per_impact_min"])
    # the helper is pure arithmetic
    st = ev.impact_stats(3, 2, 10.0, wake_service_cost_min=5.0, wake_energy_wh=0.0)
    assert st["service_impact_min"] == 75.0 and st["service_impact_events"] == 5 and st["saved_kwh_per_impact_min"] == pytest.approx(10.0 / 75.0)
    assert st["service_impact_min_from_wakes"] == 15.0 and st["service_impact_min_from_demand_asleep"] == 60.0
    assert ev.impact_stats(0, 0, 1.0)["saved_kwh_per_impact_min"] is None
    # no wakes to price: rApp-only can never score higher; no rApp-only demand windows: it wins at any cost > 0
    fake = {"with_ruapp": ev.ReplayResult("with_ruapp", result.replays["with_ruapp"].windows.assign(ruapp_wake=False)),
            "rapp_only": result.replays["rapp_only"]}
    assert ev.wake_cost_sensitivity(fake, result.params)[1] is None
    fake2 = {"with_ruapp": result.replays["with_ruapp"],
             "rapp_only": ev.ReplayResult("rapp_only", result.replays["rapp_only"].windows.assign(demand_while_asleep=False))}
    rows2, be2 = ev.wake_cost_sensitivity(fake2, result.params)
    assert be2 == 0.0 and rows2[0]["rapp_only_scores_higher"] is None and rows2[1]["rapp_only_scores_higher"] is True
    lines = "\n".join(ev.headline_lines(result.summary))
    assert "wake-cost sens." in lines and f"rApp-only scores higher from a wake cost of {be:.1f} min" in lines
    assert "same replay, accounting only" in lines
    assert "wake_cost_sensitivity_min" in result.summary["params"]
    empty = ev.EvalParams(train_days=TRAIN_DAYS, validate_days=VALIDATE_DAYS, seed=7, sweep=False, wake_cost_sensitivity_min=())
    assert ev.wake_cost_sensitivity(result.replays, empty)[0] == []


def test_wake_pricing_reaches_sweep_and_grid_rows(cfg, table):
    p = ev.EvalParams(train_days=TRAIN_DAYS, validate_days=VALIDATE_DAYS, seed=7, sweep=True, sleep_thresholds=(0.7,), low_quantiles=(0.25,),
                      grid=True, grid_dwell_min=(0.0,), grid_confirm_windows=(1,), grid_wake_margins=(0.0,), grid_profile_scales=(1.0,),
                      burst_fractions=(), demand_margins=(), ruapp_wake_service_cost_min=5.0, ruapp_wake_energy_wh=20.0)
    r = ev.run_evaluation(p, cfg, df=table)
    h = r.summary["headline"]
    for row in r.summary["sweep"] + r.summary["grid"]:
        assert row["service_impact_min_with_ruapp"] == pytest.approx(row["ruapp_wakes"] * 5.0)
        assert row["service_impact_min_rapp_only"] == pytest.approx(row["demand_while_asleep_rapp_only"] * ev.WINDOW_MIN)
        assert (row["saved_kwh_per_impact_min_with_ruapp"] is None) == (row["service_impact_min_with_ruapp"] == 0)
    g = next(x for x in r.summary["grid"] if x["primary"])     # the run's own settings (appended off-grid: the dwell axis is 0 only)
    assert g["service_impact_min_with_ruapp"] == pytest.approx(h["service_impact_min_with_ruapp"])
    assert g["saved_kwh_per_impact_min_with_ruapp"] == pytest.approx(h["saved_kwh_per_impact_min_with_ruapp"])
    assert g["savings_pct_with_ruapp"] == pytest.approx(h["savings_pct_with_ruapp"])       # the wake energy is in the grid's kWh too
    assert sum(c["service_impact_min"] for c in g["per_cell"].values()) == pytest.approx(g["service_impact_min_with_ruapp"])
    assert sum(c["service_impact_min_rapp_only"] for c in g["per_cell"].values()) == pytest.approx(g["service_impact_min_rapp_only"])
    sw = r.summary["sweep"][0]
    assert sw["service_impact_min_with_ruapp"] == pytest.approx(h["service_impact_min_with_ruapp"])
    out = ev.write_outputs(r, Path(r.cfg.var_dir) / "wake-pricing-test", plots=False)
    grid_md = (out / "grid.md").read_text()
    assert "impact min (with ruApp)" in grid_md and "saved kWh/impact min" in grid_md
    report = (out / "report.md").read_text()
    assert "impact min (with ruApp / rApp-only)" in report and "Wake-cost rows" in report


def test_wake_pricing_params_validation():
    with pytest.raises(ValueError, match="--ruapp-wake-service-cost-min"):
        ev.EvalParams(ruapp_wake_service_cost_min=-1.0)
    with pytest.raises(ValueError, match="--ruapp-wake-energy-wh"):
        ev.EvalParams(ruapp_wake_energy_wh=-0.5)
    with pytest.raises(ValueError, match="--wake-cost-sensitivity-min"):
        ev.EvalParams(wake_cost_sensitivity_min=(0.0, -1.0))
    p = ev.EvalParams()
    assert p.costs == {"wake_service_cost_min": 0.0, "wake_energy_wh": 0.0} and p.wake_cost_sensitivity_min == (0.0, 1.0, 5.0, 30.0)


def _table_cell_counts(md: str, heading: str) -> list[int]:
    """Cells per row of the first markdown table after `heading` (header + separator + rows, up to the first non-table line)."""
    counts: list[int] = []
    for line in md.split(heading, 1)[1].splitlines():
        if line.startswith("|"):
            # the separator row is "|---|---|"; header/body cells are " | "-separated (a cell may contain a bare "|")
            counts.append(line.count("|") - 1 if set(line.strip()) <= {"|", "-"} else len(line.strip().strip("|").strip().split(" | ")))
        elif counts:
            break
    return counts


def test_cli_wake_pricing_smoke_writes_impact_png(tmp_path):
    out = tmp_path / "eval"
    base = [sys.executable, str(ROOT / "scripts" / "eval_energy.py"), "--data-kind", "synthetic", "--days", str(DAYS), "--cells", "2",
            "--train-days", str(TRAIN_DAYS), "--validate-days", str(VALIDATE_DAYS), "--seed", "7", "--no-sweep"]
    proc = subprocess.run(base + ["--ruapp-wake-service-cost-min", "5", "--ruapp-wake-energy-wh", "20", "--out", str(out)],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert (out / "impact.png").exists() and (out / "impact.png").stat().st_size > 0 and "4 plots" in proc.stdout
    assert "service impact       : with ruApp" in proc.stdout and "wakes x 5 min" in proc.stdout and "windows x 30 min" in proc.stdout
    assert "wake-cost sens." in proc.stdout and "PRICED at 5 min per ruApp wake" in proc.stdout and "x 20 Wh" in proc.stdout
    summary = json.loads((out / "summary.json").read_text())
    h = summary["headline"]
    assert h["ruapp_wake_service_cost_min"] == 5.0 and h["ruapp_wake_energy_wh"] == 20.0 and h["with_ruapp_impact_priced"] is True
    assert h["with_ruapp_impact_free_by_construction"] is False
    assert h["service_impact_min_with_ruapp"] == pytest.approx(5.0 * h["ruapp_wakes"])
    assert h["service_impact_min_rapp_only"] == pytest.approx(30.0 * h["demand_while_asleep_rapp_only"])
    assert h["ruapp_wake_kwh_with_ruapp"] == pytest.approx(0.02 * h["ruapp_wakes"])
    assert "impact.png" in summary["outputs"]["plots"] and summary["params"]["ruapp_wake_service_cost_min"] == 5.0
    assert [x["wake_cost_min"] for x in summary["demand_sensitivity"]["wake_costs"]] == [0.0, 1.0, 5.0, 30.0]
    report = (out / "report.md").read_text()
    assert "service impact, PRICED" in report and "saved kWh per impact minute" in report and "Wake-cost rows" in report
    assert "PRICED at 5 min per ruApp wake" in report and "impact.png" in report and "/Users/" not in report
    # every table the pricing columns were added to stays rectangular (header, separator and rows agree)
    for heading in ("## Totals per variant", "## Per cell: with ruApp", "## Per day: with ruApp", "Wake-cost rows"):
        counts = _table_cell_counts(report, heading)
        assert len(counts) >= 3 and len(set(counts)) == 1, (heading, counts)
    # the default (unpriced) CLI run: impact.png is written too, the with-ruApp variant is impact-free by construction
    out0 = tmp_path / "eval0"
    proc0 = subprocess.run(base + ["--no-plots", "--out", str(out0)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc0.returncode == 0, proc0.stderr
    assert "service impact       : with ruApp 0.0 min" in proc0.stdout and "BY CONSTRUCTION" in proc0.stdout
    s0 = json.loads((out0 / "summary.json").read_text())
    assert s0["headline"]["with_ruapp_impact_free_by_construction"] is True and s0["headline"]["service_impact_min_with_ruapp"] == 0
    assert s0["headline"]["saved_kwh_per_impact_min_with_ruapp"] is None
    # a negative cost is rejected naming the flag
    bad = subprocess.run(base + ["--ruapp-wake-service-cost-min", "-1", "--no-plots", "--out", str(tmp_path / "x")],
                         cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert bad.returncode == 2 and "--ruapp-wake-service-cost-min" in bad.stderr
