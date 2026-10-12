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
"""Offline energy-saving evaluation of the es-rApp + ruApp on 30-minute KPI history.

What it does (the *real* rApp code paths are exercised, never re-implemented):

1. load the canonical KPI table (`ocp_auru.rapp.data.load`), split it BY TIME: the first
   `train_days` calendar dates train, the next `validate_days` validate, the rest is dropped;
2. train the rApp's model with `ocp_auru.rapp.model.train` on the train frame only (the low-load
   threshold and the hour-of-day profile come from that frame);
3. score the classifier on the validation windows (same threshold, features may use train-side
   lags, labels are validation windows only);
4. replay the rApp cycle per cell over the validation windows, mirroring `app.py:_plan` and
   `policy_engine.decide`: an ACTIVE carrier is forecast from its last PM rows
   (`predict_low_load_probability`), a sleeping one from the hour-of-day profile at the SAME
   instant, the ts of the latest PM row (`predict_low_load_probability_asleep(sim_time)`, as
   `_plan` does; a sleeping cell reports no load), then `decide()` and the ruApp wake-on-PRACH
   rule (a SLEEP carrier that meets demand is woken locally);
5. charge energy per window with the mock RU's ILLUSTRATIVE power model (`ocp_auru.mock_ru.radio`)
   and compare against an always-ACTIVE baseline and a perfect-foresight oracle;
6. sweep the sleep threshold x the low-load quantile, and report how the with-ruApp savings
   move under stricter demand definitions (burst fractions, demand margins below the threshold);
7. count the policy's CHATTER: sleep episodes (maximal runs of SLEEP-charged
   windows), short episodes (<= 2 windows = 1 h), mean episode length and actuations (SLEEP +
   ACTIVE decisions = management-plane writes) per cell/day/total; with `grid=True` replay a
   small matrix of the chatter-control settings (profile level scale x sleep dwell x
   wake-confirmation windows x wake margin, `config.py`) on the primary model and write grid.md /
   grid.png (savings % vs actuations per day, one point per configuration, per-cell columns);
8. record BOTH forecast estimators at the same instant on every window - the own-PM forecast
   (`predict_low_load_probability`) and the estate hour-profile forecast (`asleep_forecast`) - and
   flag windows where they diverge by more than 0.5: the root cause of the chatter is the
   estate-pooled profile (model.py), and the divergence makes it visible per cell. The grid's
   dwell/confirmation/margin rows are RATE LIMITERS on the symptom; the `RAPP_PROFILE_SCALE` rows
   (the profile at the carrier's own level) address the cause. Windows entered while a wake was
   held back (dwell, pending vote, or the actuation budget refusing the pre-emptive ACTIVE) are
   marked `held`, so the service-side cost is attributed: ruApp wakes while held, demand slept
   through while held (rApp-only), and the delay from the first suppressed wake to the pre-emptive
   ACTIVE;
9. mirror the per-carrier-day ACTUATION BUDGET (`RAPP_MAX_ACTUATIONS_PER_DAY` /
   `RAPP_ACTUATION_ALARM_PER_DAY`, config.py) exactly as app.py keeps it: the counter is rolled at the
   sim day boundary and incremented on every applied SLEEP / ACTIVE, and the real `decide()` refuses
   the actuation at the budget. Per variant / cell / day: `budget_reached_windows` (decisions the
   budget refused), `alarm_days` (carrier-days whose count reached the alarm threshold) and
   `max_actuations_per_carrier_day` (the peak count of any carrier-day). A budget-refused pre-emptive
   wake is a HELD wake like a dwell / pending vote (`HELD_REASONS`): the ruApp wake or the demand slept
   through in the next window is attributed to it. The budget is NOT a grid axis: the grid's off-primary
   rows replay with the budget OFF (the alarm threshold is kept as a pure metric) so the matrix stays a
   chatter-control comparison and the 'estate profile, no guards' row reproduces the pre-budget
   decisions; only the primary row (and the sweep) carry the run's own cap.

Two variants are always reported: "with ruApp" (local wake at the first PRACH of a demand
window; that window is charged ACTIVE in full, conservatively) and "rApp-only" (no local
wake: the carrier sleeps through the demand window, counted as *demand while asleep*, and the
rApp wakes it at its next cycle per the normal rules).

Demand is a 30-minute-average PRB proxy, not the ruApp's PRACH-energy rule: with the default
definition (demand = "not actually low") every window the with-ruApp variant sleeps through is
low BY CONSTRUCTION, so that variant shows zero service impact by definition and its savings
are an upper bound (a sub-window burst that would wake the live ruApp is invisible here).

Service impact is priced so the two variants are comparable: every ruApp
wake in a slept-with-demand window is booked as `ruapp_wake_service_cost_min` minutes of degraded
service (the first UE(s) wait for the re-activation: the mock RU wakes in microseconds, a real
tx-array-carrier SLEEP exit takes milliseconds to seconds; default 0 = the historical accounting,
impact-free by construction) and `ruapp_wake_energy_wh` watt-hours of re-activation energy added
to the with-ruApp kWh; a rApp-only demand-while-asleep window is booked as the whole window
(30 min) unserved by this carrier. Per variant `service_impact_min` and one comparable score,
`saved_kwh_per_impact_min` (n/a while the impact is 0), are reported; the demand-definition
sensitivity carries wake-cost rows (0, 1, 5, 30 min: pure accounting on the same replay) with the
break-even cost at which rApp-only would score higher. Decisions never change with these knobs.

Caveat that every output repeats: the KPI export is per SITE, not per carrier. The replay
treats each site's series as the carrier under control and assumes the residual low load is
carried by the remaining layer(s); the coverage question is the physics twin's job (ADR-7),
not this evaluation's. The kWh figures are from an illustrative power model, never measured.

Cell labels are the adapter's opaque ids (`cell-<i>`); nothing here names a site.
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from ocp_auru.common.config import Settings, display_path, settings
from ocp_auru.common.events import Active, CarrierState, CarrierStatus, RuAppPolicy
from ocp_auru.mock_ru.radio import P_ACTIVE_BASE_W, P_ACTIVE_PER_PRB_PCT_W, P_INACTIVE_W, P_SLEEP_W, power_w
from ocp_auru.rapp import policy_engine
from ocp_auru.rapp.data import load as load_data
from ocp_auru.rapp.data.canonical import to_canonical
from ocp_auru.rapp.features import build_features, build_labels
from ocp_auru.rapp.model import FORECAST_DIVERGENCE_FLAG, LOW_QUANTILE, LowLoadModel, train

log = logging.getLogger(__name__)

WINDOW_H = 0.5  # one KPI window = 30 minutes
WINDOW_MIN = 30.0  # service minutes a rApp-only demand-while-asleep window forfeits: the whole window
WINDOWS_PER_DAY = 48
# wake-cost sensitivity (minutes of degraded service per ruApp wake): accounting-only rows on the primary replay
WAKE_COST_SENSITIVITY_MIN: tuple[float, ...] = (0.0, 1.0, 5.0, 30.0)
SWEEP_SLEEP_THRESHOLDS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)
SWEEP_LOW_QUANTILES: tuple[float, ...] = (0.25, 0.35, 0.5)
VARIANTS = ("with_ruapp", "rapp_only")
VARIANT_TITLES = {"with_ruapp": "with ruApp (local wake-on-PRACH)", "rapp_only": "rApp-only (no local wake)",
                  "oracle": "oracle (perfect foresight)"}

POWER_MODEL: dict[str, Any] = {
    "label": "ILLUSTRATIVE",
    "note": "ocp_auru.mock_ru.radio constants; not measurements of any product or network",
    "P_ACTIVE_BASE_W": P_ACTIVE_BASE_W,
    "P_ACTIVE_PER_PRB_PCT_W": P_ACTIVE_PER_PRB_PCT_W,
    "P_SLEEP_W": P_SLEEP_W,
    "P_INACTIVE_W": P_INACTIVE_W,
    "window_hours": WINDOW_H,
}

CAVEATS: tuple[str, ...] = (
    "The KPI export is per SITE, not per carrier. The replay treats each site's 30-minute series as the "
    "carrier under control and assumes the residual low load is carried by the remaining layer(s); whether "
    "coverage survives is the physics twin's question (HLD ADR-7), not this evaluation's.",
    "All power and energy figures come from the ILLUSTRATIVE power model of the mock O-RU "
    "(ocp_auru.mock_ru.radio); nothing here is measured energy.",
    "Energy per window = power_w(active, prb_util_dl_pct) x 0.5 h. A window the ruApp wakes into is charged "
    "ACTIVE in full (conservative); a rApp-only window slept through demand is charged SLEEP and counted as "
    "service impact ('demand while asleep'), which is why rApp-only can show more kWh saved than the oracle.",
    "Demand in a window means the actual prb_util_dl_pct was not below the low-load threshold (override with "
    "--demand-prb); the ruApp is assumed to see the first PRACH of that window.",
    "With the default demand definition (demand = not actually low) the with-ruApp variant has ZERO service impact "
    "BY CONSTRUCTION: every window it is charged SLEEP in is low-load by definition, and its 'false-sleep rate' is "
    "the rApp's forecast error on slept windows, not a measured service impact. Its savings are therefore an upper "
    "bound under a 30-minute PRB-average demand proxy: a window whose average is below the threshold can still "
    "contain a PRACH burst above the ruApp's energy threshold (agent.on_prach: energy_db >= "
    "prach-energy-threshold-db), in which case the live ruApp wakes and the window costs ACTIVE power while the "
    "replay charges SLEEP. The proxy is not the ruApp's rule; the 'demand-definition sensitivity' rows show how the "
    "savings move when a fraction of slept windows is charged ACTIVE or the demand threshold sits below the low "
    "threshold.",
    "The sweep's low-load quantile changes the DEFINITION of 'low' and hence of 'no service impact' (at q=0.5 "
    "sleeping through median-load windows counts as zero impact); its 'oracle %' grows with q for that reason, not "
    "because the model improved.",
    "The energy of the residual low load offloaded to the remaining layer(s) while this series sleeps "
    "(P_ACTIVE_PER_PRB_PCT_W x PRB-% x 0.5 h under the same illustrative model) is NOT subtracted from the "
    "savings; its magnitude is reported as 'offloaded load energy' and as 'savings net of offload'. The savings are "
    "those of switching this site's series off, not of the estate.",
    "In the live loop a sleeping RU reports PRB 0, so the first forecast after a wake sees zero lags; the replay "
    "feeds the recorded series as the carrier's PM history, i.e. the demand the carrier would have observed.",
    "The asleep forecast is taken at the ts of the latest PM row (the same instant the ACTIVE-branch forecast uses), "
    "exactly as app.py:_plan passes pm['sim_time']; both branches therefore forecast the next window.",
    "A window whose prb_util_dl_pct is missing (NaN) is charged ACTIVE (conservative) and, while asleep, wakes the "
    "with-ruApp variant; such windows are reported as 'missing KPI' and excluded from the false-sleep rate and from "
    "'demand while asleep' because no demand was observed.",
    "The rApp's minimum actuation hold (RAPP_MIN_HOLD_S, wall seconds) never binds at a 30-minute cadence.",
    "The oracle sleeps exactly in the windows that turned out to be low-load; it is the upper bound of savings "
    "without service impact, not a reachable policy. It bounds the with-ruApp variant only while the demand "
    "threshold equals the low threshold (the default).",
    "The sensitivity sweep is scored on the same validation days as the primary run: do not select a threshold "
    "from that table and quote its savings.",
    "A sleep episode is a maximal run of consecutive SLEEP-charged windows of one cell (the shaded spans of the "
    "timeline); a short episode has <= 2 windows (<= 1 h). Per-day rows cut episodes at midnight. Actuations are "
    "the SLEEP + ACTIVE decisions of the rApp (each one a CM request / NETCONF write and a BUSY settle on the RU); "
    "the ruApp's local wakes are not actuations. The chatter-control grid (profile level scale x sleep dwell x "
    "wake-confirmation windows x wake margin) is scored on the same validation days as the primary run: it "
    "shows the trade-off, it does not select the setting.",
    "Chatter is judged PER CELL, never on totals alone: a healthy total can hide one quiet cell that flips "
    "several times a night. The grid carries per-cell short episodes / actuations per day and the worst cell "
    "of each configuration.",
    "The hour-of-day profile is pooled over the whole training estate (no per-cell term), so a sleeping "
    "carrier is forecast from the ESTATE-mean load while an awake one is forecast from its own PM: on a "
    "carrier quieter than the estate the two estimators disagree at the trough edge and the rApp flips. Both "
    "estimators are recorded on every window ('divergent' = |p_low(pm-window) - p_low(profile)| > 0.5). The "
    "dwell / confirmation windows / margin only RATE-LIMIT that symptom; RAPP_PROFILE_SCALE (the profile "
    "scaled to the carrier's own level from its awake PM history) addresses the cause; the grid compares both.",
    "While a dwell, a pending wake vote or the actuation budget holds a pre-emptive wake back, a demand window "
    "that arrives is served by the ruApp (with ruApp; a 'ruApp wake while held') or slept through (rApp-only; "
    "'demand while asleep while held'): the governor moves wakes from the rApp to the ruApp, it does not remove "
    "them. With any chatter-control flag or the budget on, the false-sleep rate therefore INCLUDES held wakes.",
    "Service impact is PRICED, not measured: each ruApp wake in a slept-with-demand window is booked as "
    "--ruapp-wake-service-cost-min minutes of degraded service (the first UE(s) wait for the RU re-activation; the "
    "mock RU wakes in microseconds, a real tx-array-carrier SLEEP exit takes milliseconds to seconds) plus "
    "--ruapp-wake-energy-wh of re-activation energy on the with-ruApp kWh; each rApp-only demand-while-asleep window "
    "is booked as the whole window (30 min) unserved by this carrier. With the default cost 0 the with-ruApp variant is "
    "impact-free by construction and 'saved kWh per impact minute' is n/a for it; with a cost > 0 the two variants "
    "become comparable on that score. The wake-cost sensitivity rows re-price the SAME replay (decisions never "
    "change) and show the break-even cost at which rApp-only would score higher.",
    "The actuation budget (RAPP_MAX_ACTUATIONS_PER_DAY) counts the rApp's APPLIED SLEEP + ACTIVE "
    "decisions per cell per calendar day of the data's own clock (the window ts the decision is taken on), exactly as "
    "the live rApp counts PM simTime days; a clean night is 2. At the budget decide() refuses further SLEEPs and "
    "pre-emptive wakes for the rest of that day ('budget-held windows'); the cell stays as it is and the ruApp still "
    "wakes it on demand, so a held SLEEP costs savings and a held wake moves the wake onto the ruApp (a budget-refused "
    "pre-emptive wake marks the next window `held`, so it is counted in 'ruApp wakes while held' / 'demand asleep "
    "while held' like a dwell or a pending vote). 'Alarm carrier-days' are the cell-days whose count reached "
    "RAPP_ACTUATION_ALARM_PER_DAY (the operator signal; it never changes a decision). The counter starts at 0 on the "
    "first validation day, as after an rApp restart. The budget is not a grid axis: the grid's off-primary rows replay "
    "with the budget OFF (alarm kept as a metric), so they compare the chatter controls alone; the primary row, the "
    "headline and the sweep carry the run's own cap. In windows_*.csv `actuations_today_before` is the count the "
    "decision saw (= /decisions `actuations_today`) and `actuations_today_after` / `alarm_after` the count and alarm "
    "state after the window's applied write.",
)

BURST_FRACTIONS: tuple[float, ...] = (0.1, 0.25, 0.5)   # share of slept windows assumed to hide a wake-triggering burst
DEMAND_MARGINS: tuple[float, ...] = (0.75, 0.5)         # demand threshold as a fraction of the low threshold
SHORT_EPISODE_WINDOWS = 2                                # an episode of <= this many windows (<= 1 h) counts as short
# chatter-control grid (config.py: RAPP_PROFILE_SCALE x RAPP_MIN_SLEEP_DWELL_MIN x RAPP_WAKE_CONFIRM_WINDOWS x
# RAPP_WAKE_MARGIN). The margin axis defaults to the single value 0: the synthetic grid found it
# a no-op while p_low is near-binary (`--grid-wake-margins 0,0.1` restores the earlier matrix).
GRID_PROFILE_SCALES: tuple[float, ...] = (0.0, 1.0)
GRID_DWELL_MIN: tuple[float, ...] = (0.0, 30.0, 60.0, 120.0)
GRID_CONFIRM_WINDOWS: tuple[int, ...] = (1, 2)
GRID_WAKE_MARGINS: tuple[float, ...] = (0.0,)
# decision reasons that hold a pre-emptive wake back; the budget reason counts only when the refused action was the
# pre-emptive ACTIVE (the carrier asleep at the decision), never for a refused SLEEP (see replay_cell)
HELD_REASONS = ("NONE:min-sleep-dwell", "NONE:wake-vote-pending", "NONE:actuation-budget")


def _check_unit(name: str, values: tuple[float, ...], *, open_interval: bool = True) -> None:
    for v in values:
        ok = 0.0 < float(v) < 1.0 if open_interval else 0.0 <= float(v) <= 1.0
        if not ok:
            rng = "(0, 1)" if open_interval else "[0, 1]"
            raise ValueError(f"{name}: {v} is outside {rng}")


# --------------------------------------------------------------------------------------
# parameters and results
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalParams:
    data_kind: str = "synthetic"
    data_dir: Optional[Path] = None
    train_days: int = 25
    validate_days: int = 5
    low_quantile: Optional[float] = None       # None -> the rApp's default (model.LOW_QUANTILE)
    seed: int = 42
    demand_prb: Optional[float] = None         # None -> low_threshold
    sweep: bool = True
    sleep_thresholds: tuple[float, ...] = SWEEP_SLEEP_THRESHOLDS
    low_quantiles: tuple[float, ...] = SWEEP_LOW_QUANTILES
    # demand-definition sensitivity of the with-ruApp variant (see CAVEATS); empty tuples disable
    burst_fractions: tuple[float, ...] = BURST_FRACTIONS
    demand_margins: tuple[float, ...] = DEMAND_MARGINS
    # chatter-control grid on the primary model (off by default; the primary run's own settings come from cfg)
    grid: bool = False
    grid_dwell_min: tuple[float, ...] = GRID_DWELL_MIN
    grid_confirm_windows: tuple[int, ...] = GRID_CONFIRM_WINDOWS
    grid_wake_margins: tuple[float, ...] = GRID_WAKE_MARGINS
    grid_profile_scales: tuple[float, ...] = GRID_PROFILE_SCALES
    # service-impact pricing (accounting only, decisions never change): minutes of degraded service booked per ruApp
    # wake and re-activation energy (Wh) charged per wake to the with-ruApp variant; 0 = the historical accounting
    ruapp_wake_service_cost_min: float = 0.0
    ruapp_wake_energy_wh: float = 0.0
    wake_cost_sensitivity_min: tuple[float, ...] = WAKE_COST_SENSITIVITY_MIN   # empty tuple disables the rows

    def __post_init__(self) -> None:
        if float(self.ruapp_wake_service_cost_min) < 0:
            raise ValueError(f"--ruapp-wake-service-cost-min: {self.ruapp_wake_service_cost_min} must be >= 0")
        if float(self.ruapp_wake_energy_wh) < 0:
            raise ValueError(f"--ruapp-wake-energy-wh: {self.ruapp_wake_energy_wh} must be >= 0")
        if any(float(v) < 0 for v in self.wake_cost_sensitivity_min):
            raise ValueError(f"--wake-cost-sensitivity-min: all values must be >= 0 (got {list(self.wake_cost_sensitivity_min)})")
        if self.low_quantile is not None:
            _check_unit("--low-quantile", (self.low_quantile,))
        _check_unit("--sweep-low-quantiles", tuple(self.low_quantiles))
        _check_unit("--sweep-sleep-thresholds", tuple(self.sleep_thresholds), open_interval=False)
        _check_unit("burst_fractions", tuple(self.burst_fractions), open_interval=False)
        _check_unit("demand_margins", tuple(self.demand_margins))
        if self.demand_prb is not None and float(self.demand_prb) < 0:
            raise ValueError(f"--demand-prb: {self.demand_prb} must be >= 0")
        for name, values in (("--grid-dwell-min", self.grid_dwell_min), ("--grid-wake-margins", self.grid_wake_margins)):
            if not values or any(float(v) < 0 for v in values):
                raise ValueError(f"{name}: needs >= 1 value, all >= 0 (got {list(values)})")
        if not self.grid_confirm_windows or any(int(v) < 1 for v in self.grid_confirm_windows):
            raise ValueError(f"--grid-confirm-windows: needs >= 1 value, all >= 1 (got {list(self.grid_confirm_windows)})")
        if not self.grid_profile_scales or any(not (0.0 <= float(v) <= 1.0) for v in self.grid_profile_scales):
            raise ValueError(f"--grid-profile-scales: needs >= 1 value, all in [0, 1] (got {list(self.grid_profile_scales)})")

    @property
    def effective_low_quantile(self) -> float:
        return LOW_QUANTILE if self.low_quantile is None else float(self.low_quantile)

    @property
    def costs(self) -> dict[str, float]:
        """The pricing knobs as the keyword arguments of `variant_stats` / `aggregate`."""
        return {"wake_service_cost_min": float(self.ruapp_wake_service_cost_min), "wake_energy_wh": float(self.ruapp_wake_energy_wh)}


@dataclass
class Split:
    train: pd.DataFrame
    validate: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validate_start: pd.Timestamp
    validate_end: pd.Timestamp
    per_cell: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {"train_start": self.train_start.isoformat(), "train_end": self.train_end.isoformat(),
                "validate_start": self.validate_start.isoformat(), "validate_end": self.validate_end.isoformat(),
                "n_train_rows": int(len(self.train)), "n_validate_rows": int(len(self.validate)),
                "cells": sorted(self.per_cell), "per_cell": self.per_cell}


@dataclass
class ReplayResult:
    """One variant's replay: the per-window frame (all cells) plus the decision tally."""

    variant: str
    windows: pd.DataFrame
    decisions: Counter = field(default_factory=Counter)


@dataclass
class EvalResult:
    params: EvalParams
    cfg: Settings
    split: Split
    model: LowLoadModel
    classifier: dict[str, Any]
    replays: dict[str, ReplayResult]
    sweep: list[dict[str, Any]]
    summary: dict[str, Any]


# --------------------------------------------------------------------------------------
# 1. split by time
# --------------------------------------------------------------------------------------
def split_by_time(df: pd.DataFrame, train_days: int, validate_days: int) -> Split:
    """First `train_days` calendar dates -> train, next `validate_days` -> validate, rest dropped.

    Dates are ranked on the table's own date axis, so every cell is cut at the same instant
    (train and validation never share a date across cells). Every cell must have windows on
    each required date; the per-cell window coverage is recorded, not enforced."""
    if train_days < 1 or validate_days < 1:
        raise ValueError(f"train_days ({train_days}) and validate_days ({validate_days}) must be >= 1")
    df = to_canonical(df)
    need = train_days + validate_days
    dates = df["ts"].dt.normalize()
    uniq = sorted(dates.unique())
    if len(uniq) < need:
        raise ValueError(f"the table spans {len(uniq)} distinct dates; the split needs >= {need} "
                         f"({train_days} train + {validate_days} validate days of 30-minute windows, {WINDOWS_PER_DAY}/day)")
    train_dates, val_dates = uniq[:train_days], uniq[train_days:need]
    problems = []
    for cell, g in df.groupby("cell_id", sort=True):
        have = set(g["ts"].dt.normalize().unique())
        missing = [d for d in train_dates + val_dates if d not in have]
        if missing:
            problems.append(f"{cell}: no windows on {len(missing)} of the {need} required dates "
                            f"(first missing {pd.Timestamp(missing[0]).date()}); it has {len(have)} dates")
    if problems:
        raise ValueError("every cell needs >= train_days + validate_days days of 30-minute windows on the same dates:\n  "
                         + "\n  ".join(problems))
    train = df.loc[dates.isin(train_dates)].reset_index(drop=True)
    validate = df.loc[dates.isin(val_dates)].reset_index(drop=True)
    if not train["ts"].max() < validate["ts"].min():
        raise AssertionError("time split violated: max(train ts) >= min(validate ts)")
    per_cell: dict[str, dict[str, Any]] = {}
    for cell in sorted(df["cell_id"].unique()):
        tr, va = train.loc[train["cell_id"] == cell], validate.loc[validate["cell_id"] == cell]
        per_day = va.groupby(va["ts"].dt.normalize()).size()
        per_cell[cell] = {
            "train_start": tr["ts"].min().isoformat(), "train_end": tr["ts"].max().isoformat(),
            "validate_start": va["ts"].min().isoformat(), "validate_end": va["ts"].max().isoformat(),
            "n_train": int(len(tr)), "n_validate": int(len(va)),
            "validate_windows_per_day_min": int(per_day.min()), "validate_windows_per_day_max": int(per_day.max()),
        }
        if per_day.min() < WINDOWS_PER_DAY:
            log.warning("%s: a validation day has %d windows (< %d); energy per day is what the windows carry",
                        cell, int(per_day.min()), WINDOWS_PER_DAY)
    return Split(train=train, validate=validate, train_start=train["ts"].min(), train_end=train["ts"].max(),
                 validate_start=validate["ts"].min(), validate_end=validate["ts"].max(), per_cell=per_cell)


# --------------------------------------------------------------------------------------
# 2. train exactly as the rApp does
# --------------------------------------------------------------------------------------
def fit_model(train_df: pd.DataFrame, *, low_quantile: Optional[float], seed: int, data_kind: str) -> LowLoadModel:
    """`model.train` on the train frame. `low_quantile=None` (or the rApp default) takes the rApp's
    own path (`low_threshold=None` -> training 25th percentile); any other quantile is computed
    on the TRAIN frame and passed as the configured threshold."""
    if low_quantile is None or math.isclose(float(low_quantile), LOW_QUANTILE):
        return train(train_df, low_threshold=None, seed=seed, data_kind=data_kind)
    thr = float(train_df["prb_util_dl_pct"].quantile(float(low_quantile)))
    return train(train_df, low_threshold=thr, seed=seed, data_kind=data_kind)


# --------------------------------------------------------------------------------------
# 3. classifier validation on the held-out days
# --------------------------------------------------------------------------------------
def validate_classifier(model: LowLoadModel, split: Split) -> dict[str, Any]:
    """Features on train+validate (validation rows may use train-side lags: legitimate history),
    labels = next window low with the model's threshold, scored on validation rows only."""
    full = to_canonical(pd.concat([split.train, split.validate], ignore_index=True))
    X = build_features(full)
    y = build_labels(full, model.low_threshold)
    in_val = full["ts"] >= split.validate_start
    mask = in_val & y.notna()
    proba = model.predict_proba_frame(X.loc[mask])
    y_true = y.loc[mask].astype(int).to_numpy()
    y_hat = (proba >= 0.5).astype(int)
    two_classes = len(np.unique(y_true)) == 2
    return {
        "n": int(mask.sum()),
        "n_rows_with_train_side_lags": int(min(2, len(split.validate)) * split.validate["cell_id"].nunique()),
        "positive_rate": float(y_true.mean()) if len(y_true) else None,
        "auc": float(roc_auc_score(y_true, proba)) if two_classes else None,
        "accuracy": float(accuracy_score(y_true, y_hat)) if len(y_true) else None,
        "precision": float(precision_score(y_true, y_hat, zero_division=0)) if len(y_true) else None,
        "recall": float(recall_score(y_true, y_hat, zero_division=0)) if len(y_true) else None,
        "low_threshold": float(model.low_threshold),
        "labels": "validation windows only; the last window of each cell has no next window and is dropped",
    }


# --------------------------------------------------------------------------------------
# 4. replay
# --------------------------------------------------------------------------------------
class Forecaster:
    """The rApp's two forecast paths, memoised per model: an ACTIVE forecast depends only on
    (cell, window); an asleep forecast only on the simulated time with the estate profile
    (RAPP_PROFILE_SCALE 0) and on (cell, window, weight) with level scaling (the ratio comes from
    the cell's own history), so the two variants, the sweep's sleep thresholds and the grid share
    every call. Every entry is produced by the real `LowLoadModel` method (`calls` counts them)."""

    def __init__(self, model: LowLoadModel):
        self.model = model
        self._cache: dict[tuple, Any] = {}
        self.calls = Counter()

    def active(self, cell: str, index: int, rows: Callable[[], list[dict[str, Any]]]) -> float:
        key = ("pm-window", cell, index)
        if key not in self._cache:
            self._cache[key] = float(self.model.predict_low_load_probability(rows()))
            self.calls["pm-window"] += 1
        return self._cache[key]

    def asleep(self, sim_time: str, *, cell: str = "", index: int = -1, rows: Optional[Callable[[], list[dict[str, Any]]]] = None,
               scale_weight: float = 0.0) -> tuple[float, dict[str, Any]]:
        """(p_low, diagnostics) of the expected-hour-profile forecast, see `LowLoadModel.asleep_forecast`."""
        w = float(scale_weight)
        key = ("expected-hour-profile", sim_time) if w <= 0 else ("expected-hour-profile", sim_time, cell, index, w)
        if key not in self._cache:
            hist = rows() if (w > 0 and rows is not None) else None
            self._cache[key] = self.model.asleep_forecast(sim_time, hist, w) if w > 0 else \
                (float(self.model.predict_low_load_probability_asleep(sim_time)), {"profile_scale": 1.0, "profile_scale_rows": 0})
            self.calls["expected-hour-profile"] += 1
        return self._cache[key]


def _rows_for_window(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Canonical rows as the dicts `features_from_window` expects (ts as an ISO string, like
    `pm_to_kpi_row` produces in the live rApp)."""
    out = df.copy()
    out["ts"] = out["ts"].map(lambda t: pd.Timestamp(t).isoformat())
    return out.to_dict("records")


def _reason_key(action: str, reason: str) -> str:
    """Stable tally key for a decision reason (the reason text carries numbers)."""
    patterns = (
        # the actuation budget first: its reason names the refused action (SLEEP / pre-emptive ACTIVE)
        ("actuation budget", "actuation-budget"),
        # chatter-control reasons next: they hold a SLEEP and would otherwise match "stay SLEEP"
        ("min sleep dwell", "min-sleep-dwell"), ("wake vote", "wake-vote-pending"),
        ("< sleep threshold", "p_low-below-sleep-threshold"), ("not below low threshold", "load-now-not-low"),
        ("post-local-wake hold", "post-local-wake-hold"),
        ("min hold", "min-hold"), ("forecast low load", "forecast-low"), ("pre-emptive wake", "pre-emptive-wake"),
        ("stay SLEEP", "stay-sleep"), ("BUSY", "busy"), ("INACTIVE", "inactive"),
    )
    for needle, key in patterns:
        if needle in reason:
            return f"{action}:{key}"
    return f"{action}:{''.join(ch for ch in reason if not ch.isdigit())[:40]}"


def _prb_for_power(prb: float) -> float:
    return 0.0 if prb is None or (isinstance(prb, float) and math.isnan(prb)) else float(prb)


def replay_cell(model: LowLoadModel, cell: str, train_cell: pd.DataFrame, validate_cell: pd.DataFrame, *,
                cfg: Settings, low_threshold: float, demand_prb: float, local_wake: bool,
                forecaster: Optional[Forecaster] = None) -> tuple[pd.DataFrame, Counter]:
    """Chronological replay of one cell over its validation windows (see module docstring).

    Returns one record per window: the state the window was entered in (after the rApp's
    decision), the state it was charged at (after the ruApp), the forecast and decision made at
    the end of it for the next window, and the energy of baseline / this variant / oracle.

    Forecast instant: both branches forecast at the end of window i with `sim_time = ts[i]`,
    the ts of the latest PM row, exactly as `app.py:_plan` passes `pm['sim_time']` (the ACTIVE
    branch through the row's features, the asleep branch through the profile bins ending at
    that ts); both predict the low-load state of window i+1, the window the decision governs.

    A missing `prb_util_dl_pct` (NaN) counts as demand for the energy charge (ACTIVE,
    conservative) but is booked as `missing_kpi_asleep`, never as a ruApp wake / false sleep
    or as demand while asleep: no demand was observed.

    Both estimators are evaluated on every window (`p_low_pm_window`, `p_low_profile`,
    `forecast_divergence` = pm - profile, `forecast_divergent` = |delta| > 0.5); `held` marks a
    window entered while the previous decision held a pre-emptive wake back (dwell / pending
    vote / budget-refused pre-emptive ACTIVE: a NONE from the SLEEP branch for a `HELD_REASONS`
    reason; a budget-refused SLEEP holds no wake and marks nothing); `preemptive_wake_delay_windows`
    (on the ACTIVE decision's window) counts the windows from the first held decision that would have
    woken (p_low < effective wake threshold) to the ACTIVE; `resleep_after_local_wake` marks a SLEEP
    decision taken in the window the ruApp woke.

    Actuation budget: `actuations_today_before` is the count the decision saw (the live
    `Decision.actuations_today`: the cell's applied actuations on the window's calendar day BEFORE the
    decision; None on the undecided last window), `actuations_today_after` the same count AFTER this
    window's applied write (rolled at the day boundary and incremented on an applied SLEEP / ACTIVE, as
    app.py:_observe / _actuate do), `budget_held` marks a decision the budget refused and `alarm_after` a
    window at/after which the day's count reached RAPP_ACTUATION_ALARM_PER_DAY. The two counts differ by
    exactly one on a window with an applied write: join /decisions records on `_before`."""
    fc = forecaster or Forecaster(model)
    val = validate_cell.sort_values("ts").reset_index(drop=True)
    hist: deque = deque(_rows_for_window(train_cell.sort_values("ts").tail(cfg.rapp_history_len)), maxlen=cfg.rapp_history_len)
    val_rows = _rows_for_window(val)
    prbs = val["prb_util_dl_pct"].to_numpy(dtype=float)
    stamps = list(val["ts"])
    n = len(val)
    state = Active.ACTIVE
    pstate = policy_engine.PolicyState()
    decisions: Counter = Counter()
    pending_preemptive = False
    held_next = False                                      # the next window is entered while a wake is held back
    hold_start: Optional[int] = None                       # index of the first held decision that would have woken
    wake_thr = float(cfg.rapp_wake_threshold_effective)
    scale_w = float(cfg.rapp_profile_scale)
    alarm_thr = int(cfg.rapp_actuation_alarm_per_day)
    records: list[dict[str, Any]] = []
    for i in range(n):
        ts, prb = stamps[i], float(prbs[i])
        hist.append(val_rows[i])
        entered = state
        preemptive = pending_preemptive
        pending_preemptive = False
        held = held_next
        held_next = False
        sim_time = pd.Timestamp(ts).isoformat()          # the latest PM row's ts, as _plan passes pm['sim_time']
        policy_engine.roll_actuation_day(pstate, sim_time)   # app.py:_observe: the budget counter is per sim day
        prb_missing = math.isnan(prb)
        actually_low = bool(prb < low_threshold)          # NaN -> False: the oracle charges ACTIVE too
        demand = not (prb < demand_prb)                   # NaN counts as demand for the charge (conservative)
        ruapp_wake = demand_asleep = missing_asleep = False
        if state == Active.SLEEP and demand:
            if prb_missing:
                missing_asleep = True                      # no KPI sample: not a false sleep, not observed demand
                if local_wake:
                    state = Active.ACTIVE                  # charged ACTIVE (conservative), not booked as a wake
            elif local_wake:
                ruapp_wake, state = True, Active.ACTIVE   # the ruApp's on_prach: SLEEP + PRACH -> ACTIVE
                pstate.local_wake_sim = sim_time          # as app.py:_observe stamps the wake seen in PM
                hold_start = None
            else:
                demand_asleep = True                       # no ruApp: sleeps through the demand window
        if state != Active.SLEEP:
            pstate.sleep_entered_sim = None               # app.py:_observe: no dwell clock on a carrier not asleep
        charged = state                                    # the state the window is charged at
        prb_e = _prb_for_power(prb)
        baseline_wh = power_w(Active.ACTIVE, prb_e) * WINDOW_H
        variant_wh = power_w(charged, prb_e) * WINDOW_H
        oracle_wh = (P_SLEEP_W if actually_low else power_w(Active.ACTIVE, prb_e)) * WINDOW_H
        # energy of the residual load the remaining layer(s) carry while this series sleeps (not subtracted)
        offload_wh = P_ACTIVE_PER_PRB_PCT_W * prb_e * WINDOW_H
        offloaded_wh = offload_wh if charged == Active.SLEEP else 0.0
        oracle_offloaded_wh = offload_wh if actually_low else 0.0
        missed = charged == Active.ACTIVE and actually_low

        p_low: Optional[float] = None
        p_pm: Optional[float] = None
        p_prof: Optional[float] = None
        prof_scale: Optional[float] = None
        divergence: Optional[float] = None
        delay: Optional[int] = None
        resleep = False
        budget_held = False
        before: Optional[int] = None
        basis = action = reason_key = None
        if i + 1 < n:
            # both estimators at the same instant (app.py:_plan records them side by side); the carrier's
            # state selects the decision's p_low
            p_pm = fc.active(cell, i, lambda: list(hist))
            p_prof, prof_info = fc.asleep(sim_time, cell=cell, index=i, rows=lambda: list(hist), scale_weight=scale_w)
            prof_scale = float(prof_info.get("profile_scale", 1.0))
            divergence = p_pm - p_prof
            if state == Active.ACTIVE:
                basis, p_low = "pm-window", p_pm
            else:
                basis, p_low = "expected-hour-profile", p_prof
            status = CarrierStatus(name=cell, active=state, state=CarrierState.READY, ruapp_policy=pstate.envelope)
            now = pd.Timestamp(ts).timestamp()
            # sim_now = the window's ts, as app.py:_plan passes pm['sim_time']: the sleep-dwell clock, the post-local-wake
            # hold and the one-vote-per-window rule run on data time
            d = policy_engine.decide(status, {"prb_util_dl_pct": prb}, p_low, now, pstate,
                                     low_threshold=low_threshold, cfg=cfg, sim_now=sim_time)
            pstate.wake_votes, pstate.last_vote_sim = d.wake_votes, d.vote_sim   # as app.py:_plan stores them after every decision
            action, reason_key = d.action, _reason_key(d.action, d.reason)
            decisions[reason_key] += 1
            budget_held = reason_key == "NONE:actuation-budget"
            before = int(d.actuations_today)
            # a held wake is a NONE from the SLEEP branch: dwell, pending vote, or the budget refusing the pre-emptive
            # ACTIVE (state == SLEEP here); a budget-refused SLEEP (carrier awake) holds no wake
            if reason_key in HELD_REASONS and state == Active.SLEEP:
                held_next = True
                if hold_start is None and p_low < wake_thr:
                    hold_start = i                            # the governor suppressed a wake here
            if d.action in ("SLEEP", "ACTIVE"):
                # the app's post-actuation state update (app.py:_actuate, the CM write applied): hold clock, envelope,
                # votes reset, dwell clock started on SLEEP / cleared on ACTIVE
                pstate.last_decision_wall, pstate.last_action, pstate.envelope = now, d.action, d.envelope
                pstate.wake_votes, pstate.last_vote_sim = 0, None
                pstate.sleep_entered_sim = sim_time if d.action == "SLEEP" else None
                policy_engine.count_actuation(pstate, sim_time)   # app.py:_actuate: the applied write counts on the window's sim day
                pending_preemptive = state == Active.SLEEP and d.action == "ACTIVE"
                if pending_preemptive and hold_start is not None:
                    delay = i - hold_start
                hold_start = None
                resleep = ruapp_wake and d.action == "SLEEP"
                state = Active(d.action)
        records.append({
            "cell": cell, "ts": pd.Timestamp(ts), "day": pd.Timestamp(ts).strftime("%Y-%m-%d"),
            "prb_util_dl_pct": prb, "prb_missing": prb_missing, "actually_low": actually_low, "demand": demand,
            "entered_state": entered.value, "state": charged.value,
            "p_low": p_low, "forecast_basis": basis, "decision": action, "decision_reason": reason_key,
            "p_low_pm_window": p_pm, "p_low_profile": p_prof, "profile_scale": prof_scale,
            "forecast_divergence": divergence,
            "forecast_divergent": divergence is not None and abs(divergence) > FORECAST_DIVERGENCE_FLAG,
            "held": held, "preemptive_wake_delay_windows": delay, "resleep_after_local_wake": resleep,
            # actuation budget: the count the decision saw (= Decision.actuations_today), the count and the alarm
            # state AFTER this window's applied write, and the refusal
            "actuations_today_before": before, "actuations_today_after": int(pstate.actuations_today), "budget_held": budget_held,
            "alarm_after": alarm_thr > 0 and int(pstate.actuations_today) >= alarm_thr,
            "ruapp_wake": ruapp_wake, "demand_while_asleep": demand_asleep, "missing_kpi_asleep": missing_asleep,
            "missed_opportunity": missed, "preemptive_wake": preemptive,
            "baseline_wh": baseline_wh, "variant_wh": variant_wh, "oracle_wh": oracle_wh,
            "offloaded_wh": offloaded_wh, "oracle_offloaded_wh": oracle_offloaded_wh,
        })
    return pd.DataFrame(records), decisions


def replay(model: LowLoadModel, split: Split, *, cfg: Settings, demand_prb: Optional[float] = None,
           forecaster: Optional[Forecaster] = None) -> dict[str, ReplayResult]:
    """Both variants over every cell, sharing one forecaster (the model's memoised forecasts)."""
    fc = forecaster or Forecaster(model)
    thr = float(model.low_threshold)
    dp = thr if demand_prb is None else float(demand_prb)
    out: dict[str, ReplayResult] = {}
    for variant in VARIANTS:
        frames, tally = [], Counter()
        for cell in sorted(split.validate["cell_id"].unique()):
            win, dec = replay_cell(model, cell, split.train.loc[split.train["cell_id"] == cell],
                                   split.validate.loc[split.validate["cell_id"] == cell], cfg=cfg,
                                   low_threshold=thr, demand_prb=dp, local_wake=(variant == "with_ruapp"), forecaster=fc)
            frames.append(win)
            tally.update(dec)
        out[variant] = ReplayResult(variant=variant, windows=pd.concat(frames, ignore_index=True), decisions=tally)
    return out


# --------------------------------------------------------------------------------------
# 5. aggregation
# --------------------------------------------------------------------------------------
def _rate(num: int, den: int) -> Optional[float]:
    return float(num) / float(den) if den else None


def _pct(saved: float, base: float) -> Optional[float]:
    return 100.0 * saved / base if base else None


def sleep_episodes(win: pd.DataFrame) -> list[int]:
    """Lengths (in windows) of the sleep episodes in a window frame: per cell, in time order, a
    maximal run of consecutive SLEEP-charged windows (the timeline's shaded spans). On a per-day
    subset an episode spanning midnight is cut at the day boundary."""
    lengths: list[int] = []
    for _, g in win.groupby("cell", sort=True):
        run = 0
        for asleep in g.sort_values("ts")["state"].eq("SLEEP"):
            if asleep:
                run += 1
            elif run:
                lengths.append(run)
                run = 0
        if run:
            lengths.append(run)
    return lengths


def chatter_stats(win: pd.DataFrame) -> dict[str, Any]:
    """Policy chatter: episodes and actuations of one variant over a frame."""
    n_days = int(win["day"].nunique()) or 1
    n_cell_days = int(win.groupby(["cell", "day"]).ngroups) or 1
    lengths = sleep_episodes(win)
    dec = win["decision"].dropna()
    n_sleep, n_active = int((dec == "SLEEP").sum()), int((dec == "ACTIVE").sum())
    actuations = n_sleep + n_active
    return {
        "sleep_episodes": len(lengths), "short_episodes": sum(1 for n in lengths if n <= SHORT_EPISODE_WINDOWS),
        "short_episode_max_windows": SHORT_EPISODE_WINDOWS,
        "mean_episode_h": (sum(lengths) * WINDOW_H / len(lengths)) if lengths else None,
        "sleep_episodes_per_day": len(lengths) / n_days,
        "actuations": actuations, "actuations_per_day": actuations / n_days, "actuations_per_cell_day": actuations / n_cell_days,
    }


def impact_stats(wakes: int, demand_asleep: int, savings_kwh: float, *, wake_service_cost_min: float = 0.0,
                 wake_energy_wh: float = 0.0) -> dict[str, Any]:
    """The service-impact accounting of one variant from its event counts (pure arithmetic, shared by the
    totals and the wake-cost sensitivity): ruApp wakes x the wake cost + demand-while-asleep windows x the
    whole window (WINDOW_MIN), and the comparable score saved kWh per impact minute (None while the impact is
    0: with the default cost the with-ruApp variant is impact-free by construction)."""
    cost, wh = float(wake_service_cost_min), float(wake_energy_wh)
    impact = wakes * cost + demand_asleep * WINDOW_MIN
    return {
        "ruapp_wake_service_cost_min": cost, "ruapp_wake_energy_wh": wh,
        "service_impact_events": int(wakes) + int(demand_asleep), "service_impact_min": impact,
        "service_impact_min_from_wakes": wakes * cost, "service_impact_min_from_demand_asleep": demand_asleep * WINDOW_MIN,
        "saved_kwh_per_impact_min": (float(savings_kwh) / impact) if impact > 0 else None,
    }


def variant_stats(win: pd.DataFrame, *, wake_service_cost_min: float = 0.0, wake_energy_wh: float = 0.0) -> dict[str, Any]:
    """Totals of one variant over a window frame (any subset: cell, day, everything). `wake_service_cost_min`
    and `wake_energy_wh` price the ruApp wakes (EvalParams.costs); at 0 (the default) the figures are the
    historical ones: `variant_kwh` is the power-model energy alone and the with-ruApp impact is 0."""
    n_days = int(win["day"].nunique()) or 1
    n_cell_days = int(win.groupby(["cell", "day"]).ngroups) or 1   # sleep hours are per cell-day
    base = float(win["baseline_wh"].sum()) / 1000.0
    var_pm = float(win["variant_wh"].sum()) / 1000.0
    orc = float(win["oracle_wh"].sum()) / 1000.0
    off = float(win["offloaded_wh"].sum()) / 1000.0
    orc_off = float(win["oracle_offloaded_wh"].sum()) / 1000.0
    sleep_charged = int((win["state"] == "SLEEP").sum())
    sleep_entered = int((win["entered_state"] == "SLEEP").sum())
    wakes = int(win["ruapp_wake"].sum())
    demand_asleep = int(win["demand_while_asleep"].sum())
    missing_asleep = int(win["missing_kpi_asleep"].sum())
    low = int(win["actually_low"].sum())
    missed = int(win["missed_opportunity"].sum())
    dec = win["decision"].dropna()
    wake_kwh = wakes * float(wake_energy_wh) / 1000.0        # re-activation burst per ruApp wake (0 by default)
    var = var_pm + wake_kwh
    return {
        "n_windows": int(len(win)), "n_days": n_days, "n_cell_days": n_cell_days,
        "baseline_kwh": base, "variant_kwh": var, "oracle_kwh": orc,
        "variant_kwh_power_model": var_pm, "ruapp_wake_kwh": wake_kwh,
        "savings_kwh": base - var, "savings_pct": _pct(base - var, base),
        "oracle_savings_kwh": base - orc, "oracle_savings_pct": _pct(base - orc, base),
        # priced service impact: wakes x cost (with ruApp) + demand windows slept through x 30 min (rApp-only)
        **impact_stats(wakes, demand_asleep, base - var, wake_service_cost_min=wake_service_cost_min, wake_energy_wh=wake_energy_wh),
        # residual load carried by the remaining layer(s) while asleep; reported, not subtracted (see CAVEATS)
        "offloaded_kwh": off, "oracle_offloaded_kwh": orc_off,
        "savings_kwh_net_of_offload": base - var - off, "savings_pct_net_of_offload": _pct(base - var - off, base),
        "oracle_savings_pct_net_of_offload": _pct(base - orc - orc_off, base),
        "sleep_windows": sleep_charged, "sleep_windows_entered": sleep_entered,
        "sleep_hours_per_cell_day": sleep_charged * WINDOW_H / n_cell_days,
        "sleep_hours_per_day_all_cells": sleep_charged * WINDOW_H / n_days,
        "sleep_decisions": int((dec == "SLEEP").sum()), "preemptive_active_decisions": int((dec == "ACTIVE").sum()),
        "none_decisions": int((dec == "NONE").sum()),
        # false sleeps: ruApp wakes over SLEEP windows entered with an observed KPI (missing-KPI windows excluded)
        "ruapp_wakes": wakes, "false_sleep_rate": _rate(wakes, sleep_entered - missing_asleep),
        "demand_while_asleep": demand_asleep,
        "missing_kpi_windows": int(win["prb_missing"].sum()), "missing_kpi_asleep": missing_asleep,
        "actually_low_windows": low, "missed_opportunities": missed, "missed_opportunity_rate": _rate(missed, low),
        "oracle_sleep_hours_per_cell_day": low * WINDOW_H / n_cell_days,
        # policy chatter: sleep episodes, short episodes, mean episode length, actuations (= CM writes)
        **chatter_stats(win),
        # the governor's service-side cost (windows entered while a dwell / pending vote held a wake back)
        **held_stats(win),
        # the two estimators side by side (root cause: the estate-pooled profile vs the carrier's own PM)
        **divergence_stats(win),
        # the per-carrier-day actuation budget: refusals, alarm carrier-days, the peak count
        **budget_stats(win),
    }


def budget_stats(win: pd.DataFrame) -> dict[str, Any]:
    """Actuation budget over a frame: decisions the budget refused, cell-days whose applied-actuation
    count reached the alarm threshold, and the peak count of any cell-day (`actuations_today_after` is the
    running count of the window's own calendar day, so its maximum over the frame is the peak carrier-day)."""
    if "actuations_today_after" not in win.columns or not len(win):
        return {"budget_reached_windows": 0, "alarm_days": 0, "max_actuations_per_carrier_day": 0}
    return {
        "budget_reached_windows": int(win["budget_held"].astype(bool).sum()),
        "alarm_days": int(win.groupby(["cell", "day"])["alarm_after"].any().sum()),
        "max_actuations_per_carrier_day": int(win["actuations_today_after"].max()),
    }


def held_stats(win: pd.DataFrame) -> dict[str, Any]:
    """Cost attribution of the chatter control: what happened in windows entered while the previous
    decision held a pre-emptive wake back, the delay from the first suppressed wake to the ACTIVE,
    and SLEEP decisions taken in the very window the ruApp woke the carrier."""
    held = win["held"].astype(bool) if "held" in win.columns else pd.Series(False, index=win.index)
    delays = win["preemptive_wake_delay_windows"].dropna() if "preemptive_wake_delay_windows" in win.columns else pd.Series(dtype=float)
    resleep = int(win["resleep_after_local_wake"].sum()) if "resleep_after_local_wake" in win.columns else 0
    return {
        "held_windows": int(held.sum()),
        "ruapp_wakes_while_held": int((win["ruapp_wake"].astype(bool) & held).sum()),
        "demand_while_asleep_while_held": int((win["demand_while_asleep"].astype(bool) & held).sum()),
        "preemptive_wakes_delayed": int(len(delays)),
        "preemptive_wake_delay_windows_mean": float(delays.mean()) if len(delays) else None,
        "preemptive_wake_delay_windows_max": int(delays.max()) if len(delays) else None,
        "resleep_after_local_wake": resleep,
    }


def divergence_stats(win: pd.DataFrame) -> dict[str, Any]:
    """Windows where |p_low(pm-window) - p_low(expected-hour-profile)| > FORECAST_DIVERGENCE_FLAG."""
    if "forecast_divergence" not in win.columns:
        return {"forecast_divergent_windows": 0, "forecast_divergence_rate": None, "forecast_divergence_mean_abs": None}
    d = win["forecast_divergence"].dropna()
    return {
        "forecast_divergent_windows": int(win["forecast_divergent"].astype(bool).sum()),
        "forecast_divergence_rate": _rate(int(win["forecast_divergent"].astype(bool).sum()), int(len(d))),
        "forecast_divergence_mean_abs": float(d.abs().mean()) if len(d) else None,
    }


def aggregate(replays: dict[str, ReplayResult], *, wake_service_cost_min: float = 0.0, wake_energy_wh: float = 0.0) -> dict[str, Any]:
    costs = {"wake_service_cost_min": wake_service_cost_min, "wake_energy_wh": wake_energy_wh}
    out: dict[str, Any] = {}
    for variant, rr in replays.items():
        win = rr.windows
        out[variant] = {
            "total": variant_stats(win, **costs),
            "per_cell": {cell: variant_stats(g, **costs) for cell, g in win.groupby("cell", sort=True)},
            "per_day": [{"day": day, **variant_stats(g, **costs)} for day, g in win.groupby("day", sort=True)],
            "per_cell_day": [{"cell": cell, "day": day, **variant_stats(g, **costs)}
                             for (cell, day), g in win.groupby(["cell", "day"], sort=True)],
            "decisions_by_reason": dict(sorted(rr.decisions.items())),
        }
    return out


# --------------------------------------------------------------------------------------
# 6. sweep
# --------------------------------------------------------------------------------------
def run_sweep(params: EvalParams, cfg: Settings, split: Split, models: dict[float, LowLoadModel],
              forecasters: dict[float, Forecaster]) -> list[dict[str, Any]]:
    """rapp_sleep_threshold x low_quantile. Retrains per quantile (the label changes), reusing
    the primary model when the quantile matches; the primary run stays a separate result."""
    rows = []
    for q in params.low_quantiles:
        q = float(q)
        if q not in models:
            models[q] = fit_model(split.train, low_quantile=q, seed=params.seed, data_kind=params.data_kind)
            forecasters[q] = Forecaster(models[q])
        m = models[q]
        clf = validate_classifier(m, split)
        for s in params.sleep_thresholds:
            cfg_s = replace(cfg, rapp_sleep_threshold=float(s))
            agg = aggregate(replay(m, split, cfg=cfg_s, demand_prb=params.demand_prb, forecaster=forecasters[q]), **params.costs)
            w, r = agg["with_ruapp"]["total"], agg["rapp_only"]["total"]
            rows.append({
                "sleep_threshold": float(s), "low_quantile": q, "low_threshold": float(m.low_threshold),
                "validation_auc": clf["auc"],
                "savings_pct_with_ruapp": w["savings_pct"], "savings_pct_rapp_only": r["savings_pct"],
                "oracle_savings_pct": w["oracle_savings_pct"],
                "sleep_decisions": w["sleep_decisions"], "sleep_windows_entered": w["sleep_windows_entered"],
                "ruapp_wakes": w["ruapp_wakes"], "false_sleep_rate": w["false_sleep_rate"],
                "demand_while_asleep_rapp_only": r["demand_while_asleep"],
                "missed_opportunity_rate": w["missed_opportunity_rate"],
                "sleep_hours_per_cell_day": w["sleep_hours_per_cell_day"],
                **_chatter_columns(w),
                **_impact_columns(w, r),
            })
    return rows


def _impact_columns(w: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
    """The priced service impact of both variants as sweep/grid table columns."""
    return {"service_impact_min_with_ruapp": w["service_impact_min"], "service_impact_min_rapp_only": r["service_impact_min"],
            "saved_kwh_per_impact_min_with_ruapp": w["saved_kwh_per_impact_min"],
            "saved_kwh_per_impact_min_rapp_only": r["saved_kwh_per_impact_min"]}


def _chatter_columns(w: dict[str, Any]) -> dict[str, Any]:
    """The chatter metrics of the with-ruApp total as sweep/grid table columns."""
    return {"sleep_episodes": w["sleep_episodes"], "short_episodes": w["short_episodes"], "mean_episode_h": w["mean_episode_h"],
            "actuations": w["actuations"], "actuations_per_day": w["actuations_per_day"],
            "budget_reached_windows": w["budget_reached_windows"], "alarm_days": w["alarm_days"],
            "max_actuations_per_carrier_day": w["max_actuations_per_carrier_day"]}


def chatter_settings(cfg: Settings) -> dict[str, Any]:
    return {"rapp_min_sleep_dwell_min": float(cfg.rapp_min_sleep_dwell_min), "rapp_wake_confirm_windows": int(cfg.rapp_wake_confirm_windows),
            "rapp_wake_margin": float(cfg.rapp_wake_margin), "rapp_wake_threshold_effective": float(cfg.rapp_wake_threshold_effective),
            "rapp_profile_scale": float(cfg.rapp_profile_scale), "rapp_post_local_wake_hold_min": float(cfg.rapp_post_local_wake_hold_min),
            # the actuation budget is not a grid axis (off-primary grid rows replay with it OFF); it is recorded so
            # every output names the cap in force on the primary run
            "rapp_max_actuations_per_day": int(cfg.rapp_max_actuations_per_day),
            "rapp_actuation_alarm_per_day": int(cfg.rapp_actuation_alarm_per_day)}


def chatter_flags_on(cfg: Settings) -> bool:
    """True when any setting that can hold a pre-emptive wake back is on - a chatter-control rate limiter
    (dwell, confirmation windows, margin) or the actuation budget (a cap > 0 refuses the pre-emptive ACTIVE
    once reached): held wakes can then exist, and the false-sleep rate includes them."""
    return (float(cfg.rapp_min_sleep_dwell_min) > 0 or int(cfg.rapp_wake_confirm_windows) > 1 or float(cfg.rapp_wake_margin) > 0
            or int(cfg.rapp_max_actuations_per_day) > 0)


def grid_label(dwell_min: float, confirm: int, margin: float, scale: float = 0.0) -> str:
    return f"s{scale:g}/d{dwell_min:g}/c{confirm}/m{margin:g}"


_GRID_CELL_KEYS = ("sleep_episodes", "short_episodes", "actuations", "actuations_per_day", "ruapp_wakes", "ruapp_wakes_while_held",
                   "false_sleep_rate", "forecast_divergent_windows", "savings_pct", "service_impact_min",
                   "budget_reached_windows", "alarm_days", "max_actuations_per_carrier_day")


def _grid_result_row(cfg: Settings, cfg_g: Settings, label: str, primary: bool, agg: dict[str, Any], *, off_grid: bool = False) -> dict[str, Any]:
    w, r = agg["with_ruapp"]["total"], agg["rapp_only"]["total"]
    per_cell = {cell: {k: st[k] for k in _GRID_CELL_KEYS} for cell, st in agg["with_ruapp"]["per_cell"].items()}
    for cell, st in agg["rapp_only"]["per_cell"].items():
        per_cell[cell]["demand_while_asleep_rapp_only"] = st["demand_while_asleep"]
        per_cell[cell]["demand_while_asleep_while_held_rapp_only"] = st["demand_while_asleep_while_held"]
        per_cell[cell]["service_impact_min_rapp_only"] = st["service_impact_min"]
    worst_short = max(per_cell.items(), key=lambda kv: (kv[1]["short_episodes"], kv[1]["actuations_per_day"]))
    worst_act = max(per_cell.items(), key=lambda kv: kv[1]["actuations_per_day"])
    return {
        "label": label, "primary": primary, "off_grid": off_grid,
        "profile_scale": float(cfg_g.rapp_profile_scale),
        "min_sleep_dwell_min": float(cfg_g.rapp_min_sleep_dwell_min), "wake_confirm_windows": int(cfg_g.rapp_wake_confirm_windows),
        "wake_margin": float(cfg_g.rapp_wake_margin), "wake_threshold_effective": cfg_g.rapp_wake_threshold_effective,
        "post_local_wake_hold_min": float(cfg_g.rapp_post_local_wake_hold_min),
        "savings_pct_with_ruapp": w["savings_pct"], "savings_pct_rapp_only": r["savings_pct"],
        "oracle_savings_pct": w["oracle_savings_pct"],
        "sleep_hours_per_cell_day": w["sleep_hours_per_cell_day"],
        "sleep_decisions": w["sleep_decisions"], "preemptive_active_decisions": w["preemptive_active_decisions"],
        "ruapp_wakes": w["ruapp_wakes"], "false_sleep_rate": w["false_sleep_rate"],
        "false_sleep_rate_includes_held_wakes": chatter_flags_on(cfg_g),
        "demand_while_asleep_rapp_only": r["demand_while_asleep"],
        "missed_opportunity_rate": w["missed_opportunity_rate"],
        **_chatter_columns(w),
        **_impact_columns(w, r),
        "actuations_rapp_only": r["actuations"], "sleep_episodes_rapp_only": r["sleep_episodes"],
        # the governor's cost and the estimator divergence
        "held_windows": w["held_windows"], "ruapp_wakes_while_held": w["ruapp_wakes_while_held"],
        "demand_while_asleep_while_held_rapp_only": r["demand_while_asleep_while_held"],
        "preemptive_wakes_delayed": w["preemptive_wakes_delayed"],
        "preemptive_wake_delay_windows_mean": w["preemptive_wake_delay_windows_mean"],
        "resleep_after_local_wake": w["resleep_after_local_wake"],
        "forecast_divergent_windows": w["forecast_divergent_windows"], "forecast_divergence_rate": w["forecast_divergence_rate"],
        # per cell: chatter is judged on the quietest / busiest cell, never on totals alone
        "per_cell": per_cell,
        "worst_cell_short_episodes": worst_short[0], "max_cell_short_episodes": worst_short[1]["short_episodes"],
        "worst_cell_actuations_per_day": worst_act[0], "max_cell_actuations_per_day": worst_act[1]["actuations_per_day"],
        "max_cell_ruapp_wakes_while_held": max(c["ruapp_wakes_while_held"] for c in per_cell.values()),
        # the actuation budget in force on THIS row (the run's cap on the primary row, 0 elsewhere), and its rApp-only mirror
        "max_actuations_per_day": int(cfg_g.rapp_max_actuations_per_day), "actuation_alarm_per_day": int(cfg_g.rapp_actuation_alarm_per_day),
        "budget_reached_windows_rapp_only": r["budget_reached_windows"], "alarm_days_rapp_only": r["alarm_days"],
    }


def _is_primary(cfg: Settings, cfg_g: Settings) -> bool:
    return (math.isclose(float(cfg_g.rapp_min_sleep_dwell_min), float(cfg.rapp_min_sleep_dwell_min))
            and int(cfg_g.rapp_wake_confirm_windows) == int(cfg.rapp_wake_confirm_windows)
            and math.isclose(float(cfg_g.rapp_wake_margin), float(cfg.rapp_wake_margin))
            and math.isclose(float(cfg_g.rapp_profile_scale), float(cfg.rapp_profile_scale)))


def run_grid(params: EvalParams, cfg: Settings, split: Split, model: LowLoadModel, forecaster: Forecaster) -> list[dict[str, Any]]:
    """Chatter-control matrix on the PRIMARY model and sleep threshold: profile level scale x sleep dwell x
    wake-confirmation windows x wake margin, both variants replayed per configuration (only `decide` and the
    asleep forecast's level change, so every forecast comes from the memoised forecaster). The row whose
    settings equal the primary run's is flagged `primary`; when the primary settings lie on none of the axes
    the primary configuration is appended as an extra row flagged `off_grid` (label suffix '(primary,
    off-grid)') so the headline is always comparable with the matrix. Every row carries per-cell columns
    and the worst cell: chatter is judged per cell.

    The actuation budget is not an axis: every OFF-PRIMARY row replays with the budget OFF
    (`rapp_max_actuations_per_day=0`; the alarm threshold is kept, it is a pure metric), so the matrix
    compares the chatter controls alone and the 'estate profile, no guards' row reproduces the pre-budget
    decisions on chatter-prone data (the cap would otherwise bind on a quiet cell with many writes per
    day and silently move that row); the primary row keeps the run's own cap and reproduces the headline. Each row
    records the cap in force (`max_actuations_per_day`)."""
    rows = []
    for scale in params.grid_profile_scales:
        for dwell in params.grid_dwell_min:
            for confirm in params.grid_confirm_windows:
                for margin in params.grid_wake_margins:
                    cfg_g = replace(cfg, rapp_profile_scale=float(scale), rapp_min_sleep_dwell_min=float(dwell),
                                    rapp_wake_confirm_windows=int(confirm), rapp_wake_margin=float(margin))
                    primary = _is_primary(cfg, cfg_g)
                    if not primary:
                        cfg_g = replace(cfg_g, rapp_max_actuations_per_day=0)
                    agg = aggregate(replay(model, split, cfg=cfg_g, demand_prb=params.demand_prb, forecaster=forecaster), **params.costs)
                    rows.append(_grid_result_row(cfg, cfg_g, grid_label(float(dwell), int(confirm), float(margin), float(scale)), primary, agg))
    if rows and not any(r["primary"] for r in rows):
        agg = aggregate(replay(model, split, cfg=cfg, demand_prb=params.demand_prb, forecaster=forecaster), **params.costs)
        label = grid_label(float(cfg.rapp_min_sleep_dwell_min), int(cfg.rapp_wake_confirm_windows), float(cfg.rapp_wake_margin),
                           float(cfg.rapp_profile_scale))
        rows.append(_grid_result_row(cfg, cfg, label, True, agg, off_grid=True))
    return rows


def demand_sensitivity(model: LowLoadModel, split: Split, replays: dict[str, ReplayResult], params: EvalParams,
                       cfg: Settings, forecaster: Optional[Forecaster] = None) -> dict[str, Any]:
    """How the with-ruApp savings move under stricter demand definitions than the 30-minute
    PRB-average proxy (see CAVEATS):

    - `burst_fractions` (analytic, no replay): a share f of the SLEEP-charged windows is assumed
      to contain a wake-triggering PRACH burst invisible to the 30-minute average; the live ruApp
      wakes, so those windows cost their ACTIVE energy and the savings shrink by f x (the saving
      of every slept window);
    - `demand_margins` (a with-ruApp replay each): the demand threshold is m x the low threshold,
      so a window with m*thr <= prb < thr wakes the ruApp and is charged ACTIVE."""
    fc = forecaster or Forecaster(model)
    thr = float(model.low_threshold)
    costs = params.costs
    win = replays["with_ruapp"].windows
    base = float(win["baseline_wh"].sum()) / 1000.0
    slept = win["state"] == "SLEEP"
    slept_saving = float((win.loc[slept, "baseline_wh"] - win.loc[slept, "variant_wh"]).sum()) / 1000.0
    wakes = int(win["ruapp_wake"].sum())
    wake_kwh = wakes * costs["wake_energy_wh"] / 1000.0            # the re-activation energy, part of the variant's kWh
    savings = base - float(win["variant_wh"].sum()) / 1000.0 - wake_kwh
    bursts = [{"burst_fraction": float(f), "slept_windows_charged_active": int(round(f * int(slept.sum()))),
               "savings_kwh": savings - f * slept_saving, "savings_pct": _pct(savings - f * slept_saving, base)}
              for f in params.burst_fractions]
    margins = []
    for m in params.demand_margins:
        dp = float(m) * thr
        frames = [replay_cell(model, cell, split.train.loc[split.train["cell_id"] == cell],
                              split.validate.loc[split.validate["cell_id"] == cell], cfg=cfg, low_threshold=thr,
                              demand_prb=dp, local_wake=True, forecaster=fc)[0]
                  for cell in sorted(split.validate["cell_id"].unique())]
        st = variant_stats(pd.concat(frames, ignore_index=True), **costs)
        margins.append({"demand_margin": float(m), "demand_prb": dp, "savings_kwh": st["savings_kwh"],
                        "savings_pct": st["savings_pct"], "ruapp_wakes": st["ruapp_wakes"],
                        "false_sleep_rate": st["false_sleep_rate"], "sleep_windows": st["sleep_windows"],
                        "service_impact_min": st["service_impact_min"], "saved_kwh_per_impact_min": st["saved_kwh_per_impact_min"]})
    wake_costs, breakeven = wake_cost_sensitivity(replays, params)
    return {"variant": "with_ruapp", "default_savings_pct": _pct(savings, base), "slept_windows": int(slept.sum()),
            "burst_fractions": bursts, "demand_margins": margins,
            "wake_costs": wake_costs, "wake_cost_breakeven_min": breakeven,
            "note": "the default demand proxy (30-min PRB average vs the low threshold) makes the with-ruApp "
                    "variant impact-free by construction; these rows are stricter definitions, not measurements",
            "wake_cost_note": "wake-cost rows re-price the SAME replay (decisions never change): with ruApp impact = "
                              "ruApp wakes x cost, rApp-only impact = demand windows slept through x 30 min whatever the "
                              "cost; the break-even is the wake cost at which rApp-only's saved kWh per impact minute "
                              "would equal the with-ruApp score"}


def wake_cost_sensitivity(replays: dict[str, ReplayResult], params: EvalParams) -> tuple[list[dict[str, Any]], Optional[float]]:
    """Accounting-only rows: the priced service impact and the comparable score of both variants for each
    wake cost in `params.wake_cost_sensitivity_min` on the primary replay (the run's own wake energy is kept), and
    the break-even wake cost c* at which rApp-only's saved kWh per impact minute equals the with-ruApp score:
    S_w / (N c*) = S_r / (M x 30) => c* = 30 M S_w / (N S_r) with N = ruApp wakes, M = rApp-only demand windows.
    None when rApp-only can never score higher (no wakes to price, or no rApp-only saving); 0 when rApp-only
    slept through no demand at all (impact-free itself)."""
    wh = float(params.ruapp_wake_energy_wh)
    w = variant_stats(replays["with_ruapp"].windows, wake_service_cost_min=0.0, wake_energy_wh=wh)
    r = variant_stats(replays["rapp_only"].windows, wake_service_cost_min=0.0, wake_energy_wh=wh)
    n_w, n_r = int(w["ruapp_wakes"]), int(r["demand_while_asleep"])
    s_w, s_r = float(w["savings_kwh"]), float(r["savings_kwh"])
    rows = []
    for c in params.wake_cost_sensitivity_min:
        iw = impact_stats(n_w, w["demand_while_asleep"], s_w, wake_service_cost_min=float(c), wake_energy_wh=wh)
        ir = impact_stats(r["ruapp_wakes"], n_r, s_r, wake_service_cost_min=float(c), wake_energy_wh=wh)
        score_w = math.inf if iw["service_impact_min"] == 0 else iw["saved_kwh_per_impact_min"]
        score_r = math.inf if ir["service_impact_min"] == 0 else ir["saved_kwh_per_impact_min"]
        both_free = iw["service_impact_min"] == 0 and ir["service_impact_min"] == 0
        rows.append({"wake_cost_min": float(c),
                     "service_impact_min_with_ruapp": iw["service_impact_min"], "service_impact_min_rapp_only": ir["service_impact_min"],
                     "savings_kwh_with_ruapp": s_w, "savings_kwh_rapp_only": s_r,
                     "saved_kwh_per_impact_min_with_ruapp": iw["saved_kwh_per_impact_min"],
                     "saved_kwh_per_impact_min_rapp_only": ir["saved_kwh_per_impact_min"],
                     "rapp_only_scores_higher": None if both_free else bool(score_r > score_w)})
    if n_w == 0 or s_w <= 0 or s_r <= 0:
        breakeven: Optional[float] = None
    elif n_r == 0:
        breakeven = 0.0
    else:
        breakeven = WINDOW_MIN * n_r * s_w / (n_w * s_r)
    return rows, breakeven


# --------------------------------------------------------------------------------------
# 7. orchestration
# --------------------------------------------------------------------------------------
def load_table(params: EvalParams, cfg: Settings) -> pd.DataFrame:
    return load_data(params.data_kind, params.data_dir, cfg)


DATA_DIR_PLACEHOLDER = "<data-dir (path withheld)>"


def _params_for_output(params: EvalParams) -> dict[str, Any]:
    """`EvalParams` as written to report.md / summary.json. The data directory is never
    recorded, not even its last component: private KPI export directories are commonly named
    after the operator or site, and these two files are the ones that get copied around."""
    out = asdict(params)
    out["data_dir"] = None if params.data_dir is None else DATA_DIR_PLACEHOLDER
    return {k: (display_path(v) if isinstance(v, Path) else v) for k, v in out.items()}


def _notes(params: EvalParams, model: LowLoadModel, w: dict[str, Any], r: dict[str, Any]) -> list[str]:
    """Run-specific caveats (appended to the static CAVEATS in every output)."""
    notes: list[str] = []
    dp = None if params.demand_prb is None else float(params.demand_prb)
    thr = float(model.low_threshold)
    if dp is not None and not math.isclose(dp, thr):
        rel = "below" if dp < thr else "above"
        bound = ("the with-ruApp variant can sleep through windows with low_threshold <= prb < demand_prb, charged SLEEP "
                 "and not counted as impact, so the oracle is NO LONGER a bound for it and its 'zero service impact' "
                 "no longer holds" if dp > thr else
                 "windows with demand_prb <= prb < low_threshold wake the ruApp and are charged ACTIVE, so the "
                 "with-ruApp savings sit further below the oracle than with the default")
        notes.append(f"--demand-prb {dp:.2f} differs from the low threshold {thr:.2f} ({rel} it): {bound}.")
    if r["demand_while_asleep"] > 0:
        notes.append(f"rApp-only slept through {r['demand_while_asleep']} demand window(s) (service impact); its savings "
                     f"are not comparable to the with-ruApp figure without that count.")
    cost, wh = float(params.ruapp_wake_service_cost_min), float(params.ruapp_wake_energy_wh)
    if cost > 0 or wh > 0:
        notes.append(f"ruApp wakes are PRICED: {cost:g} min of degraded service and {wh:g} Wh of re-activation energy per wake "
                     f"(--ruapp-wake-service-cost-min / --ruapp-wake-energy-wh; assumptions, not measurements: the mock RU wakes in "
                     f"microseconds, a real tx-array-carrier SLEEP exit takes milliseconds to seconds). With ruApp: {w['ruapp_wakes']} "
                     f"wakes -> {w['service_impact_min']:g} min of impact and {w['ruapp_wake_kwh']:.3f} kWh added to its energy; its "
                     f"savings are no longer impact-free by construction and its 'saved kWh per impact minute' is comparable with "
                     f"rApp-only's.")
    if w["missing_kpi_windows"] > 0:
        notes.append(f"{w['missing_kpi_windows']} validation window(s) have no prb_util_dl_pct sample; {w['missing_kpi_asleep']} of "
                     f"them fell into SLEEP and were charged ACTIVE without being counted as ruApp wakes.")
    return notes


def run_evaluation(params: EvalParams, cfg: Settings = settings, df: Optional[pd.DataFrame] = None) -> EvalResult:
    t0 = time.perf_counter()
    table = load_table(params, cfg) if df is None else df
    split = split_by_time(table, params.train_days, params.validate_days)
    q0 = params.effective_low_quantile
    model = fit_model(split.train, low_quantile=params.low_quantile, seed=params.seed, data_kind=params.data_kind)
    models, forecasters = {q0: model}, {q0: Forecaster(model)}
    t_train = time.perf_counter() - t0
    classifier = validate_classifier(model, split)
    replays = replay(model, split, cfg=cfg, demand_prb=params.demand_prb, forecaster=forecasters[q0])
    agg = aggregate(replays, **params.costs)
    sens = demand_sensitivity(model, split, replays, params, cfg, forecaster=forecasters[q0])
    t_replay = time.perf_counter() - t0 - t_train
    sweep = run_sweep(params, cfg, split, models, forecasters) if params.sweep else []
    t_sweep = time.perf_counter() - t0 - t_train - t_replay
    grid = run_grid(params, cfg, split, model, forecasters[q0]) if params.grid else []
    t_grid = time.perf_counter() - t0 - t_train - t_replay - t_sweep

    w, r = agg["with_ruapp"]["total"], agg["rapp_only"]["total"]
    notes = _notes(params, model, w, r)
    proxy_default = params.demand_prb is None or math.isclose(float(params.demand_prb), float(model.low_threshold))
    priced = float(params.ruapp_wake_service_cost_min) > 0
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluation": "ocp_auru.rapp.eval: offline energy-saving replay of the es-rApp + ruApp on 30-minute KPI history",
        "power_model": POWER_MODEL,
        "params": {**_params_for_output(params),
                   "effective_low_quantile": q0,
                   "rapp_sleep_threshold": cfg.rapp_sleep_threshold, "rapp_wake_threshold": cfg.rapp_wake_threshold,
                   **chatter_settings(cfg),
                   "rapp_min_hold_s": cfg.rapp_min_hold_s, "rapp_history_len": cfg.rapp_history_len,
                   "demand_prb_effective": float(model.low_threshold) if params.demand_prb is None else float(params.demand_prb),
                   "demand_prb_is_low_threshold": params.demand_prb is None or math.isclose(float(params.demand_prb), float(model.low_threshold))},
        "split": split.as_dict(),
        "model": {k: v for k, v in model.metrics.items() if k != "trained_at"},
        "classifier_validation": classifier,
        "headline": {
            "savings_pct_with_ruapp": w["savings_pct"], "savings_pct_rapp_only": r["savings_pct"],
            "savings_pct_oracle": w["oracle_savings_pct"],
            "savings_kwh_with_ruapp": w["savings_kwh"], "savings_kwh_rapp_only": r["savings_kwh"],
            "savings_kwh_oracle": w["oracle_savings_kwh"], "baseline_kwh": w["baseline_kwh"],
            "offloaded_kwh_with_ruapp": w["offloaded_kwh"], "savings_pct_with_ruapp_net_of_offload": w["savings_pct_net_of_offload"],
            "false_sleep_rate": w["false_sleep_rate"], "ruapp_wakes": w["ruapp_wakes"],
            "demand_while_asleep_rapp_only": r["demand_while_asleep"],
            "missing_kpi_windows": w["missing_kpi_windows"], "missing_kpi_asleep_with_ruapp": w["missing_kpi_asleep"],
            "missed_opportunity_rate": w["missed_opportunity_rate"],
            "sleep_hours_per_cell_day_with_ruapp": w["sleep_hours_per_cell_day"],
            "sleep_hours_per_day_all_cells_with_ruapp": w["sleep_hours_per_day_all_cells"],
            "n_cells": int(split.validate["cell_id"].nunique()),
            # impact-free by construction only under the default demand proxy AND with the wakes unpriced (cost 0)
            "with_ruapp_impact_free_by_construction": proxy_default and not priced,
            "with_ruapp_impact_priced": priced,
            # priced service impact (accounting only): wakes x cost (with ruApp), demand windows x 30 min (rApp-only)
            "ruapp_wake_service_cost_min": float(params.ruapp_wake_service_cost_min), "ruapp_wake_energy_wh": float(params.ruapp_wake_energy_wh),
            "service_impact_min_with_ruapp": w["service_impact_min"], "service_impact_min_rapp_only": r["service_impact_min"],
            "service_impact_events_with_ruapp": w["service_impact_events"], "service_impact_events_rapp_only": r["service_impact_events"],
            "saved_kwh_per_impact_min_with_ruapp": w["saved_kwh_per_impact_min"], "saved_kwh_per_impact_min_rapp_only": r["saved_kwh_per_impact_min"],
            "ruapp_wake_kwh_with_ruapp": w["ruapp_wake_kwh"],
            "validation_auc": classifier["auc"], "train_holdout_auc": model.metrics.get("auc"),
            # policy chatter, with ruApp (the built system): episodes and CM writes over the validation days
            "sleep_episodes_with_ruapp": w["sleep_episodes"], "short_episodes_with_ruapp": w["short_episodes"],
            "mean_episode_h_with_ruapp": w["mean_episode_h"], "sleep_episodes_per_day_with_ruapp": w["sleep_episodes_per_day"],
            "actuations_with_ruapp": w["actuations"], "actuations_per_day_with_ruapp": w["actuations_per_day"],
            "actuations_per_cell_day_with_ruapp": w["actuations_per_cell_day"],
            "sleep_decisions_with_ruapp": w["sleep_decisions"], "preemptive_active_decisions_with_ruapp": w["preemptive_active_decisions"],
            "actuations_rapp_only": r["actuations"], "sleep_episodes_rapp_only": r["sleep_episodes"],
            # the governor's cost (held wakes) and the estimator divergence (root cause visibility)
            "held_windows_with_ruapp": w["held_windows"], "ruapp_wakes_while_held": w["ruapp_wakes_while_held"],
            "demand_while_asleep_while_held_rapp_only": r["demand_while_asleep_while_held"],
            "preemptive_wake_delay_windows_mean": w["preemptive_wake_delay_windows_mean"],
            "resleep_after_local_wake_with_ruapp": w["resleep_after_local_wake"],
            "false_sleep_rate_includes_held_wakes": chatter_flags_on(cfg),
            "forecast_divergent_windows": w["forecast_divergent_windows"], "forecast_divergence_rate": w["forecast_divergence_rate"],
            "forecast_divergence_mean_abs": w["forecast_divergence_mean_abs"],
            "forecast_divergent_windows_per_cell": {c: st["forecast_divergent_windows"] for c, st in agg["with_ruapp"]["per_cell"].items()},
            "short_episodes_per_cell_with_ruapp": {c: st["short_episodes"] for c, st in agg["with_ruapp"]["per_cell"].items()},
            "actuations_per_day_per_cell_with_ruapp": {c: st["actuations_per_day"] for c, st in agg["with_ruapp"]["per_cell"].items()},
            # the per-carrier-day actuation budget: carrier-days that hit the alarm, windows the budget held, the peak
            "max_actuations_per_day": int(cfg.rapp_max_actuations_per_day), "actuation_alarm_per_day": int(cfg.rapp_actuation_alarm_per_day),
            "alarm_days_with_ruapp": w["alarm_days"], "budget_reached_windows_with_ruapp": w["budget_reached_windows"],
            "max_actuations_per_carrier_day_with_ruapp": w["max_actuations_per_carrier_day"],
            "alarm_days_rapp_only": r["alarm_days"], "budget_reached_windows_rapp_only": r["budget_reached_windows"],
            "max_actuations_per_carrier_day_rapp_only": r["max_actuations_per_carrier_day"],
            "alarm_days_per_cell_with_ruapp": {c: st["alarm_days"] for c, st in agg["with_ruapp"]["per_cell"].items()},
        },
        "variants": agg,
        "demand_sensitivity": sens,
        "sweep": sweep,
        "grid": grid,
        "forecast_calls": {str(q): dict(f.calls) for q, f in forecasters.items()},
        "caveats": list(CAVEATS) + notes,
        "notes": notes,
        "timing_s": {"train": round(t_train, 3), "replay": round(t_replay, 3), "sweep": round(t_sweep, 3),
                     "grid": round(t_grid, 3), "total": round(time.perf_counter() - t0, 3)},
    }
    return EvalResult(params=params, cfg=cfg, split=split, model=model, classifier=classifier, replays=replays,
                      sweep=sweep, summary=summary)


# --------------------------------------------------------------------------------------
# 8. outputs
# --------------------------------------------------------------------------------------
def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    return o


def _f(v: Any, nd: int = 1, suffix: str = "") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{nd}f}{suffix}"


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _stats_row(label: str, s: dict[str, Any]) -> list[Any]:
    return [label, _f(s["baseline_kwh"], 2), _f(s["variant_kwh"], 2), _f(s["savings_kwh"], 2), _f(s["savings_pct"], 1, "%"),
            _f(s["offloaded_kwh"], 2), _f(s["savings_pct_net_of_offload"], 1, "%"),
            _f(s["sleep_hours_per_cell_day"], 2), s["sleep_decisions"], s["preemptive_active_decisions"], s["ruapp_wakes"],
            _f(None if s["false_sleep_rate"] is None else 100 * s["false_sleep_rate"], 1, "%"), s["demand_while_asleep"],
            s["missing_kpi_asleep"],
            s["missed_opportunities"], _f(None if s["missed_opportunity_rate"] is None else 100 * s["missed_opportunity_rate"], 1, "%"),
            s["sleep_episodes"], s["short_episodes"], _f(s["mean_episode_h"], 2), s["actuations"], _f(s["actuations_per_day"], 2),
            s["ruapp_wakes_while_held"], _f(s["preemptive_wake_delay_windows_mean"], 1), s["resleep_after_local_wake"],
            s["forecast_divergent_windows"],
            _f(s["service_impact_min"], 1), _f(s["saved_kwh_per_impact_min"], 3), _f(s["ruapp_wake_kwh"], 3),
            s["budget_reached_windows"], s["alarm_days"], s["max_actuations_per_carrier_day"]]


_STATS_HEADERS = ["scope", "baseline kWh", "variant kWh", "saved kWh", "saved %", "offloaded kWh (not subtracted)",
                  "saved % net of offload", "sleep h per cell-day", "SLEEP dec.",
                  "pre-emptive ACTIVE dec.", "ruApp wakes (false sleeps)", "false-sleep rate", "demand while asleep",
                  "missing KPI while asleep", "missed opp.", "missed rate",
                  "sleep episodes", "short episodes (<= 1 h)", "mean episode h", "actuations (CM writes)", "actuations/day",
                  "ruApp wakes while held", "mean wake delay (windows)", "re-sleeps in the local-wake window",
                  "divergent forecasts (abs(pm - profile) > 0.5)",   # no '|' inside a markdown table cell
                  "service impact min (priced)", "saved kWh per impact min", "ruApp wake kWh (re-activation, in variant kWh)",
                  "budget-held windows", "alarm carrier-days", "max actuations per carrier-day"]

_GRID_HEADERS = ["config", "profile scale", "dwell min", "confirm windows", "wake margin", "eff. wake thr", "saved % (with ruApp)",
                 "saved % (rApp-only)", "oracle %", "sleep h per cell-day", "sleep episodes", "short (<= 1 h)", "mean episode h",
                 "actuations", "actuations/day", "ruApp wakes", "false-sleep rate", "demand while asleep (rApp-only)", "missed rate",
                 "ruApp wakes while held", "demand asleep while held (rApp-only)", "mean wake delay (windows)",
                 "divergent forecasts", "worst cell: short episodes", "worst cell: actuations/day",
                 "impact min (with ruApp)", "impact min (rApp-only)", "saved kWh/impact min (with ruApp / rApp-only)",
                 "budget-held windows", "alarm carrier-days", "max actuations per carrier-day"]

_GRID_CELL_HEADERS = ["config", "cell", "saved %", "sleep episodes", "short (<= 1 h)", "actuations", "actuations/day", "ruApp wakes",
                      "ruApp wakes while held", "false-sleep rate", "demand while asleep (rApp-only)", "divergent forecasts",
                      "impact min (with ruApp)", "impact min (rApp-only)",
                      "budget-held windows", "alarm carrier-days", "max actuations per carrier-day"]


def _grid_config_label(g: dict[str, Any]) -> str:
    if g.get("off_grid"):
        return g["label"] + " (primary, off-grid)"
    return g["label"] + (" (primary)" if g["primary"] else "")


def _grid_row(g: dict[str, Any]) -> list[Any]:
    fsr = _f(None if g["false_sleep_rate"] is None else 100 * g["false_sleep_rate"], 1, "%")
    if g.get("false_sleep_rate_includes_held_wakes"):
        fsr += " (incl. held wakes)"
    return [_grid_config_label(g), _f(g.get("profile_scale", 0.0), 2), _f(g["min_sleep_dwell_min"], 0), g["wake_confirm_windows"],
            _f(g["wake_margin"], 2), _f(g["wake_threshold_effective"], 2), _f(g["savings_pct_with_ruapp"], 1, "%"),
            _f(g["savings_pct_rapp_only"], 1, "%"), _f(g["oracle_savings_pct"], 1, "%"), _f(g["sleep_hours_per_cell_day"], 2),
            g["sleep_episodes"], g["short_episodes"], _f(g["mean_episode_h"], 2), g["actuations"], _f(g["actuations_per_day"], 2),
            g["ruapp_wakes"], fsr,
            g["demand_while_asleep_rapp_only"], _f(None if g["missed_opportunity_rate"] is None else 100 * g["missed_opportunity_rate"], 1, "%"),
            g["ruapp_wakes_while_held"], g["demand_while_asleep_while_held_rapp_only"], _f(g["preemptive_wake_delay_windows_mean"], 1),
            g["forecast_divergent_windows"],
            f"{g['worst_cell_short_episodes']}: {g['max_cell_short_episodes']}",
            f"{g['worst_cell_actuations_per_day']}: {_f(g['max_cell_actuations_per_day'], 2)}",
            _f(g["service_impact_min_with_ruapp"], 1), _f(g["service_impact_min_rapp_only"], 1),
            f"{_f(g['saved_kwh_per_impact_min_with_ruapp'], 3)} / {_f(g['saved_kwh_per_impact_min_rapp_only'], 3)}",
            g.get("budget_reached_windows", 0), g.get("alarm_days", 0), g.get("max_actuations_per_carrier_day", 0)]


def _grid_cell_rows(g: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for cell, c in g["per_cell"].items():
        rows.append([_grid_config_label(g), cell, _f(c["savings_pct"], 1, "%"), c["sleep_episodes"], c["short_episodes"], c["actuations"],
                     _f(c["actuations_per_day"], 2), c["ruapp_wakes"], c["ruapp_wakes_while_held"],
                     _f(None if c["false_sleep_rate"] is None else 100 * c["false_sleep_rate"], 1, "%"),
                     c["demand_while_asleep_rapp_only"], c["forecast_divergent_windows"],
                     _f(c["service_impact_min"], 1), _f(c["service_impact_min_rapp_only"], 1),
                     c.get("budget_reached_windows", 0), c.get("alarm_days", 0), c.get("max_actuations_per_carrier_day", 0)])
    return rows


def write_grid_report(result: EvalResult, out_dir: Path, plot_name: Optional[str]) -> Path:
    """grid.md: the chatter-control matrix (with-ruApp savings vs episodes/actuations per configuration,
    the governor's cost, the estimator divergence, and every row per cell)."""
    s = result.summary
    p = s["params"]
    L: list[str] = ["# Chatter-control grid: profile level scale x sleep dwell x wake-confirmation windows x wake margin\n",
                    f"Generated {s['generated_at']} by `ocp_auru.rapp.eval` on the PRIMARY model (low quantile "
                    f"{p['effective_low_quantile']}, sleep threshold {p['rapp_sleep_threshold']}, wake threshold "
                    f"{p['rapp_wake_threshold']}); the primary run's own settings are profile scale {p['rapp_profile_scale']:g}, "
                    f"dwell {p['rapp_min_sleep_dwell_min']:g} min, {p['rapp_wake_confirm_windows']} confirmation window(s), "
                    f"margin {p['rapp_wake_margin']:g}, post-local-wake hold {p['rapp_post_local_wake_hold_min']:g} min, "
                    f"actuation budget {p.get('rapp_max_actuations_per_day', 0)} per carrier-day (alarm at "
                    f"{p.get('rapp_actuation_alarm_per_day', 0)}; 0 = off). The budget is not a grid axis: only the primary row "
                    f"carries the run's cap; every other row replays with the budget OFF (alarm kept as a metric) so the matrix "
                    f"compares the chatter controls alone - see the 'budget-held windows' column (always 0 off the primary row) "
                    f"and `summary.json` -> `grid[i].max_actuations_per_day`.\n",
                    "> Every row replays both variants through the real `policy_engine.decide` on the same validation days "
                    "as the primary run (`RAPP_PROFILE_SCALE`, `RAPP_MIN_SLEEP_DWELL_MIN`, `RAPP_WAKE_CONFIRM_WINDOWS`, "
                    "`RAPP_WAKE_MARGIN` in `config.py`). **It shows the savings-vs-writes trade-off; it does not select a "
                    "setting.** kWh figures are from the ILLUSTRATIVE power model. A sleep episode is a maximal run of "
                    "SLEEP-charged windows of one cell; short = <= 2 windows (1 h); actuations = SLEEP + ACTIVE decisions "
                    "(CM writes). Dwell counts simulated/data time.\n",
                    "> **Read the rows per cell, not on totals alone.** The dwell / confirmation-window / margin rows are "
                    "RATE LIMITING of the symptom: while a wake is held back, a demand window is served by the ruApp "
                    "('ruApp wakes while held', the false-sleep rate then INCLUDES them) or, rApp-only, slept through "
                    "('demand asleep while held'), and the pre-emptive ACTIVE arrives 'mean wake delay' windows later. The "
                    "root cause of the chatter is the estate-pooled hour-of-day profile ('divergent forecasts' = windows "
                    "where |p_low(pm-window) - p_low(profile)| > 0.5); the `profile scale` 1 rows forecast a sleeping "
                    "carrier from the profile at its OWN level (config.py `RAPP_PROFILE_SCALE`, model.py "
                    "`profile_level_ratio`). Compare a profile-scale row against its scale-0 twin before any governor "
                    "default is flipped; 'worst cell' names the cell with the most short episodes / actuations per day.\n"]
    L.append(_md_table(_GRID_HEADERS, [_grid_row(g) for g in s["grid"]]))
    L.append("\n## Per cell\n")
    L.append(_md_table(_GRID_CELL_HEADERS, [row for g in s["grid"] for row in _grid_cell_rows(g)]))
    if plot_name:
        L.append(f"\n- `{plot_name}`: savings % (with ruApp) vs actuations per day, one annotated point per configuration\n")
    L.append("- `summary.json` -> `grid`: the same rows, machine-readable (`per_cell` per row)\n")
    path = out_dir / "grid.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def write_report(result: EvalResult, out_dir: Path, plot_files: dict[str, str]) -> Path:
    s = result.summary
    p, h, clf, m = s["params"], s["headline"], s["classifier_validation"], s["model"]
    v = s["variants"]
    L: list[str] = []
    L.append("# Energy-saving evaluation: es-rApp + ruApp on 30-minute KPI history\n")
    L.append(f"Generated {s['generated_at']} by `ocp_auru.rapp.eval`. Data kind `{p['data_kind']}`, cells "
             f"{', '.join(s['split']['cells'])} (adapter labels, never site identifiers).\n")
    L.append("> **All kWh figures are from an ILLUSTRATIVE power model** (mock O-RU constants below), never "
             "measured. **The KPI export is per SITE, not per carrier**: the replay treats each site's series "
             "as the carrier under control and assumes residual low load is carried by the remaining layer(s); "
             "the coverage question is the physics twin's job (ADR-7), not this evaluation's.\n")
    L.append("## Headline\n")
    cost, wh = float(h.get("ruapp_wake_service_cost_min", 0.0)), float(h.get("ruapp_wake_energy_wh", 0.0))
    if h["with_ruapp_impact_free_by_construction"]:
        impact_note = ("zero service impact BY CONSTRUCTION under the 30-min PRB-average demand proxy; upper bound, "
                       "see caveats and the demand-definition sensitivity")
    elif h.get("with_ruapp_impact_priced"):
        impact_note = (f"service impact PRICED at {cost:g} min per ruApp wake: {h['ruapp_wakes']} wakes = "
                       f"{_f(h['service_impact_min_with_ruapp'], 1)} min, comparable with rApp-only; see the run notes")
    else:
        impact_note = "demand threshold differs from the low threshold; see the run notes"
    if wh > 0:
        impact_note += f"; incl. {h['ruapp_wakes']} wakes x {wh:g} Wh = {_f(h['ruapp_wake_kwh_with_ruapp'], 3)} kWh of re-activation energy"
    L.append(_md_table(["metric", "value"], [
        ["savings, with ruApp", f"{_f(h['savings_pct_with_ruapp'], 1, '%')} ({_f(h['savings_kwh_with_ruapp'], 2)} of {_f(h['baseline_kwh'], 2)} kWh; {impact_note})"],
        ["savings, rApp-only (no local wake)", f"{_f(h['savings_pct_rapp_only'], 1, '%')} ({_f(h['savings_kwh_rapp_only'], 2)} kWh; service impact: {h['demand_while_asleep_rapp_only']} windows slept through demand)"],
        ["savings, oracle (perfect foresight)", f"{_f(h['savings_pct_oracle'], 1, '%')} ({_f(h['savings_kwh_oracle'], 2)} kWh)"],
        ["service impact, PRICED (with ruApp: wakes x cost; rApp-only: demand windows slept through x 30 min)",
         f"with ruApp {_f(h['service_impact_min_with_ruapp'], 1)} min ({h['ruapp_wakes']} wakes x {cost:g} min), rApp-only "
         f"{_f(h['service_impact_min_rapp_only'], 1)} min ({h['demand_while_asleep_rapp_only']} windows x {WINDOW_MIN:g} min); "
         f"--ruapp-wake-service-cost-min {cost:g}, --ruapp-wake-energy-wh {wh:g}"],
        ["saved kWh per impact minute (the comparable score; n/a while the impact is 0)",
         f"with ruApp {_f(h['saved_kwh_per_impact_min_with_ruapp'], 3)}, rApp-only {_f(h['saved_kwh_per_impact_min_rapp_only'], 3)}"],
        ["offloaded residual-load energy while asleep, with ruApp (not subtracted)", f"{_f(h['offloaded_kwh_with_ruapp'], 2)} kWh (savings net of offload {_f(h['savings_pct_with_ruapp_net_of_offload'], 1, '%')})"],
        ["false-sleep rate (ruApp wakes / SLEEP windows entered with a KPI sample)", _f(None if h['false_sleep_rate'] is None else 100 * h['false_sleep_rate'], 1, '%') + f" ({h['ruapp_wakes']} wakes; = forecast error on slept windows, not measured impact)"],
        ["missed-opportunity rate (ACTIVE in actually-low windows)", _f(None if h['missed_opportunity_rate'] is None else 100 * h['missed_opportunity_rate'], 1, '%')],
        ["sleep hours per cell-day, with ruApp", f"{_f(h['sleep_hours_per_cell_day_with_ruapp'], 2)} ({_f(h['sleep_hours_per_day_all_cells_with_ruapp'], 2)} h/day summed over {h['n_cells']} cells)"],
        ["windows with a missing KPI sample", f"{h['missing_kpi_windows']} ({h['missing_kpi_asleep_with_ruapp']} while asleep, charged ACTIVE, not counted as wakes)"],
        ["classifier AUC on the validation days", _f(h['validation_auc'], 3)],
        ["classifier AUC on the rApp's own train holdout", _f(h['train_holdout_auc'], 3)],
        ["sleep episodes, with ruApp (policy chatter)", f"{h['sleep_episodes_with_ruapp']} ({h['short_episodes_with_ruapp']} short, <= 1 h; "
                                                        f"mean {_f(h['mean_episode_h_with_ruapp'], 2)} h; {_f(h['sleep_episodes_per_day_with_ruapp'], 2)} per day over all cells)"],
        ["actuations, with ruApp (SLEEP + pre-emptive ACTIVE decisions = CM writes)", f"{h['actuations_with_ruapp']} ({h['sleep_decisions_with_ruapp']} SLEEP + "
                                                                                      f"{h['preemptive_active_decisions_with_ruapp']} ACTIVE; {_f(h['actuations_per_day_with_ruapp'], 2)} per day over all cells, "
                                                                                      f"{_f(h['actuations_per_cell_day_with_ruapp'], 2)} per cell-day)"],
        ["short episodes / actuations per day, PER CELL (chatter is judged per cell)",
         "; ".join(f"{c}: {h['short_episodes_per_cell_with_ruapp'][c]} short, {_f(h['actuations_per_day_per_cell_with_ruapp'][c], 2)}/day"
                   for c in sorted(h["short_episodes_per_cell_with_ruapp"]))],
        ["chatter-control cost (windows entered while a wake was held back)",
         f"{h['held_windows_with_ruapp']} held windows; {h['ruapp_wakes_while_held']} ruApp wakes while held"
         + (" (the false-sleep rate above INCLUDES them)" if h["false_sleep_rate_includes_held_wakes"] else "")
         + f"; rApp-only slept through {h['demand_while_asleep_while_held_rapp_only']} demand windows while held; "
         f"mean pre-emptive wake delay {_f(h['preemptive_wake_delay_windows_mean'], 1)} windows; "
         f"{h['resleep_after_local_wake_with_ruapp']} SLEEP decisions in the window the ruApp woke"],
        ["forecast estimators side by side (root cause: estate-pooled profile vs own PM)",
         f"{h['forecast_divergent_windows']} windows with |p_low(pm-window) - p_low(profile)| > {FORECAST_DIVERGENCE_FLAG} "
         f"({_f(None if h['forecast_divergence_rate'] is None else 100 * h['forecast_divergence_rate'], 1, '%')}; mean |delta| "
         f"{_f(h['forecast_divergence_mean_abs'], 3)}); per cell "
         + ", ".join(f"{c}: {n}" for c, n in sorted(h["forecast_divergent_windows_per_cell"].items()))
         + f"; profile scale {p['rapp_profile_scale']:g}"],
        ["actuation budget (max applied actuations per carrier-day; 0 = off)",
         f"{h['alarm_days_with_ruapp']} carrier-days hit the alarm, {h['budget_reached_windows_with_ruapp']} windows held by the budget "
         f"(with ruApp; rApp-only {h['alarm_days_rapp_only']} / {h['budget_reached_windows_rapp_only']}); budget "
         f"{h['max_actuations_per_day']}, alarm at {h['actuation_alarm_per_day']}; peak {h['max_actuations_per_carrier_day_with_ruapp']} "
         f"actuations in one carrier-day; alarm carrier-days per cell "
         + ", ".join(f"{c}: {n}" for c, n in sorted(h["alarm_days_per_cell_with_ruapp"].items()))],
    ]))
    if s.get("notes"):
        L.append("\nRun notes:\n")
        L += [f"- {n}" for n in s["notes"]]
    L.append("\n## Parameters\n")
    L.append("`data_dir` is never recorded (a private export directory may be named after an operator or site).\n")
    L.append(_md_table(["parameter", "value"], [[k, p[k]] for k in sorted(p)]))
    L.append("\n## Data and split (by time)\n")
    sp = s["split"]
    L.append(f"Train: `{sp['train_start']}` .. `{sp['train_end']}` ({sp['n_train_rows']} rows). "
             f"Validate: `{sp['validate_start']}` .. `{sp['validate_end']}` ({sp['n_validate_rows']} rows). "
             f"Assertion max(train ts) < min(validate ts) holds; later windows are dropped.\n")
    L.append(_md_table(["cell", "train start", "train end", "n train", "validate start", "validate end", "n validate", "windows/validate day"],
                       [[c, b["train_start"], b["train_end"], b["n_train"], b["validate_start"], b["validate_end"], b["n_validate"],
                         f"{b['validate_windows_per_day_min']}..{b['validate_windows_per_day_max']}"] for c, b in sp["per_cell"].items()]))
    L.append("\n## Model (trained on the train frame only, as the rApp does)\n")
    L.append(_md_table(["metric", "value"], [[k, (_f(val, 4) if isinstance(val, float) else val)] for k, val in m.items()]))
    L.append("\nThe low-load threshold and the hour-of-day profile come from the train frame. The rApp's own "
             "metrics above use its internal time holdout (last 20% of the train frame).\n")
    L.append("## Classifier validation on the held-out days\n")
    L.append(_md_table(["metric", "value"], [[k, (_f(val, 4) if isinstance(val, float) else val)] for k, val in clf.items()]))
    L.append("\n## Totals per variant\n")
    L.append("Baseline = always ACTIVE. Oracle = SLEEP exactly in the windows that turned out low-load (upper bound "
             "without service impact while the demand threshold equals the low threshold). With the default demand "
             "definition the with-ruApp variant sleeps only through windows that are low by definition (zero service "
             "impact BY CONSTRUCTION; a window the ruApp wakes into is charged ACTIVE in full), so its savings are an "
             "upper bound under the 30-min PRB-average proxy. rApp-only can exceed the oracle only by sleeping through "
             "demand ('demand while asleep' = service impact). 'Offloaded kWh' is the energy of the residual load the "
             "remaining layer(s) carry while this series sleeps; it is reported, not subtracted. "
             "'Sleep h per cell-day' is per cell, not summed over cells. 'Service impact min' is PRICED, not measured: "
             "each ruApp wake is booked as `--ruapp-wake-service-cost-min` minutes of degraded service (the first UE(s) wait "
             "for the RU re-activation; the mock RU wakes in microseconds, a real tx-array-carrier SLEEP exit takes "
             "milliseconds to seconds) and each rApp-only demand-while-asleep window as the whole window (30 min); "
             "`--ruapp-wake-energy-wh` per wake is added to the with-ruApp kWh ('ruApp wake kWh'). With the default cost 0 "
             "the with-ruApp variant is impact-free BY CONSTRUCTION and 'saved kWh per impact min' is n/a for it; with a "
             "cost > 0 that score makes the two variants comparable "
             f"(this run: cost {float(h.get('ruapp_wake_service_cost_min', 0.0)):g} min, energy "
             f"{float(h.get('ruapp_wake_energy_wh', 0.0)):g} Wh per wake).\n")
    oracle_total = {**v["with_ruapp"]["total"]}
    oracle_row = ["oracle (perfect foresight)", _f(oracle_total["baseline_kwh"], 2), _f(oracle_total["oracle_kwh"], 2),
                  _f(oracle_total["oracle_savings_kwh"], 2), _f(oracle_total["oracle_savings_pct"], 1, "%"),
                  _f(oracle_total["oracle_offloaded_kwh"], 2), _f(oracle_total["oracle_savings_pct_net_of_offload"], 1, "%"),
                  _f(oracle_total["oracle_sleep_hours_per_cell_day"], 2), "-", "-", "-", "-", 0, "-", 0, "0.0%", "-", "-", "-", "-", "-",
                  "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    L.append(_md_table(_STATS_HEADERS, [_stats_row(VARIANT_TITLES["with_ruapp"], v["with_ruapp"]["total"]),
                                        _stats_row(VARIANT_TITLES["rapp_only"], v["rapp_only"]["total"]), oracle_row]))
    ds = s.get("demand_sensitivity")
    if ds:
        L.append("\n### Demand-definition sensitivity (with ruApp)\n")
        L.append(f"{ds['note']}. Default savings {_f(ds['default_savings_pct'], 1, '%')} over {ds['slept_windows']} SLEEP-charged windows.\n")
        if ds["burst_fractions"]:
            L.append("If a share of the slept windows contained a PRACH burst above the ruApp's energy threshold "
                     "(invisible to the 30-min average), the live ruApp would wake and those windows would cost ACTIVE:\n")
            L.append(_md_table(["burst fraction of slept windows", "windows charged ACTIVE", "saved kWh", "saved %"],
                               [[f"{b['burst_fraction']:.0%}", b["slept_windows_charged_active"], _f(b["savings_kwh"], 2),
                                 _f(b["savings_pct"], 1, "%")] for b in ds["burst_fractions"]]))
        if ds["demand_margins"]:
            L.append("\nWith the demand threshold a margin below the low threshold (a replay each; windows with "
                     "margin x thr <= prb < thr wake the ruApp and are charged ACTIVE):\n")
            L.append(_md_table(["demand margin", "demand thr (PRB %)", "saved kWh", "saved %", "ruApp wakes", "false-sleep rate", "SLEEP windows"],
                               [[f"{m['demand_margin']:.2f} x low thr", _f(m["demand_prb"], 2), _f(m["savings_kwh"], 2), _f(m["savings_pct"], 1, "%"),
                                 m["ruapp_wakes"], _f(None if m["false_sleep_rate"] is None else 100 * m["false_sleep_rate"], 1, "%"),
                                 m["sleep_windows"]] for m in ds["demand_margins"]]))
        if ds.get("wake_costs"):
            be = ds.get("wake_cost_breakeven_min")
            be_text = (f"rApp-only would score higher than with-ruApp from a wake cost of {be:.1f} min per wake" if be is not None
                       else "rApp-only can never score higher than with-ruApp on this replay")
            L.append(f"\nWake-cost rows (pure accounting on the same replay; decisions never change): {ds.get('wake_cost_note', '')}. "
                     f"{be_text}.\n")
            L.append(_md_table(["wake cost (min per ruApp wake)", "impact min (with ruApp)", "impact min (rApp-only)", "saved kWh (with ruApp)",
                                "saved kWh (rApp-only)", "saved kWh per impact min (with ruApp)", "saved kWh per impact min (rApp-only)",
                                "rApp-only scores higher"],
                               [[f"{r['wake_cost_min']:g}", _f(r["service_impact_min_with_ruapp"], 1), _f(r["service_impact_min_rapp_only"], 1),
                                 _f(r["savings_kwh_with_ruapp"], 2), _f(r["savings_kwh_rapp_only"], 2),
                                 _f(r["saved_kwh_per_impact_min_with_ruapp"], 3), _f(r["saved_kwh_per_impact_min_rapp_only"], 3),
                                 "n/a" if r["rapp_only_scores_higher"] is None else ("yes" if r["rapp_only_scores_higher"] else "no")]
                                for r in ds["wake_costs"]]))
    for variant in VARIANTS:
        L.append(f"\n## Per cell: {VARIANT_TITLES[variant]}\n")
        L.append(_md_table(_STATS_HEADERS, [_stats_row(cell, st) for cell, st in v[variant]["per_cell"].items()]))
        L.append(f"\n## Per day: {VARIANT_TITLES[variant]}\n")
        L.append(_md_table(["day", "baseline kWh", "variant kWh", "saved kWh", "saved %", "oracle kWh", "oracle saved %",
                            "sleep h per cell-day", "ruApp wakes", "demand while asleep", "missing KPI while asleep", "missed opp.",
                            "sleep episodes (cut at midnight)", "short (<= 1 h)", "mean episode h", "actuations",
                            "budget-held windows", "alarm carrier-days", "max actuations per carrier-day", "service impact min (priced)"],
                           [[d["day"], _f(d["baseline_kwh"], 2), _f(d["variant_kwh"], 2), _f(d["savings_kwh"], 2), _f(d["savings_pct"], 1, "%"),
                             _f(d["oracle_kwh"], 2), _f(d["oracle_savings_pct"], 1, "%"), _f(d["sleep_hours_per_cell_day"], 2),
                             d["ruapp_wakes"], d["demand_while_asleep"], d["missing_kpi_asleep"], d["missed_opportunities"],
                             d["sleep_episodes"], d["short_episodes"], _f(d["mean_episode_h"], 2), d["actuations"],
                             d["budget_reached_windows"], d["alarm_days"], d["max_actuations_per_carrier_day"],
                             _f(d["service_impact_min"], 1)] for d in v[variant]["per_day"]]))
        L.append(f"\nDecisions by action:reason ({VARIANT_TITLES[variant]}):\n")
        L.append(_md_table(["action:reason", "count"], [[k, n] for k, n in v[variant]["decisions_by_reason"].items()]))
    L.append("\n## Sensitivity sweep: sleep threshold x low-load quantile\n")
    if s["sweep"]:
        L.append("Each low-load quantile retrains the model on the train frame (the label changes); each sleep "
                 "threshold replays both variants. The primary run above is separate. **Sensitivity only: all rows "
                 "are scored on the same validation days; do not select a threshold from this table and quote its "
                 "savings.** The low-load quantile changes the DEFINITION of 'low' and therefore of 'no service "
                 "impact' (at q=0.5 sleeping through median-load windows counts as zero impact); the 'oracle %' "
                 "column grows with q for that reason, not because the model improved.\n")
        L.append(_md_table(["sleep thr", "low quantile", "low thr (PRB %)", "val AUC", "saved % (with ruApp)", "saved % (rApp-only)",
                            "oracle %", "SLEEP dec.", "false-sleep rate", "demand while asleep (rApp-only)", "missed rate", "sleep h per cell-day",
                            "sleep episodes", "short (<= 1 h)", "mean episode h", "actuations", "actuations/day",
                            "impact min (with ruApp / rApp-only)", "saved kWh per impact min (with ruApp / rApp-only)",
                            "budget-held windows", "alarm carrier-days"],
                           [[r["sleep_threshold"], r["low_quantile"], _f(r["low_threshold"], 2), _f(r["validation_auc"], 3),
                             _f(r["savings_pct_with_ruapp"], 1, "%"), _f(r["savings_pct_rapp_only"], 1, "%"), _f(r["oracle_savings_pct"], 1, "%"),
                             r["sleep_decisions"], _f(None if r["false_sleep_rate"] is None else 100 * r["false_sleep_rate"], 1, "%"),
                             r["demand_while_asleep_rapp_only"],
                             _f(None if r["missed_opportunity_rate"] is None else 100 * r["missed_opportunity_rate"], 1, "%"),
                             _f(r["sleep_hours_per_cell_day"], 2),
                             r["sleep_episodes"], r["short_episodes"], _f(r["mean_episode_h"], 2), r["actuations"], _f(r["actuations_per_day"], 2),
                             f"{_f(r['service_impact_min_with_ruapp'], 1)} / {_f(r['service_impact_min_rapp_only'], 1)}",
                             f"{_f(r['saved_kwh_per_impact_min_with_ruapp'], 3)} / {_f(r['saved_kwh_per_impact_min_rapp_only'], 3)}",
                             r.get("budget_reached_windows", 0), r.get("alarm_days", 0)] for r in s["sweep"]]))
    else:
        L.append("Sweep disabled (`--no-sweep`).\n")
    L.append("\n## Chatter control: profile level scale x sleep dwell x wake-confirmation windows x wake margin\n")
    if s.get("grid"):
        L.append(f"{len(s['grid'])} configurations replayed on the primary model; the table, its per-cell rows and its plot are in "
                 "`grid.md` / `grid.png` (summary.json -> `grid`). Scored on the same validation days as the primary run: a "
                 "trade-off view, not a setting selection. Dwell / confirmation / margin rows rate-limit the symptom (held wakes "
                 "move to the ruApp); the profile-scale rows address the root cause (estate-pooled profile). The actuation "
                 "budget is not an axis: only the primary row carries the run's cap, every other row replays with it OFF.\n")
        L.append(_md_table(_GRID_HEADERS, [_grid_row(g) for g in s["grid"]]))
    else:
        L.append("Grid disabled (run with `--grid`). The primary run's settings: profile scale "
                 f"{p['rapp_profile_scale']:g}, dwell {p['rapp_min_sleep_dwell_min']:g} min, {p['rapp_wake_confirm_windows']} "
                 f"confirmation window(s), margin {p['rapp_wake_margin']:g} (effective wake threshold "
                 f"{_f(p['rapp_wake_threshold_effective'], 2)}), post-local-wake hold {p['rapp_post_local_wake_hold_min']:g} min.\n")
    L.append("\n## Power model (ILLUSTRATIVE)\n")
    L.append(_md_table(["constant", "value"], [[k, val] for k, val in POWER_MODEL.items()]))
    L.append("\nEnergy per window = `power_w(active, prb_util_dl_pct) x 0.5 h` from `ocp_auru.mock_ru.radio` (single source of truth).\n")
    L.append("## Caveats\n")
    L += [f"- {c}" for c in s["caveats"]]
    L.append("\n## Files\n")
    L += [f"- `{name}`: {desc}" for name, desc in plot_files.items()]
    L.append("- `windows_*.csv` columns `p_low_pm_window`, `p_low_profile`, `forecast_divergence`, `held`, "
             "`preemptive_wake_delay_windows`, `resleep_after_local_wake`: the two estimators at every window and the "
             "governor's cost attribution (`held` includes windows after a budget-refused pre-emptive wake); "
             "`actuations_today_before` (the count the decision saw, = /decisions `actuations_today`), "
             "`actuations_today_after` / `alarm_after` (after the window's applied write), `budget_held`")
    if s.get("grid"):
        L.append("- `grid.md`: the chatter-control grid (profile level scale x sleep dwell x wake-confirmation windows x wake "
                 "margin) as a table, with per-cell rows")
    L.append("- `summary.json`: everything above, machine-readable")
    L.append("- `windows_with_ruapp.csv`, `windows_rapp_only.csv`: one row per (cell, window): states, forecast, decision, "
             "ruApp wake, missing-KPI flag, energy of baseline / variant / oracle and of the offloaded residual load\n")
    path = out_dir / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


# ---- plots (matplotlib, Agg) ----------------------------------------------------------------
_INK, _INK2, _GRID = "#0b0b0b", "#52514e", "#e5e4e0"
_BLUE, _ORANGE, _AQUA, _RED, _VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#4a3aa7"


def _style(ax) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_INK2, labelsize=8)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _spans(win: pd.DataFrame, mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Merge consecutive masked windows into (start, end) spans."""
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = prev = None
    for ts, on in zip(win["ts"], mask):
        if on:
            if start is None:
                start = ts
            prev = ts
        elif start is not None:
            spans.append((start, prev + pd.Timedelta(minutes=30)))
            start = None
    if start is not None:
        spans.append((start, prev + pd.Timedelta(minutes=30)))
    return spans


def write_plots(result: EvalResult, out_dir: Path) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.size": 9, "text.color": _INK, "axes.labelcolor": _INK2, "axes.titlecolor": _INK,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    files: dict[str, str] = {}
    thr = float(result.model.low_threshold)
    win_all = result.replays["with_ruapp"].windows
    sleep_thr = result.cfg.rapp_sleep_threshold

    # timeline per cell (with-ruApp variant)
    for cell, win in win_all.groupby("cell", sort=True):
        fig, ax = plt.subplots(figsize=(14, 4.2), dpi=110)
        _style(ax)
        for a, b in _spans(win, win["state"] == "SLEEP"):
            ax.axvspan(a, b, color=_BLUE, alpha=0.12, linewidth=0)
        ax.plot(win["ts"], win["prb_util_dl_pct"], color=_INK, linewidth=1.4, label="prb_util_dl_pct")
        ax.axhline(thr, color=_ORANGE, linestyle="--", linewidth=1.2)
        ax.annotate(f"low threshold {thr:.1f}%", xy=(win["ts"].iloc[0], thr), xytext=(2, 3), textcoords="offset points",
                    color=_ORANGE, fontsize=8)
        wk = win.loc[win["ruapp_wake"]]
        pe = win.loc[win["preemptive_wake"]]
        ax.scatter(wk["ts"], wk["prb_util_dl_pct"], marker="v", s=48, color=_RED, zorder=5, edgecolor="white", linewidth=0.8)
        ax.scatter(pe["ts"], pe["prb_util_dl_pct"], marker="^", s=48, color=_AQUA, zorder=5, edgecolor="white", linewidth=0.8)
        st = result.summary["variants"]["with_ruapp"]["per_cell"][cell]
        ax.set_title(f"{cell}: validation days, with ruApp  (saved {_f(st['savings_pct'], 1, '%')} of an ILLUSTRATIVE "
                     f"baseline, {st['ruapp_wakes']} ruApp wakes, {st['preemptive_active_decisions']} pre-emptive wakes; "
                     f"sleep thr {sleep_thr:.2f})", fontsize=9.5, loc="left")
        ax.set_ylabel("DL PRB utilisation (%)")
        ax.set_ylim(bottom=0)
        # tz-aware tables: label the axis in the table's own zone (matplotlib would show UTC); no-op for naive data
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d\n%H:%M", tz=getattr(win["ts"].dt, "tz", None)))
        ax.legend(handles=[Line2D([], [], color=_INK, linewidth=1.4, label="prb_util_dl_pct"),
                           Line2D([], [], color=_ORANGE, linestyle="--", label="low-load threshold (train quantile)"),
                           Patch(facecolor=_BLUE, alpha=0.25, label="carrier SLEEP (charged 60 W, illustrative)"),
                           Line2D([], [], marker="v", color=_RED, linestyle="", markersize=7, label="ruApp wake on PRACH (false sleep)"),
                           Line2D([], [], marker="^", color=_AQUA, linestyle="", markersize=7, label="rApp pre-emptive wake")],
                  loc="upper left", fontsize=8, frameon=False, ncol=5, bbox_to_anchor=(0, -0.18))
        fig.tight_layout()
        name = f"timeline_{cell}.png"
        fig.savefig(out_dir / name)
        plt.close(fig)
        files[name] = f"{cell} validation timeline: PRB line, low threshold, SLEEP windows shaded, ruApp / pre-emptive wake markers"

    # energy per day (summed over cells): baseline vs with-ruApp vs oracle
    days = result.summary["variants"]["with_ruapp"]["per_day"]
    labels = [d["day"] for d in days]
    x = np.arange(len(labels))
    wbar = 0.26
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(labels) + 3), 4.2), dpi=110)
    _style(ax)
    ax.bar(x - wbar, [d["baseline_kwh"] for d in days], wbar - 0.03, color=_INK2, label="baseline (always ACTIVE)")
    bars = ax.bar(x, [d["variant_kwh"] for d in days], wbar - 0.03, color=_BLUE, label="with ruApp")
    ax.bar(x + wbar, [d["oracle_kwh"] for d in days], wbar - 0.03, color=_AQUA, label="oracle (perfect foresight)")
    for b, d in zip(bars, days):
        ax.annotate(f"-{_f(d['savings_pct'], 1, '%')}", xy=(b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8, color=_INK)
    ax.set_xticks(x, labels)
    ax.set_ylabel("energy per day, all cells (kWh, ILLUSTRATIVE model)")
    ax.set_title("Energy per validation day: baseline vs with-ruApp vs oracle (savings % annotated)", loc="left", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=3, bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    fig.savefig(out_dir / "energy_daily.png")
    plt.close(fig)
    files["energy_daily.png"] = "grouped bars per validation day: baseline vs with-ruApp vs oracle, savings % annotated"

    # priced service impact per variant (bars, savings % annotated) and the comparable score vs the wake cost
    h = result.summary["headline"]
    ds = result.summary.get("demand_sensitivity") or {}
    wake_rows = ds.get("wake_costs") or []
    cost = float(h.get("ruapp_wake_service_cost_min", 0.0))
    fig, (axb, axc) = plt.subplots(1, 2, figsize=(10.5, 3.9), dpi=110, gridspec_kw={"width_ratios": [1.0, 1.35]})
    for ax in (axb, axc):
        _style(ax)
    impact = [float(h["service_impact_min_with_ruapp"]), float(h["service_impact_min_rapp_only"])]
    bars = axb.bar(["with ruApp", "rApp-only"], impact, 0.55, color=[_BLUE, _ORANGE])
    notes = [f"saved {_f(h['savings_pct_with_ruapp'], 1, '%')}\n{h['ruapp_wakes']} wakes x {cost:g} min\n"
             f"{_f(h['saved_kwh_per_impact_min_with_ruapp'], 3)} kWh/impact-min",
             f"saved {_f(h['savings_pct_rapp_only'], 1, '%')}\n{h['demand_while_asleep_rapp_only']} windows x {WINDOW_MIN:g} min\n"
             f"{_f(h['saved_kwh_per_impact_min_rapp_only'], 3)} kWh/impact-min"]
    for b, text in zip(bars, notes):
        axb.annotate(text, xy=(b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=7.5, color=_INK)
    axb.set_ylim(0, max(impact + [1.0]) * 1.7)
    axb.set_ylabel("service impact (min, PRICED, not measured)")
    axb.set_title(f"Service impact per variant, wake cost {cost:g} min\n(savings % annotated; PRICED, not measured)", loc="left", fontsize=9)
    if wake_rows:
        xs = [float(r["wake_cost_min"]) for r in wake_rows]
        yw = [r["saved_kwh_per_impact_min_with_ruapp"] for r in wake_rows]
        yr = [r["saved_kwh_per_impact_min_rapp_only"] for r in wake_rows]
        axc.plot([x_ for x_, y_ in zip(xs, yw) if y_ is not None], [y_ for y_ in yw if y_ is not None], marker="o", markersize=5,
                 linewidth=2, color=_BLUE, label="with ruApp (impact = wakes x cost; n/a at cost 0)")
        axc.plot(xs, [np.nan if y_ is None else y_ for y_ in yr], marker="s", markersize=5, linewidth=2, color=_ORANGE,
                 label=f"rApp-only (impact = demand windows x {WINDOW_MIN:g} min, flat)")
        be = ds.get("wake_cost_breakeven_min")
        if be is not None and min(xs) <= be <= max(xs):
            axc.axvline(be, color=_INK2, linewidth=1, linestyle=":")
            axc.annotate(f"break-even {be:.1f} min", xy=(be, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -10),
                         textcoords="offset points", fontsize=8, color=_INK2)
        axc.set_xlabel("minutes of degraded service booked per ruApp wake (--ruapp-wake-service-cost-min)")
        axc.set_ylabel("saved kWh per impact minute")
        axc.set_title("Saved kWh per impact minute vs wake cost\n(same replay, accounting only)", loc="left", fontsize=9)
        axc.legend(fontsize=7.5, frameon=False, loc="best")
    else:
        axc.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "impact.png")
    plt.close(fig)
    files["impact.png"] = ("priced service impact per variant (bars, savings % annotated) and saved kWh per impact minute vs the "
                           "wake cost (sensitivity rows)")

    # chatter-control grid: savings % vs actuations/day, one annotated point per configuration (--grid only)
    grid = result.summary.get("grid") or []
    if grid:
        fig, ax = plt.subplots(figsize=(8, 5.2), dpi=110)
        _style(ax)
        ax.xaxis.grid(True, color=_GRID, linewidth=0.8)
        markers = {1: "o", 2: "s", 3: "^", 4: "D"}
        # configurations that land on the same point (common on smooth data) share one marker and one stacked label
        clusters: dict[tuple[float, float], list[dict[str, Any]]] = {}
        for g in grid:
            key = (round(float(g["actuations_per_day"]), 2), round(float(g["savings_pct_with_ruapp"] or 0.0), 2))   # visually coincident points share a label
            clusters.setdefault(key, []).append(g)
        for (x, y), members in clusters.items():
            primary = any(m["primary"] for m in members)
            scales = {float(m.get("profile_scale", 0.0)) > 0 for m in members}
            confirms = {int(m["wake_confirm_windows"]) for m in members}
            col = _INK2 if len(scales) > 1 else (_AQUA if True in scales else _BLUE)
            mk = markers.get(next(iter(confirms)), "x") if len(confirms) == 1 else "P"
            ax.scatter([x], [y], marker=mk, s=110 if primary else 54, color=col, zorder=5,
                       edgecolor=_INK if primary else "white", linewidth=1.2 if primary else 0.8)
            # one compact line per (scale, confirm, margin) listing the dwell values that coincide: "s0/c2/m0: d0*,d30,d60"
            groups: dict[tuple[float, int, float], list[str]] = {}
            for m in members:
                key = (float(m.get("profile_scale", 0.0)), int(m["wake_confirm_windows"]), float(m["wake_margin"]))
                groups.setdefault(key, []).append(f"d{m['min_sleep_dwell_min']:g}" + ("*" if m["primary"] else ""))
            text = "\n".join(f"s{k[0]:g}/c{k[1]}/m{k[2]:g}: " + ",".join(v) for k, v in groups.items())
            ax.annotate(text, xy=(x, y), xytext=(8, 4), textcoords="offset points", fontsize=7.5, color=_INK2, va="bottom",
                        annotation_clip=False)
        ax.margins(x=0.3, y=0.3)
        handles = [Line2D([], [], marker="o", color=_BLUE, linestyle="", markersize=7, label="profile scale 0 (estate profile)"),
                   Line2D([], [], marker="o", color=_AQUA, linestyle="", markersize=7, label="profile scale > 0 (carrier's own level)"),
                   Line2D([], [], marker="o", color=_INK2, linestyle="", markersize=7, label="both scales coincide")]
        handles += [Line2D([], [], marker=markers.get(c, "x"), color=_INK2, linestyle="", markersize=7, label=f"{c} confirmation window(s)")
                    for c in sorted({int(g["wake_confirm_windows"]) for g in grid})]
        handles.append(Line2D([], [], marker="P", color=_INK2, linestyle="", markersize=7, label="confirmation windows coincide"))
        ax.legend(handles=handles, fontsize=8, frameon=False, loc="best")
        ax.set_xlabel("actuations per day, all cells (SLEEP + ACTIVE decisions = CM writes)")
        ax.set_ylabel("savings % with ruApp (ILLUSTRATIVE model)")
        ax.set_title("Chatter control on the primary model: savings vs management-plane writes, one point per "
                     "s<profile scale>/d<dwell min>/c<confirm windows>/m<margin>; * = primary (same validation days; "
                     "trade-off, not selection)", loc="left", fontsize=9, wrap=True)
        fig.tight_layout()
        fig.savefig(out_dir / "grid.png")
        plt.close(fig)
        files["grid.png"] = "chatter-control grid: savings % (with ruApp) vs actuations/day, one annotated point per configuration"

    # sweep: savings % and false-sleep rate vs sleep threshold, one line per low quantile (skipped with --no-sweep)
    sweep = result.summary["sweep"]
    if not sweep:
        return files
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6), dpi=110, sharex=True)
    for ax in (ax1, ax2):
        _style(ax)
    qs = sorted({r["low_quantile"] for r in sweep})
    colors = [_BLUE, _ORANGE, _AQUA, _VIOLET, _RED]
    for q, col in zip(qs, colors):
        rows = sorted((r for r in sweep if r["low_quantile"] == q), key=lambda r: r["sleep_threshold"])
        xs = [r["sleep_threshold"] for r in rows]
        ax1.plot(xs, [r["savings_pct_with_ruapp"] for r in rows], marker="o", markersize=5, linewidth=2, color=col,
                 label=f"low quantile {q:.2f} (redefines 'low')")
        ax2.plot(xs, [100 * (r["false_sleep_rate"] or 0.0) for r in rows], marker="o", markersize=5, linewidth=2, color=col,
                 label=f"low quantile {q:.2f}")
    for ax in (ax1, ax2):
        ax.axvline(sleep_thr, color=_INK2, linewidth=1, linestyle=":")
    ax1.annotate(f"primary run ({sleep_thr:.2f})", xy=(sleep_thr, 1.0), xycoords=("data", "axes fraction"),
                 xytext=(4, -10), textcoords="offset points", fontsize=8, color=_INK2)
    ax1.legend(fontsize=8, frameon=False, loc="center right")
    ax1.set_ylabel("savings % with ruApp\n(ILLUSTRATIVE model)")
    ax2.set_ylabel("false-sleep rate %\n(ruApp wakes / SLEEP windows)")
    ax2.set_xlabel("RAPP_SLEEP_THRESHOLD (p_low needed to SLEEP)")
    ax1.set_title("Sensitivity only (same validation days; not for threshold selection): sleep threshold x low-load quantile",
                  loc="left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out_dir / "sweep.png")
    plt.close(fig)
    files["sweep.png"] = "savings % (top) and false-sleep rate (bottom) vs sleep threshold, one line per low-load quantile"
    return files


def write_outputs(result: EvalResult, out_dir: Optional[Path] = None, *, plots: bool = True) -> Path:
    """report.md, summary.json (+ windows.csv per variant) and the PNG plots under `out_dir`
    (default `<RAPP_EVAL_DIR>/<UTC ts>/`, gitignored)."""
    out = Path(out_dir) if out_dir is not None else result.cfg.rapp_eval_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    plot_files = write_plots(result, out) if plots else {}
    for variant, rr in result.replays.items():
        rr.windows.to_csv(out / f"windows_{variant}.csv", index=False)
    result.summary["outputs"] = {"dir": display_path(out, result.cfg.var_dir), "plots": sorted(plot_files),
                                 "windows": [f"windows_{v}.csv" for v in result.replays],
                                 "grid": "grid.md" if result.summary.get("grid") else None}
    (out / "summary.json").write_text(json.dumps(_jsonable(result.summary), indent=2), encoding="utf-8")
    if result.summary.get("grid"):
        write_grid_report(result, out, "grid.png" if "grid.png" in plot_files else None)
    write_report(result, out, plot_files)
    return out


def headline_lines(summary: dict[str, Any]) -> list[str]:
    h = summary["headline"]
    pct = lambda v: _f(v, 1, "%")  # noqa: E731
    rate = lambda v: _f(None if v is None else 100 * v, 1, "%")  # noqa: E731
    cost, wh = float(h.get("ruapp_wake_service_cost_min", 0.0)), float(h.get("ruapp_wake_energy_wh", 0.0))
    if h["with_ruapp_impact_free_by_construction"]:
        by_construction = "zero service impact BY CONSTRUCTION under the 30-min PRB-average demand proxy; upper bound"
    elif h.get("with_ruapp_impact_priced"):
        by_construction = f"service impact PRICED at {cost:g} min per ruApp wake, comparable with rApp-only"
    else:
        by_construction = "demand threshold != low threshold, see notes"
    if wh > 0:
        by_construction += f"; incl. {h['ruapp_wakes']} wakes x {wh:g} Wh = {_f(h['ruapp_wake_kwh_with_ruapp'], 3)} kWh re-activation energy"
    lines = [
        "Energy-saving evaluation (ILLUSTRATIVE power model; per-site KPI treated as the carrier under control):",
        f"  savings with ruApp   : {pct(h['savings_pct_with_ruapp'])}  ({_f(h['savings_kwh_with_ruapp'], 2)} of {_f(h['baseline_kwh'], 2)} kWh; {by_construction})",
        f"  savings rApp-only    : {pct(h['savings_pct_rapp_only'])}  (service impact: {h['demand_while_asleep_rapp_only']} windows slept through demand)",
        f"  savings oracle       : {pct(h['savings_pct_oracle'])}  (perfect foresight upper bound)",
        f"  service impact       : with ruApp {_f(h['service_impact_min_with_ruapp'], 1)} min ({h['ruapp_wakes']} wakes x {cost:g} min), "
        f"rApp-only {_f(h['service_impact_min_rapp_only'], 1)} min ({h['demand_while_asleep_rapp_only']} windows x {WINDOW_MIN:g} min); "
        f"saved kWh per impact min: with ruApp {_f(h['saved_kwh_per_impact_min_with_ruapp'], 3)}, rApp-only "
        f"{_f(h['saved_kwh_per_impact_min_rapp_only'], 3)} (PRICED, not measured; n/a = impact 0)",
        f"  offloaded load       : {_f(h['offloaded_kwh_with_ruapp'], 2)} kWh carried by the remaining layer(s), not subtracted "
        f"(savings net of offload {pct(h['savings_pct_with_ruapp_net_of_offload'])})",
        f"  false-sleep rate     : {rate(h['false_sleep_rate'])}  ({h['ruapp_wakes']} ruApp wakes; forecast error on slept windows, not measured impact)",
        f"  missed-opportunity   : {rate(h['missed_opportunity_rate'])}",
        f"  sleep h per cell-day : {_f(h['sleep_hours_per_cell_day_with_ruapp'], 2)}  ({_f(h['sleep_hours_per_day_all_cells_with_ruapp'], 2)} h/day summed over {h['n_cells']} cells)",
        f"  sleep episodes       : {h['sleep_episodes_with_ruapp']}  ({_f(h['sleep_episodes_per_day_with_ruapp'], 2)}/day over all cells; "
        f"{h['short_episodes_with_ruapp']} short <= 1 h; mean episode {_f(h['mean_episode_h_with_ruapp'], 2)} h)",
        f"  actuations           : {h['actuations_with_ruapp']}  ({_f(h['actuations_per_day_with_ruapp'], 2)}/day over all cells, "
        f"{_f(h['actuations_per_cell_day_with_ruapp'], 2)} per cell-day; {h['sleep_decisions_with_ruapp']} SLEEP + "
        f"{h['preemptive_active_decisions_with_ruapp']} pre-emptive ACTIVE = CM writes)",
        f"  validation AUC       : {_f(h['validation_auc'], 3)}  (train holdout AUC {_f(h['train_holdout_auc'], 3)})",
    ]
    p = summary.get("params") or {}
    if "short_episodes_per_cell_with_ruapp" in h:
        per_cell = "; ".join(f"{c}: {h['short_episodes_per_cell_with_ruapp'][c]} short, {_f(h['actuations_per_day_per_cell_with_ruapp'][c], 2)}/day"
                             for c in sorted(h["short_episodes_per_cell_with_ruapp"]))
        lines.append(f"  chatter per cell     : {per_cell}")
        held = (f"{h['held_windows_with_ruapp']} held windows; {h['ruapp_wakes_while_held']} ruApp wakes while held"
                + (" (false-sleep rate includes them)" if h.get("false_sleep_rate_includes_held_wakes") else "")
                + f"; rApp-only demand asleep while held {h['demand_while_asleep_while_held_rapp_only']}; "
                f"mean wake delay {_f(h['preemptive_wake_delay_windows_mean'], 1)} windows; "
                f"re-sleeps in the local-wake window {h['resleep_after_local_wake_with_ruapp']}")
        lines.append(f"  governor cost        : {held}")
        div_cells = ", ".join(f"{c}: {n}" for c, n in sorted(h["forecast_divergent_windows_per_cell"].items()))
        lines.append(f"  forecast divergence  : {h['forecast_divergent_windows']} windows with |p_low(pm) - p_low(profile)| > "
                     f"{FORECAST_DIVERGENCE_FLAG} ({rate(h['forecast_divergence_rate'])}; mean |delta| {_f(h['forecast_divergence_mean_abs'], 3)}; "
                     f"per cell {div_cells}) - the estate-pooled profile is the chatter's root cause")
    if "rapp_min_sleep_dwell_min" in p:
        lines.append(f"  chatter control      : profile scale {p.get('rapp_profile_scale', 0.0):g}, dwell {p['rapp_min_sleep_dwell_min']:g} min, "
                     f"{p['rapp_wake_confirm_windows']} confirmation window(s), wake margin {p['rapp_wake_margin']:g} "
                     f"(effective wake threshold {_f(p['rapp_wake_threshold_effective'], 2)}), post-local-wake hold "
                     f"{p.get('rapp_post_local_wake_hold_min', 0.0):g} min")
    if "alarm_days_with_ruapp" in h:
        cap, alarm = h.get("max_actuations_per_day", 0), h.get("actuation_alarm_per_day", 0)
        per_cell = ", ".join(f"{c}: {n}" for c, n in sorted(h["alarm_days_per_cell_with_ruapp"].items()))
        lines.append(f"  actuation budget     : {h['alarm_days_with_ruapp']} carrier-days hit the alarm, {h['budget_reached_windows_with_ruapp']} "
                     f"windows held by the budget  (max {cap if cap else 'off'} per carrier-day, alarm at {alarm if alarm else 'off'}; "
                     f"peak {h['max_actuations_per_carrier_day_with_ruapp']} in one carrier-day; alarm carrier-days per cell {per_cell}; "
                     f"rApp-only {h['alarm_days_rapp_only']} alarm carrier-days, {h['budget_reached_windows_rapp_only']} held)")
    grid = summary.get("grid") or []
    if grid:
        lines.append(f"  chatter grid         : {len(grid)} configs on the primary model (grid.md / grid.png; same validation days, trade-off "
                     "not selection; dwell/confirm/margin = rate limiting, profile scale = root cause; judge per cell; "
                     "budget in force on the primary row only, off elsewhere)")
        for g in grid:
            mark = "*" if g["primary"] else " "
            label = g["label"] + ("(off-grid)" if g.get("off_grid") else "")
            lines.append(f"    {label:<22} {mark} saved {pct(g['savings_pct_with_ruapp']):>6} | episodes {g['sleep_episodes']:>3} "
                         f"({g['short_episodes']} short, mean {_f(g['mean_episode_h'], 2)} h) | actuations {g['actuations']:>3} "
                         f"({_f(g['actuations_per_day'], 2)}/day) | ruApp wakes {g['ruapp_wakes']} ({g['ruapp_wakes_while_held']} while held) "
                         f"| divergent {g['forecast_divergent_windows']:>3} | worst cell {g['worst_cell_short_episodes']}: "
                         f"{g['max_cell_short_episodes']} short, {_f(g['max_cell_actuations_per_day'], 2)}/day")
    ds = summary.get("demand_sensitivity") or {}
    if ds.get("burst_fractions"):
        parts = ", ".join(f"{b['burst_fraction']:.0%} -> {pct(b['savings_pct'])}" for b in ds["burst_fractions"])
        lines.append(f"  burst sensitivity    : if a share of slept windows hid a wake-triggering burst: {parts}")
    if ds.get("demand_margins"):
        parts = ", ".join(f"{m['demand_margin']:.2f} x thr -> {pct(m['savings_pct'])}" for m in ds["demand_margins"])
        lines.append(f"  demand-margin sens.  : demand threshold below the low threshold: {parts}")
    if ds.get("wake_costs"):
        parts = ", ".join(f"{r['wake_cost_min']:g} min -> {_f(r['service_impact_min_with_ruapp'], 0)} min, "
                          f"{_f(r['saved_kwh_per_impact_min_with_ruapp'], 3)} kWh/min" for r in ds["wake_costs"])
        r0 = ds["wake_costs"][0]
        be = ds.get("wake_cost_breakeven_min")
        be_text = f"rApp-only scores higher from a wake cost of {be:.1f} min" if be is not None else "rApp-only never scores higher"
        lines.append(f"  wake-cost sens.      : with ruApp impact / score per wake cost: {parts}; rApp-only "
                     f"{_f(r0['service_impact_min_rapp_only'], 0)} min, {_f(r0['saved_kwh_per_impact_min_rapp_only'], 3)} kWh/min at any cost; "
                     f"{be_text} (same replay, accounting only)")
    if h["missing_kpi_windows"]:
        lines.append(f"  missing KPI windows  : {h['missing_kpi_windows']} ({h['missing_kpi_asleep_with_ruapp']} while asleep; charged ACTIVE, not counted as wakes)")
    lines += [f"  note                 : {n}" for n in summary.get("notes", [])]
    return lines
