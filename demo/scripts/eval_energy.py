#!/usr/bin/env python
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

    .venv/bin/python scripts/eval_energy.py --data-kind synthetic --days 30
    .venv/bin/python scripts/eval_energy.py --data-kind kpi_report_csv --data-dir <private dir>
    OMP_NUM_THREADS=1 .venv/bin/python scripts/eval_energy.py --data-kind kpi_report_csv --data-dir <private dir> --grid   # chatter-control matrix

Splits the table by time (first --train-days train, next --validate-days validate), trains the
rApp's model exactly as the rApp does, replays the rApp cycle + ruApp wake rule over the
validation windows through the real `policy_engine.decide` / `LowLoadModel` code paths, charges
energy with the mock O-RU's ILLUSTRATIVE power model and writes report.md, summary.json and the
plots to --out (default `<RAPP_EVAL_DIR>/<UTC ts>/`, gitignored). See `ocp_auru/rapp/eval.py`.

Cell labels in every output are the adapter's opaque ids; nothing here names a site, and the
--data-dir path is never written to the outputs (a private export directory may be named after one).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

from ocp_auru.common.config import display_path, settings
from ocp_auru.rapp.eval import (GRID_CONFIRM_WINDOWS, GRID_DWELL_MIN, GRID_PROFILE_SCALES, GRID_WAKE_MARGINS, SWEEP_LOW_QUANTILES,
                            SWEEP_SLEEP_THRESHOLDS, WAKE_COST_SENSITIVITY_MIN, EvalParams, headline_lines, run_evaluation, write_outputs)


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-kind", default=settings.rapp_data, choices=("synthetic", "kpi_report_csv"))
    ap.add_argument("--data-dir", type=Path, default=None, help="kpi_report_csv directory (default RAPP_DATA_DIR)")
    ap.add_argument("--days", type=int, default=30, help="synthetic only: days to generate (default 30)")
    ap.add_argument("--cells", type=int, default=settings.rapp_synthetic_cells, help="synthetic only: number of cells")
    ap.add_argument("--train-days", type=int, default=25)
    ap.add_argument("--validate-days", type=int, default=5)
    ap.add_argument("--low-quantile", type=float, default=None,
                    help="low-load label quantile of the TRAIN frame (default: the rApp's own, 0.25)")
    ap.add_argument("--seed", type=int, default=settings.rapp_seed)
    ap.add_argument("--demand-prb", type=float, default=None,
                    help="prb_util_dl_pct at/above which a window has demand (default: the low threshold)")
    # service-impact pricing (accounting only; decisions never change): makes with-ruApp and rApp-only comparable
    ap.add_argument("--ruapp-wake-service-cost-min", type=float, default=0.0,
                    help="minutes of degraded service booked per ruApp wake in a slept-with-demand window (the first UE(s) wait "
                         "for the RU re-activation: the mock RU wakes in microseconds, a real tx-array-carrier SLEEP exit takes "
                         "ms-s). 0 (default) keeps the historical accounting: with ruApp impact-free by construction. rApp-only "
                         "demand windows slept through are always booked at 30 min (the whole window)")
    ap.add_argument("--ruapp-wake-energy-wh", type=float, default=0.0,
                    help="re-activation energy (Wh) charged per ruApp wake to the with-ruApp variant's kWh (default 0)")
    ap.add_argument("--wake-cost-sensitivity-min", type=_floats, default=WAKE_COST_SENSITIVITY_MIN,
                    help="wake costs (min) re-priced on the same replay in the demand-sensitivity section (default 0,1,5,30; '' disables)")
    ap.add_argument("--sleep-threshold", type=float, default=settings.rapp_sleep_threshold,
                    help="RAPP_SLEEP_THRESHOLD for the primary run")
    ap.add_argument("--wake-threshold", type=float, default=settings.rapp_wake_threshold,
                    help="RAPP_WAKE_THRESHOLD for the primary run")
    # chatter control: the primary run's settings; defaults = the rApp's (all off)
    ap.add_argument("--min-sleep-dwell-min", type=float, default=settings.rapp_min_sleep_dwell_min,
                    help="RAPP_MIN_SLEEP_DWELL_MIN for the primary run: minutes of data time a SLEEP holds before a "
                         "pre-emptive wake (0 = off)")
    ap.add_argument("--wake-confirm-windows", type=int, default=settings.rapp_wake_confirm_windows,
                    help="RAPP_WAKE_CONFIRM_WINDOWS: consecutive windows with p_low below the effective wake threshold "
                         "before a pre-emptive wake (1 = off)")
    ap.add_argument("--wake-margin", type=float, default=settings.rapp_wake_margin,
                    help="RAPP_WAKE_MARGIN: effective wake threshold = wake threshold - margin (0 = off; clamped to [0, wake thr])")
    ap.add_argument("--profile-scale", type=float, default=settings.rapp_profile_scale,
                    help="RAPP_PROFILE_SCALE for the primary run: weight [0, 1] of the carrier's own load level in the asleep "
                         "forecast (0 = estate profile as-is, the root cause of the chatter; 1 = profile at the carrier's level)")
    ap.add_argument("--post-local-wake-hold-min", type=float, default=settings.rapp_post_local_wake_hold_min,
                    help="RAPP_POST_LOCAL_WAKE_HOLD_MIN for the primary run: minutes of data time after a ruApp local wake "
                         "during which the rApp does not SLEEP the carrier again (30 = one window; 0 = off)")
    # actuation budget (config.py): the primary run's cap and alarm; defaults = the rApp's own
    ap.add_argument("--max-actuations-per-day", type=int, default=settings.rapp_max_actuations_per_day,
                    help="RAPP_MAX_ACTUATIONS_PER_DAY for the primary run: applied SLEEP + ACTIVE decisions per cell per calendar day "
                         "of data time after which decide() refuses further ones (the ruApp still wakes on demand; 0 = off)")
    ap.add_argument("--actuation-alarm-per-day", type=int, default=settings.rapp_actuation_alarm_per_day,
                    help="RAPP_ACTUATION_ALARM_PER_DAY: the count at which a cell-day is reported as an alarm carrier-day (0 = off)")
    ap.add_argument("--grid", action="store_true", default=False,
                    help="also replay the chatter-control matrix (profile scale x dwell x confirmation windows x margin) on the "
                         "primary model and write grid.md / grid.png (per-cell rows included). The actuation budget is not an "
                         "axis: only the primary row carries --max-actuations-per-day, every other row replays with it off. "
                         "Tip: OMP_NUM_THREADS=1 avoids OpenMP oversubscription in the many small scikit-learn calls (a 5x "
                         "slowdown was measured on macOS)")
    ap.add_argument("--grid-profile-scales", type=_floats, default=GRID_PROFILE_SCALES, help="grid axis, [0, 1] (default 0,1)")
    ap.add_argument("--grid-dwell-min", type=_floats, default=GRID_DWELL_MIN, help="grid axis, minutes (default 0,30,60,120)")
    ap.add_argument("--grid-confirm-windows", type=_ints, default=GRID_CONFIRM_WINDOWS, help="grid axis (default 1,2)")
    ap.add_argument("--grid-wake-margins", type=_floats, default=GRID_WAKE_MARGINS,
                    help="grid axis (default 0: the margin has little effect; pass 0,0.1 for the earlier matrix)")
    ap.add_argument("--sweep", dest="sweep", action="store_true", default=True)
    ap.add_argument("--no-sweep", dest="sweep", action="store_false")
    ap.add_argument("--sweep-sleep-thresholds", type=_floats, default=SWEEP_SLEEP_THRESHOLDS)
    ap.add_argument("--sweep-low-quantiles", type=_floats, default=SWEEP_LOW_QUANTILES)
    ap.add_argument("--out", type=Path, default=None, help="output directory (default <RAPP_EVAL_DIR>/<UTC ts>/)")
    ap.add_argument("--no-plots", dest="plots", action="store_false", default=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.min_sleep_dwell_min < 0 or args.wake_confirm_windows < 1 or args.wake_margin < 0:
        print("error: --min-sleep-dwell-min must be >= 0, --wake-confirm-windows >= 1, --wake-margin >= 0", file=sys.stderr)
        return 2
    if not (0.0 <= args.profile_scale <= 1.0) or args.post_local_wake_hold_min < 0:
        print("error: --profile-scale must be in [0, 1], --post-local-wake-hold-min >= 0", file=sys.stderr)
        return 2
    if args.max_actuations_per_day < 0 or args.actuation_alarm_per_day < 0:
        print("error: --max-actuations-per-day and --actuation-alarm-per-day must be >= 0 (0 = off)", file=sys.stderr)
        return 2
    cfg = replace(settings, rapp_synthetic_days=args.days, rapp_synthetic_cells=args.cells, rapp_seed=args.seed,
                  rapp_sleep_threshold=args.sleep_threshold, rapp_wake_threshold=args.wake_threshold,
                  rapp_min_sleep_dwell_min=args.min_sleep_dwell_min, rapp_wake_confirm_windows=args.wake_confirm_windows,
                  rapp_wake_margin=args.wake_margin, rapp_profile_scale=args.profile_scale,
                  rapp_post_local_wake_hold_min=args.post_local_wake_hold_min,
                  rapp_max_actuations_per_day=args.max_actuations_per_day, rapp_actuation_alarm_per_day=args.actuation_alarm_per_day)
    t0 = time.perf_counter()
    try:
        # EvalParams validates the quantiles/thresholds and names the flag in its message
        params = EvalParams(data_kind=args.data_kind, data_dir=args.data_dir, train_days=args.train_days,
                            validate_days=args.validate_days, low_quantile=args.low_quantile, seed=args.seed,
                            demand_prb=args.demand_prb, sweep=args.sweep, sleep_thresholds=args.sweep_sleep_thresholds,
                            low_quantiles=args.sweep_low_quantiles, grid=args.grid, grid_dwell_min=args.grid_dwell_min,
                            grid_confirm_windows=args.grid_confirm_windows, grid_wake_margins=args.grid_wake_margins,
                            grid_profile_scales=args.grid_profile_scales,
                            ruapp_wake_service_cost_min=args.ruapp_wake_service_cost_min, ruapp_wake_energy_wh=args.ruapp_wake_energy_wh,
                            wake_cost_sensitivity_min=args.wake_cost_sensitivity_min)
        result = run_evaluation(params, cfg)
    except (ValueError, AssertionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = write_outputs(result, args.out, plots=args.plots)
    for line in headline_lines(result.summary):
        print(line)
    extra = ", grid.md" if result.summary["outputs"].get("grid") else ""
    print(f"  outputs              : {display_path(out, cfg.var_dir)}  (report.md, summary.json{extra}, "
          f"{len(result.summary['outputs']['plots'])} plots; {time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
