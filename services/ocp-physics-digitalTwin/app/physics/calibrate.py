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

"""Calibration: fit a small set of global parameters to measurement marginals (RSRP/SINR).

Fits MARGINALS, never points: the model's structure comes from the spec and the topology inputs;
calibration only anchors absolute level, spread and indoor mix. Categorically different from
inventing UE positions to train a surrogate model.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

log = logging.getLogger(__name__)

QS = [5, 25, 50, 75, 95]

# Columns the measurement report reader needs, as {role: header}. Every management system names
# these differently, so these defaults are generic: rename the headers in your export to match,
# or pass a `columns` override with the headers it actually has.
MR_COLUMNS = {"rsrp": "rsrp_dbm", "sinr": "sinr_db", "count": "sample_count"}


def _numeric(s: pd.Series) -> pd.Series:
    """Locale-tolerant numeric parse: strip quotes/whitespace and thousands separators before
    coercion. A report may write counts >= 1,000 as comma-formatted TEXT; a bare
    `to_numeric(errors="coerce")` would silently drop those rows, which are the rows with the
    largest weight. Silent weight loss is the failure class the refuse-on-defaults guard exists
    for, so parsing is cleaned AND accounted (see `_parse_stats`); anything still unparseable is
    logged loudly, never quietly masked."""
    cleaned = (s.astype(str).str.strip().str.strip("'\"")
                 .str.replace(",", "", regex=False)
                 .replace({"": None, "nan": None, "None": None}))
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_stats(raw: pd.Series, parsed: pd.Series) -> dict[str, int]:
    nonempty = raw.notna() & (raw.astype(str).str.strip() != "")
    unparsed = int((nonempty & parsed.isna()).sum())
    if unparsed:
        sample = raw[nonempty & parsed.isna()].astype(str).head(3).tolist()
        log.warning("MR-report parse: %d non-empty values failed numeric parse (e.g. %s) — "
                    "their weight is EXCLUDED from the calibration targets", unparsed, sample)
    return {"rows": int(len(raw)), "nonempty": int(nonempty.sum()),
            "parsed": int(parsed.notna().sum()), "unparsed": unparsed}

BOUNDS = {
    # P0 bounds ARE the physical window: with wider bounds a fit can pass every quantile gate
    # while buying accuracy with an unphysical EIRP. Constraining the search to physical power
    # forces the optimizer to explain the targets with geometry and environment instead.
    "p0_low": (8.0, 28.0), "p0_mid": (8.0, 28.0), "p0_hi": (8.0, 28.0),
    "sigma_extra": (0.0, 6.0), "p_in": (0.30, 0.995),
    "ltw_low": (5.0, 20.0), "ltw_hi": (10.0, 30.0), "load": (0.02, 0.60),
    "agg": (1.0, 24.0),   # pseudo-device session size — the targets are per-device trajectory means
    "bias_mid": (0.0, 20.0), "bias_hi": (0.0, 20.0),  # NSA serving-policy band bias, dB
    "w_tier": (0.0, 1.0),  # external-tier visibility (density + orientation mismatch of the hex construction)
}
NAMES = list(BOUNDS)


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, qs=QS) -> list[float]:
    v, w = np.asarray(values, float), np.asarray(weights, float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[m], w[m]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w) / w.sum()
    return [float(v[np.searchsorted(cw, q / 100.0)]) for q in qs]


def _columns(overrides: Mapping[str, str] | None) -> dict[str, str]:
    return {**MR_COLUMNS, **(overrides or {})}


def _column(d: pd.DataFrame, cols: Mapping[str, str], key: str) -> pd.Series:
    name = cols[key]
    if name not in d.columns:
        raise KeyError(f"measurement report has no column {name!r} for {key!r}; "
                       f"pass a column map for this export. Found: {list(d.columns)}")
    return d[name]


def targets_from_mr_report(path: str, columns: Mapping[str, str] | None = None) -> dict[str, list[float]]:
    """Report-weighted quantiles from a measurement report (one row per device).

    Columns are resolved through `MR_COLUMNS`, overridable per call with `columns`."""
    d = pd.read_excel(path)
    cols = _columns(columns)

    w_raw = _column(d, cols, "count")
    w = _numeric(w_raw)
    out = {}
    for key in ("rsrp", "sinr"):
        v = _numeric(_column(d, cols, key))
        keep = v > -150  # sentinel filter
        out[key] = weighted_quantiles(v[keep].values, w[keep].values)
    out["parse"] = {"mr_count": _parse_stats(w_raw, w), "total_mrs": int(w.sum())}
    return out


def targets_from_mr_report_split(path: str, holdout_frac: float = 0.3, seed: int = 42,
                                 columns: Mapping[str, str] | None = None) -> tuple[dict, dict, dict]:
    """Device-level train/holdout split of the measurement targets.

    Splits REPORT ROWS (pseudo-devices) with a seeded permutation, then computes the
    same report-weighted quantiles per subset. The fit only ever sees the train marginals;
    the holdout marginals are untouched ground truth from devices the calibration never
    saw. Predicted quantiles depend only on the fitted params, so the holdout check is
    prediction vs unseen-device marginals - a genuine out-of-sample test.
    """
    d = pd.read_excel(path)
    cols = _columns(columns)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(d))
    n_hold = int(round(len(d) * holdout_frac))
    hold_idx, train_idx = idx[:n_hold], idx[n_hold:]
    w_raw = _column(d, cols, "count")
    w_all = _numeric(w_raw)

    def subset_targets(rows) -> dict[str, list[float]]:
        out = {}
        for key in ("rsrp", "sinr"):
            v = _numeric(_column(d, cols, key)).iloc[rows]
            w = w_all.iloc[rows]
            keep = v > -150
            out[key] = weighted_quantiles(v[keep].values, w[keep].values)
        return out

    meta = {"n_devices": int(len(d)), "n_train": int(len(train_idx)),
            "n_holdout": int(len(hold_idx)), "holdout_frac": holdout_frac, "seed": seed,
            "parse": {"mr_count": _parse_stats(w_raw, w_all), "total_mrs": int(w_all.sum())}}
    return subset_targets(train_idx), subset_targets(hold_idx), meta


def holdout_check(report: dict[str, Any], holdout_targets: dict[str, list[float]]) -> dict[str, Any]:
    """The acceptance table recomputed against UNSEEN devices' marginals."""
    shadow = {
        "params": report["params"],
        "quantiles": {
            "rsrp": {"predicted": report["quantiles"]["rsrp"]["predicted"],
                     "target": holdout_targets["rsrp"]},
            "sinr": {"predicted": report["quantiles"]["sinr"]["predicted"],
                     "target": holdout_targets["sinr"]},
        },
    }
    gate = acceptance_gate(shadow)
    gate["rung"] = "simulation, VALIDATED against held-out supplied data"
    return {"targets": holdout_targets, "gate": gate,
            "quantiles": shadow["quantiles"]}


def loss(vec: np.ndarray, estate, targets: dict[str, list[float]],
         serving_mask: np.ndarray) -> float:
    params = dict(zip(NAMES, vec))
    r = estate.evaluate(params, serving_mask=serving_mask)
    a = r["_arrays"]
    pr = np.percentile(a["rsrp"], QS) - np.array(targets["rsrp"])
    ps = np.percentile(a["sinr"], QS) - np.array(targets["sinr"])
    # Tail emphasis is DECLARED, not hidden: the guardrail KPIs read the 5th percentile, so
    # the p05 residuals carry extra weight. This tunes optimizer priorities among honest residuals;
    # it does not touch the model or the targets.
    w_r = np.array([2.0, 1.0, 1.0, 1.0, 1.0])
    w_s = np.array([3.0, 1.0, 1.0, 1.0, 1.0])
    return float((w_r * pr ** 2).sum() + 2.0 * (w_s * ps ** 2).sum())


def fit(estate, targets: dict[str, list[float]]) -> dict[str, Any]:
    mask = estate.nr_serving_mask()
    bounds = [BOUNDS[n] for n in NAMES]
    de = differential_evolution(loss, bounds, args=(estate, targets, mask),
                                seed=7, maxiter=140, popsize=14, tol=1e-4, polish=False)
    # Bounded polish: an unbounded Nelder-Mead can walk a parameter outside its physical range
    # (for example p_in above 1). scipy's Nelder-Mead honours bounds since 1.7; passing them is
    # the fix, not clipping after.
    nm = minimize(loss, de.x, args=(estate, targets, mask), method="Nelder-Mead", bounds=bounds,
                  options={"maxiter": 500, "xatol": 1e-3, "fatol": 1e-3})
    best = nm if nm.fun < de.fun else de
    params = dict(zip(NAMES, [float(v) for v in best.x]))

    r = estate.evaluate(params, serving_mask=mask)
    a = r["_arrays"]
    report = {
        "params": params,
        "loss": float(best.fun),
        "quantiles": {
            "rsrp": {"predicted": [round(float(x), 2) for x in np.percentile(a["rsrp"], QS)],
                     "target": targets["rsrp"]},
            "sinr": {"predicted": [round(float(x), 2) for x in np.percentile(a["sinr"], QS)],
                     "target": targets["sinr"]},
        },
    }
    report["gate"] = acceptance_gate(report)
    return report


def acceptance_gate(report: dict[str, Any]) -> dict[str, Any]:
    """Hard gates decide wiring into any decision path; softs are reported, not hidden."""
    rq, sq = report["quantiles"]["rsrp"], report["quantiles"]["sinr"]
    d_rsrp50 = abs(rq["predicted"][2] - rq["target"][2])
    d_sinr05 = abs(sq["predicted"][0] - sq["target"][0])
    p = report["params"]
    # P0 is EIRP at boresight MINUS peak gain (gain is added separately) → compare P0+~17 dBi.
    eirp_ok = all(25.0 <= p[k] + 17.0 <= 45.0 for k in ("p0_low", "p0_mid", "p0_hi"))
    checks = {
        "rsrp_p50_within_2db": {"delta_db": round(d_rsrp50, 2), "pass": d_rsrp50 <= 2.0, "hard": True},
        "sinr_p05_within_1p5db": {"delta_db": round(d_sinr05, 2), "pass": d_sinr05 <= 1.5, "hard": True},
        "eirp_sanity_window": {"pass": eirp_ok, "hard": True},
        "rsrp_p05_within_3db": {"delta_db": round(abs(rq["predicted"][0] - rq["target"][0]), 2),
                                 "pass": abs(rq["predicted"][0] - rq["target"][0]) <= 3.0, "hard": False},
        "rsrp_p95_within_3db": {"delta_db": round(abs(rq["predicted"][4] - rq["target"][4]), 2),
                                 "pass": abs(rq["predicted"][4] - rq["target"][4]) <= 3.0, "hard": False},
        "sinr_p50_within_2db": {"delta_db": round(abs(sq["predicted"][2] - sq["target"][2]), 2),
                                 "pass": abs(sq["predicted"][2] - sq["target"][2]) <= 2.0, "hard": False},
        "sinr_p95_within_2db": {"delta_db": round(abs(sq["predicted"][4] - sq["target"][4]), 2),
                                 "pass": abs(sq["predicted"][4] - sq["target"][4]) <= 2.0, "hard": False},
    }
    return {"checks": checks,
            "hard_pass": all(c["pass"] for c in checks.values() if c["hard"]),
            "rung": "simulation, calibrated to supplied measurement data"}
