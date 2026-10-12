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
"""Low-load forecaster: "is the next 30-minute window low-load?" (HLD ADR-6, §3.7).

A small supervised classifier (scikit-learn HistGradientBoosting, LogisticRegression
fallback) with a time-based holdout. The label threshold is data-relative by default
(25th percentile of DL PRB utilisation in the training set), so no absolute KPI
value is ever written into code or logs. Model files live in the gitignored `var/`.

Sleeping carriers: a carrier in SLEEP reports PRB 0 by construction, so its own PM says
nothing about demand. The model therefore also keeps an *hour-of-day load profile* (mean
KPIs per weekday/weekend half-hour bin of the training table) and, for a sleeping carrier,
forecasts from the load the carrier would carry if it were ACTIVE at the current simulated
time (`predict_low_load_probability_asleep`). That is what makes the HLD §3.7 pre-emptive
SLEEP -> ACTIVE decision reachable. The profile is part of the model file (gitignored).

The profile is pooled over the whole training ESTATE (`build_hourly_profile` groups by weekend x
half-hour only, never by cell), so the asleep forecast is P(low | estate-mean load) while the awake
forecast is P(low | this carrier's own PM). On a carrier much quieter than the estate the two
disagree at the edge of the night trough and the rApp flips SLEEP -> ACTIVE -> SLEEP
(the root cause of the chatter). `profile_level_ratio` measures the carrier's
own level against the profile at the same bins from its awake PM history, and
`asleep_forecast(..., scale_weight=RAPP_PROFILE_SCALE)` scales the profile by it (0 = off, the
shipped default; the offline evaluation's `--grid` compares both).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ocp_auru.common.config import Settings, settings

from .data.canonical import to_canonical, validate
from .features import FEATURE_COLUMNS, build_features, build_labels, features_from_window

log = logging.getLogger(__name__)

LOW_QUANTILE = 0.25
HOLDOUT_FRAC = 0.2
PROFILE_COLUMNS = ["prb_util_dl_pct", "prb_util_ul_pct", "thr_dl_mbps", "active_users"]
_BINS_PER_DAY = 48  # 30-minute bins; profile index = is_weekend * 48 + hour * 2 + (minute >= 30)
# level scaling of the estate profile (config.py RAPP_PROFILE_SCALE): the carrier's own level is trusted
# after this many AWAKE windows of its own PM, and the ratio is clipped to this range
PROFILE_SCALE_MIN_ROWS = 6
PROFILE_SCALE_RATIO_RANGE = (0.2, 5.0)
FORECAST_DIVERGENCE_FLAG = 0.5  # |p_low(pm-window) - p_low(expected-hour-profile)| above this is flagged


def _profile_key(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts)
    return (ts.dt.dayofweek >= 5).astype(int) * _BINS_PER_DAY + ts.dt.hour * 2 + (ts.dt.minute >= 30).astype(int)


def build_hourly_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of `PROFILE_COLUMNS` per (weekend, half-hour) bin: 96 rows indexed 0..95.
    Missing bins are interpolated within the day; a missing weekend (or weekday) block copies the other."""
    num = df.loc[:, PROFILE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    prof = num.groupby(_profile_key(df["ts"]).to_numpy()).mean().reindex(range(2 * _BINS_PER_DAY))
    wd, we = prof.iloc[:_BINS_PER_DAY], prof.iloc[_BINS_PER_DAY:]
    if we.isna().all().all() and not wd.isna().all().all():
        prof.iloc[_BINS_PER_DAY:] = wd.to_numpy()
    elif wd.isna().all().all() and not we.isna().all().all():
        prof.iloc[:_BINS_PER_DAY] = we.to_numpy()
    blocks = [prof.iloc[i:i + _BINS_PER_DAY].interpolate(limit_direction="both") for i in (0, _BINS_PER_DAY)]
    return pd.concat(blocks)


def _gbdt_pipeline(seed: int) -> Pipeline:
    from sklearn.ensemble import HistGradientBoostingClassifier  # may be unavailable on old builds

    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
                                         min_samples_leaf=10, random_state=seed)
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", clf)])


def _fallback_pipeline() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=1000))])


@dataclass
class LowLoadModel:
    pipeline: Pipeline
    low_threshold: float
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    metrics: dict[str, Any] = field(default_factory=dict)
    hourly_profile: Optional[pd.DataFrame] = None  # see build_hourly_profile; None for pre-profile model files

    # ---- inference ----------------------------------------------------------------------
    def predict_proba_frame(self, X: pd.DataFrame) -> np.ndarray:
        classes = list(self.pipeline.classes_)
        if 1 not in classes:  # degenerate training set: never low
            return np.zeros(len(X))
        return self.pipeline.predict_proba(X.loc[:, self.feature_columns])[:, classes.index(1)]

    def predict_low_load_probability(self, window_rows: Sequence[Mapping[str, Any]]) -> float:
        """P(next window is low-load) from the carrier's recent PM rows (oldest first)."""
        return float(self.predict_proba_frame(features_from_window(window_rows))[0])

    def expected_window(self, sim_time: str, n: int = 3, *, scale: Optional[Mapping[str, float]] = None) -> list[dict[str, Any]]:
        """Canonical rows for the `n` half-hour bins ending at `sim_time`, taken from the
        hour-of-day profile: the load a carrier would carry if it were ACTIVE now. `scale` (per
        `PROFILE_COLUMNS`, default 1.0) multiplies the profile values (level scaling, see
        `profile_level_ratio`)."""
        if self.hourly_profile is None:
            raise ValueError("model has no hourly load profile (retrain: RAPP_RETRAIN=1)")
        end = pd.Timestamp(sim_time)
        stamps = [end - pd.Timedelta(minutes=30 * k) for k in range(n - 1, -1, -1)]
        keys = _profile_key(pd.Series(stamps))
        factors = {c: float((scale or {}).get(c, 1.0)) for c in PROFILE_COLUMNS}
        rows = []
        for ts, key in zip(stamps, keys):
            vals = self.hourly_profile.loc[int(key)]
            rows.append({"ts": ts.isoformat(), "cell_id": "expected", **{c: float(vals[c]) * factors[c] for c in PROFILE_COLUMNS}})
        return rows

    def profile_level_ratio(self, history_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], int]:
        """The carrier's own load level relative to the estate profile: per `PROFILE_COLUMNS`,
        mean(own value over the AWAKE rows of `history_rows`) / mean(profile at the same bins),
        clipped to `PROFILE_SCALE_RATIO_RANGE`. Rows carrying `active` != "ACTIVE" (the live rApp
        tags its PM history) or a non-finite value are skipped: a sleeping carrier's zero counters
        must not pull the level down. Returns ({column: ratio}, n_rows_used); every ratio is 1.0
        (no scaling) when fewer than `PROFILE_SCALE_MIN_ROWS` awake PRB rows are available, when
        the model has no profile, or when a column has no usable rows (that column then follows
        the PRB-DL ratio)."""
        neutral = {c: 1.0 for c in PROFILE_COLUMNS}
        if self.hourly_profile is None or not history_rows:
            return neutral, 0
        df = pd.DataFrame([dict(r) for r in history_rows])
        if "ts" not in df.columns:
            return neutral, 0
        awake = df["active"].astype(str).eq("ACTIVE") if "active" in df.columns else pd.Series(True, index=df.index)
        ts = pd.to_datetime(df["ts"], errors="coerce")
        awake &= ts.notna()
        if not awake.any():
            return neutral, 0
        keys = _profile_key(ts.loc[awake]).astype(int)
        prof = self.hourly_profile.reindex(keys.to_numpy())
        lo, hi = PROFILE_SCALE_RATIO_RANGE
        ratios: dict[str, float] = {}
        n_used = 0
        for c in PROFILE_COLUMNS:
            own = pd.to_numeric(df.loc[awake, c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index[awake])
            ok = own.notna() & np.isfinite(own.astype(float))
            ref = prof[c].to_numpy(dtype=float)
            if int(ok.sum()) >= PROFILE_SCALE_MIN_ROWS and float(np.nanmean(ref[ok.to_numpy()])) > 0:
                ratios[c] = float(np.clip(float(own[ok].mean()) / float(np.nanmean(ref[ok.to_numpy()])), lo, hi))
                if c == "prb_util_dl_pct":
                    n_used = int(ok.sum())
        if "prb_util_dl_pct" not in ratios:
            return neutral, 0
        return {c: ratios.get(c, ratios["prb_util_dl_pct"]) for c in PROFILE_COLUMNS}, n_used

    def asleep_forecast(self, sim_time: str, history_rows: Optional[Sequence[Mapping[str, Any]]] = None,
                        scale_weight: float = 0.0) -> tuple[float, dict[str, Any]]:
        """P(next window is low-load) for a carrier whose own PM is uninformative (SLEEP/INACTIVE),
        from the expected load at `sim_time`, plus the diagnostics of the forecast. With
        `scale_weight` w > 0 (config.py RAPP_PROFILE_SCALE) and a history the profile bins are
        multiplied per column by `1 + w * (ratio - 1)` with `ratio` from `profile_level_ratio`
        (w = 1: the profile at this carrier's own level). Returns (p_low, {"profile_scale": the
        PRB-DL factor applied (1.0 = estate profile), "profile_ratio": {column: ratio},
        "profile_scale_rows": awake rows the ratio used, "profile_scale_weight": w})."""
        w = min(max(float(scale_weight), 0.0), 1.0)
        ratio, n_rows = self.profile_level_ratio(history_rows or []) if w > 0 else ({c: 1.0 for c in PROFILE_COLUMNS}, 0)
        factors = {c: 1.0 + w * (r - 1.0) for c, r in ratio.items()}
        p = float(self.predict_low_load_probability(self.expected_window(sim_time, scale=factors)))
        return p, {"profile_scale": factors["prb_util_dl_pct"], "profile_ratio": ratio, "profile_scale_rows": n_rows,
                   "profile_scale_weight": w}

    def predict_low_load_probability_asleep(self, sim_time: str, history_rows: Optional[Sequence[Mapping[str, Any]]] = None,
                                            scale_weight: float = 0.0) -> float:
        """P(next window is low-load) for a carrier whose own PM is uninformative (SLEEP/INACTIVE):
        forecast from the expected load at `sim_time` instead of its zero counters (see
        `asleep_forecast` for the optional level scaling; the defaults keep the estate profile)."""
        return self.asleep_forecast(sim_time, history_rows, scale_weight)[0]

    # ---- persistence --------------------------------------------------------------------
    def save(self, path: Path, metrics_path: Optional[Path] = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "low_threshold": self.low_threshold,
                     "feature_columns": self.feature_columns, "metrics": self.metrics,
                     "hourly_profile": self.hourly_profile}, path)
        mp = Path(metrics_path) if metrics_path else path.with_suffix(".metrics.json")
        mp.write_text(json.dumps(self.metrics, indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: Path) -> "LowLoadModel":
        blob = joblib.load(Path(path))
        return cls(pipeline=blob["pipeline"], low_threshold=float(blob["low_threshold"]),
                   feature_columns=list(blob["feature_columns"]), metrics=dict(blob.get("metrics", {})),
                   hourly_profile=blob.get("hourly_profile"))


def train(df: pd.DataFrame, *, low_threshold: Optional[float] = None, holdout_frac: float = HOLDOUT_FRAC,
          seed: int = 42, data_kind: str = "synthetic") -> LowLoadModel:
    """Fit on the first `1-holdout_frac` of time, score on the rest; never shuffles windows."""
    t0 = time.perf_counter()
    df = to_canonical(df)
    report = validate(df)
    thr = float(df["prb_util_dl_pct"].quantile(LOW_QUANTILE)) if low_threshold is None else float(low_threshold)

    X = build_features(df)
    y = build_labels(df, thr)
    mask = y.notna()
    X, y, ts = X.loc[mask], y.loc[mask].astype(int), df.loc[mask, "ts"]
    uniq = np.sort(ts.unique())
    split_ts = uniq[min(int(len(uniq) * (1.0 - holdout_frac)), len(uniq) - 1)]
    tr = (ts < split_ts).to_numpy()
    te = ~tr
    if tr.sum() == 0 or te.sum() == 0:
        raise ValueError(f"time split produced empty partition (train={int(tr.sum())}, test={int(te.sum())})")

    model_type = "HistGradientBoostingClassifier"
    try:
        pipe = _gbdt_pipeline(seed)
        pipe.fit(X.loc[tr], y.loc[tr])
    except Exception as exc:  # pragma: no cover - environment dependent
        log.warning("gradient boosting unavailable (%s); falling back to LogisticRegression", exc)
        model_type = "LogisticRegression"
        pipe = _fallback_pipeline()
        pipe.fit(X.loc[tr], y.loc[tr])

    m = LowLoadModel(pipeline=pipe, low_threshold=thr, hourly_profile=build_hourly_profile(df.loc[mask]))
    proba = m.predict_proba_frame(X.loc[te])
    y_te = y.loc[te].to_numpy()
    auc = float(roc_auc_score(y_te, proba)) if len(np.unique(y_te)) == 2 else None
    acc = float(accuracy_score(y_te, (proba >= 0.5).astype(int)))
    m.metrics = {
        "auc": auc, "accuracy": acc, "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "low_threshold": thr, "low_threshold_source": "data-quantile-0.25" if low_threshold is None else "configured",
        "positive_rate": float(y.mean()), "model_type": model_type, "data_kind": data_kind,
        "n_cells": report["n_cells"], "share_30min": report["share_30min"],
        "train_seconds": round(time.perf_counter() - t0, 3), "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    log.info("trained %s: auc=%s acc=%.3f n_train=%d n_test=%d thr=%.2f%% (%.2fs)", model_type,
             f"{auc:.3f}" if auc is not None else "n/a", acc, m.metrics["n_train"], m.metrics["n_test"], thr,
             m.metrics["train_seconds"])
    return m


def load_or_train(loader: Callable[[], pd.DataFrame], cfg: Settings = settings) -> LowLoadModel:
    """Use the persisted model unless RAPP_RETRAIN=1, it is missing/unreadable, or it was
    trained on a different `RAPP_DATA` kind (a model trained on locally supplied data must not be
    silently reused for a synthetic run, and vice versa)."""
    path = cfg.rapp_model_path
    if not cfg.rapp_retrain and path.exists():
        try:
            m = LowLoadModel.load(path)
            if m.hourly_profile is None:
                log.info("persisted model %s has no hour-of-day profile (older format); retraining", path.name)
            elif m.metrics.get("data_kind", cfg.rapp_data) == cfg.rapp_data:
                log.info("loaded model from %s (auc=%s)", path.name, m.metrics.get("auc"))
                return m
            else:
                log.info("persisted model is for data_kind=%s, current=%s; retraining", m.metrics.get("data_kind"), cfg.rapp_data)
        except Exception as exc:
            log.warning("could not load %s (%s); retraining", path.name, exc)
    m = train(loader(), low_threshold=cfg.rapp_low_threshold, seed=cfg.rapp_seed, data_kind=cfg.rapp_data)
    m.save(path, cfg.rapp_model_metrics_path)
    return m
