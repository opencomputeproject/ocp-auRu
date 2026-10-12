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

"""Physics Twin: standalone RF twin service.

GUARD-RAIL: a standalone service with no imports from any caller; any local input data is
mounted read-only at runtime and never enters the image or this repository. The response
vocabulary is `cell_id` / `cell_el_deg` / `on_off` / `guardrail_kpis`.

Two rungs, one service. A model is either a CALIBRATION FIT (`POST /v1/train` on locally supplied
measurement data → rung "simulation, calibrated to supplied measurement data") or the GENERIC placeholder
(`POST /v1/load-generic` or `TWIN_GENERIC=1` → a fictional hex estate with uncalibrated 3GPP
defaults → rung "simulation, generic 3GPP defaults (uncalibrated)"). Every response carries the
rung of the model that produced it; `/health` says which one is loaded.
"""
from __future__ import annotations

import io
import logging
import os
import pathlib
import time
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from app import store
from app.generic import (GENERIC_RUNG, default_params, generic_build_kwargs, generic_estate,
                         generic_model, is_generic_model, is_generic_topology)
from app.physics.calibrate import (fit, holdout_check, targets_from_mr_report,
                                   targets_from_mr_report_split)
from app.physics.evaluator import Estate

log = logging.getLogger("physics-twin")

DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", "/data"))
RUNG = "simulation, calibrated to supplied measurement data"

app = FastAPI(title="Physics Twin", version="0.2.0")

_state: dict[str, Any] = {"estate": None, "targets": None, "model": None}


def _rebuild(topology: pd.DataFrame, config: pd.DataFrame, targets: dict | None,
             **build_kwargs: Any) -> dict:
    estate = Estate.build(topology, config, **build_kwargs)
    _state.update(estate=estate, targets=targets)
    return {
        "cells": int(len(estate.cells)),
        "nr_serving_cells": int(estate.nr_serving_mask().sum()),
        "grid_points": int(len(estate.grid_x)),
        "targets_loaded": targets is not None,
    }


def _rung() -> str | None:
    """The rung of the loaded model — None when no model is loaded (a rung is earned by a model:
    an empty twin has none to report). A calibration fit written by an older `/v1/train` may lack
    the key; it is the only kind of model that can, so the calibrated constant is its fallback."""
    m = _state["model"]
    if not m:
        return None
    if m.get("rung"):
        return str(m["rung"])
    return RUNG


def _generic_env() -> bool:
    return os.getenv("TWIN_GENERIC", "").strip().lower() in {"1", "true", "yes", "on"}


def _protected_state() -> str | None:
    """Why VAR_DIR must not be overwritten by generic state, or None when it may be: a persisted
    CALIBRATION fit, or persisted estate inputs that are not the generic hex grid (somebody's
    estate loaded via /v1/upload or /v1/load-local, fitted or not)."""
    existing = store.load_model()
    if existing and not is_generic_model(existing):
        return "a persisted calibrated model exists in VAR_DIR"
    topo, _cfg, _targets = store.load_inputs()
    if topo is not None and not is_generic_topology(topo):
        return "persisted non-generic estate inputs exist in VAR_DIR (no fit yet)"
    return None


def _load_generic(kwargs: dict[str, Any], *, persist: bool, overwrite_calibrated: bool = False) -> dict:
    """Build the fictional estate, install the generic model, optionally persist both.

    Persistence never silently replaces persisted CALIBRATION state (a fit, which takes minutes
    to compute, or non-generic estate inputs awaiting one): unless `overwrite_calibrated`, the
    generic model then lives in memory only and the response says so."""
    kwargs = dict(kwargs)
    topo, cfg = generic_estate(**kwargs)
    build = generic_build_kwargs(int(kwargs.get("n_sites", 3)), float(kwargs.get("isd_m", 500.0)))
    info = _rebuild(topo, cfg, None, **build)
    model = generic_model(default_params(), estate_kwargs=kwargs)
    model["build"] = build
    _state["model"] = model
    _state.pop("mr_path", None)
    persisted, note = False, None
    if persist:
        why = None if overwrite_calibrated else _protected_state()
        if why:
            note = (f"{why}; the generic model is loaded in memory only "
                    "(pass overwrite_calibrated=true to replace it)")
            log.warning("load-generic: %s", note)
        else:
            store.save_inputs(topo, cfg, {})
            store.save_model(model)
            persisted = True
    return {"loaded": info, "rung": GENERIC_RUNG, "params": model["params"], "gate": None,
            "estate": kwargs, "build": build, "persisted": persisted, "note": note or model["note"]}


@app.on_event("startup")
def _restore() -> None:
    model = store.load_model()
    if _generic_env():
        # Explicit generic mode: always a generic estate. A persisted generic model's estate
        # kwargs are honoured (so a 7-site choice survives a restart); a persisted calibrated
        # model is left untouched on disk and NOT loaded.
        kwargs = (model.get("estate") or {}) if is_generic_model(model) else {}
        out = _load_generic(kwargs, persist=True)
        log.info("TWIN_GENERIC=1: generic estate loaded (%s cells), rung=%r, persisted=%s",
                 out["loaded"]["cells"], GENERIC_RUNG, out["persisted"])
        return
    if is_generic_model(model):
        # The generic estate is a pure function of its kwargs: regenerate rather than trusting
        # CSVs that may have been edited by hand.
        _load_generic(model.get("estate") or {}, persist=False)
        return
    t, c, g = store.load_inputs()
    if t is not None:
        _rebuild(t, c, g)
        _state["model"] = model


@app.get("/health")
def health() -> dict:
    m = _state["model"]
    generic = is_generic_model(m)
    return {
        "status": "ok",
        "estate_loaded": _state["estate"] is not None,
        "calibrated": m is not None and not generic,
        "generic": generic,
        "model": "generic" if generic else ("calibrated" if m is not None else None),
        "rung": _rung(),
    }


def _drop_generic_model(out: dict) -> dict:
    """Non-generic estate inputs arriving over a generic placeholder: the generic params must not stay
    attached to supplied geometry (the output would look calibrated and be neither). Only a
    persisted GENERIC model.json is removed: with TWIN_GENERIC=1 started over a calibrated fit the
    generic model lives in memory only and the fit on disk stays untouched."""
    if is_generic_model(_state["model"]):
        _state["model"] = None
        if is_generic_model(store.load_model()):
            store.delete_model()
        out["model_reset"] = "generic model dropped"
        log.info("non-generic inputs loaded over a generic model: generic model dropped (POST /v1/train to calibrate)")
    return out


@app.post("/v1/upload")
async def upload(topology: UploadFile = File(...), config: UploadFile = File(...),
                 mr_report: UploadFile | None = File(default=None)) -> dict:
    topo = pd.read_csv(io.BytesIO(await topology.read()))
    cfg = pd.read_csv(io.BytesIO(await config.read()))
    targets = None
    if mr_report is not None:
        tmp = store.VAR / "mr_report.xlsx"
        tmp.write_bytes(await mr_report.read())
        targets = targets_from_mr_report(str(tmp))
    try:
        info = _rebuild(topo, cfg, targets)
    except ValueError as exc:            # the refuse-on-defaults guard
        raise HTTPException(422, str(exc)) from exc
    store.save_inputs(topo, cfg, targets or {})
    # A load carries no model rung: the calibrated rung is assigned by /v1/train only.
    return _drop_generic_model({"loaded": info, "rung": None})


class LoadLocal(BaseModel):
    # paths relative to DATA_DIR
    topology: str = "topology.csv"
    config: str = "config.csv"
    mr_report: str = "mr_report.xlsx"


@app.post("/v1/load-local")
def load_local(body: LoadLocal) -> dict:
    paths = [DATA_DIR / body.topology, DATA_DIR / body.config, DATA_DIR / body.mr_report]
    for p in paths:
        if not p.resolve().is_relative_to(DATA_DIR.resolve()):
            raise HTTPException(400, "paths must stay inside DATA_DIR")
        if not p.exists():
            raise HTTPException(404, f"not found: {p}")
    topo, cfg = pd.read_csv(paths[0]), pd.read_csv(paths[1])
    targets = targets_from_mr_report(str(paths[2]))
    try:
        info = _rebuild(topo, cfg, targets)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    store.save_inputs(topo, cfg, targets)
    _state["mr_path"] = str(paths[2])
    return _drop_generic_model({"loaded": info, "targets": targets, "rung": None})


class LoadGeneric(BaseModel):
    """`app.generic.generic_estate` kwargs — all optional; the defaults are the public demo estate.
    Unknown keys are a 422 (a typo must not silently load the default estate)."""
    model_config = ConfigDict(extra="forbid")
    n_sites: int = Field(default=3, ge=1, le=61)
    isd_m: float = Field(default=500.0, gt=0.0)
    bands_mhz: list[float] = Field(default_factory=lambda: [1800.0, 3500.0], min_length=1)
    h_tx_m: float = Field(default=30.0, gt=1.5)
    tilt_deg: float = 6.0
    beam_width_deg: float = Field(default=65.0, gt=0.0)
    lat0: float = Field(default=0.0, ge=-80.0, le=80.0)
    lon0: float = Field(default=0.0, ge=-180.0, le=180.0)
    # Replace persisted CALIBRATED state (fit and/or non-generic estate inputs) in VAR_DIR — never done implicitly.
    overwrite_calibrated: bool = False


@app.post("/v1/load-generic")
def load_generic(body: LoadGeneric | None = None) -> dict:
    """Generic mode: a fictional hex estate + uncalibrated 3GPP defaults, no supplied data.
    Every later response carries the generic rung; `/v1/train` stays unavailable (no targets)."""
    body = body or LoadGeneric()
    kwargs = body.model_dump(exclude={"overwrite_calibrated"})
    kwargs["bands_mhz"] = tuple(float(f) for f in kwargs["bands_mhz"])
    try:
        out = _load_generic(kwargs, persist=True, overwrite_calibrated=body.overwrite_calibrated)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    out["estate"] = {**out["estate"], "bands_mhz": list(kwargs["bands_mhz"])}
    return out


@app.get("/v1/cells")
def cells() -> dict:
    est = _require_estate()
    out = est.cells.drop(columns=["x", "y"]).to_dict("records")
    return {"cells": out, "count": len(out), "rung": _rung()}


class TrainBody(BaseModel):
    holdout_frac: float | None = None   # e.g. 0.3 → 70/30 device split; None = fit on everything
    seed: int = 42


@app.post("/v1/train")
def train(body: TrainBody | None = None) -> dict:
    est = _require_estate()
    if not _state["targets"]:
        raise HTTPException(422, "no calibration targets loaded: upload the measurement report first")
    body = body or TrainBody()
    t0 = time.time()
    if body.holdout_frac:
        if not _state.get("mr_path"):
            raise HTTPException(422, "holdout split needs the measurement report path: POST /v1/load-local first")
        train_t, hold_t, meta = targets_from_mr_report_split(
            _state["mr_path"], holdout_frac=body.holdout_frac, seed=body.seed)
        report = fit(est, train_t)
        report["split"] = meta
        report["holdout"] = holdout_check(report, hold_t)
    else:
        report = fit(est, _state["targets"])
    report["fit_seconds"] = round(time.time() - t0, 1)
    report["rung"] = RUNG
    _state["model"] = report
    store.save_model(report)
    return report


class CellConfigItem(BaseModel):
    cell_id: str
    cell_el_deg: float | None = None     # electrical tilt, degrees
    on_off: bool | None = None


class EvaluateBody(BaseModel):
    cell_configs: list[CellConfigItem] = Field(default_factory=list)
    nr_serving_only: bool = True
    # device_mean (default) reproduces the MR report's view; point_sample is the optimization view.
    aggregate: bool = True


@app.post("/v1/evaluate")
def evaluate(body: EvaluateBody) -> dict:
    est, params = _require_model()
    known = set(est.cells.cell_id)
    unknown = [i.cell_id for i in body.cell_configs if i.cell_id not in known]
    if unknown:
        # Unknown cells are refused, never ignored: a typo must not evaluate the baseline silently.
        raise HTTPException(422, f"unknown cells: {unknown}")
    tilt = {i.cell_id: i.cell_el_deg for i in body.cell_configs if i.cell_el_deg is not None}
    off = {i.cell_id for i in body.cell_configs if i.on_off is False}
    mask = est.nr_serving_mask() if body.nr_serving_only else None

    base = est.evaluate(params, serving_mask=mask, aggregate=body.aggregate)
    prop = est.evaluate(params, tilt_override=tilt, off_cells=off, serving_mask=mask,
                        aggregate=body.aggregate)
    delta = {k: round(prop["guardrail_kpis"][k] - base["guardrail_kpis"][k], 3)
             for k in ("rsrp_p5", "sinr_p5", "outage_rate", "coverage_rate")}
    return {
        "guardrail_kpis": {k: round(v, 3) for k, v in prop["guardrail_kpis"].items()},
        "baseline_kpis": {k: round(v, 3) for k, v in base["guardrail_kpis"].items()},
        "delta_vs_baseline": delta,
        "serving_share": prop["serving_share"],
        "mode": "device_mean" if body.aggregate else "point_sample",
        "rung": _rung(),
    }


class AttachPoint(BaseModel):
    lon: float
    lat: float


class AttachBody(BaseModel):
    points: list[AttachPoint]
    cell_configs: list[CellConfigItem] = Field(default_factory=list)
    nr_serving_only: bool = True
    # True when the caller's cell list is the WHOLE live estate (the caller removed the off
    # cells before calling): any known cell absent from the list is treated as off.
    cells_are_exhaustive: bool = False


@app.post("/v1/attach")
def attach(body: AttachBody) -> dict:
    """The attach contract: per-UE coordinates in, per-UE radio out.

    A caller posts a frame of UE positions with a config overlay; the response columns are
    `rsrp_dbm`, `sinr_db` and `serving_cell_id`, one value per point."""
    est, params = _require_model()
    import numpy as np
    known = set(est.cells.cell_id)
    unknown = [i.cell_id for i in body.cell_configs if i.cell_id not in known]
    if unknown:
        raise HTTPException(422, f"unknown cells: {unknown}")
    tilt = {i.cell_id: i.cell_el_deg for i in body.cell_configs if i.cell_el_deg is not None}
    off = {i.cell_id for i in body.cell_configs if i.on_off is False}
    if body.cells_are_exhaustive:
        off |= known - {i.cell_id for i in body.cell_configs}
    lons = np.array([p.lon for p in body.points]); lats = np.array([p.lat for p in body.points])
    out = est.attach_points(params, lons, lats, tilt_by_cell=tilt, off_cells=off,
                            nr_serving_only=body.nr_serving_only)
    return {"rsrp_dbm": [round(float(v), 3) for v in out["rsrp_dbm"]],
            "sinr_db": [round(float(v), 3) for v in out["sinr_db"]],
            "serving_cell_id": list(map(str, out["serving_cell_id"])),
            "samples": len(body.points), "rung": _rung()}


class SweepBody(BaseModel):
    cell_id: str
    delta_range_deg: float = 6.0
    step_deg: float = 2.0
    nr_serving_only: bool = True
    # Optimization defaults to the point-sample field: device-mean averaging blunts the very
    # response a tilt optimiser needs to see.
    aggregate: bool = False


@app.post("/v1/tilt-sweep")
def tilt_sweep(body: SweepBody) -> dict:
    est, params = _require_model()
    row = est.cells[est.cells.cell_id == body.cell_id]
    if row.empty:
        raise HTTPException(422, f"unknown cell: {body.cell_id}")
    base_tilt = float(row.tilt_deg.iloc[0])
    mask = est.nr_serving_mask() if body.nr_serving_only else None
    import numpy as np
    points = []
    for d in np.arange(-body.delta_range_deg, body.delta_range_deg + 1e-9, body.step_deg):
        r = est.evaluate(params, tilt_override={body.cell_id: base_tilt + float(d)},
                         serving_mask=mask, aggregate=body.aggregate)
        g = r["guardrail_kpis"]
        entry = {"tilt_deg": round(base_tilt + float(d), 1),
                 "delta_deg": round(float(d), 1),
                 "sinr_p5": round(g["sinr_p5"], 3),
                 "rsrp_p5": round(g["rsrp_p5"], 3),
                 "coverage_rate": round(g["coverage_rate"], 4)}
        # Cell-scoped KPIs (point mode): estate-wide p5 barely moves for one cell of many — the 5th-
        # percentile points mostly live in OTHER cells' areas — so the estate KPI understates the
        # very response a per-cell tilt optimiser is searching. Scope to the swept cell's served
        # points and the control surface becomes visible.
        if r.get("_serving") is not None:
            sel = r["_serving"] == body.cell_id
            if sel.any():
                a = r["_arrays"]
                entry["cell_served_points"] = int(sel.sum())
                entry["cell_sinr_p50"] = round(float(np.median(a["sinr"][sel])), 3)
                entry["cell_sinr_p5"] = round(float(np.percentile(a["sinr"][sel], 5)), 3)
                entry["cell_rsrp_p50"] = round(float(np.median(a["rsrp"][sel])), 3)
        points.append(entry)
    return {"cell_id": body.cell_id, "baseline_tilt_deg": base_tilt,
            "sweep": points, "mode": "device_mean" if body.aggregate else "point_sample",
            "rung": _rung()}


def _require_estate() -> Estate:
    if _state["estate"] is None:
        raise HTTPException(409, "no estate loaded — POST /v1/upload, /v1/load-local or /v1/load-generic first")
    return _state["estate"]


def _require_model():
    est = _require_estate()
    if _state["model"] is None:
        raise HTTPException(409, "no model — POST /v1/train (calibrated) or /v1/load-generic (generic) first")
    return est, _state["model"]["params"]
