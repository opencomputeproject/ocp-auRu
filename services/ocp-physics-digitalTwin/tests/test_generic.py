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

"""Generic mode: the twin runs for a stranger with no supplied data.

Every assertion here is about a fictional hex estate and public 3GPP defaults; nothing reads
`var/` or `/data`. `VAR_DIR` is pointed at a temp dir BEFORE `app.store` is imported so the
persisted calibrated state (gitignored, local) is never touched by the test process.
"""
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

_TMP_VAR = tempfile.mkdtemp(prefix="twin-test-var-")
os.environ["VAR_DIR"] = _TMP_VAR
os.environ["DATA_DIR"] = os.path.join(_TMP_VAR, "no-data")

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import store
from app.generic import (GENERIC_RUNG, band_label, default_params, generic_build_kwargs, generic_estate,
                         generic_model, hex_site_positions, is_generic_model, is_generic_topology)
from app.physics.calibrate import BOUNDS, NAMES
from app.physics.evaluator import Estate

REQUIRED = ["cell_id", "cell_lat", "cell_lon", "cell_az_deg", "cell_carrier_freq_mhz", "hTx",
            "beam_width_deg", "technology", "site_id"]


@pytest.fixture(autouse=True)
def _isolated_var(tmp_path, monkeypatch):
    """Belt and braces: even if another module imported `app.store` first, every test writes to
    its own temp dir."""
    monkeypatch.setattr(store, "VAR", tmp_path)
    yield


# ── estate + params ──────────────────────────────────────────────────────────

def test_hex_layout_is_equilateral_for_three_sites():
    pts = hex_site_positions(3, 500.0)
    d = [np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) for i, j in ((0, 1), (0, 2), (1, 2))]
    assert d == pytest.approx([500.0, 500.0, 500.0], abs=1e-6)
    ring = hex_site_positions(7, 500.0)
    assert ring[0] == (0.0, 0.0)
    assert all(np.hypot(x, y) == pytest.approx(500.0, abs=1e-6) for x, y in ring[1:])


def test_generic_estate_has_required_columns_and_no_nulls():
    topo, cfg = generic_estate()
    assert set(REQUIRED) <= set(topo.columns)
    assert list(cfg.columns) == ["cell_id", "cell_el_deg"]
    assert len(topo) == 3 * 3 * 2 and len(cfg) == len(topo)
    assert not topo[REQUIRED].isna().any().any()
    assert not cfg.isna().any().any()
    assert topo.cell_id.is_unique
    assert "g-s1-a-n78" in set(topo.cell_id) and "g-s3-c-n3" in set(topo.cell_id)
    assert sorted(topo.cell_az_deg.unique()) == [0.0, 120.0, 240.0]
    assert sorted(topo.cell_carrier_freq_mhz.unique()) == [1800.0, 3500.0]
    assert (topo.technology == "NR").all()
    # fictional coordinates: within a few km of (0, 0), open ocean
    assert topo.cell_lat.abs().max() < 0.1 and topo.cell_lon.abs().max() < 0.1
    # the estate builder's own guard accepts it (no defaulted geometry)
    est = Estate.build(topo, cfg, n_points=400)
    assert len(est.cells) == 18 and est.nr_serving_mask().all()


def test_generic_estate_kwargs_and_labels():
    topo, cfg = generic_estate(n_sites=2, isd_m=800.0, bands_mhz=(700.0,), h_tx_m=25.0,
                               tilt_deg=4.0, beam_width_deg=60.0, lat0=10.0, lon0=20.0)
    assert len(topo) == 2 * 3 * 1 and (cfg.cell_el_deg == 4.0).all()
    assert (topo.hTx == 25.0).all() and (topo.beam_width_deg == 60.0).all()
    assert topo.cell_id.str.endswith("-n28").all()
    assert topo.cell_lat.between(9.9, 10.1).all() and topo.cell_lon.between(19.9, 20.1).all()
    assert band_label(3500.0) == "n78" and band_label(1800.0) == "n3" and band_label(4321.0) == "4321mhz"
    with pytest.raises(ValueError):
        generic_estate(bands_mhz=())


def test_default_params_cover_every_fitted_name_inside_bounds():
    p = default_params()
    assert set(p) == set(NAMES)
    for name, value in p.items():
        lo, hi = BOUNDS[name]
        assert lo <= value <= hi, f"{name}={value} outside {BOUNDS[name]}"
    # the physical EIRP window the calibration gate enforces holds for the defaults too
    assert all(25.0 <= p[k] + 17.0 <= 45.0 for k in ("p0_low", "p0_mid", "p0_hi"))


def test_generic_model_shape():
    m = generic_model(estate_kwargs={"n_sites": 3})
    assert m["rung"] == GENERIC_RUNG and "generic" in m["rung"] and "uncalibrated" in m["rung"]
    assert m["gate"] is None and m["params"] == default_params() and m["estate"] == {"n_sites": 3}
    assert is_generic_model(m) and not is_generic_model(None) and not is_generic_model({"params": {}, "gate": {}})


def test_is_generic_topology_recognises_only_the_generated_ids():
    topo, _ = generic_estate(n_sites=2)
    assert is_generic_topology(topo)
    real = topo.copy()
    real.loc[0, "cell_id"] = "real-site-a-n3"
    assert not is_generic_topology(real)
    assert not is_generic_topology(None) and not is_generic_topology(topo.iloc[0:0])


# ── physics with default params ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def estate():
    """Built the way the service builds it: the hex overrides (tier offset √n·ISD, sampling
    radius ISD/√3) are part of generic mode, not an option."""
    topo, cfg = generic_estate()
    return Estate.build(topo, cfg, n_points=1500, **generic_build_kwargs(3, 500.0))


def test_build_kwargs_are_the_textbook_hex_values():
    bk = generic_build_kwargs(3, 500.0)
    assert bk["d_wrap_m"] == pytest.approx(500.0 * np.sqrt(3.0))
    assert bk["r_max_m"] == pytest.approx(500.0 / np.sqrt(3.0))
    topo, cfg = generic_estate()
    est = Estate.build(topo, cfg, n_points=300, **bk)
    assert est.cells.attrs["d_wrap"] == pytest.approx(bk["d_wrap_m"])
    # every grid point lies within r_max of some site (annulus union), and beyond the 100 m floor
    sites = np.unique(np.column_stack([est.cells.x, est.cells.y]), axis=0)
    d = np.min(np.hypot(est.grid_x[:, None] - sites[None, :, 0], est.grid_y[:, None] - sites[None, :, 1]), axis=1)
    assert d.max() <= bk["r_max_m"] + 1e-6 and d.min() >= 100.0 - 1e-6
    # the calibrated construction is untouched: no overrides → the old defaults
    est0 = Estate.build(topo, cfg, n_points=300)
    assert est0.cells.attrs["d_wrap"] == pytest.approx(500.0)   # mean pairwise site distance of the triangle


@pytest.mark.parametrize("aggregate", [True, False])
def test_evaluate_defaults_sane_ranges(estate, aggregate):
    r = estate.evaluate(default_params(), serving_mask=estate.nr_serving_mask(), aggregate=aggregate)
    g = r["guardrail_kpis"]
    assert 0.0 < g["coverage_rate"] <= 1.0
    assert 0.0 <= g["outage_rate"] < 1.0
    assert -140.0 < g["rsrp_p5"] <= g["rsrp_p50"] <= g["rsrp_p95"] < -40.0
    assert -30.0 < g["sinr_p5"] <= g["sinr_p50"] <= g["sinr_p95"] < 40.0
    assert g["samples"] > 0
    assert abs(sum(r["serving_share"].values()) - 1.0) < 1e-3


def test_switching_a_cell_off_changes_serving_and_never_improves_coverage(estate):
    p = default_params()
    mask = estate.nr_serving_mask()
    base = estate.evaluate(p, serving_mask=mask)
    off = estate.evaluate(p, off_cells={"g-s1-a-n78"}, serving_mask=mask)
    assert "g-s1-a-n78" in base["serving_share"]
    assert "g-s1-a-n78" not in off["serving_share"]
    assert off["serving_share"] != base["serving_share"]
    # fewer serving cells: coverage cannot go up (points re-attach to a weaker cell or drop out);
    # outage may move either way (the off cell no longer interferes) but stays a rate.
    assert off["guardrail_kpis"]["coverage_rate"] <= base["guardrail_kpis"]["coverage_rate"]
    assert off["guardrail_kpis"]["rsrp_p50"] <= base["guardrail_kpis"]["rsrp_p50"] + 1e-9
    assert 0.0 <= off["guardrail_kpis"]["outage_rate"] < 1.0


def test_generic_is_deterministic():
    topo, cfg = generic_estate()
    bk = generic_build_kwargs(3, 500.0)
    a = Estate.build(topo, cfg, n_points=600, **bk).evaluate(default_params())["guardrail_kpis"]
    b = Estate.build(topo, cfg, n_points=600, **bk).evaluate(default_params())["guardrail_kpis"]
    assert a == b
    t2, c2 = generic_estate()
    pd.testing.assert_frame_equal(topo, t2)
    pd.testing.assert_frame_equal(cfg, c2)


# ── the service ──────────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("TWIN_GENERIC", raising=False)
    from app import main
    main._state.update(estate=None, targets=None, model=None)
    main._state.pop("mr_path", None)
    with TestClient(main.app) as c:
        yield c


def test_service_load_generic_then_health_evaluate_cells(client, tmp_path):
    h = client.get("/health").json()
    assert h["estate_loaded"] is False and h["calibrated"] is False and h["generic"] is False
    assert h["model"] is None and h["rung"] is None     # no model: no rung to advertise
    assert client.get("/v1/cells").status_code == 409

    r = client.post("/v1/load-generic", json={"n_sites": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rung"] == GENERIC_RUNG and body["loaded"]["cells"] == 18
    assert body["loaded"]["nr_serving_cells"] == 18 and body["loaded"]["targets_loaded"] is False
    assert body["params"] == default_params() and body["gate"] is None
    assert (tmp_path / "model.json").exists() and (tmp_path / "topology.csv").exists()

    h = client.get("/health").json()
    assert h["estate_loaded"] is True
    assert h["calibrated"] is False and h["generic"] is True and h["model"] == "generic"
    assert h["rung"] == GENERIC_RUNG

    cells = client.get("/v1/cells").json()
    assert cells["count"] == 18 and cells["cells"][0]["cell_id"].startswith("g-s1-")

    ev = client.post("/v1/evaluate", json={"cell_configs": [{"cell_id": "g-s1-a-n78", "on_off": False}]})
    assert ev.status_code == 200, ev.text
    e = ev.json()
    assert e["rung"] == GENERIC_RUNG
    assert 0.0 < e["guardrail_kpis"]["coverage_rate"] <= 1.0
    assert e["delta_vs_baseline"]["coverage_rate"] <= 0.0
    assert "g-s1-a-n78" not in e["serving_share"]

    unknown = client.post("/v1/evaluate", json={"cell_configs": [{"cell_id": "nope", "on_off": False}]})
    assert unknown.status_code == 422

    sweep = client.post("/v1/tilt-sweep", json={"cell_id": "g-s1-a-n78", "delta_range_deg": 2, "step_deg": 2})
    assert sweep.status_code == 200 and sweep.json()["rung"] == GENERIC_RUNG
    assert len(sweep.json()["sweep"]) == 3

    att = client.post("/v1/attach", json={"points": [{"lon": 0.001, "lat": 0.001}]})
    assert att.status_code == 200 and att.json()["rung"] == GENERIC_RUNG
    assert att.json()["serving_cell_id"][0].startswith("g-s")

    # /v1/train needs calibration targets — generic mode has none, by design
    assert client.post("/v1/train").status_code == 422


def test_service_load_generic_rejects_unknown_keys(client):
    """A typo must not silently load the default estate with HTTP 200."""
    r = client.post("/v1/load-generic", json={"nsites": 7})
    assert r.status_code == 422, r.text
    assert client.post("/v1/load-generic", json={"bogus": 1}).status_code == 422
    assert client.post("/v1/load-generic", json={"n_sites": 2}).json()["loaded"]["cells"] == 12


def test_service_generic_env_autoloads_on_startup(monkeypatch, tmp_path):
    monkeypatch.setenv("TWIN_GENERIC", "1")
    from app import main
    main._state.update(estate=None, targets=None, model=None)
    with TestClient(main.app) as c:
        h = c.get("/health").json()
        assert h["estate_loaded"] and h["generic"] and h["rung"] == GENERIC_RUNG
        assert h["calibrated"] is False
        assert c.get("/v1/cells").json()["count"] == 18


def test_service_calibrated_path_untouched(client):
    """A persisted calibration report (the shape `/v1/train` writes) keeps its own rung and
    `calibrated=true`; nothing generic leaks into it. No supplied data involved: the params here
    are the test fixture's, not a fitted set."""
    from app import main
    topo, cfg = generic_estate(n_sites=1)
    main._rebuild(topo, cfg, {"rsrp": [-110, -100, -95, -90, -80], "sinr": [-5, 0, 5, 10, 15]})
    fake_fit = {"params": default_params(), "gate": {"hard_pass": False}, "loss": 1.0}
    main._state["model"] = fake_fit          # a fit report without an explicit rung key
    h = client.get("/health").json()
    assert h["calibrated"] is True and h["generic"] is False and h["model"] == "calibrated"
    assert h["rung"] == main.RUNG
    e = client.post("/v1/evaluate", json={}).json()
    assert e["rung"] == main.RUNG


def test_service_load_local_drops_generic_model(client, tmp_path):
    """Loading real inputs over a generic placeholder must not leave the generic params attached
    to a real estate (a generic model over supplied geometry would masquerade as calibrated-looking
    output)."""
    from app import main
    assert client.post("/v1/load-generic", json={}).status_code == 200
    assert (tmp_path / "model.json").exists()
    # load-local is exercised through /v1/upload (same code path for the model reset) with a
    # second fictional estate, so no local files are needed.
    topo, cfg = generic_estate(n_sites=1, lat0=1.0, lon0=1.0)
    files = {"topology": ("t.csv", topo.to_csv(index=False), "text/csv"),
             "config": ("c.csv", cfg.to_csv(index=False), "text/csv")}
    r = client.post("/v1/upload", files=files)
    assert r.status_code == 200, r.text
    assert r.json()["model_reset"] == "generic model dropped"
    assert r.json()["rung"] is None          # a load carries no model rung; /v1/train assigns one
    h = client.get("/health").json()
    assert h["estate_loaded"] and h["calibrated"] is False and h["generic"] is False
    assert h["model"] is None and h["rung"] is None
    assert not (tmp_path / "model.json").exists()
    assert client.post("/v1/evaluate", json={}).status_code == 409
    assert client.get("/v1/cells").json()["rung"] is None


def _persist_fake_fit(var_dir):
    """The files `/v1/load-local` + `/v1/train` leave behind — with a fictional estate and the
    generic defaults standing in for fitted values (no fitted number anywhere)."""
    from app import main
    topo, cfg = generic_estate(n_sites=1, lat0=2.0, lon0=2.0)
    store.save_inputs(topo, cfg, {"rsrp": [-110, -100, -95, -90, -80], "sinr": [-5, 0, 5, 10, 15]})
    store.save_model({"params": default_params(), "gate": {"hard_pass": True}, "loss": 1.0, "rung": main.RUNG})


def test_restart_restores_a_persisted_calibrated_fit_untouched(monkeypatch, tmp_path):
    monkeypatch.delenv("TWIN_GENERIC", raising=False)
    from app import main
    _persist_fake_fit(tmp_path)
    main._state.update(estate=None, targets=None, model=None)
    with TestClient(main.app) as c:
        h = c.get("/health").json()
        assert h["calibrated"] is True and h["generic"] is False and h["rung"] == main.RUNG
        assert c.get("/v1/cells").json()["count"] == 6
        assert c.post("/v1/evaluate", json={}).json()["rung"] == main.RUNG


def test_restart_regenerates_a_persisted_generic_model(monkeypatch, tmp_path):
    monkeypatch.delenv("TWIN_GENERIC", raising=False)
    from app import main
    main._state.update(estate=None, targets=None, model=None)
    with TestClient(main.app) as c:
        assert c.post("/v1/load-generic", json={"n_sites": 4}).json()["loaded"]["cells"] == 24
    main._state.update(estate=None, targets=None, model=None)
    with TestClient(main.app) as c:          # restart without the env: the generic choice survives
        h = c.get("/health").json()
        assert h["generic"] is True and h["rung"] == GENERIC_RUNG
        assert c.get("/v1/cells").json()["count"] == 24


def _upload_fictional_estate(c, **kw):
    topo, cfg = generic_estate(n_sites=1, **kw)
    topo = topo.assign(cell_id=topo.cell_id.str.replace("g-s", "real-s", regex=False))   # not the generated ids
    files = {"topology": ("t.csv", topo.to_csv(index=False), "text/csv"),
             "config": ("c.csv", cfg.assign(cell_id=topo.cell_id).to_csv(index=False), "text/csv")}
    return c.post("/v1/upload", files=files)


def test_generic_env_never_clobbers_persisted_real_inputs_without_a_fit(monkeypatch, tmp_path):
    """Real inputs loaded but never trained (no model.json) are somebody's estate too: TWIN_GENERIC=1
    must not replace topology/config/targets on disk with the generic estate and an empty targets.json."""
    monkeypatch.setenv("TWIN_GENERIC", "1")
    from app import main
    topo, cfg = generic_estate(n_sites=1, lat0=3.0, lon0=3.0)
    topo = topo.assign(cell_id=topo.cell_id.str.replace("g-s", "real-s", regex=False))
    targets = {"rsrp": [-110, -100, -95, -90, -80], "sinr": [-5, 0, 5, 10, 15]}
    store.save_inputs(topo, cfg.assign(cell_id=topo.cell_id), targets)
    before = {n: (tmp_path / n).read_text() for n in ("topology.csv", "config.csv", "targets.json")}
    main._state.update(estate=None, targets=None, model=None)
    with TestClient(main.app) as c:
        h = c.get("/health").json()
        assert h["generic"] is True and h["rung"] == GENERIC_RUNG        # generic in memory ...
        assert {n: (tmp_path / n).read_text() for n in before} == before  # ... nothing on disk touched
        assert not (tmp_path / "model.json").exists()
        r = c.post("/v1/load-generic", json={}).json()
        assert r["persisted"] is False and "non-generic estate inputs" in r["note"]
        r = c.post("/v1/load-generic", json={"overwrite_calibrated": True}).json()
        assert r["persisted"] is True and is_generic_topology(pd.read_csv(tmp_path / "topology.csv"))


def test_generic_env_over_a_calibrated_fit_then_upload_keeps_the_fit_on_disk(monkeypatch, tmp_path):
    """TWIN_GENERIC=1 over a persisted fit keeps the generic model in memory only; uploading real
    inputs afterwards drops that in-memory generic model and must NOT unlink the calibrated
    model.json underneath it."""
    monkeypatch.setenv("TWIN_GENERIC", "1")
    from app import main
    _persist_fake_fit(tmp_path)
    before = (tmp_path / "model.json").read_text()
    main._state.update(estate=None, targets=None, model=None)
    with TestClient(main.app) as c:
        assert c.get("/health").json()["generic"] is True
        r = _upload_fictional_estate(c, lat0=4.0, lon0=4.0)
        assert r.status_code == 200, r.text
        assert r.json()["model_reset"] == "generic model dropped" and r.json()["rung"] is None
        assert (tmp_path / "model.json").read_text() == before           # the fit survives
        h = c.get("/health").json()
        assert h["generic"] is False and h["calibrated"] is False and h["model"] is None and h["rung"] is None


def test_generic_env_never_clobbers_a_persisted_calibrated_fit(monkeypatch, tmp_path):
    monkeypatch.setenv("TWIN_GENERIC", "1")
    from app import main
    _persist_fake_fit(tmp_path)
    before = (tmp_path / "model.json").read_text()
    main._state.update(estate=None, targets=None, model=None)
    with TestClient(main.app) as c:
        h = c.get("/health").json()
        assert h["generic"] is True and h["rung"] == GENERIC_RUNG      # explicit env wins in memory
        assert (tmp_path / "model.json").read_text() == before          # the fit on disk is untouched
        r = c.post("/v1/load-generic", json={}).json()
        assert r["persisted"] is False and "calibrated model exists" in r["note"]
        r = c.post("/v1/load-generic", json={"overwrite_calibrated": True}).json()
        assert r["persisted"] is True
        assert is_generic_model(store.load_model())
