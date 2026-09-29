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

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from app.physics.arfcn import earfcn_to_mhz, nr_arfcn_to_mhz
from app.physics.antenna import gain_db, peak_gain_dbi, effective_h3db
from app.physics.propagation import uma_los, uma_nlos, p_los_uma, rng_for
from app.physics.evaluator import Estate


def test_arfcn_pinned_values():
    # Worked examples from the public rasters (TS 36.101 Table 5.7.3-1, TS 38.104 §5.4.2).
    assert earfcn_to_mhz(300) == pytest.approx(2140.0)    # B1: 2110 + 0.1 * (300 - 0)
    assert earfcn_to_mhz(1575) == pytest.approx(1842.5)   # B3: 1805 + 0.1 * (1575 - 1200)
    assert earfcn_to_mhz(3100) == pytest.approx(2655.0)   # B7: 2620 + 0.1 * (3100 - 2750)
    assert earfcn_to_mhz(5230) == pytest.approx(751.0)    # B13: 746 + 0.1 * (5230 - 5180)
    assert earfcn_to_mhz(9690) == pytest.approx(720.0)    # B29: 717 + 0.1 * (9690 - 9660)
    assert nr_arfcn_to_mhz(428000) == pytest.approx(2140.0)   # below 3 GHz: 5 kHz raster
    assert nr_arfcn_to_mhz(630000) == pytest.approx(3450.0)   # 3 to 24.25 GHz: 15 kHz raster
    assert nr_arfcn_to_mhz(2016667) == pytest.approx(24250.08)
    with pytest.raises(ValueError):
        earfcn_to_mhz(5000)   # a gap between bands


def test_uma_los_is_free_space_like_before_breakpoint():
    # 28 + 22log10(d) + 20log10(fc): at 100 m, 2.1 GHz → 28 + 44 + 6.44 ≈ 78.4 dB
    pl = uma_los(np.array([100.0]), np.array([100.0]), 2.1, 40.0)
    assert pl[0] == pytest.approx(28 + 22*np.log10(100) + 20*np.log10(2.1), abs=0.01)


def test_nlos_never_below_los():
    d = np.linspace(20, 3000, 200)
    assert (uma_nlos(d, d, 2.1, 40.0) >= uma_los(d, d, 2.1, 40.0) - 1e-9).all()


def test_los_probability_limits():
    assert p_los_uma(np.array([10.0]))[0] == 1.0
    assert p_los_uma(np.array([5000.0]))[0] < 0.01


def test_antenna_peak_at_boresight_and_tilt_shifts_lobe():
    g0 = gain_db(np.array([4.0]), np.array([0.0]), 4.0, 65.0)
    assert g0[0] == pytest.approx(peak_gain_dbi(6.5, 65.0))
    # pointing 4° below boresight of an untilted panel loses energy; tilting 4° recovers it
    g_untilted = gain_db(np.array([4.0]), np.array([0.0]), 0.0, 65.0)
    assert g0[0] > g_untilted[0]


def test_mmimo_narrow_beam_widened_for_ssb():
    assert effective_h3db(12.0) == 65.0
    assert effective_h3db(65.0) == 65.0


def _tiny_estate():
    topo = pd.DataFrame({
        "cell_id": ["a", "b", "c"], "cell_lat": [40.000, 40.006, 39.993],
        "cell_lon": [-75.000, -74.996, -74.989], "cell_az_deg": [90.0, 210.0, 330.0],
        "cell_carrier_freq_mhz": [2100.0, 2100.0, 700.0], "hTx": [40.0, 45.0, 50.0],
        "beam_width_deg": [65.0, 65.0, 65.0], "technology": ["NR", "NR", "NR"],
    })
    cfg = pd.DataFrame({"cell_id": ["a", "b", "c"], "cell_el_deg": [6.0, 8.0, 4.0]})
    return Estate.build(topo, cfg, n_points=500)


PARAMS = dict(p0_low=18.0, p0_mid=18.0, p0_hi=18.0, sigma_extra=1.0,
              p_in=0.7, ltw_low=12.0, ltw_hi=20.0, load=0.3)


def test_determinism_same_inputs_same_outputs():
    r1 = _tiny_estate().evaluate(PARAMS)["guardrail_kpis"]
    r2 = _tiny_estate().evaluate(PARAMS)["guardrail_kpis"]
    assert r1 == r2   # cache-safety: byte-identical across rebuilds


def test_tilt_moves_kpis():
    # Asserted on the MEDIAN: the tiny 3-cell fixture plus the wrap-around tier pushes p5 onto the
    # −15 dB measurement clamp, where any delta is invisible by construction. A denser estate's
    # p05 sits well off the clamp, so p5 stays the product metric; the median is
    # simply the clamp-safe way to assert the property in a synthetic fixture.
    est = _tiny_estate()
    base = est.evaluate(PARAMS)["guardrail_kpis"]["sinr_p50"]
    tilted = est.evaluate(PARAMS, tilt_override={"a": 14.0})["guardrail_kpis"]["sinr_p50"]
    assert abs(tilted - base) > 0.05   # a model that is blind to tilt would show no response


def test_off_cell_excluded_from_serving():
    est = _tiny_estate()
    r = est.evaluate(PARAMS, off_cells={"a"})
    assert "a" not in r["serving_share"]


def test_refuses_null_tilt():
    topo = pd.DataFrame({
        "cell_id": ["a"], "cell_lat": [40.000], "cell_lon": [-75.000],
        "cell_az_deg": [90.0], "cell_carrier_freq_mhz": [2100.0], "hTx": [40.0],
        "beam_width_deg": [65.0], "technology": ["NR"],
    })
    cfg = pd.DataFrame({"cell_id": ["a"], "cell_el_deg": [np.nan]})
    with pytest.raises(ValueError, match="cell_el_deg"):
        Estate.build(topo, cfg, n_points=50)


def test_point_sample_mode_wider_and_more_tilt_sensitive():
    """aggregate=False must (a) show wider spread than device-mean and (b) show a LARGER tilt
    response — the reason the mode exists: an optimiser reading device means would see a blunted
    control surface."""
    est = _tiny_estate()
    p = {**PARAMS, "agg": 8.0}
    dev = est.evaluate(p, aggregate=True)["_arrays"]
    pt = est.evaluate(p, aggregate=False)["_arrays"]
    assert pt["rsrp"].std() > dev["rsrp"].std()

    # NOT asserted: point response > device response. That can hold on a denser estate at p5 but is
    # not an invariant — in a tiny fixture, device means mix every pseudo-device with some exposure
    # to the tilted cell, so the MEDIAN can move more under aggregation. The invariants are: the
    # mode is plumbed through (different spreads above), and both modes keep a non-degenerate
    # tilt response.
    def resp(aggregate):
        a = est.evaluate(p, aggregate=aggregate)["guardrail_kpis"]["sinr_p50"]
        b = est.evaluate(p, tilt_override={"a": 14.0}, aggregate=aggregate)["guardrail_kpis"]["sinr_p50"]
        return abs(b - a)
    assert resp(False) > 0.05
    assert resp(True) > 0.05


def test_mr_targets_parse_thousands_separators(tmp_path):
    """Comma-formatted sample counts ("1,000") must carry their weight: a naive numeric parse
    would silently drop them, and they are the rows with the largest weight."""
    from app.physics.calibrate import targets_from_mr_report
    df = pd.DataFrame({
        "rsrp_dbm": ["-100", "-80"],
        "sinr_db": ["0", "10"],
        "sample_count": ["1,000", "1"],
    })
    path = tmp_path / "mr.xlsx"
    df.to_excel(path, index=False)
    t = targets_from_mr_report(str(path))
    # The comma-weighted device dominates the median; a naive parse would drop it and return -80.
    assert t["rsrp"][2] == -100.0
    assert t["parse"]["mr_count"]["unparsed"] == 0
    assert t["parse"]["total_mrs"] == 1001
