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

"""Generic mode — a fictional hex estate + uncalibrated 3GPP-default parameters.

Lets a stranger run the twin with NO supplied data: no topology export, no measurement report, no fitted
model. Everything here is derived from public 3GPP defaults (TR 38.901 / TR 38.913) and declared
in comments; nothing is copied from, or tuned against, any calibrated model.

Rung discipline: every output produced from this module carries `GENERIC_RUNG` —
"simulation, generic 3GPP defaults (uncalibrated)". It is a different rung from the calibrated
one (`app.main.RUNG`, "simulation, calibrated to supplied measurement data"): the physics is
identical, the absolute level, spread and indoor mix are textbook assumptions instead of a fit
against supplied measurement marginals. Relative comparisons (tilt A vs tilt B, cell on vs off, same assumptions) are the
only claims the generic rung supports.
"""
from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from app.physics.calibrate import BOUNDS, NAMES
from app.physics.geometry import R_E

GENERIC_RUNG = "simulation, generic 3GPP defaults (uncalibrated)"

# Sector letters for the three 120° sectors per site (azimuth 0 / 120 / 240 deg, 0 = North).
_SECTORS = (("a", 0.0), ("b", 120.0), ("c", 240.0))

# NR band label by DL carrier frequency range (MHz) — a readable suffix for the generated cell
# ids only; the evaluator reads `cell_carrier_freq_mhz`, never the label. DL ranges from
# TS 38.101-1 Table 5.2-1, matched with ±25 MHz slack so the colloquial band centres people
# type ("700", "1800", "3500") resolve to the band they mean.
_NR_BANDS = (
    (703.0, 803.0, "n28"), (925.0, 960.0, "n8"), (1805.0, 1880.0, "n3"),
    (1930.0, 1995.0, "n2"), (2110.0, 2170.0, "n1"), (2496.0, 2690.0, "n41"),
    (3300.0, 3800.0, "n78"), (3700.0, 3980.0, "n77"),
)
_BAND_SLACK_MHZ = 25.0


def band_label(freq_mhz: float) -> str:
    for lo, hi, name in _NR_BANDS:
        if lo - _BAND_SLACK_MHZ <= freq_mhz <= hi + _BAND_SLACK_MHZ:
            return name
    return f"{int(round(freq_mhz))}mhz"


def generic_build_kwargs(n_sites: int, isd_m: float) -> dict[str, float]:
    """`Estate.build` overrides that make the fictional hex grid a textbook hex network.

    - `d_wrap_m = √n · ISD`: the evaluator's wrap-around tier is six translated copies of the
      estate at 60° steps; for an n-site cluster the copies keep the grid's site density
      when offset by √n·ISD (exact tiling for the n=3 triangle: √3·ISD reproduces the hex
      grid site for site). The calibrated default (mean pairwise site distance) belongs to
      an arbitrary supplied geometry, not to a regular grid.
    - `r_max_m = ISD/√3`: the hex cell circumradius — every point of a site's cell lies within
      it, so sampling annuli 100 m..ISD/√3 around the real sites cover the cluster's served
      area without reaching into the tier's (never-serving) territory.
    """
    return {"d_wrap_m": math.sqrt(n_sites) * isd_m, "r_max_m": isd_m / math.sqrt(3.0)}


def hex_site_positions(n_sites: int, isd_m: float) -> list[tuple[float, float]]:
    """The first `n_sites` points of a hexagonal grid with spacing `isd_m`, ordered centre
    first, then by ring distance and bearing. ENU metres (x east, y north). n=3 gives an
    equilateral triangle (site 1 at the origin, sites 2 and 3 on the first ring 60° apart); n=7
    gives the classic centre + one full ring."""
    if n_sites < 1:
        raise ValueError("n_sites must be >= 1")
    a1 = (isd_m, 0.0)
    a2 = (isd_m / 2.0, isd_m * math.sqrt(3.0) / 2.0)
    rings = int(math.ceil(math.sqrt(n_sites))) + 1
    pts = []
    for i in range(-rings, rings + 1):
        for j in range(-rings, rings + 1):
            x, y = i * a1[0] + j * a2[0], i * a1[1] + j * a2[1]
            r = math.hypot(x, y)
            bearing = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0   # 0 = North, clockwise
            pts.append((round(r, 6), round(bearing, 6), x, y))
    pts.sort()
    return [(x, y) for _r, _b, x, y in pts[:n_sites]]


def generic_estate(n_sites: int = 3, isd_m: float = 500.0,
                   bands_mhz: tuple[float, ...] = (1800.0, 3500.0),
                   h_tx_m: float = 30.0, tilt_deg: float = 6.0, beam_width_deg: float = 65.0,
                   lat0: float = 0.0, lon0: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A fictional estate with exactly the columns `Estate.build` needs.

    Sites sit on a hexagonal grid with inter-site distance `isd_m` (TR 38.913 UMa: 500 m),
    three sectors per site at azimuth 0/120/240°, one cell per band per sector. Coordinates are
    placed around (`lat0`, `lon0`) — the default (0, 0) is a point in the Gulf of Guinea, open
    ocean, so the geometry can never be mistaken for a real deployment. Ids are deterministic:
    `g-s<site>-<sector>-<band>`, e.g. `g-s1-a-n78`.

    Returns `(topology, config)`: topology has `cell_id, site_id, cell_lat, cell_lon,
    cell_az_deg, cell_carrier_freq_mhz, hTx, beam_width_deg, technology`; config has
    `cell_id, cell_el_deg`.
    """
    if not bands_mhz:
        raise ValueError("bands_mhz must name at least one carrier")
    m_per_deg_lat = math.radians(1.0) * R_E
    m_per_deg_lon = math.radians(1.0) * R_E * math.cos(math.radians(lat0))
    rows = []
    for s, (x, y) in enumerate(hex_site_positions(n_sites, isd_m), start=1):
        site_id = f"g-s{s}"
        lat = lat0 + y / m_per_deg_lat
        lon = lon0 + x / m_per_deg_lon
        for sec, az in _SECTORS:
            for f in bands_mhz:
                rows.append({
                    "cell_id": f"{site_id}-{sec}-{band_label(float(f))}",
                    "site_id": site_id,
                    "cell_lat": lat, "cell_lon": lon,
                    "cell_az_deg": az,
                    "cell_carrier_freq_mhz": float(f),
                    "hTx": float(h_tx_m),
                    "beam_width_deg": float(beam_width_deg),
                    "technology": "NR",
                })
    topology = pd.DataFrame(rows)
    config = pd.DataFrame({"cell_id": topology.cell_id, "cell_el_deg": float(tilt_deg)})
    return topology, config


def default_params() -> dict[str, float]:
    """Uncalibrated defaults for every fitted parameter in `calibrate.NAMES`, each inside
    `calibrate.BOUNDS`. Every value is a public 3GPP assumption, reasoned here; none is a fitted
    number (a fitted set exists only after a local calibration and is never imported here).

    Parameter semantics come from `Estate.rsrp_matrix` / `Estate.evaluate`:
    RSRP = P0[band group] + antenna gain − path loss − shadowing − O2I; P0 is the conducted
    per-resource-element power (the §7.3 peak gain is added separately, so P0 excludes it).
    """
    # P0 — conducted power per resource element.
    #  low/mid band groups: a 43 dBm / 20 MHz-class macro radio (the TR 38.913 macro-layer
    #  assumption is 43–44 dBm per 20 MHz); 20 MHz at 15 kHz SCS is 100 PRB × 12 = 1200 REs
    #  → 43 − 10·log10(1200) = 12.2 dBm/RE.
    #  C-band group: a 200 W (53 dBm) / 100 MHz active-antenna class; 100 MHz is ≈ 6600
    #  15-kHz-RE equivalents → 53 − 10·log10(6600) = 14.8 dBm/RE.
    # Sanity: the gate's EIRP window is 25 ≤ P0 + 17 dBi ≤ 45 → 29.2 / 31.8 dBm, inside.
    p0_macro = round(43.0 - 10.0 * math.log10(1200.0), 1)
    p0_cband = round(53.0 - 10.0 * math.log10(6600.0), 1)
    params = {
        "p0_low": p0_macro, "p0_mid": p0_macro, "p0_hi": p0_cband,
        # Shadowing spread: TR 38.901 Table 7.4.1-1 UMa σ (4 dB LOS / 6 dB NLOS) is already in
        # the cached fields; nothing extra without supporting measurement evidence.
        "sigma_extra": 0.0,
        # Indoor user fraction: TR 38.901 Table 7.2-1, UMa — 80 % of users indoor.
        "p_in": 0.80,
        # O2I building-penetration means, TR 38.901 §7.4.3.1 (Table 7.4.3-2), wall term PL_tw:
        #  low-loss model at 1.8 GHz: 5 − 10·log10(0.3·10^(−(2+0.2f)/10) + 0.7·10^(−(5+4f)/10))
        #  ≈ 11.6 dB → 12; used for the low/mid band groups.
        #  high-loss model at 3.5 GHz: 5 − 10·log10(0.7·10^(−(23+0.3f)/10) + 0.3·10^(−(5+4f)/10))
        #  ≈ 26.9 dB → 27; used for the C-band group.
        # The in-building term PL_in (0.5 dB/m, mean ≈ 6 dB for UMa d_in ~ U(0, 25 m)) is left
        # out of both so the C-band value stays inside the fitted window (10, 30); the evaluator's
        # ±4 dB per-point spread (σ_P) is applied on top of these means.
        "ltw_low": 12.0, "ltw_hi": 27.0,
        # Interferer activity: 50 % average PRB load on every co-channel cell — a mid-load
        # operating point (full buffer = 1.0 sits outside the fitted window (0.02, 0.60)).
        "load": 0.50,
        # Pseudo-device session size: 1 = point samples. There is no measurement report to
        # emulate in generic mode, so no trajectory averaging is applied.
        "agg": 1.0,
        # NSA band-priority serving bias: 0 dB — plain max-RSRP serving, the generic (SA) case.
        "bias_mid": 0.0, "bias_hi": 0.0,
        # Wrap-around external tier fully visible: the six translated copies of the estate
        # stand in for a continuous hex network — the textbook wrap-around assumption.
        "w_tier": 1.0,
    }
    missing = [n for n in NAMES if n not in params]
    extra = [n for n in params if n not in NAMES]
    if missing or extra:
        raise RuntimeError(f"default_params out of sync with calibrate.NAMES: "
                           f"missing={missing} extra={extra}")
    for name, value in params.items():
        lo, hi = BOUNDS[name]
        if not (lo <= value <= hi):
            raise RuntimeError(f"default {name}={value} outside BOUNDS {BOUNDS[name]}")
    return params


def generic_model(params: dict[str, float] | None = None,
                  estate_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    """The model dict shape `store`/`main` expect. `gate` is None on purpose: no calibration was
    run, so no acceptance gate exists — nothing here may be wired into a decision path as if it
    were calibrated."""
    return {
        "params": dict(params or default_params()),
        "rung": GENERIC_RUNG,
        "gate": None,
        "source": "generic",
        "estate": dict(estate_kwargs or {}),
        "note": ("Uncalibrated: parameters are public 3GPP defaults (TR 38.901 / TR 38.913), "
                 "not fitted to any measurement data. Absolute KPIs are illustrative; only "
                 "relative comparisons under identical assumptions are meaningful."),
    }


def is_generic_model(model: dict[str, Any] | None) -> bool:
    return bool(model) and (model.get("source") == "generic" or model.get("rung") == GENERIC_RUNG)


_GENERIC_ID = re.compile(r"^g-s\d+-[abc]-")


def is_generic_topology(topology: pd.DataFrame | None) -> bool:
    """Whether a topology is one this module generated (every cell id `g-s<site>-<sector>-<band>`).
    Persisted inputs that are NOT generic are somebody's estate: generic mode never overwrites
    them on disk."""
    if topology is None or "cell_id" not in topology.columns or len(topology) == 0:
        return False
    return bool(topology.cell_id.astype(str).map(lambda i: _GENERIC_ID.match(i) is not None).all())
