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

"""The estate model: cached fields + fast RSRP/SINR evaluation.

Everything parameter-independent (geometry, path loss per LOS state, antenna gain at the BASELINE
tilt, random fields) is computed once per (estate, grid). A calibration loss evaluation or a tilt
overlay then costs matrix arithmetic only — which is what makes differential evolution affordable
and the tilt sweep interactive.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .antenna import gain_db
from .geometry import cell_point_geometry, to_enu
from .propagation import SIGMA_LOS, SIGMA_NLOS, p_los_uma, rng_for, uma_los, uma_nlos

H_UT = 1.5
NOISE_DBM_RE = -174.0 + 10.0 * np.log10(15e3) + 7.0   # 15 kHz SCS, NF 7 dB ≈ −125.2 dBm
OUTAGE_SINR_DB = -6.0
COVERAGE_RSRP_DBM = -110.0


def band_group(freq_mhz: float) -> int:
    """0=low(<1 GHz) 1=mid 2=C-band(>3 GHz). One calibrated EIRP per group."""
    return 0 if freq_mhz < 1000 else (1 if freq_mhz < 3000 else 2)


@dataclass
class Estate:
    cells: pd.DataFrame                 # one row per cell, ENU + parameters
    grid_x: np.ndarray
    grid_y: np.ndarray
    fields: dict[str, np.ndarray] = field(default_factory=dict)

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def build(cls, topology: pd.DataFrame, config: pd.DataFrame,
              n_points: int = 4000, margin_m: float = 1500.0, seed: str = "grid-v1",
              d_wrap_m: float | None = None, r_max_m: float = 1250.0) -> "Estate":
        """`d_wrap_m` / `r_max_m` (additive, generic mode): override the wrap-around tier offset
        and the sampling annulus outer radius. Defaults reproduce the calibrated construction
        exactly (tier offset = mean inter-site distance, annuli 100–1250 m); `app.generic` passes
        the textbook hex values for a fictional hex grid (√n·ISD, ISD/√3)."""
        t = topology.merge(config[["cell_id", "cell_el_deg"]], on="cell_id", how="left")
        required = ["cell_id", "cell_lat", "cell_lon", "cell_az_deg",
                    "cell_carrier_freq_mhz", "hTx", "cell_el_deg"]
        missing = [c for c in required if t[c].isna().any()]
        if missing:
            # The refuse-to-train-on-defaults guard: a fabricated tilt or azimuth would make a
            # confident model of a fictional antenna. Fail loudly instead.
            raise ValueError(f"topology/config incomplete — null values in {missing}")

        lat0, lon0 = float(t.cell_lat.mean()), float(t.cell_lon.mean())
        x, y = to_enu(t.cell_lat.values, t.cell_lon.values, lat0, lon0)
        # kept for attach_points: caller coordinates must land in the SAME ENU frame
        _origin = {"lat0": lat0, "lon0": lon0}
        cells = pd.DataFrame({
            "cell_id": t.cell_id.values, "x": x, "y": y,
            "h_tx": t.hTx.astype(float).values,
            "az_deg": t.cell_az_deg.astype(float).values,
            "tilt_deg": t.cell_el_deg.astype(float).values,
            "freq_mhz": t.cell_carrier_freq_mhz.astype(float).values,
            "beam_w": t.get("beam_width_deg", pd.Series([65.0] * len(t))).astype(float).values,
            "technology": t.get("technology", pd.Series(["NR"] * len(t))).values,
        })
        cells["grp"] = cells.freq_mhz.map(band_group)
        cells.attrs.update(_origin)

        # Traffic-proxy sampling, not area-uniform: calibration targets are per-DEVICE session
        # means from users in the served area. An area-uniform rectangle puts mass at tower bases
        # (p95 far too hot) and in far corners no user occupies (p05 far too cold), which
        # overstates the RSRP spread. Sample the union of annuli 100 m..r_max around the sites
        # instead, and declare it.
        rng = rng_for(seed)
        sites = np.unique(np.column_stack([x, y]), axis=0)
        # The surrounding network as a WRAP-AROUND TIER: six translated copies of the whole
        # estate at the estate's own mean inter-site distance, interference-only (they never
        # serve, and tilt overrides never touch them: the surroundings are not ours to tune).
        # A flat external interference floor would leave some points interference-free and pin
        # SINR p95 at the reporting clamp. No new fitted parameter: the tier's geometry is
        # derived from the estate itself and declared, not tuned.
        if len(sites) > 1:
            from itertools import combinations
            d_wrap = float(np.mean([np.hypot(a[0]-b[0], a[1]-b[1])
                                    for a, b in combinations(sites, 2)]))
        else:
            d_wrap = 1200.0
        if d_wrap_m is not None:
            d_wrap = float(d_wrap_m)
        cells.attrs["d_wrap"] = d_wrap
        per = int(np.ceil(n_points / len(sites)))
        pts = []
        for sx, sy in sites:
            r = np.sqrt(rng.uniform(100.0 ** 2, float(r_max_m) ** 2, per))  # uniform in area; 100 m floor — tower-base outdoor camping is not where the device population lives
            a = rng.uniform(0.0, 2.0 * np.pi, per)
            pts.append(np.column_stack([sx + r * np.sin(a), sy + r * np.cos(a)]))
        pts = np.vstack(pts)[:n_points]
        gx, gy = pts[:, 0], pts[:, 1]
        est = cls(cells=cells, grid_x=gx, grid_y=gy)
        est._precompute()
        return est

    def _precompute(self) -> None:
        C, N = len(self.cells), len(self.grid_x)
        pl = np.empty((C, N)); gain0 = np.empty((C, N))
        sf = np.empty((C, N)); theta = np.empty((C, N)); phi = np.empty((C, N))
        for i, c in enumerate(self.cells.itertuples()):
            d2d, d3d, ph, th = cell_point_geometry(c.x, c.y, c.h_tx, c.az_deg,
                                                   self.grid_x, self.grid_y, H_UT)
            fc = c.freq_mhz / 1000.0
            r = rng_for(f"cell|{c.cell_id}")
            los = r.uniform(size=N) < p_los_uma(d2d)
            z = r.standard_normal(N)
            pl[i] = np.where(los, uma_los(d2d, d3d, fc, c.h_tx, H_UT),
                                  uma_nlos(d2d, d3d, fc, c.h_tx, H_UT))
            # Unit-normal shadow field; σ scaling happens at eval time (σ_spec + σ_extra).
            sf[i] = z * np.where(los, SIGMA_LOS, SIGMA_NLOS) / SIGMA_NLOS
            theta[i], phi[i] = th, ph
            gain0[i] = gain_db(th, ph, c.tilt_deg, c.beam_w)
        # External tier fields: 6 hex offsets × every cell, at baseline tilt, forever.
        d_wrap = self.cells.attrs.get("d_wrap", 1200.0)
        offs = [(d_wrap * np.sin(np.radians(k * 60.0)), d_wrap * np.cos(np.radians(k * 60.0)))
                for k in range(6)]
        E = 6 * C
        ext_pl = np.empty((E, N)); ext_gain = np.empty((E, N)); ext_sf = np.empty((E, N))
        ext_grp = np.empty(E, dtype=int); ext_freq = np.empty(E)
        e = 0
        for k, (ox, oy) in enumerate(offs):
            for c in self.cells.itertuples():
                d2d, d3d, ph, th = cell_point_geometry(c.x + ox, c.y + oy, c.h_tx, c.az_deg,
                                                       self.grid_x, self.grid_y, H_UT)
                fc = c.freq_mhz / 1000.0
                r = rng_for(f"ext|{k}|{c.cell_id}")
                los = r.uniform(size=N) < p_los_uma(d2d)
                z = r.standard_normal(N)
                ext_pl[e] = np.where(los, uma_los(d2d, d3d, fc, c.h_tx, H_UT),
                                          uma_nlos(d2d, d3d, fc, c.h_tx, H_UT))
                ext_sf[e] = z * np.where(los, SIGMA_LOS, SIGMA_NLOS) / SIGMA_NLOS
                ext_gain[e] = gain_db(th, ph, c.tilt_deg, c.beam_w)
                ext_grp[e] = c.grp; ext_freq[e] = c.freq_mhz
                e += 1
        r = rng_for("points")
        self.fields = {
            "pl": pl, "sf_unit": sf, "theta": theta, "phi": phi, "gain0": gain0,
            "u_indoor": r.uniform(size=N), "z_o2i": r.standard_normal(N),
            "ext_pl": ext_pl, "ext_gain": ext_gain, "ext_sf": ext_sf,
            "ext_grp": ext_grp, "ext_freq": ext_freq,
        }

    # ── evaluation ───────────────────────────────────────────────────────────
    def rsrp_matrix(self, params: dict[str, float],
                    tilt_override: dict[str, float] | None = None) -> np.ndarray:
        f = self.fields
        p0 = np.array([params["p0_low"], params["p0_mid"], params["p0_hi"]])
        gain = f["gain0"]
        if tilt_override:
            gain = gain.copy()
            idx = {cid: i for i, cid in enumerate(self.cells.cell_id)}
            for cid, tilt in tilt_override.items():
                i = idx[cid]; c = self.cells.iloc[i]
                gain[i] = gain_db(f["theta"][i], f["phi"][i], float(tilt), c.beam_w)
        # `agg` models the MEASUREMENT process, not the physics: each target row is a device's
        # session mean over many measurement reports, and averaging k effectively-decorrelated
        # positions shrinks the shadowing contribution by √k. Without this operator the forward
        # model would compare raw point statistics against session-mean targets.
        sf = f["sf_unit"] * (SIGMA_NLOS + params["sigma_extra"])
        indoor = f["u_indoor"] < params["p_in"]
        ltw = np.where(self.cells.grp.values[:, None] == 2, params["ltw_hi"], params["ltw_low"])
        o2i = indoor[None, :] * (ltw + 4.0 * f["z_o2i"][None, :])
        return p0[self.cells.grp.values][:, None] + gain - f["pl"] - sf - o2i

    def evaluate(self, params: dict[str, float],
                 tilt_override: dict[str, float] | None = None,
                 off_cells: set[str] | None = None,
                 serving_mask: np.ndarray | None = None,
                 aggregate: bool = True) -> dict[str, Any]:
        """Serving selection + SINR over the grid. `serving_mask` restricts which cells may SERVE
        (calibration lets only NR cells serve, because the measurement targets describe the NR
        leg); everything on the same carrier still interferes."""
        rsrp = self.rsrp_matrix(params, tilt_override)
        active = ~self.cells.cell_id.isin(off_cells or set()).values
        eligible = active & (serving_mask if serving_mask is not None else np.ones(len(active), bool))
        if not eligible.any():
            raise ValueError("no serving-eligible cells")

        # NSA serving is POLICY, not max-RSRP. The SCG leg is chosen by capacity preference
        # (mid/C-band first when adequate), so selection uses RSRP + a per-band bias, while the
        # REPORTED value stays the chosen cell's true RSRP. Pure argmax over-weights stretched
        # low-band links and over-spreads the RSRP marginal.
        bias = np.array([0.0, params.get("bias_mid", 0.0), params.get("bias_hi", 0.0)])
        masked = np.where(eligible[:, None], rsrp + bias[self.cells.grp.values][:, None], -np.inf)
        serving = masked.argmax(axis=0)
        s_dbm = np.take_along_axis(rsrp, serving[None, :], 0)[0]

        lin = 10.0 ** (rsrp / 10.0) * active[:, None]
        same_carrier = (self.cells.freq_mhz.values[:, None]
                        == self.cells.freq_mhz.values[serving][None, :])
        itot = (lin * same_carrier * params["load"]).sum(axis=0) \
               - np.take_along_axis(lin, serving[None, :], 0)[0] * params["load"]
        # External co-channel field. A handful of modelled sites normally sit inside a denser
        # grid with neighbouring sites on every side; modelling interference from the modelled
        # sites alone leaves single-cell carriers pristine and overstates SINR p95. The external
        # term is environment modelling, and it is reported, bounded, and sanity-checked.
        # Wrap-around tier interference: same P0/σ/O2I treatment as the modelled cells, co-channel only.
        f2 = self.fields
        p0v = np.array([params["p0_low"], params["p0_mid"], params["p0_hi"]])
        ext_sf_s = f2["ext_sf"] * (SIGMA_NLOS + params["sigma_extra"])
        indoor = f2["u_indoor"] < params["p_in"]
        ltw_e = np.where(f2["ext_grp"][:, None] == 2, params["ltw_hi"], params["ltw_low"])
        o2i_e = indoor[None, :] * (ltw_e + 4.0 * f2["z_o2i"][None, :])
        ext_rsrp = p0v[f2["ext_grp"]][:, None] + f2["ext_gain"] - f2["ext_pl"] - ext_sf_s - o2i_e
        ext_same = f2["ext_freq"][:, None] == self.cells.freq_mhz.values[serving][None, :]
        # `w_tier` decouples tier visibility from estate load (with one shared `load` a fit
        # pushes it towards zero to compensate for an over-dense tier). Physically it bundles what
        # the hex-copy construction over-counts: more virtual sites than the geometry supports,
        # and neighbour sectors pointing AWAY from us where the copies point in every direction.
        i_tier = (10.0 ** (ext_rsrp / 10.0) * ext_same
                  * params["load"] * params.get("w_tier", 1.0)).sum(axis=0)
        sinr = 10.0 * np.log10(10.0 ** (s_dbm / 10.0)
                               / (itot + i_tier + 10.0 ** (NOISE_DBM_RE / 10.0)))


        # Measurement operator. A calibration target row is a DEVICE's session average over the
        # measurement reports collected along its day, so the forward model must average over a
        # trajectory, not sample a point: session means compress the GEOMETRY spread (near vs
        # far), not just shadowing. Grid points are grouped into pseudo-devices of `agg`
        # deterministically-shuffled points and averaged in the dB domain (how measurement
        # report tooling averages); the reporting clamp of -15..23 dB then applies to the
        # AGGREGATED value.
        # Two output modes, two different questions:
        #   aggregate=True:  DEVICE-MEAN KPIs, what a measurement report would show. This is the
        #                    calibration-comparable view, and the report clamp applies to it.
        #   aggregate=False: POINT-SAMPLE KPIs, the physical field at grid points, unclamped
        #                    (the clamp is an artifact of reporting, not of radio).
        #                    This is the OPTIMIZATION view: device averaging blunts single-cell
        #                    tilt effects, so an optimiser comparing tilts must read this mode.
        g = max(1, int(round(params.get("agg", 1.0)))) if aggregate else 1
        if g > 1:
            perm = rng_for("device-mix").permutation(len(s_dbm))
            keep = (len(s_dbm) // g) * g
            s_dev = s_dbm[perm][:keep].reshape(-1, g).mean(axis=1)
            sinr_dev = sinr[perm][:keep].reshape(-1, g).mean(axis=1)
        else:
            s_dev, sinr_dev = s_dbm, sinr
        if aggregate:
            # The -15..23 dB clamp emulates measurement reporting and belongs to the device-mean
            # view only. Point-sample mode is the physical field: clamping it would pin the p5 at
            # the lower bound across a tilt sweep and hide the signal the mode exists to show.
            sinr_dev = np.clip(sinr_dev, -15.0, 23.0)

        q = lambda v, p: float(np.percentile(v, p))
        return {
            "guardrail_kpis": {
                "rsrp_p5": q(s_dev, 5), "rsrp_p50": q(s_dev, 50), "rsrp_p95": q(s_dev, 95),
                "sinr_p5": q(sinr_dev, 5), "sinr_p50": q(sinr_dev, 50), "sinr_p95": q(sinr_dev, 95),
                "outage_rate": float((sinr_dev < OUTAGE_SINR_DB).mean()),
                "coverage_rate": float((s_dev > COVERAGE_RSRP_DBM).mean()),
                "samples": float(len(s_dev)),
            },
            "serving_share": pd.Series(self.cells.cell_id.values[serving])
                               .value_counts(normalize=True).round(4).to_dict(),
            "_arrays": {"rsrp": s_dev, "sinr": sinr_dev},
            # Point mode only: per-point serving ids, so callers can scope KPIs to one cell's
            # served area. Undefined for pseudo-devices (a device serves from many cells).
            "_serving": None if g > 1 else self.cells.cell_id.values[serving],
        }

    def nr_serving_mask(self) -> np.ndarray:
        return (self.cells.technology == "NR").values

    # ── the attach contract (per-point radio for caller-supplied coordinates) ─
    def attach_points(self, params: dict[str, float],
                      lons: np.ndarray, lats: np.ndarray,
                      tilt_by_cell: dict[str, float] | None = None,
                      off_cells: set[str] | None = None,
                      nr_serving_only: bool = True) -> dict[str, np.ndarray]:
        """Per-point RSRP/SINR/serving for CALLER-SUPPLIED coordinates (per-UE rows in, per-UE
        radio out).

        Fresh, uncached computation (the cached fields belong to the internal grid). Determinism
        is preserved the same way as everywhere else: draws are a pure function of (cell, point
        index), so identical requests give identical bytes and a caller may cache by request.

        Point-sample and UNCLAMPED by design: this is the physical field, not the measurement
        report emulation.
        """
        lat0 = float(self.cells.attrs.get("lat0", 0.0)) or None
        # ENU frame must match the estate's — rebuild from stored origin.
        ox, oy = self.cells.attrs["lon0"], self.cells.attrs["lat0"]
        px, py = to_enu(np.asarray(lats, float), np.asarray(lons, float), oy, ox)

        tilt_by_cell = tilt_by_cell or {}
        off = off_cells or set()
        C, N = len(self.cells), len(px)
        rsrp = np.empty((C, N))
        for i, c in enumerate(self.cells.itertuples()):
            d2d, d3d, ph, th = cell_point_geometry(c.x, c.y, c.h_tx, c.az_deg, px, py, H_UT)
            fc = c.freq_mhz / 1000.0
            r = rng_for(f"attach|{c.cell_id}")
            los = r.uniform(size=N) < p_los_uma(d2d)
            z = r.standard_normal(N)
            pl = np.where(los, uma_los(d2d, d3d, fc, c.h_tx, H_UT),
                               uma_nlos(d2d, d3d, fc, c.h_tx, H_UT))
            sf = z * np.where(los, SIGMA_LOS, SIGMA_NLOS) / SIGMA_NLOS                  * (SIGMA_NLOS + params["sigma_extra"])
            tilt = float(tilt_by_cell.get(c.cell_id, c.tilt_deg))
            g = gain_db(th, ph, tilt, c.beam_w)
            p0 = [params["p0_low"], params["p0_mid"], params["p0_hi"]][c.grp]
            rsrp[i] = p0 + g - pl - sf
        ri = rng_for("attach|points")
        indoor = ri.uniform(size=N) < params["p_in"]
        z2 = ri.standard_normal(N)
        ltw = np.where(self.cells.grp.values[:, None] == 2, params["ltw_hi"], params["ltw_low"])
        rsrp -= indoor[None, :] * (ltw + 4.0 * z2[None, :])

        active = ~self.cells.cell_id.isin(off).values
        eligible = active & (self.nr_serving_mask() if nr_serving_only else np.ones(C, bool))
        bias = np.array([0.0, params.get("bias_mid", 0.0), params.get("bias_hi", 0.0)])
        masked = np.where(eligible[:, None], rsrp + bias[self.cells.grp.values][:, None], -np.inf)
        serving = masked.argmax(axis=0)
        s_dbm = np.take_along_axis(rsrp, serving[None, :], 0)[0]

        lin = 10.0 ** (rsrp / 10.0) * active[:, None]
        same = self.cells.freq_mhz.values[:, None] == self.cells.freq_mhz.values[serving][None, :]
        itot = (lin * same * params["load"]).sum(axis=0)                - np.take_along_axis(lin, serving[None, :], 0)[0] * params["load"]
        sinr = 10.0 * np.log10(10.0 ** (s_dbm / 10.0) / (itot + 10.0 ** (NOISE_DBM_RE / 10.0)))
        return {"rsrp_dbm": s_dbm, "sinr_db": sinr,
                "serving_cell_id": self.cells.cell_id.values[serving]}
