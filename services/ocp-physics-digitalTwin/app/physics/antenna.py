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

"""3GPP TR 38.901 §7.3 element pattern, tilt and azimuth explicit.

`theta` is the depression angle toward the point (positive down); electrical+mechanical downtilt
shifts the vertical lobe. Peak gain is estimated from beamwidths (G ≈ 10·log10(41000/(θ3dB·φ3dB)));
its absolute error is absorbed by the calibrated per-band EIRP, so only the SHAPE matters here.

mMIMO note: reported RSRP refers to the SSB broadcast beam. A narrow traffic-beam width (<30°)
in the topology input is therefore widened to an SSB-envelope approximation; per-traffic-beam
accuracy on that panel is explicitly out of scope.
"""
from __future__ import annotations

import numpy as np

SLA_V = 30.0
A_MAX = 30.0
V3DB_DEFAULT = 6.5   # typical macro panel vertical beamwidth


def effective_h3db(beam_width_deg: float) -> float:
    return 65.0 if beam_width_deg < 30.0 else float(beam_width_deg)


def peak_gain_dbi(v3db: float, h3db: float) -> float:
    return 10.0 * np.log10(41000.0 / (v3db * h3db))


def gain_db(theta, phi, tilt_deg: float, beam_width_deg: float, v3db: float = V3DB_DEFAULT):
    h3db = effective_h3db(beam_width_deg)
    a_v = -np.minimum(12.0 * ((theta - tilt_deg) / v3db) ** 2, SLA_V)
    a_h = -np.minimum(12.0 * (phi / h3db) ** 2, A_MAX)
    return peak_gain_dbi(v3db, h3db) - np.minimum(-(a_v + a_h), A_MAX)
