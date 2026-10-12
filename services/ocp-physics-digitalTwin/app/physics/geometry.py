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

"""Flat-earth ENU projection + per-cell angles. Accurate to about a metre over an extent of a few kilometres."""
from __future__ import annotations

import numpy as np

R_E = 6_371_000.0


def to_enu(lat, lon, lat0: float, lon0: float):
    x = np.radians(np.asarray(lon) - lon0) * R_E * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * R_E
    return x, y


def cell_point_geometry(cx, cy, h_tx, az_deg, px, py, h_ut: float = 1.5):
    """Vectorised over points for ONE cell. Returns d2d, d3d, phi (az offset), theta (depression).

    Bearing convention 0°=North clockwise, the convention of the topology input. Depression angle is
    positive downward — the same sense as downtilt, so the §7.3 vertical pattern compares them
    directly.
    """
    dx, dy = px - cx, py - cy
    d2d = np.hypot(dx, dy).clip(min=10.0)          # TR 38.901 validity floor
    d3d = np.hypot(d2d, h_tx - h_ut)
    phi = (np.degrees(np.arctan2(dx, dy)) - az_deg + 180.0) % 360.0 - 180.0
    theta = np.degrees(np.arctan2(h_tx - h_ut, d2d))
    return d2d, d3d, phi, theta
