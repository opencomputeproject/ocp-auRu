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

"""Channel-number → carrier frequency (MHz).

EARFCN per TS 36.101 Table 5.7.3-1 (a selection of common E-UTRA operating bands, downlink);
NR-ARFCN per TS 38.104 §5.4.2 global raster.
"""
from __future__ import annotations

_EARFCN = {  # band: (N_offs_DL, N_max_DL, F_DL_low_MHz)
    1: (0, 599, 2110.0), 2: (600, 1199, 1930.0), 3: (1200, 1949, 1805.0),
    4: (1950, 2399, 2110.0), 5: (2400, 2649, 869.0), 7: (2750, 3449, 2620.0),
    8: (3450, 3799, 925.0), 12: (5010, 5179, 729.0), 13: (5180, 5279, 746.0),
    14: (5280, 5379, 758.0), 17: (5730, 5849, 734.0), 20: (6150, 6449, 791.0),
    25: (8040, 8689, 1930.0), 26: (8690, 9039, 859.0), 28: (9210, 9659, 758.0),
    29: (9660, 9769, 717.0), 30: (9770, 9869, 2350.0), 38: (37750, 38249, 2570.0),
    40: (38650, 39649, 2300.0), 41: (39650, 41589, 2496.0), 42: (41590, 43589, 3400.0),
    43: (43590, 45589, 3600.0), 48: (55240, 56739, 3550.0), 66: (66436, 67335, 2110.0),
    71: (68586, 68935, 617.0),
}


def earfcn_to_mhz(n: int) -> float:
    for _band, (lo, hi, f0) in _EARFCN.items():
        if lo <= n <= hi:
            return f0 + 0.1 * (n - lo)
    raise ValueError(f"EARFCN {n} not in the band table")


def nr_arfcn_to_mhz(n: int) -> float:
    if n < 600_000:
        return n * 0.005                      # ΔF_global = 5 kHz
    if n < 2_016_667:
        return 3000.0 + (n - 600_000) * 0.015  # ΔF_global = 15 kHz
    return 24_250.08 + (n - 2_016_667) * 0.060
