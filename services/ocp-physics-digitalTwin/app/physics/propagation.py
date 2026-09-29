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

"""TR 38.901 UMa path loss (Table 7.4.1-1), LOS probability (Table 7.4.2-1), deterministic draws.

All randomness is a pure function of (salt, cell_id) so identical inputs give identical outputs,
which makes results safe to cache.
"""
from __future__ import annotations

import hashlib

import numpy as np

SIGMA_LOS = 4.0    # dB, Table 7.4.1-1
SIGMA_NLOS = 6.0   # dB


def rng_for(name: str, salt: str = "pt-v1") -> np.random.Generator:
    seed = int.from_bytes(
        hashlib.blake2s(f"{salt}|{name}".encode(), digest_size=8).digest(), "big"
    )
    return np.random.default_rng(seed)


def uma_los(d2d, d3d, fc_ghz: float, h_bs: float, h_ut: float = 1.5):
    d_bp = 4.0 * (h_bs - 1.0) * (h_ut - 1.0) * (fc_ghz * 1e9) / 3e8
    pl1 = 28.0 + 22.0 * np.log10(d3d) + 20.0 * np.log10(fc_ghz)
    pl2 = (28.0 + 40.0 * np.log10(d3d) + 20.0 * np.log10(fc_ghz)
           - 9.0 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))
    return np.where(d2d <= d_bp, pl1, pl2)


def uma_nlos(d2d, d3d, fc_ghz: float, h_bs: float, h_ut: float = 1.5):
    pl_nlos = 13.54 + 39.08 * np.log10(d3d) + 20.0 * np.log10(fc_ghz) - 0.6 * (h_ut - 1.5)
    return np.maximum(uma_los(d2d, d3d, fc_ghz, h_bs, h_ut), pl_nlos)


def p_los_uma(d2d, h_ut: float = 1.5):
    # hUT ≤ 13 m ⇒ C'(hUT)=0 branch.
    p = 18.0 / d2d + np.exp(-d2d / 63.0) * (1.0 - 18.0 / d2d)
    return np.where(d2d <= 18.0, 1.0, p)
