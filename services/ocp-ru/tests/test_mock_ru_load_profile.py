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
"""Optional hourly PRB replay profile of the mock radio (RU_LOAD_PROFILE)."""
import json

import pytest

from ocp_auru.mock_ru.radio import PRB_UTIL_AT_FULL_LOAD_PCT, load_prb_profile, profile_load


def test_no_path_means_generic_curve():
    assert load_prb_profile("") is None


def test_profile_is_read_clamped_and_interpolated(tmp_path):
    values = [float(h) for h in range(24)]
    values[23] = 250.0                                   # clamped to 100
    f = tmp_path / "profile.json"
    f.write_text(json.dumps({"prb_util_dl_pct_by_hour": values}))
    prof = load_prb_profile(str(f))
    assert prof is not None and len(prof) == 24 and prof[23] == 100.0
    assert profile_load(prof, 3.5) == pytest.approx(3.0 / PRB_UTIL_AT_FULL_LOAD_PCT)      # middle of hour 3
    assert profile_load(prof, 4.0) == pytest.approx(3.5 / PRB_UTIL_AT_FULL_LOAD_PCT)      # halfway to hour 4
    assert profile_load(prof, 0.0) == pytest.approx(50.0 / PRB_UTIL_AT_FULL_LOAD_PCT)     # wraps 23 -> 0


def test_wrong_length_is_rejected(tmp_path):
    f = tmp_path / "profile.json"
    f.write_text(json.dumps({"prb_util_dl_pct_by_hour": [1.0] * 23}))
    with pytest.raises(ValueError):
        load_prb_profile(str(f))


def test_half_hour_series(tmp_path):
    f = tmp_path / "day.json"
    f.write_text(json.dumps({"prb_util_dl_pct_by_half_hour": [float(i) for i in range(48)]}))
    prof = load_prb_profile(str(f))
    assert prof is not None and len(prof) == 48
    assert profile_load(prof, 5.25) == pytest.approx(10.0 / PRB_UTIL_AT_FULL_LOAD_PCT)    # middle of slot 10 (05:00-05:30)
    assert profile_load(prof, 5.5) == pytest.approx(10.5 / PRB_UTIL_AT_FULL_LOAD_PCT)


def test_unknown_key_is_rejected(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text(json.dumps({"something": [1.0] * 24}))
    with pytest.raises(ValueError):
        load_prb_profile(str(f))
