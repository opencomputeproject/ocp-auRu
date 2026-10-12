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
"""es-rApp policy engine: every rule, deterministic."""
from __future__ import annotations

import pytest

from ocp_auru.common.config import Settings
from ocp_auru.common.events import Active, CarrierState, CarrierStatus, RuAppPolicy
from ocp_auru.rapp.policy_engine import Decision, PolicyState, decide

THR = 10.0


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setenv("RAPP_SLEEP_THRESHOLD", "0.7")
    monkeypatch.setenv("RAPP_WAKE_THRESHOLD", "0.3")
    monkeypatch.setenv("RAPP_MIN_HOLD_S", "20")
    monkeypatch.setenv("RAPP_ALLOW_INACTIVE", "0")
    return Settings()


def status(active=Active.ACTIVE, state=CarrierState.READY):
    return CarrierStatus(name="txcarrier0", active=active, state=state, ruapp_policy=RuAppPolicy())


def pm(prb=3.0):
    return {"prb_util_dl_pct": prb}


def test_active_low_forecast_low_load_sleeps_with_wake_envelope(cfg):
    d = decide(status(), pm(3.0), 0.9, 1000.0, PolicyState(envelope=RuAppPolicy(wake_on_prach=False, prach_energy_threshold_db=12)),
               low_threshold=THR, cfg=cfg)
    assert d.action == "SLEEP"
    assert d.envelope.wake_on_prach is True and d.envelope.prach_energy_threshold_db == 12
    assert d.active_before == Active.ACTIVE and d.p_low == 0.9 and d.low_threshold == THR
    assert isinstance(d, Decision) and "SLEEP" in d.reason


def test_active_p_low_below_threshold_is_none(cfg):
    d = decide(status(), pm(3.0), 0.69, 1000.0, PolicyState(), low_threshold=THR, cfg=cfg)
    assert d.action == "NONE" and "sleep threshold" in d.reason


def test_active_load_not_low_now_is_none(cfg):
    d = decide(status(), pm(THR), 0.95, 1000.0, PolicyState(), low_threshold=THR, cfg=cfg)
    assert d.action == "NONE" and "not below low threshold" in d.reason
    d = decide(status(), None, 0.95, 1000.0, PolicyState(), low_threshold=THR, cfg=cfg)
    assert d.action == "NONE"


def test_min_hold_blocks_then_releases(cfg):
    st = PolicyState(last_decision_wall=1000.0, last_action="ACTIVE")
    assert decide(status(), pm(3.0), 0.9, 1019.9, st, low_threshold=THR, cfg=cfg).action == "NONE"
    assert "min hold" in decide(status(), pm(3.0), 0.9, 1019.9, st, low_threshold=THR, cfg=cfg).reason
    assert decide(status(), pm(3.0), 0.9, 1020.0, st, low_threshold=THR, cfg=cfg).action == "SLEEP"


def test_sleep_wakes_preemptively_when_load_forecast_returns(cfg):
    d = decide(status(Active.SLEEP, CarrierState.DISABLED), pm(3.0), 0.29, 1000.0, PolicyState(), low_threshold=THR, cfg=cfg)
    assert d.action == "ACTIVE" and "pre-emptive" in d.reason
    # hold does not delay restoring service
    d = decide(status(Active.SLEEP, CarrierState.DISABLED), pm(3.0), 0.1, 1001.0, PolicyState(last_decision_wall=1000.0), low_threshold=THR, cfg=cfg)
    assert d.action == "ACTIVE"


def test_sleep_stays_asleep_when_forecast_low(cfg):
    d = decide(status(Active.SLEEP, CarrierState.DISABLED), pm(3.0), 0.8, 1000.0, PolicyState(), low_threshold=THR, cfg=cfg)
    assert d.action == "NONE" and "stay SLEEP" in d.reason


def test_busy_carrier_is_left_alone(cfg):
    d = decide(status(Active.ACTIVE, CarrierState.BUSY), pm(3.0), 0.99, 1000.0, PolicyState(), low_threshold=THR, cfg=cfg)
    assert d.action == "NONE" and "BUSY" in d.reason


def test_inactive_never_touched_unless_allowed(cfg, monkeypatch):
    st = status(Active.INACTIVE, CarrierState.DISABLED)
    assert decide(st, pm(3.0), 0.1, 1000.0, PolicyState(), low_threshold=THR, cfg=cfg).action == "NONE"
    assert decide(st, pm(3.0), 0.9, 1000.0, PolicyState(), low_threshold=THR, cfg=cfg).action == "NONE"
    monkeypatch.setenv("RAPP_ALLOW_INACTIVE", "1")
    allowed = Settings()
    assert decide(st, pm(3.0), 0.1, 1000.0, PolicyState(), low_threshold=THR, cfg=allowed).action == "ACTIVE"
    assert decide(st, pm(3.0), 0.9, 1000.0, PolicyState(), low_threshold=THR, cfg=allowed).action == "NONE"


def test_engine_never_proposes_inactive_and_is_deterministic(cfg):
    st = PolicyState()
    seen = set()
    for active in Active:
        for p in (0.0, 0.3, 0.7, 1.0):
            for prb in (1.0, 50.0):
                a = decide(status(active, CarrierState.READY), pm(prb), p, 1000.0, st, low_threshold=THR, cfg=cfg).action
                b = decide(status(active, CarrierState.READY), pm(prb), p, 1000.0, st, low_threshold=THR, cfg=cfg).action
                assert a == b
                seen.add(a)
    assert "INACTIVE" not in seen and {"SLEEP", "ACTIVE", "NONE"} <= seen


# ---- chatter control : sleep dwell + hysteresis on the pre-emptive wake ----
ASLEEP = status(Active.SLEEP, CarrierState.DISABLED)
# The baseline without chatter control, pinned explicitly now that the shipped defaults are on: governor off (no dwell, one
# confirmation window, no margin), the estate-pooled profile as-is, no post-local-wake hold.
LEGACY_ENV = {"RAPP_PROFILE_SCALE": "0", "RAPP_MIN_SLEEP_DWELL_MIN": "0", "RAPP_POST_LOCAL_WAKE_HOLD_MIN": "0",
              "RAPP_WAKE_CONFIRM_WINDOWS": "1", "RAPP_WAKE_MARGIN": "0"}


def _cfg(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    return Settings()


def test_shipped_defaults_are_the_chatter_control_settings(monkeypatch):
    """The chatter-control defaults shipped in config.py: profile scale 1 (the root-cause fix: the carrier's own
    load level in the asleep forecast), sleep dwell 60 min and post-local-wake hold 30 min on, 1 confirmation
    window and no wake margin (the two rate limiters that stay off). Read with the environment cleared so the test pins config.py, not the shell."""
    for name in LEGACY_ENV:
        monkeypatch.delenv(name, raising=False)
    s = Settings()
    assert s.rapp_profile_scale == 1.0
    assert s.rapp_min_sleep_dwell_min == 60
    assert s.rapp_post_local_wake_hold_min == 30
    assert s.rapp_wake_confirm_windows == 1
    assert s.rapp_wake_margin == 0
    assert s.rapp_wake_threshold_effective == s.rapp_wake_threshold   # no margin: the wake threshold as configured


def test_legacy_baseline_keeps_decisions_and_reasons_identical(monkeypatch):
    """LEGACY BASELINE, pinned explicitly: with the governor off (dwell
    0, 1 confirmation window, margin 0) and the estate profile as-is every decision - action AND reason text - is
    the pre-chatter-control one, with or without sim_now and whatever the new state fields hold."""
    cfg = _cfg(monkeypatch, RAPP_SLEEP_THRESHOLD="0.7", RAPP_WAKE_THRESHOLD="0.3", RAPP_MIN_HOLD_S="20", RAPP_ALLOW_INACTIVE="0",
               **LEGACY_ENV)
    assert cfg.rapp_min_sleep_dwell_min == 0 and cfg.rapp_wake_confirm_windows == 1 and cfg.rapp_wake_margin == 0
    assert cfg.rapp_profile_scale == 0 and cfg.rapp_post_local_wake_hold_min == 0
    assert cfg.rapp_wake_threshold_effective == cfg.rapp_wake_threshold
    legacy = {
        (Active.SLEEP, 0.29): ("ACTIVE", "pre-emptive wake: load forecast to return (p_low 0.29 < 0.30)"),
        (Active.SLEEP, 0.30): ("NONE", "stay SLEEP (p_low 0.30 >= 0.30); ruApp may wake on PRACH"),
        (Active.ACTIVE, 0.90): ("SLEEP", "forecast low load (p_low 0.90 >= 0.70, load 3.0% < 10.0%): active=SLEEP with wake-on-PRACH envelope"),
        (Active.ACTIVE, 0.10): ("NONE", "p_low 0.10 < sleep threshold 0.70"),
    }
    for (active, p), (action, reason) in legacy.items():
        for sim_now in (None, "2024-01-10T02:30:00", "not-a-timestamp"):
            for st in (PolicyState(), PolicyState(wake_votes=7, sleep_entered_sim="2024-01-10T02:00:00", local_wake_sim="2024-01-10T02:20:00")):
                d = decide(status(active, CarrierState.READY), pm(3.0), p, 1000.0, st, low_threshold=THR, cfg=cfg, sim_now=sim_now)
                assert (d.action, d.reason) == (action, reason), (active, p, sim_now, st)
    # the keyword is optional: callers that never pass sim_now decide exactly as before
    d = decide(ASLEEP, pm(3.0), 0.29, 1000.0, PolicyState(), low_threshold=THR, cfg=cfg)
    assert d.action == "ACTIVE" and d.wake_votes == 1


def test_minutes_between_handles_naive_aware_and_garbage():
    from ocp_auru.rapp.policy_engine import minutes_between
    assert minutes_between("2024-01-10T02:00:00", "2024-01-10T02:45:00") == 45.0
    assert minutes_between("2026-01-01T02:00:00Z", "2026-01-01T03:00:00Z") == 60.0
    assert minutes_between("2026-01-01T02:00:00Z", "2026-01-01T02:30:00") == 30.0      # mixed pair: the naive side is UTC
    # a mixed pair with an offset is compared in UTC (the naive clock would say 360 here)
    assert minutes_between("2024-01-10T02:00:00", "2024-01-10T08:00:00+05:00") == 60.0
    assert minutes_between("2024-01-10T02:00:00+05:00", "2024-01-10T02:00:00") == 300.0
    assert minutes_between("2024-01-10T03:00:00", "2024-01-10T02:00:00") == -60.0
    assert minutes_between(None, "2024-01-10T02:00:00") is None and minutes_between("2024-01-10T02:00:00", None) is None
    assert minutes_between("garbage", "2024-01-10T02:00:00") is None and minutes_between("2024-01-10T02:00:00", "None") is None


def test_min_sleep_dwell_blocks_then_allows(monkeypatch):
    c = _cfg(monkeypatch, RAPP_SLEEP_THRESHOLD="0.7", RAPP_WAKE_THRESHOLD="0.3", RAPP_MIN_SLEEP_DWELL_MIN="60")
    st = PolicyState(sleep_entered_sim="2024-01-10T02:00:00", wake_votes=0)
    d = decide(ASLEEP, pm(0.0), 0.05, 1000.0, st, low_threshold=THR, cfg=c, sim_now="2024-01-10T02:30:00")
    assert d.action == "NONE" and "min sleep dwell: 30 of 60 min" in d.reason and d.wake_votes == 0
    d = decide(ASLEEP, pm(0.0), 0.05, 1000.0, st, low_threshold=THR, cfg=c, sim_now="2024-01-10T02:59:00")
    assert d.action == "NONE" and "min sleep dwell: 59 of 60 min" in d.reason
    d = decide(ASLEEP, pm(0.0), 0.05, 1000.0, st, low_threshold=THR, cfg=c, sim_now="2024-01-10T03:00:00")
    assert d.action == "ACTIVE" and d.reason == "pre-emptive wake: load forecast to return (p_low 0.05 < 0.30)"
    # the dwell rule is the first rule of the SLEEP branch and names itself even when the forecast says stay asleep
    d = decide(ASLEEP, pm(0.0), 0.9, 1000.0, st, low_threshold=THR, cfg=c, sim_now="2024-01-10T02:30:00")
    assert d.action == "NONE" and d.reason.startswith("min sleep dwell")
    # unknown clocks (rApp restart, no PM simTime yet, unparseable): the rule does not apply
    for st2, sim in ((PolicyState(), "2024-01-10T02:30:00"), (st, None), (st, "garbage")):
        assert decide(ASLEEP, pm(0.0), 0.05, 1000.0, st2, low_threshold=THR, cfg=c, sim_now=sim).action == "ACTIVE"
    # the RU's tz-aware simulated clock, and a mixed pair
    st3 = PolicyState(sleep_entered_sim="2026-01-01T02:00:00Z")
    assert "min sleep dwell" in decide(ASLEEP, pm(0.0), 0.05, 1000.0, st3, low_threshold=THR, cfg=c, sim_now="2026-01-01T02:45:00Z").reason
    assert "min sleep dwell" in decide(ASLEEP, pm(0.0), 0.05, 1000.0, st3, low_threshold=THR, cfg=c, sim_now="2026-01-01T02:45:00").reason
    # the ACTIVE branch never sees the dwell (it is the RU's min-hold-after-wake and RAPP_MIN_HOLD_S that guard it)
    assert decide(status(), pm(3.0), 0.9, 1000.0, st, low_threshold=THR, cfg=c, sim_now="2024-01-10T02:30:00").action == "SLEEP"
    # the engine never mutates the state
    assert st.wake_votes == 0 and st.sleep_entered_sim == "2024-01-10T02:00:00"


def test_wake_confirmation_windows(monkeypatch):
    c = _cfg(monkeypatch, RAPP_SLEEP_THRESHOLD="0.7", RAPP_WAKE_THRESHOLD="0.3", RAPP_WAKE_CONFIRM_WINDOWS="2")
    st = PolicyState()
    d1 = decide(ASLEEP, pm(0.0), 0.1, 1000.0, st, low_threshold=THR, cfg=c)
    assert d1.action == "NONE" and "wake vote 1 of 2" in d1.reason and d1.wake_votes == 1
    assert st.wake_votes == 0                    # pure engine: the caller persists the count ...
    st.wake_votes = d1.wake_votes                # ... as app.py:_plan and eval.replay_cell do
    d2 = decide(ASLEEP, pm(0.0), 0.1, 1000.0, st, low_threshold=THR, cfg=c)
    assert d2.action == "ACTIVE" and "pre-emptive wake" in d2.reason and "confirmed 2 of 2 consecutive windows" in d2.reason
    assert d2.wake_votes == 2
    # a window at/above the threshold resets the count
    st.wake_votes = 1
    d3 = decide(ASLEEP, pm(0.0), 0.5, 1000.0, st, low_threshold=THR, cfg=c)
    assert d3.action == "NONE" and "stay SLEEP" in d3.reason and d3.wake_votes == 0
    # outside the SLEEP branch the count is 0 (an awake carrier does not vote); BUSY keeps it
    assert decide(status(), pm(3.0), 0.1, 1000.0, PolicyState(wake_votes=1), low_threshold=THR, cfg=c).wake_votes == 0
    assert decide(status(Active.SLEEP, CarrierState.BUSY), pm(0.0), 0.1, 1000.0, PolicyState(wake_votes=1), low_threshold=THR, cfg=c).wake_votes == 1
    # three windows
    c3 = _cfg(monkeypatch, RAPP_WAKE_CONFIRM_WINDOWS="3")
    st = PolicyState(wake_votes=1)
    d = decide(ASLEEP, pm(0.0), 0.1, 1000.0, st, low_threshold=THR, cfg=c3)
    assert d.action == "NONE" and "wake vote 2 of 3" in d.reason and d.wake_votes == 2
    assert decide(ASLEEP, pm(0.0), 0.1, 1000.0, PolicyState(wake_votes=2), low_threshold=THR, cfg=c3).action == "ACTIVE"
    # while the dwell holds no vote is counted
    cd = _cfg(monkeypatch, RAPP_WAKE_CONFIRM_WINDOWS="2", RAPP_MIN_SLEEP_DWELL_MIN="60")
    st = PolicyState(sleep_entered_sim="2024-01-10T02:00:00", wake_votes=0)
    d = decide(ASLEEP, pm(0.0), 0.1, 1000.0, st, low_threshold=THR, cfg=cd, sim_now="2024-01-10T02:30:00")
    assert d.action == "NONE" and "min sleep dwell" in d.reason and d.wake_votes == 0


def test_wake_margin_lowers_the_effective_threshold(monkeypatch):
    c = _cfg(monkeypatch, RAPP_SLEEP_THRESHOLD="0.7", RAPP_WAKE_THRESHOLD="0.3", RAPP_WAKE_MARGIN="0.1")
    assert c.rapp_wake_threshold_effective == pytest.approx(0.2)
    d = decide(ASLEEP, pm(0.0), 0.25, 1000.0, PolicyState(), low_threshold=THR, cfg=c)    # would wake without the margin
    assert d.action == "NONE" and "stay SLEEP" in d.reason and "0.20 = wake 0.30 - margin 0.10" in d.reason and d.wake_votes == 0
    d = decide(ASLEEP, pm(0.0), 0.19, 1000.0, PolicyState(), low_threshold=THR, cfg=c)
    assert d.action == "ACTIVE" and "pre-emptive wake" in d.reason and "margin 0.10" in d.reason
    # clamped at 0: a margin >= the wake threshold disables the pre-emptive wake entirely (the ruApp still wakes locally)
    c0 = _cfg(monkeypatch, RAPP_WAKE_MARGIN="0.5")
    assert c0.rapp_wake_threshold_effective == 0.0
    assert decide(ASLEEP, pm(0.0), 0.0, 1000.0, PolicyState(), low_threshold=THR, cfg=c0).action == "NONE"
    # margin and confirmation combine; the reason names both - and names the confirmation ONLY when N > 1
    cb = _cfg(monkeypatch, RAPP_WAKE_MARGIN="0.1", RAPP_WAKE_CONFIRM_WINDOWS="2")
    d = decide(ASLEEP, pm(0.0), 0.15, 1000.0, PolicyState(wake_votes=1), low_threshold=THR, cfg=cb)
    assert d.action == "ACTIVE" and "margin 0.10" in d.reason and "confirmed 2 of 2" in d.reason
    d = decide(ASLEEP, pm(0.0), 0.15, 1000.0, PolicyState(), low_threshold=THR, cfg=c)
    assert d.action == "ACTIVE" and "confirmed" not in d.reason
    # Settings clamps the margin into [0, wake threshold]: a negative margin can never wake EARLIER
    cn = _cfg(monkeypatch, RAPP_WAKE_MARGIN="-0.2", RAPP_WAKE_CONFIRM_WINDOWS="0", RAPP_MIN_SLEEP_DWELL_MIN="-5",
              RAPP_PROFILE_SCALE="7", RAPP_POST_LOCAL_WAKE_HOLD_MIN="-1")
    assert cn.rapp_wake_margin == 0.0 and cn.rapp_wake_threshold_effective == cn.rapp_wake_threshold
    assert cn.rapp_wake_confirm_windows == 1 and cn.rapp_min_sleep_dwell_min == 0.0 and cn.rapp_profile_scale == 1.0
    assert cn.rapp_post_local_wake_hold_min == 0.0
    assert decide(ASLEEP, pm(0.0), 0.35, 1000.0, PolicyState(), low_threshold=THR, cfg=cn).action == "NONE"
    # the ACTIVE branch and the RAPP_ALLOW_INACTIVE restore path are untouched by the margin
    assert decide(status(), pm(3.0), 0.9, 1000.0, PolicyState(), low_threshold=THR, cfg=c).action == "SLEEP"
    ci = _cfg(monkeypatch, RAPP_WAKE_MARGIN="0.1", RAPP_ALLOW_INACTIVE="1")
    assert decide(status(Active.INACTIVE, CarrierState.DISABLED), pm(0.0), 0.25, 1000.0, PolicyState(), low_threshold=THR, cfg=ci).action == "ACTIVE"


# ---- clock hygiene and holds ------------------------------------------------------------------
def test_one_vote_per_pm_window(monkeypatch):
    """A second decision on the PM window the last vote was counted on (same sim_now) casts no vote; a new window
    does; an unknown window (sim_now None) always counts, as before."""
    c = _cfg(monkeypatch, RAPP_SLEEP_THRESHOLD="0.7", RAPP_WAKE_THRESHOLD="0.3", RAPP_WAKE_CONFIRM_WINDOWS="2")
    st = PolicyState()
    d1 = decide(ASLEEP, pm(0.0), 0.1, 1000.0, st, low_threshold=THR, cfg=c, sim_now="2024-01-10T10:00:00")
    assert d1.action == "NONE" and d1.wake_votes == 1 and d1.vote_sim == "2024-01-10T10:00:00"
    st.wake_votes, st.last_vote_sim = d1.wake_votes, d1.vote_sim     # as app.py:_plan persists them
    d2 = decide(ASLEEP, pm(0.0), 0.1, 1010.0, st, low_threshold=THR, cfg=c, sim_now="2024-01-10T10:00:00")
    assert d2.action == "NONE" and d2.wake_votes == 1 and d2.vote_sim == "2024-01-10T10:00:00" and "already counted" in d2.reason
    d3 = decide(ASLEEP, pm(0.0), 0.1, 1020.0, st, low_threshold=THR, cfg=c, sim_now="2024-01-10T10:30:00")
    assert d3.action == "ACTIVE" and d3.wake_votes == 2 and d3.vote_sim == "2024-01-10T10:30:00" and "confirmed 2 of 2" in d3.reason
    # a window at/above the threshold resets the count and the window
    d4 = decide(ASLEEP, pm(0.0), 0.9, 1030.0, st, low_threshold=THR, cfg=c, sim_now="2024-01-10T10:00:00")
    assert d4.wake_votes == 0 and d4.vote_sim is None
    # unknown window: every decision counts (nothing to compare)
    st0 = PolicyState(wake_votes=1, last_vote_sim=None)
    assert decide(ASLEEP, pm(0.0), 0.1, 1000.0, st0, low_threshold=THR, cfg=c, sim_now=None).action == "ACTIVE"
    # a failed ACTIVE keeps its votes (the caller does not reset them): the retry on the same window fires
    st2 = PolicyState(wake_votes=2, last_vote_sim="2024-01-10T10:30:00")
    d5 = decide(ASLEEP, pm(0.0), 0.1, 1040.0, st2, low_threshold=THR, cfg=c, sim_now="2024-01-10T10:30:00")
    assert d5.action == "ACTIVE" and d5.wake_votes == 2
    # BUSY and the dwell keep both fields; the engine never mutates the state
    cd = _cfg(monkeypatch, RAPP_WAKE_CONFIRM_WINDOWS="2", RAPP_MIN_SLEEP_DWELL_MIN="60")
    st3 = PolicyState(wake_votes=1, last_vote_sim="2024-01-10T02:00:00", sleep_entered_sim="2024-01-10T02:00:00")
    d6 = decide(ASLEEP, pm(0.0), 0.1, 1000.0, st3, low_threshold=THR, cfg=cd, sim_now="2024-01-10T02:30:00")
    assert "min sleep dwell" in d6.reason and d6.wake_votes == 1 and d6.vote_sim == "2024-01-10T02:00:00"
    d7 = decide(status(Active.SLEEP, CarrierState.BUSY), pm(0.0), 0.1, 1000.0, st3, low_threshold=THR, cfg=cd, sim_now="2024-01-10T02:30:00")
    assert d7.wake_votes == 1 and d7.vote_sim == "2024-01-10T02:00:00"
    assert st3.wake_votes == 1 and st3.last_vote_sim == "2024-01-10T02:00:00"


def test_post_local_wake_hold_refuses_a_resleep_then_releases(monkeypatch):
    """RAPP_POST_LOCAL_WAKE_HOLD_MIN (data time since the ruApp's local wake) guards ACTIVE -> SLEEP; it is disjoint
    from the RU's wall-second min-hold-after-wake and from RAPP_MIN_HOLD_S; 30 min is the shipped default since
    0 = off - both pinned explicitly here."""
    c = _cfg(monkeypatch, RAPP_SLEEP_THRESHOLD="0.7", RAPP_WAKE_THRESHOLD="0.3", RAPP_MIN_HOLD_S="0", RAPP_POST_LOCAL_WAKE_HOLD_MIN="30")
    st = PolicyState(local_wake_sim="2024-01-10T03:10:00")
    d = decide(status(), pm(3.0), 0.95, 1000.0, st, low_threshold=THR, cfg=c, sim_now="2024-01-10T03:10:00")
    assert d.action == "NONE" and d.reason.startswith("post-local-wake hold: 0 of 30 min")
    d = decide(status(), pm(3.0), 0.95, 1000.0, st, low_threshold=THR, cfg=c, sim_now="2024-01-10T03:39:00")
    assert d.action == "NONE" and "29 of 30 min" in d.reason
    assert decide(status(), pm(3.0), 0.95, 1000.0, st, low_threshold=THR, cfg=c, sim_now="2024-01-10T03:40:00").action == "SLEEP"
    # the hold is checked after the forecast/load rules (a NONE for another reason names that reason)
    assert "sleep threshold" in decide(status(), pm(3.0), 0.1, 1000.0, st, low_threshold=THR, cfg=c, sim_now="2024-01-10T03:10:00").reason
    # unknown clocks: no wake observed, no sim time, unparseable -> the rule does not apply
    for st2, sim in ((PolicyState(), "2024-01-10T03:10:00"), (st, None), (st, "garbage")):
        assert decide(status(), pm(3.0), 0.95, 1000.0, st2, low_threshold=THR, cfg=c, sim_now=sim).action == "SLEEP"
    # 0 = off (pinned; the hold is on by default)
    c0 = _cfg(monkeypatch, RAPP_POST_LOCAL_WAKE_HOLD_MIN="0")
    assert decide(status(), pm(3.0), 0.95, 1000.0, st, low_threshold=THR, cfg=c0, sim_now="2024-01-10T03:10:00").action == "SLEEP"
    # the SLEEP branch never sees it
    assert decide(ASLEEP, pm(0.0), 0.1, 1000.0, st, low_threshold=THR, cfg=c, sim_now="2024-01-10T03:10:00").action == "ACTIVE"
    assert st.local_wake_sim == "2024-01-10T03:10:00"
