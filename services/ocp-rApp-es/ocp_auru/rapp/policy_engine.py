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
"""Energy-saving policy engine (HLD §3.7, ADR-3): pure, deterministic decisions.

The decision is expressed entirely in `o-ran-uplane-conf` terms — the action is the
value written to `tx-array-carriers/active`, and `SLEEP` carries the ruApp envelope
(`ocp-ruapp-policy`, `wake-on-prach=true`) that permits a local SLEEP -> ACTIVE wake.
`INACTIVE` is a hard-off outside the envelope: the engine never proposes it, and the app
only forwards a forced INACTIVE when `RAPP_ALLOW_INACTIVE=1`.

Five DISJOINT hold mechanisms touch a carrier (HLD §3.7 table); only the last four live here:

| mechanism | clock | guards | where |
|---|---|---|---|
| RU `min-hold-after-wake-s` (envelope) | wall s on the RU | a new SLEEP/INACTIVE after a LOCAL wake (`HoldViolation`) | mock_ru/datastore.py |
| `RAPP_MIN_HOLD_S` | wall s since the rApp's own last actuation | ACTIVE -> SLEEP | ACTIVE branch |
| `RAPP_POST_LOCAL_WAKE_HOLD_MIN` | sim/data min since a ruApp local wake seen in PM | ACTIVE -> SLEEP | ACTIVE branch |
| `RAPP_MIN_SLEEP_DWELL_MIN` + `RAPP_WAKE_CONFIRM_WINDOWS` | sim/data min since the rApp's SLEEP; distinct PM windows | SLEEP -> ACTIVE (pre-emptive) | SLEEP branch |
| `RAPP_MAX_ACTUATIONS_PER_DAY` (+ `RAPP_ACTUATION_ALARM_PER_DAY`) | applied rApp writes per carrier per sim/data calendar day | any NEW SLEEP and any NEW pre-emptive ACTIVE | last guard of both branches |

The fifth is the per-carrier-day ACTUATION BUDGET (`RAPP_MAX_ACTUATIONS_PER_DAY`, a "max toggles per day" guard): the engine counts nothing itself - the caller keeps
`PolicyState.actuations_day` / `actuations_today` (`count_actuation` on an APPLIED CM write, `roll_actuation_day`
when the sim day changes) - and, at the budget, refuses any NEW rApp-initiated actuation (SLEEP and pre-emptive
ACTIVE) for the rest of that sim day; the carrier stays as it is and the ruApp remains the wake path. Both the
budget and the lower `RAPP_ACTUATION_ALARM_PER_DAY` are reported on every Decision (`actuations_today`,
`budget_reached`, `alarm`). The `RAPP_ALLOW_INACTIVE` restore of a hard-off carrier is not budgeted (service-first).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Optional

from pydantic import BaseModel, Field

from ocp_auru.common.config import Settings, settings
from ocp_auru.common.events import Active, CarrierPm, CarrierState, CarrierStatus, RuAppPolicy

log = logging.getLogger(__name__)

Action = Literal["SLEEP", "ACTIVE", "NONE", "INACTIVE"]


class PolicyState(BaseModel):
    """Per-carrier memory the caller keeps between cycles (the engine never mutates it).

    Chatter control (inert while the settings are at
    their defaults): `sleep_entered_sim` is the simulated/data time (ISO) of the last SUCCESSFUL
    SLEEP actuation (the dwell clock; the caller clears it whenever PM shows the carrier not asleep);
    `wake_votes` is the number of consecutive PM windows that saw p_low below the effective wake
    threshold and `last_vote_sim` the PM simTime the last vote was counted on (a second decision on
    the same window casts no second vote); the caller copies both back from the Decision after EVERY
    decision and resets them on a successful actuation (a failed ACTIVE keeps its votes so the retry
    needs no re-confirmation). `local_wake_sim` is the PM simTime at which a ruApp LOCAL wake was
    observed (post-local-wake hold). Restart semantics: every field starts empty, so no dwell/hold
    applies until the next observed event (at most one extra flip, service-first; config.py).

    Actuation budget: `actuations_day` is the calendar day
    (YYYY-MM-DD) of SIMULATED/data time the counter belongs to and `actuations_today` the number of APPLIED
    rApp actuations (SLEEP / ACTIVE / INACTIVE CM writes, forced ones included) on that day; the caller
    increments through `count_actuation` on an applied write only and the pair resets when the sim day
    changes (`roll_actuation_day`, or implicitly in `decide`, which reads the count as 0 for any other day).
    After a restart the count starts at 0 for the day (at most one extra budget's worth, service-first)."""

    last_decision_wall: Optional[float] = None   # time.time() of the last actuation attempt
    last_action: Optional[str] = None            # the last actuated/attempted action
    envelope: RuAppPolicy = Field(default_factory=RuAppPolicy)
    sleep_entered_sim: Optional[str] = None      # ISO sim/data time of the last successful SLEEP actuation (dwell clock)
    wake_votes: int = 0                          # consecutive PM windows with p_low < effective wake threshold while asleep
    last_vote_sim: Optional[str] = None          # PM simTime of the last counted vote (one vote per window)
    local_wake_sim: Optional[str] = None         # PM simTime at which a ruApp local wake was observed
    actuations_day: Optional[str] = None         # YYYY-MM-DD (sim/data time) the actuation counter belongs to
    actuations_today: int = 0                    # APPLIED rApp actuations on that sim day (the budget's counter)
    # bookkeeping the app uses to detect a local wake from PM (previous cycle's view of the carrier)
    last_seen_active: Optional[str] = None
    last_wake_events: Optional[int] = None


class Decision(BaseModel):
    carrier: str
    action: Action
    reason: str
    p_low: float
    envelope: RuAppPolicy
    active_before: Optional[Active] = None
    prb_util_dl_pct: Optional[float] = None
    low_threshold: Optional[float] = None
    forced: bool = False
    wall_ts: float = 0.0
    sim_time: Optional[str] = None
    forecast_basis: Optional[str] = None  # "pm-window" (own PM) | "expected-hour-profile" (carrier asleep)
    # the wake-vote count AFTER this decision and the PM window it was counted on; the caller stores both
    wake_votes: int = 0
    vote_sim: Optional[str] = None
    # side-by-side estimators at the same instant (diagnostic, filled by the caller): the own-PM forecast,
    # the hour-profile forecast, their difference (pm - profile; |delta| > 0.5 on an AWAKE decision =
    # divergent; asleep, the own-PM estimate reads zero counters and is recorded only) and the level
    # factor applied to the profile (1.0 = estate profile, RAPP_PROFILE_SCALE=0)
    p_low_pm_window: Optional[float] = None
    p_low_profile: Optional[float] = None
    forecast_divergence: Optional[float] = None
    forecast_divergent: bool = False
    profile_scale: Optional[float] = None
    # actuation budget: the carrier's APPLIED actuations on the sim day of this decision BEFORE it, whether
    # the day's budget is reached (RAPP_MAX_ACTUATIONS_PER_DAY > 0 and count >= it: no new SLEEP / pre-emptive
    # ACTIVE) and whether the alarm threshold is reached (RAPP_ACTUATION_ALARM_PER_DAY > 0 and count >= it)
    actuations_today: int = 0
    budget_reached: bool = False
    alarm: bool = False


def _prb(latest_pm: CarrierPm | Mapping[str, Any] | None) -> Optional[float]:
    if latest_pm is None:
        return None
    v = latest_pm.prb_util_dl_pct if isinstance(latest_pm, CarrierPm) else latest_pm.get("prb_util_dl_pct")
    return None if v is None else float(v)


_mixed_pair_logged = False


def minutes_between(earlier: Optional[str], later: Optional[str]) -> Optional[float]:
    """`later - earlier` in minutes for two ISO timestamps (naive or tz-aware). When exactly one
    side is tz-aware the naive one is taken as UTC (the RU's simulated clock convention, `...Z`)
    and the pair is compared in UTC; the first such pair is logged once. None when either is
    missing or unparseable: the dwell/hold rule then simply does not apply, exactly as with the
    setting off."""
    global _mixed_pair_logged
    if not earlier or not later:
        return None
    try:
        a, b = datetime.fromisoformat(str(earlier)), datetime.fromisoformat(str(later))
    except (TypeError, ValueError):
        return None
    if (a.tzinfo is None) != (b.tzinfo is None):
        if not _mixed_pair_logged:
            log.warning("minutes_between: mixed naive/aware timestamps (%s, %s); the naive one is taken as UTC", earlier, later)
            _mixed_pair_logged = True
        a = a.replace(tzinfo=timezone.utc) if a.tzinfo is None else a.astimezone(timezone.utc)
        b = b.replace(tzinfo=timezone.utc) if b.tzinfo is None else b.astimezone(timezone.utc)
    return (b - a).total_seconds() / 60.0


# ---- actuation budget: the caller-owned per-carrier-day counter ---------------------------
def sim_day(sim_now: Optional[str]) -> Optional[str]:
    """Calendar day (YYYY-MM-DD) of an ISO sim/data timestamp as written (the RU's simulated clock; no
    timezone conversion), None when missing or unparseable: the budget then simply does not apply."""
    if not sim_now:
        return None
    try:
        return datetime.fromisoformat(str(sim_now)).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def actuations_today(state: PolicyState, sim_now: Optional[str]) -> int:
    """The carrier's applied actuations on the sim day of `sim_now` as the engine reads them: the stored
    count when it belongs to that day, 0 for any other day (the implicit midnight reset) or an unknown day."""
    day = sim_day(sim_now)
    return int(state.actuations_today) if day is not None and state.actuations_day == day else 0


def roll_actuation_day(state: PolicyState, sim_now: Optional[str]) -> bool:
    """Caller helper (app.py:_observe, eval.replay_cell): reset the counter when the sim day moved on so the
    stored pair always reads as the current day's. Returns True when it reset. An unknown day changes nothing."""
    day = sim_day(sim_now)
    if day is None or state.actuations_day == day:
        return False
    state.actuations_day, state.actuations_today = day, 0
    return True


def count_actuation(state: PolicyState, sim_now: Optional[str]) -> int:
    """Caller helper, on an APPLIED CM write only (app.py:_actuate inside `resp.ok`; eval mirrors): roll the
    day if needed, then increment the carrier's count for the sim day of `sim_now`. Returns the new count.
    With an unknown day the write is counted under day None (never read by `decide`)."""
    roll_actuation_day(state, sim_now)
    state.actuations_today = int(state.actuations_today) + 1
    return state.actuations_today


def _budget(state: PolicyState, cfg: Settings, sim_now: Optional[str]) -> tuple[Optional[str], int, bool, bool]:
    """(day, count today, budget reached, alarm reached) for this decision. Either threshold at 0 is off;
    an unknown day (no PM simTime yet, unparseable) applies neither."""
    day = sim_day(sim_now)
    today = actuations_today(state, sim_now)
    cap, alarm = int(cfg.rapp_max_actuations_per_day), int(cfg.rapp_actuation_alarm_per_day)
    reached = day is not None and cap > 0 and today >= cap
    alarmed = day is not None and alarm > 0 and today >= alarm
    return day, today, reached, alarmed


def _budget_reason(day: Optional[str], today: int, cfg: Settings, refused: str, p_low: float) -> str:
    return (f"actuation budget: {today} of {int(cfg.rapp_max_actuations_per_day)} for {day} reached; ruApp remains the wake path "
            f"({refused} refused, p_low {p_low:.2f})")


def decide(carrier_status: CarrierStatus, latest_pm: CarrierPm | Mapping[str, Any] | None, p_low: float,
           now: float, state: PolicyState, *, low_threshold: float, cfg: Settings = settings,
           sim_now: Optional[str] = None) -> Decision:
    """Rules (in order):
    - carrier BUSY (settling after a previous `active` write)      -> NONE
    - ACTIVE, p_low >= RAPP_SLEEP_THRESHOLD, load below threshold now,
      >= RAPP_POST_LOCAL_WAKE_HOLD_MIN of sim time since a ruApp local wake (0 = off),
      and >= RAPP_MIN_HOLD_S since the last actuation                -> SLEEP (+ wake-on-PRACH envelope)
    - SLEEP and p_low < RAPP_WAKE_THRESHOLD (load forecast to return) -> ACTIVE (pre-emptive; no hold)
      Chatter control (all off by default, see config.py):
        * RAPP_MIN_SLEEP_DWELL_MIN > 0 and both `sim_now` and `state.sleep_entered_sim` known and
          less than the dwell has elapsed (sim/data time)             -> NONE, no wake vote counted
        * the wake compares p_low against the EFFECTIVE wake threshold
          (RAPP_WAKE_THRESHOLD - RAPP_WAKE_MARGIN, >= 0) and needs it for
          RAPP_WAKE_CONFIRM_WINDOWS consecutive PM WINDOWS (`wake_votes`; a decision on the same
          `sim_now` as the last counted vote casts none)                -> NONE until confirmed
    - INACTIVE                                                        -> NONE, unless RAPP_ALLOW_INACTIVE
      and load is forecast to return, in which case ACTIVE
    - otherwise                                                       -> NONE
    - ACTUATION BUDGET (last guard of both branches, so a NONE for any other reason names that reason):
      RAPP_MAX_ACTUATIONS_PER_DAY > 0 and the carrier's APPLIED actuations on the sim day of `sim_now`
      (`state.actuations_day` / `actuations_today`, kept by the caller) >= it
                                                                      -> NONE instead of the SLEEP or the
      pre-emptive ACTIVE ("actuation budget: N of N for <day> reached; ruApp remains the wake path");
      the carrier stays in whatever state it is in; a forced action bypasses this (the caller overrides
      the action) but still counts; the RAPP_ALLOW_INACTIVE restore is not budgeted.

    `sim_now` is the ISO timestamp of the current window (PM simTime); it feeds the dwell clock, the
    post-local-wake hold, the one-vote-per-window rule and the budget's calendar day. `Decision.wake_votes` /
    `vote_sim` are the vote count and its window after this decision (0 / None outside the SLEEP branch,
    unchanged while the dwell holds); the caller persists them and resets them on a successful actuation.
    `Decision.actuations_today` / `budget_reached` / `alarm` describe the budget state BEFORE this decision.
    """
    prb = _prb(latest_pm)
    envelope = state.envelope.model_copy(update={"wake_on_prach": True})
    day, today, reached, alarmed = _budget(state, cfg, sim_now)
    base = dict(carrier=carrier_status.name, p_low=float(p_low), envelope=envelope, active_before=carrier_status.active,
                prb_util_dl_pct=prb, low_threshold=float(low_threshold), wall_ts=float(now),
                actuations_today=today, budget_reached=reached, alarm=alarmed)

    if carrier_status.state == CarrierState.BUSY:
        return Decision(action="NONE", reason="carrier BUSY (settling after the last active write)",
                        wake_votes=state.wake_votes, vote_sim=state.last_vote_sim, **base)

    if carrier_status.active == Active.ACTIVE:
        if p_low < cfg.rapp_sleep_threshold:
            return Decision(action="NONE", reason=f"p_low {p_low:.2f} < sleep threshold {cfg.rapp_sleep_threshold:.2f}", **base)
        if prb is None or not (prb < low_threshold):
            shown = "n/a" if prb is None else f"{prb:.1f}%"
            return Decision(action="NONE", reason=f"load now {shown} not below low threshold {low_threshold:.1f}%", **base)
        hold_min = float(cfg.rapp_post_local_wake_hold_min)
        if hold_min > 0:
            since = minutes_between(state.local_wake_sim, sim_now)
            if since is not None and since < hold_min:
                return Decision(action="NONE", reason=f"post-local-wake hold: {max(since, 0.0):.0f} of {hold_min:.0f} min since the ruApp "
                                                      f"woke the carrier (sim time); SLEEP deferred (p_low {p_low:.2f})", **base)
        if state.last_decision_wall is not None and (now - state.last_decision_wall) < cfg.rapp_min_hold_s:
            remaining = cfg.rapp_min_hold_s - (now - state.last_decision_wall)
            return Decision(action="NONE", reason=f"min hold: {remaining:.1f}s of {cfg.rapp_min_hold_s:.0f}s remaining since last decision", **base)
        if reached:
            return Decision(action="NONE", reason=_budget_reason(day, today, cfg, "SLEEP", p_low), **base)
        return Decision(action="SLEEP", reason=f"forecast low load (p_low {p_low:.2f} >= {cfg.rapp_sleep_threshold:.2f}, "
                                               f"load {prb:.1f}% < {low_threshold:.1f}%): active=SLEEP with wake-on-PRACH envelope", **base)

    if carrier_status.active == Active.SLEEP:
        return _decide_asleep(p_low, state, cfg, sim_now, base, budget=(day, today, reached))

    # Active.INACTIVE: hard off, outside the ruApp envelope.
    if cfg.rapp_allow_inactive and p_low < cfg.rapp_wake_threshold:
        return Decision(action="ACTIVE", reason=f"RAPP_ALLOW_INACTIVE: restoring hard-off carrier (p_low {p_low:.2f})", **base)
    return Decision(action="NONE", reason="carrier INACTIVE (hard off) is outside the rApp envelope; RAPP_ALLOW_INACTIVE=0", **base)


def _decide_asleep(p_low: float, state: PolicyState, cfg: Settings, sim_now: Optional[str], base: dict[str, Any], *,
                   budget: tuple[Optional[str], int, bool] = (None, 0, False)) -> Decision:
    """The SLEEP branch: dwell, then the (margin-adjusted, vote-confirmed) pre-emptive wake, then the
    actuation budget (the vote is still counted, the ACTIVE is refused until the sim day changes).
    With the defaults (dwell 0, margin 0, 1 window) the reason strings are the baseline
    ones verbatim; a held rule always names itself in the reason."""
    day, today, reached = budget
    dwell = float(cfg.rapp_min_sleep_dwell_min)
    if dwell > 0:
        elapsed = minutes_between(state.sleep_entered_sim, sim_now)
        if elapsed is not None and elapsed < dwell:
            return Decision(action="NONE", wake_votes=state.wake_votes, vote_sim=state.last_vote_sim,
                            reason=f"min sleep dwell: {max(elapsed, 0.0):.0f} of {dwell:.0f} min since SLEEP (sim time); "
                                   f"wake deferred (p_low {p_low:.2f})", **base)
    wake_thr = cfg.rapp_wake_threshold_effective
    margin = float(cfg.rapp_wake_margin)
    need = max(1, int(cfg.rapp_wake_confirm_windows))
    thr_txt = f"{wake_thr:.2f}" if margin <= 0 else f"{wake_thr:.2f} = wake {cfg.rapp_wake_threshold:.2f} - margin {margin:.2f}"
    if p_low < wake_thr:
        # one vote per PM window: a second decision on the window the last vote was counted on (auto cycle +
        # POST /cycle, or a cycle shorter than the PM interval) casts none; unknown windows always count
        repeat = sim_now is not None and state.last_vote_sim is not None and str(sim_now) == str(state.last_vote_sim)
        votes = int(state.wake_votes) if repeat else int(state.wake_votes) + 1
        vote_sim = state.last_vote_sim if repeat else sim_now
        if votes >= need:
            if reached:
                return Decision(action="NONE", wake_votes=votes, vote_sim=vote_sim,
                                reason=_budget_reason(day, today, cfg, "pre-emptive ACTIVE", p_low), **base)
            confirmed = "" if need == 1 else f"; confirmed {votes} of {need} consecutive windows"
            return Decision(action="ACTIVE", wake_votes=votes, vote_sim=vote_sim,
                            reason=f"pre-emptive wake: load forecast to return (p_low {p_low:.2f} < {thr_txt}{confirmed})", **base)
        held = " (already counted on this PM window)" if repeat else ""
        return Decision(action="NONE", wake_votes=votes, vote_sim=vote_sim,
                        reason=f"wake vote {votes} of {need}{held}: p_low {p_low:.2f} < {thr_txt}; SLEEP kept until confirmed", **base)
    return Decision(action="NONE", wake_votes=0, vote_sim=None,
                    reason=f"stay SLEEP (p_low {p_low:.2f} >= {thr_txt}); ruApp may wake on PRACH", **base)
