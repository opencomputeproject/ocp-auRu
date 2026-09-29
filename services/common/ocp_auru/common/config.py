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
"""Environment-driven settings shared by every process in the demo.

Every port/URL has exactly one source of truth here. Processes read `settings` at import
time; tests override via environment variables before importing, or construct `Settings()`
directly. Ports were chosen to stay clear of common local stacks (8081, 8100).

Paths default to the repository checkout this file lives in (the services are installed
editable). `OCP_AURU_REPO_ROOT` overrides the root for any other install layout.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# .../services/common/ocp_auru/common/config.py -> repository root
REPO_ROOT = Path(os.environ.get("OCP_AURU_REPO_ROOT") or Path(__file__).resolve().parents[4])
DEMO_ROOT = REPO_ROOT / "demo"  # scripts, var/ (runtime state, gitignored), data/private/ (gitignored)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_opt_float(name: str) -> Optional[float]:
    """Float setting whose empty/unset value means \"derive from data\"."""
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else None


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # ---- profile ------------------------------------------------------------------------
    profile: str = field(default_factory=lambda: _env("PROFILE", "lite"))  # lite | full
    var_dir: Path = field(default_factory=lambda: Path(_env("VAR_DIR", str(DEMO_ROOT / "var"))))

    # ---- mock O-RU (M-Plane server + ruApp) ---------------------------------------------
    ru_id: str = field(default_factory=lambda: _env("RU_ID", "oru-0001"))
    ru_netconf_host: str = field(default_factory=lambda: _env("RU_NETCONF_HOST", "127.0.0.1"))
    ru_netconf_port: int = field(default_factory=lambda: _env_int("RU_NETCONF_PORT", 1830))
    ru_netconf_user: str = field(default_factory=lambda: _env("RU_NETCONF_USER", "oru"))
    # Demo-only default for a mock device; override with RU_NETCONF_PASSWORD.
    ru_netconf_password: str = field(default_factory=lambda: _env("RU_NETCONF_PASSWORD", "oru-demo"))
    ru_host_key: Path = field(default_factory=lambda: Path(_env("RU_HOST_KEY", str(DEMO_ROOT / "var" / "ru-host-key"))))
    ru_mgmt_port: int = field(default_factory=lambda: _env_int("RU_MGMT_PORT", 8102))
    ru_carriers: tuple[str, ...] = field(default_factory=lambda: tuple(c for c in _env("RU_CARRIERS", "txcarrier0").split(",") if c))
    ru_pm_interval_s: float = field(default_factory=lambda: _env_float("RU_PM_INTERVAL_S", 5.0))
    ru_sim_speedup: float = field(default_factory=lambda: _env_float("RU_SIM_SPEEDUP", 720.0))
    ru_settle_ms: float = field(default_factory=lambda: _env_float("RU_SETTLE_MS", 2.0))
    ru_initial_active: str = field(default_factory=lambda: _env("RU_INITIAL_ACTIVE", "ACTIVE"))
    # Simulated wall clock starts at this hour (02:00 = low-load night) so the narrative is short.
    ru_sim_start_hour: float = field(default_factory=lambda: _env_float("RU_SIM_START_HOUR", 2.0))
    # Optional load replay: JSON file {"prb_util_dl_pct_by_hour": [24 numbers]} (mean DL PRB % per hour of day); empty = the
    # generic diurnal curve. RU_SIM_PRB_SIGMA is the per-tick noise on the simulated DL PRB utilisation, in PRB %.
    ru_load_profile: str = field(default_factory=lambda: _env("RU_LOAD_PROFILE", ""))
    ru_sim_prb_sigma: float = field(default_factory=lambda: _env_float("RU_SIM_PRB_SIGMA", 3.0))

    # ---- SMO services -------------------------------------------------------------------
    smo_host: str = field(default_factory=lambda: _env("SMO_HOST", "127.0.0.1"))
    smo_port: int = field(default_factory=lambda: _env_int("SMO_PORT", 8103))
    # CM_ADAPTER selects the south-bound adapter behind the SMO's CM routes (ocp_auru/smo/o1_adapter.py).
    # "mplane-direct" (reference): the SMO performs the NETCONF edit-config on the O-RU itself
    # (O-RAN hybrid M-Plane model).
    cm_adapter: str = field(default_factory=lambda: _env("CM_ADAPTER", "mplane-direct"))
    # CM_RUAPP_POLICY=0 omits the ruApp policy container from the edit-config (M-Plane servers without the extension module).
    cm_include_ruapp_policy: bool = field(default_factory=lambda: _env_bool("CM_RUAPP_POLICY", True))
    # M-Plane client of the SMO (NETCONF over SSH towards RU_NETCONF_HOST:RU_NETCONF_PORT).
    mplane_timeout_s: float = field(default_factory=lambda: _env_float("MPLANE_TIMEOUT_S", 10.0))
    # The demo skips SSH host-key verification of the mock O-RU (its key is generated on first
    # start); a real O-RU deployment must pin the key (set to 1).
    mplane_hostkey_verify: bool = field(default_factory=lambda: _env_bool("MPLANE_HOSTKEY_VERIFY", False))

    # ---- es-rApp ------------------------------------------------------------------------
    rapp_host: str = field(default_factory=lambda: _env("RAPP_HOST", "127.0.0.1"))
    rapp_port: int = field(default_factory=lambda: _env_int("RAPP_PORT", 8101))
    rapp_cycle_s: float = field(default_factory=lambda: _env_float("RAPP_CYCLE_S", 10.0))
    rapp_sleep_threshold: float = field(default_factory=lambda: _env_float("RAPP_SLEEP_THRESHOLD", 0.7))
    rapp_data: str = field(default_factory=lambda: _env("RAPP_DATA", "synthetic"))  # synthetic | kpi_report_csv
    rapp_data_dir: Path = field(default_factory=lambda: Path(_env("RAPP_DATA_DIR", str(DEMO_ROOT / "data" / "private"))))
    rapp_allow_inactive: bool = field(default_factory=lambda: _env_bool("RAPP_ALLOW_INACTIVE", False))
    rapp_auto_cycle: bool = field(default_factory=lambda: _env_bool("RAPP_AUTO_CYCLE", True))
    # -- actuation hold, model lifecycle, R1/CM timing --
    rapp_min_hold_s: float = field(default_factory=lambda: _env_float("RAPP_MIN_HOLD_S", 20.0))  # wall seconds between actuations per carrier
    rapp_retrain: bool = field(default_factory=lambda: _env_bool("RAPP_RETRAIN", False))          # 1 = ignore persisted model, retrain at start
    rapp_wake_threshold: float = field(default_factory=lambda: _env_float("RAPP_WAKE_THRESHOLD", 0.3))  # SLEEP -> ACTIVE when p_low < this
    # Low-load threshold in prb_util_dl_pct; unset/empty = data-relative (25th percentile of the training set).
    rapp_low_threshold: Optional[float] = field(default_factory=lambda: _env_opt_float("RAPP_LOW_THRESHOLD"))
    rapp_r1_retry_s: float = field(default_factory=lambda: _env_float("RAPP_R1_RETRY_S", 2.0))       # background retry of catalogue/ICS registration
    rapp_http_timeout_s: float = field(default_factory=lambda: _env_float("RAPP_HTTP_TIMEOUT_S", 5.0))
    rapp_cm_timeout_s: float = field(default_factory=lambda: _env_float("RAPP_CM_TIMEOUT_S", 30.0))  # CM request spans SMO -> NETCONF session -> edit-config
    rapp_history_len: int = field(default_factory=lambda: _env_int("RAPP_HISTORY_LEN", 64))         # PM windows kept per carrier for features
    rapp_synthetic_cells: int = field(default_factory=lambda: _env_int("RAPP_SYNTHETIC_CELLS", 3))
    rapp_synthetic_days: int = field(default_factory=lambda: _env_int("RAPP_SYNTHETIC_DAYS", 14))
    rapp_seed: int = field(default_factory=lambda: _env_int("RAPP_SEED", 42))
    # -- offline energy evaluation: where `scripts/eval_energy.py` writes its
    # report/summary/plots (one UTC-stamped sub-directory per run; stays in the gitignored var/).
    rapp_eval_dir: Path = field(default_factory=lambda: Path(_env("RAPP_EVAL_DIR", str(DEMO_ROOT / "var" / "eval"))))
    # -- chatter control: a minimum sleep DWELL and HYSTERESIS on the pre-emptive SLEEP -> ACTIVE wake.
    # Defaults: profile scale 1.0, dwell 60, post-local-wake hold 30; confirmation and margin off
    # (the offline experiment `eval_energy.py --grid` compares the combinations).
    # These three are RATE LIMITERS on the symptom; the root cause of the chatter is the estate-pooled
    # hour-of-day profile (see `rapp_profile_scale` below). Values are clamped in `__post_init__`.
    # Dwell counts SIMULATED/data time (PM simTime), not wall seconds; 0 = off.
    rapp_min_sleep_dwell_min: float = field(default_factory=lambda: _env_float("RAPP_MIN_SLEEP_DWELL_MIN", 60.0))
    # Pre-emptive wake needs p_low < the effective wake threshold for this many consecutive PM WINDOWS
    # (distinct PM simTimes, not decision cycles: two cycles on one window cast one vote); 1 = off.
    rapp_wake_confirm_windows: int = field(default_factory=lambda: _env_int("RAPP_WAKE_CONFIRM_WINDOWS", 1))
    # Effective wake threshold = rapp_wake_threshold - rapp_wake_margin; clamped to [0, rapp_wake_threshold]
    # (a negative margin would silently wake EARLIER); 0.0 = off. It has little effect while the
    # classifier's p_low is near-binary; it stays as a documented knob, not as a fix.
    rapp_wake_margin: float = field(default_factory=lambda: _env_float("RAPP_WAKE_MARGIN", 0.0))
    # -- the ROOT-CAUSE step and the post-local-wake hold, confirmation and margin OFF ------
    # RAPP_PROFILE_SCALE in [0, 1]: weight of the carrier's OWN load level in the asleep forecast. The
    # hour-of-day profile is pooled over the whole training estate (model.build_hourly_profile groups by
    # weekend x half-hour only), so a sleeping carrier is forecast from the ESTATE-mean load while an awake
    # one is forecast from its own PM; on a quiet site the two disagree at the trough edge and the rApp
    # flips. With weight w the profile bins are multiplied per column by
    # 1 + w * (ratio - 1), ratio = mean(own awake PM at those bins) / mean(profile at those bins) over the
    # carrier's PM history (model.LowLoadModel.profile_level_ratio; needs >= 6 awake windows, clipped to
    # [0.2, 5]). 0 = off (estate profile as-is), 1 = full level scaling. Evaluated by `eval_energy.py --grid`.
    rapp_profile_scale: float = field(default_factory=lambda: _env_float("RAPP_PROFILE_SCALE", 1.0))
    # RAPP_POST_LOCAL_WAKE_HOLD_MIN: minutes of SIMULATED/data time after a ruApp LOCAL wake (PM shows
    # SLEEP -> ACTIVE not written by the rApp, or wake_events grew) during which the rApp does not SLEEP the
    # carrier again (30 = one PM window); 0 = off. Distinct from the RU's own `min-hold-after-wake-s`
    # (wall seconds, enforced on the RU) and from RAPP_MIN_HOLD_S (wall seconds since the rApp's own
    # last actuation).
    rapp_post_local_wake_hold_min: float = field(default_factory=lambda: _env_float("RAPP_POST_LOCAL_WAKE_HOLD_MIN", 30.0))
    # Restart / unknown-state semantics of the chatter control: every clock (`sleep_entered_sim`,
    # `local_wake_sim`, `last_vote_sim`) and the vote count live in the rApp's in-memory PolicyState. After
    # an rApp restart, or for a carrier that entered SLEEP without an rApp write (operator, another rApp, RU
    # restart), `sleep_entered_sim` is None => NO dwell applies and the post-local-wake hold is off until
    # the next observed wake: the carrier can be flipped at most ONE extra time, service-first, never held
    # asleep on a stale clock.
    # -- actuation budget (a "max toggles per day" guard) --
    # RAPP_MAX_ACTUATIONS_PER_DAY: the rApp's own actuations (APPLIED SLEEP + pre-emptive ACTIVE CM writes) per
    # carrier per calendar day of SIMULATED/data time (the day of the PM simTime the decision is taken on, never
    # wall time). At the budget the engine refuses any NEW SLEEP and any NEW pre-emptive ACTIVE for that carrier
    # until the sim day changes: the carrier stays in whatever state it is in and the ruApp remains the wake
    # path (a local wake-on-PRACH is not an actuation). A forced action (POST /cycle force=) bypasses the budget
    # - operator intent - but still counts. A clean night is exactly 2 (one SLEEP, one ACTIVE); 6 leaves room
    # for a ruApp wake + re-sleep and a retry. 0 = off. The counter lives in the rApp's in-memory PolicyState:
    # after a restart it starts at 0 for the day (at most one extra budget's worth, service-first).
    rapp_max_actuations_per_day: int = field(default_factory=lambda: _env_int("RAPP_MAX_ACTUATIONS_PER_DAY", 6))
    # RAPP_ACTUATION_ALARM_PER_DAY: the operator-visible signal below the budget - once the carrier's count for the
    # sim day reaches it the decision carries `alarm=true`, `/metrics` counts the carrier and the app logs ONE
    # WARNING per carrier-day (a second WARNING when the budget itself is reached). Same unit as the budget; 0 = off.
    # With both on, an alarm ABOVE the budget could only be reached by forced writes (the operator signal would never
    # fire before the cap), so `__post_init__` clamps it down to the budget.
    rapp_actuation_alarm_per_day: int = field(default_factory=lambda: _env_int("RAPP_ACTUATION_ALARM_PER_DAY", 4))

    # ---- Non-RT RIC platform services (R1) ----------------------------------------------
    ics_url_full: str = field(default_factory=lambda: _env("ICS_URL", "http://127.0.0.1:18083"))
    catalogue_url_full: str = field(default_factory=lambda: _env("CATALOGUE_URL", "http://127.0.0.1:18680"))

    # ---- twin ---------------------------------------------------------------------------
    twin_url: str = field(default_factory=lambda: _env("TWIN_URL", ""))
    twin_cell_id: str = field(default_factory=lambda: _env("TWIN_CELL_ID", ""))
    # -- guardrail call timing and the coverage-drop veto (HLD ADR-7) --
    twin_timeout_s: float = field(default_factory=lambda: _env_float("TWIN_TIMEOUT_S", 3.0))
    twin_coverage_drop_max: float = field(default_factory=lambda: _env_float("TWIN_COVERAGE_DROP_MAX", -0.05))  # block SLEEP if coverage_rate delta < this
    # -- the physics twin as the fourth host process ------
    # Where the twin listens when run_demo.sh starts it (TWIN=0 disables); `twin_url` stays the
    # rApp's switch (empty = guardrail off), the scripts export TWIN_URL from host/port.
    twin_host: str = field(default_factory=lambda: _env("TWIN_HOST", "127.0.0.1"))
    twin_port: int = field(default_factory=lambda: _env_int("TWIN_PORT", 8093))
    # Empty TWIN_CELL_ID + auto-cell: the rApp discovers the cell to evaluate via GET /v1/cells (first cell).
    twin_auto_cell: bool = field(default_factory=lambda: _env_bool("TWIN_AUTO_CELL", True))
    # The public repo runs the twin on a GENERIC estate with uncalibrated 3GPP defaults (the twin's own
    # TWIN_GENERIC switch, read by the same env). The rApp only labels: it expects a rung containing
    # "generic" and says so in /health when the twin reports a different rung.
    twin_generic: bool = field(default_factory=lambda: _env_bool("TWIN_GENERIC", True))
    # The physics twin service (services/ocp-physics-digitalTwin) for the scripts.
    twin_dir: Path = field(default_factory=lambda: Path(_env("TWIN_DIR", str(REPO_ROOT / "services" / "ocp-physics-digitalTwin"))))

    def __post_init__(self) -> None:
        # Range-check the chatter-control inputs from the environment (the CLI checks its flags, the
        # service must too): the frozen dataclass is written through object.__setattr__. `dataclasses.replace`
        # re-runs this, so an out-of-range value passed by the evaluation's grid is clamped the same way.
        clamp = {
            "rapp_min_sleep_dwell_min": max(0.0, float(self.rapp_min_sleep_dwell_min)),
            "rapp_wake_confirm_windows": max(1, int(self.rapp_wake_confirm_windows)),
            "rapp_wake_margin": min(max(0.0, float(self.rapp_wake_margin)), max(0.0, float(self.rapp_wake_threshold))),
            "rapp_profile_scale": min(max(0.0, float(self.rapp_profile_scale)), 1.0),
            "rapp_post_local_wake_hold_min": max(0.0, float(self.rapp_post_local_wake_hold_min)),
            # actuation budget / alarm: whole actuations per carrier-day, never negative (0 = off)
            "rapp_max_actuations_per_day": max(0, int(self.rapp_max_actuations_per_day)),
            "rapp_actuation_alarm_per_day": max(0, int(self.rapp_actuation_alarm_per_day)),
        }
        # the alarm is the signal BELOW the budget: with both on it is clamped to the budget (alarm > cap would be
        # reachable by forced writes only and never fire before the cap on natural ones)
        cap, alarm = clamp["rapp_max_actuations_per_day"], clamp["rapp_actuation_alarm_per_day"]
        if cap > 0 and alarm > cap:
            clamp["rapp_actuation_alarm_per_day"] = cap
        for name, value in clamp.items():
            if value != getattr(self, name):
                object.__setattr__(self, name, value)

    # ---- derived URLs -------------------------------------------------------------------
    @property
    def smo_url(self) -> str:
        return _env("SMO_URL", f"http://{self.smo_host}:{self.smo_port}")

    @property
    def rapp_url(self) -> str:
        return _env("RAPP_URL", f"http://{self.rapp_host}:{self.rapp_port}")

    @property
    def ru_mgmt_url(self) -> str:
        return _env("RU_MGMT_URL", f"http://{self.ru_netconf_host}:{self.ru_mgmt_port}")

    @property
    def ru_netconf_target(self) -> str:
        """`host:port` of the O-RU's M-Plane (NETCONF over SSH) server."""
        return f"{self.ru_netconf_host}:{self.ru_netconf_port}"

    @property
    def smo_public_url(self) -> str:
        """URL at which containers (real ICS in `full`) can reach the SMO on the host."""
        default = f"http://host.docker.internal:{self.smo_port}" if self.profile == "full" else self.smo_url
        return _env("SMO_PUBLIC_URL", default)

    @property
    def rapp_public_url(self) -> str:
        """URL the R1 job's `job_result_uri` points at. Its only consumer is the SMO (a host
        process in both profiles), so the default is `rapp_url` in `full` too.
        Set `RAPP_PUBLIC_URL` for a genuinely remote SMO."""
        return _env("RAPP_PUBLIC_URL", self.rapp_url)

    @property
    def ics_url(self) -> str:
        """R1 DME endpoint: the real ICS in `full`, the SMO's compatible routes in `lite`."""
        return self.ics_url_full if self.profile == "full" else self.smo_url

    @property
    def catalogue_url(self) -> str:
        return self.catalogue_url_full if self.profile == "full" else self.smo_url

    @property
    def twin_default_url(self) -> str:
        """Where run_demo.sh's twin process listens; exported as TWIN_URL for the rApp."""
        return f"http://{self.twin_host}:{self.twin_port}"

    @property
    def twin_var_dir(self) -> Path:
        """The twin's VAR_DIR/DATA_DIR live under the (gitignored) demo var dir, never in the twin's source folder."""
        return self.var_dir / "twin"

    @property
    def twin_ingest_path(self) -> Path:
        return Path(_env("TWIN_INGEST_PATH", str(self.var_dir / "twin-ingest.jsonl")))

    @property
    def rapp_model_path(self) -> Path:
        """Persisted es-rApp classifier (joblib). Stays in the gitignored `var/` (HLD ADR-6)."""
        return Path(_env("RAPP_MODEL_PATH", str(self.var_dir / "es-rapp-model.joblib")))

    @property
    def rapp_model_metrics_path(self) -> Path:
        return self.rapp_model_path.with_suffix(".metrics.json")

    @property
    def rapp_wake_threshold_effective(self) -> float:
        """The pre-emptive wake threshold the SLEEP branch compares against: wake threshold minus
        the hysteresis margin, never below 0 (equals `rapp_wake_threshold` while the margin is 0;
        the margin itself is clamped to [0, wake threshold] in `__post_init__`)."""
        return max(0.0, float(self.rapp_wake_threshold) - float(self.rapp_wake_margin))


settings = Settings()


def display_path(path: Path | str, var_dir: Optional[Path] = None) -> str:
    """Path as it may appear in shared outputs (logs, evidence, API stats): never the absolute
    checkout path (which carries the OS account name). Relative to the demo root when inside
    it, `$VAR_DIR/<rel>` when inside an overridden var dir, else the file name only."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(DEMO_ROOT))
    except ValueError:
        pass
    if var_dir is not None:
        try:
            return "$VAR_DIR/" + str(p.resolve().relative_to(Path(var_dir).resolve()))
        except ValueError:
            pass
    return p.name
