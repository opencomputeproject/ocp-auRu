#!/usr/bin/env bash
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

# Shared helpers for the demo scripts. Sourced, not executed.
# Every URL/port comes from ocp_auru.common.config (single source of truth); nothing is hard-coded here.

DEMO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # .../demo
REPO="$(cd "$DEMO/.." && pwd)"
PY="${OCP_AURU_PY:-$REPO/.venv/bin/python}"
export VAR_DIR="${VAR_DIR:-$DEMO/var}"
LOG_DIR="$VAR_DIR/logs"
RUN_DIR="$VAR_DIR/run"
COMPOSE_FILE="$DEMO/nonrtric/docker-compose.yaml"
COMPOSE=(docker compose -p ocp-auru -f "$COMPOSE_FILE")
PROCS=(smo rapp mock_ru)   # start order; run_demo.sh waits for each /health before the next, so the RU's
                           # simulated clock (starts at process start, 02:00 sim) begins once the rApp is ready
# RU_EXTERNAL=1: the O-RU runs on another host; nothing here starts or stops it. The SMO (M-Plane NETCONF) and
# the narrative reach the RU through RU_NETCONF_HOST / RU_MGMT_URL, the RU pushes north to SMO_URL.
if [[ "${RU_EXTERNAL:-0}" == "1" ]]; then PROCS=(smo rapp); fi
# The physics twin is the fourth host process (TWIN=1 default; TWIN=0 disables the RF guardrail). It is not an
# `ocp_auru.*` module: it runs from its own folder (settings.twin_dir) in its own venv, in GENERIC mode
# (TWIN_GENERIC=1: a generic estate, uncalibrated 3GPP defaults), with VAR_DIR/DATA_DIR under the demo var dir
# so nothing is ever written into the twin's source folder. Started before PROCS so the rApp finds it at start.
export TWIN="${TWIN:-1}"
export TWIN_GENERIC="${TWIN_GENERIC:-1}"
# The profile of the running demo (run_demo.sh records it) so status/stop work without re-exporting PROFILE.
if [[ -z "${PROFILE:-}" && -f "$RUN_DIR/profile" ]]; then export PROFILE="$(cat "$RUN_DIR/profile")"; fi

if [[ ! -x "$PY" ]]; then
  echo "ERROR: $PY not found. Set up the venv first (see README.md, Quick start: python3 -m venv .venv, then install the services editable)" >&2
  exit 2
fi

# Resolve URLs from settings into shell variables OCP_AURU_* (respects every env override).
load_settings() {
  eval "$("$PY" - <<'PYEOF'
import shlex
from ocp_auru.common.config import settings as s
vals = {
    "OCP_AURU_PROFILE": s.profile,
    "OCP_AURU_RU_MGMT_URL": s.ru_mgmt_url,
    "OCP_AURU_RU_NETCONF": s.ru_netconf_target,
    "OCP_AURU_CM_ADAPTER": s.cm_adapter,
    "OCP_AURU_SMO_URL": s.smo_url,
    "OCP_AURU_RAPP_URL": s.rapp_url,
    "OCP_AURU_ICS_URL": s.ics_url_full,
    "OCP_AURU_CATALOGUE_URL": s.catalogue_url_full,
    "OCP_AURU_RU_PM_INTERVAL_S": s.ru_pm_interval_s,
    "OCP_AURU_TWIN_INGEST_PATH": str(s.twin_ingest_path),
    "OCP_AURU_TWIN_URL": s.twin_url or s.twin_default_url,
    "OCP_AURU_TWIN_DIR": str(s.twin_dir),
    "OCP_AURU_TWIN_VAR_DIR": str(s.twin_var_dir),
    "OCP_AURU_TWIN_HOST": s.twin_host,
    "OCP_AURU_TWIN_PORT": s.twin_port,
}
for k, v in vals.items():
    print(f"{k}={shlex.quote(str(v))}")
PYEOF
)"
}

health_url_of() {
  case "$1" in
    mock_ru) echo "$OCP_AURU_RU_MGMT_URL/health" ;;
    smo) echo "$OCP_AURU_SMO_URL/health" ;;
    rapp) echo "$OCP_AURU_RAPP_URL/health" ;;
    twin) echo "$OCP_AURU_TWIN_URL/health" ;;
    *) echo "unknown process $1" >&2; return 1 ;;
  esac
}

# wait_http URL NAME [TIMEOUT_S] [LOGFILE] [GREP_PATTERN]
wait_http() {
  local url="$1" name="$2" timeout="${3:-30}" logfile="${4:-}" pattern="${5:-}"
  local t0=$SECONDS body
  while true; do
    if body="$(curl -fsS -m 2 "$url" 2>/dev/null)"; then
      if [[ -z "$pattern" ]] || grep -qE "$pattern" <<<"$body"; then
        echo "  ok    $name  $url  ($((SECONDS - t0))s)"
        return 0
      fi
    fi
    if (( SECONDS - t0 >= timeout )); then
      echo "  FAIL  $name did not become healthy at $url within ${timeout}s" >&2
      if [[ -n "$logfile" && -f "$logfile" ]]; then
        echo "  ----- tail -n 40 $logfile -----" >&2
        tail -n 40 "$logfile" >&2
        echo "  ---------------------------------" >&2
      fi
      return 1
    fi
    sleep 0.5
  done
}

pid_alive() { [[ -f "$RUN_DIR/$1.pid" ]] && kill -0 "$(cat "$RUN_DIR/$1.pid")" 2>/dev/null; }

start_proc() {   # start_proc NAME  ->  python -m ocp_auru.NAME, log to var/logs/NAME.log, pid to var/run/NAME.pid
  local name="$1"
  nohup "$PY" -m "ocp_auru.$name" >"$LOG_DIR/$name.log" 2>&1 &
  echo $! >"$RUN_DIR/$name.pid"
  echo "  started $name (pid $!) -> ${LOG_DIR#"$DEMO"/}/$name.log"
}

stop_proc() {    # SIGTERM, wait up to 10 s, then SIGKILL
  local name="$1" pid
  [[ -f "$RUN_DIR/$name.pid" ]] || return 0
  pid="$(cat "$RUN_DIR/$name.pid")"
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    local i
    for i in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
    if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || true; echo "  killed  $name (pid $pid, did not exit on SIGTERM)"; else echo "  stopped $name (pid $pid)"; fi
  else
    echo "  $name not running (stale pid $pid)"
  fi
  rm -f "$RUN_DIR/$name.pid"
}

# ---- physics twin (fourth host process) ----------------------------------------------------------
TWIN_PY_MODULES="fastapi, uvicorn, numpy, scipy, pandas"

twin_present() { [[ -f "$OCP_AURU_TWIN_DIR/app/main.py" && -f "$OCP_AURU_TWIN_DIR/requirements.txt" ]]; }

# Create the twin's own venv once (python -m venv + pip install -r requirements.txt); a venv that already
# imports the twin's dependencies is left alone. The venv is created with the interpreter that runs the demo
# ($PY), so it has the same Python version. Quiet: pip output goes to var/logs/twin-venv.log.
ensure_twin_venv() {
  local venv="${OCP_AURU_TWIN_VENV:-$OCP_AURU_TWIN_DIR/.venv}" py
  py="$venv/bin/python"
  if [[ -x "$py" ]] && "$py" -c "import $TWIN_PY_MODULES" >/dev/null 2>&1; then return 0; fi
  echo "  twin: creating ${venv#"$REPO"/} (python -m venv + pip install -r requirements.txt; first time only, ~1-2 min)"
  mkdir -p "$LOG_DIR"
  [[ -x "$py" ]] || "$PY" -m venv "$venv" >>"$LOG_DIR/twin-venv.log" 2>&1 || { echo "  FAIL  python -m venv ${venv#"$REPO"/} (see ${LOG_DIR#"$DEMO"/}/twin-venv.log)" >&2; return 1; }
  "$py" -m pip install -q --disable-pip-version-check -r "$OCP_AURU_TWIN_DIR/requirements.txt" >>"$LOG_DIR/twin-venv.log" 2>&1 \
    || { echo "  FAIL  pip install -r ${OCP_AURU_TWIN_DIR#"$REPO"/}/requirements.txt (see ${LOG_DIR#"$DEMO"/}/twin-venv.log)" >&2; return 1; }
  "$py" -c "import $TWIN_PY_MODULES" >/dev/null 2>&1 || { echo "  FAIL  twin venv still cannot import $TWIN_PY_MODULES" >&2; return 1; }
}

# twin_port_owner: pid of the process LISTENING on the twin port (empty when free or lsof is missing).
twin_port_owner() {
  command -v lsof >/dev/null 2>&1 || return 0
  lsof -nP -tiTCP:"$OCP_AURU_TWIN_PORT" -sTCP:LISTEN 2>/dev/null | head -n 1 || true
}

# twin_port_owner_is_twin PID: the listener is a physics-twin uvicorn (`-m uvicorn app.main:app`).
twin_port_owner_is_twin() { ps -o command= -p "$1" 2>/dev/null | grep -q -- "-m uvicorn app.main:app"; }

# stop_twin_orphan: a twin uvicorn holding the port with no (or a wrong) pid file - left behind by an older
# start_twin that recorded a wrapper pid - is stopped like any other twin; a foreign listener is left alone.
stop_twin_orphan() {
  local owner
  owner="$(twin_port_owner)"
  [[ -n "$owner" ]] || return 0
  if [[ -f "$RUN_DIR/twin.pid" && "$owner" == "$(cat "$RUN_DIR/twin.pid")" ]]; then return 0; fi
  if twin_port_owner_is_twin "$owner"; then
    echo "  stopping orphaned twin (pid $owner listening on :$OCP_AURU_TWIN_PORT, not in ${RUN_DIR#"$DEMO"/})"
    kill -TERM "$owner" 2>/dev/null || true
    local i
    for i in $(seq 1 40); do kill -0 "$owner" 2>/dev/null || break; sleep 0.25; done
    kill -0 "$owner" 2>/dev/null && { kill -KILL "$owner" 2>/dev/null || true; }
    echo "  stopped orphaned twin (pid $owner)"
  else
    echo "  WARN  port $OCP_AURU_TWIN_PORT is held by pid $owner, which is not a physics-twin uvicorn ($(ps -o command= -p "$owner" 2>/dev/null | head -c 80)); left alone" >&2
    return 1
  fi
}

# start_twin: uvicorn app.main:app from the twin folder (TWIN_GENERIC=1 default: generic mode), state under
# the demo var dir. The recorded pid IS the uvicorn server: the subshell `exec`s into it, so `$!` names the
# process that holds the port (a `( cd && cmd & )` form records a wrapper pid and stop_demo.sh leaves the
# server orphaned on the port). Refuses to start when the port is already held: a twin this script did not
# start must never answer for its evidence.
start_twin() {
  mkdir -p "$OCP_AURU_TWIN_VAR_DIR" "$OCP_AURU_TWIN_VAR_DIR-data" "$LOG_DIR" "$RUN_DIR"
  local owner
  owner="$(twin_port_owner)"
  if [[ -n "$owner" ]]; then
    echo "  FAIL  port $OCP_AURU_TWIN_PORT already held by pid $owner ($(ps -o command= -p "$owner" 2>/dev/null | head -c 80)); not starting a twin behind it (stop it, or set TWIN_PORT)" >&2
    return 1
  fi
  ( cd "$OCP_AURU_TWIN_DIR" && exec env TWIN_GENERIC="$TWIN_GENERIC" VAR_DIR="$OCP_AURU_TWIN_VAR_DIR" DATA_DIR="$OCP_AURU_TWIN_VAR_DIR-data" PYTHONPATH="$OCP_AURU_TWIN_DIR" \
      nohup "${OCP_AURU_TWIN_VENV:-$OCP_AURU_TWIN_DIR/.venv}/bin/python" -m uvicorn app.main:app --host "$OCP_AURU_TWIN_HOST" --port "$OCP_AURU_TWIN_PORT" ) \
      >"$LOG_DIR/twin.log" 2>&1 &
  echo $! >"$RUN_DIR/twin.pid"
  echo "  started twin (pid $!, $([[ "$TWIN_GENERIC" == 1 ]] && echo generic mode || echo "TWIN_GENERIC=$TWIN_GENERIC")) -> ${LOG_DIR#"$DEMO"/}/twin.log"
}

# twin_owned: after /health answers, the recorded pid must be alive AND be the listener on the twin port,
# and twin.log must not show a bind error - otherwise something else is answering and the run must not
# adopt it (its numbers would come from a process these scripts did not start and cannot stop).
twin_owned() {
  local pid owner
  pid="$(cat "$RUN_DIR/twin.pid" 2>/dev/null || true)"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    echo "  FAIL  twin pid ${pid:-?} is not alive although $OCP_AURU_TWIN_URL/health answers: a twin these scripts did not start holds :$OCP_AURU_TWIN_PORT" >&2
    return 1
  fi
  if grep -q "address already in use" "$LOG_DIR/twin.log" 2>/dev/null; then
    echo "  FAIL  twin pid $pid could not bind :$OCP_AURU_TWIN_PORT (address already in use, see ${LOG_DIR#"$DEMO"/}/twin.log); another process answers on the port" >&2
    return 1
  fi
  owner="$(twin_port_owner)"
  if [[ -n "$owner" && "$owner" != "$pid" ]]; then
    echo "  FAIL  :$OCP_AURU_TWIN_PORT is held by pid $owner, not the twin started by this script (pid $pid)" >&2
    return 1
  fi
  echo "  ok    twin pid $pid owns :$OCP_AURU_TWIN_PORT$([[ -z "$owner" ]] && echo ' (lsof unavailable: bind log checked only)')"
}

# twin_generic_ready: after /health answers, make sure a generic estate is loaded (auto-loaded at startup with
# TWIN_GENERIC=1; else POST /v1/load-generic). With TWIN_GENERIC=0 nothing is loaded: that knob means "not the
# generic estate", so the rApp then reports the (empty) twin honestly instead of a generic estate labelled
# as expected-calibrated. Says precisely when the twin at TWIN_DIR has no generic mode.
twin_generic_ready() {
  local h loaded
  h="$(curl -fsS -m 3 "$OCP_AURU_TWIN_URL/health" 2>/dev/null || true)"
  if grep -qE '"estate_loaded": ?true' <<<"$h"; then
    echo "  ok    twin estate loaded: $(head -c 200 <<<"$h")"
    return 0
  fi
  if [[ "$TWIN_GENERIC" != "1" ]]; then
    echo "  WARN  TWIN_GENERIC=$TWIN_GENERIC: no generic estate is loaded and none will be; the twin has no estate (load one via its API), the rApp records twin.status=error and allows SLEEP, advisory" >&2
    return 1
  fi
  echo "  twin: /health reports no estate yet -> POST $OCP_AURU_TWIN_URL/v1/load-generic"
  local code
  code="$(curl -sS -m 30 -o "$LOG_DIR/twin-load-generic.json" -w '%{http_code}' -X POST "$OCP_AURU_TWIN_URL/v1/load-generic" -H 'content-type: application/json' -d '{}' 2>/dev/null || echo 000)"
  if [[ "$code" == "200" ]]; then
    echo "  ok    twin generic estate loaded: $(head -c 200 "$LOG_DIR/twin-load-generic.json")"
    return 0
  fi
  echo "  WARN  twin at $OCP_AURU_TWIN_URL is up but has NO generic estate: POST /v1/load-generic -> HTTP $code" >&2
  echo "        (the twin at TWIN_DIR does not implement TWIN_GENERIC / /v1/load-generic;" >&2
  echo "         the rApp will record twin.status=error and allow SLEEP, advisory - nothing is faked)" >&2
  return 1
}
