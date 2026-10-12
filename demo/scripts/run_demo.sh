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

# Run the AI-RU closed-loop demo end to end (HLD §3.8).
#
#   demo/scripts/run_demo.sh lite   # no Docker: the SMO serves ICS/catalogue-compatible routes itself
#   demo/scripts/run_demo.sh full   # real O-RAN SC NONRTRIC (ICS + rApp Catalogue) via docker compose
#
# Starts the physics twin (generic mode, TWIN=0 to skip), SMO, es-rApp and mock O-RU (+ruApp)
# with nohup (pids in demo/var/run, logs in demo/var/logs), waits for every /health, then runs the
# narrative (scripts/narrative.py) and leaves evidence in demo/var/evidence/<UTC timestamp>/
# (summary.json, logs, twin-ingest.jsonl).
# Stop with demo/scripts/stop_demo.sh [--down]. Every port/URL comes from ocp_auru.common.config.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

PROFILE_ARG="${1:-lite}"
case "$PROFILE_ARG" in
  lite|full) ;;
  -h|--help) echo "usage: $0 [lite|full]"; exit 0 ;;
  *) echo "usage: $0 [lite|full]" >&2; exit 2 ;;
esac
export PROFILE="$PROFILE_ARG"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
EVID="$VAR_DIR/evidence/$TS"
mkdir -p "$LOG_DIR" "$RUN_DIR" "$EVID"
echo "$PROFILE" >"$RUN_DIR/profile"
# This run's twin ingest sink lives with its evidence (override TWIN_INGEST_PATH to change).
export TWIN_INGEST_PATH="${TWIN_INGEST_PATH:-$EVID/twin-ingest.jsonl}"
load_settings
# Paths printed to the terminal are relative to the demo dir: the output is pasted into reports.
EVID_REL="${EVID#"$DEMO"/}"
LOG_REL="${LOG_DIR#"$DEMO"/}"

echo "== OCP Autonomous AI-RU closed-loop demo  profile=$PROFILE  evidence=$EVID_REL"
echo "   RU mgmt $OCP_AURU_RU_MGMT_URL | RU M-Plane (NETCONF) $OCP_AURU_RU_NETCONF | SMO $OCP_AURU_SMO_URL (CM adapter $OCP_AURU_CM_ADAPTER) | rApp $OCP_AURU_RAPP_URL | twin $([[ $TWIN == 1 ]] && echo "$OCP_AURU_TWIN_URL ($([[ $TWIN_GENERIC == 1 ]] && echo generic || echo "TWIN_GENERIC=$TWIN_GENERIC"))" || echo disabled)"

# -- stale processes from a previous run ------------------------------------------------------
for name in twin "${PROCS[@]}"; do
  if pid_alive "$name"; then echo "  stopping stale $name from a previous run"; stop_proc "$name"; fi
done
# A twin uvicorn still holding the twin port without a pid file (older scripts recorded a wrapper pid) is
# stopped too; a foreign listener on that port stops the run - the guardrail evidence must come from a twin
# these scripts started, never from whatever happens to answer on :$OCP_AURU_TWIN_PORT.
if [[ "$TWIN" == "1" ]]; then
  rm -f "$RUN_DIR/twin.pid"
  stop_twin_orphan || { echo "  FAIL  :$OCP_AURU_TWIN_PORT is not free; stop that process or set TWIN_PORT (TWIN=0 runs without the RF guardrail)" >&2; exit 2; }
fi

# -- physics twin (fourth host process, generic mode) ------------------------------------------
# The rApp reads TWIN_URL: set it only when the twin is actually up so "twin disabled" and "twin
# unreachable" stay two different, honestly reported states.
export TWIN_URL=""
if [[ "$TWIN" == "1" ]]; then
  if ! twin_present; then
    echo "  WARN  physics twin not found at $OCP_AURU_TWIN_DIR (set TWIN_DIR); running without the RF guardrail" >&2
  elif ! ensure_twin_venv; then
    echo "  WARN  twin venv not usable; running without the RF guardrail" >&2
  else
    if [[ "$TWIN_GENERIC" == "1" ]]; then
      echo "== physics twin: generic estate, uncalibrated 3GPP defaults (VAR_DIR=${OCP_AURU_TWIN_VAR_DIR#"$DEMO"/})"
    else
      echo "== physics twin: TWIN_GENERIC=$TWIN_GENERIC - no generic estate will be loaded (VAR_DIR=${OCP_AURU_TWIN_VAR_DIR#"$DEMO"/})"
    fi
    start_twin || exit 2
    if wait_http "$OCP_AURU_TWIN_URL/health" twin 90 "$LOG_DIR/twin.log" '"status": ?"ok"'; then
      # /health answering is not proof that OUR twin answers: assert the pid we recorded owns the port.
      twin_owned || { echo "  the process answering on :$OCP_AURU_TWIN_PORT was not started by this script; refusing to record its numbers as evidence (demo/scripts/stop_demo.sh cleans up)" >&2; exit 2; }
      twin_generic_ready || true
      export TWIN_URL="$OCP_AURU_TWIN_URL"
    else
      echo "  WARN  twin did not come up; the rApp will report twin.status=unreachable (SLEEP allowed, logged)" >&2
      export TWIN_URL="$OCP_AURU_TWIN_URL"
    fi
  fi
else
  echo "== physics twin: disabled (TWIN=0) - no RF guardrail; the SMO twin-ingest JSONL still runs"
fi

# -- NONRTRIC (full) ----------------------------------------------------------------------------
if [[ "$PROFILE" == "full" ]]; then
  echo "== NONRTRIC: docker compose -p ocp-auru up -d (ICS + rApp Catalogue)"
  "${COMPOSE[@]}" up -d
  wait_http "$OCP_AURU_ICS_URL/status" "ICS (ocp-auru-ics)" 90 || { docker logs --tail 40 ocp-auru-ics >&2 || true; exit 1; }
  wait_http "$OCP_AURU_CATALOGUE_URL/services" "rApp Catalogue (ocp-auru-rapp-catalogue)" 90 || { docker logs --tail 40 ocp-auru-rapp-catalogue >&2 || true; exit 1; }
fi

# -- demo processes -----------------------------------------------------------------------------
# Each process is started and then waited for (/health) before the next one, so the mock O-RU -
# last in PROCS - starts its simulated clock only once the rApp is ready.
echo "== starting processes (logs in $LOG_REL)"
if [[ "${RU_EXTERNAL:-0}" == "1" ]]; then
  echo "  RU_EXTERNAL=1: not starting the mock O-RU; the SMO reaches the O-RU's M-Plane at $OCP_AURU_RU_NETCONF, the narrative its mgmt API at $OCP_AURU_RU_MGMT_URL"
fi
for name in "${PROCS[@]}"; do
  start_proc "$name"
  wait_http "$(health_url_of "$name")" "$name" 60 "$LOG_DIR/$name.log" '"ok": ?true' || { echo "startup failed; see $LOG_REL/*.log (demo/scripts/stop_demo.sh to clean up)" >&2; exit 1; }
done

# -- narrative ----------------------------------------------------------------------------------
echo "== narrative"
rc=0
"$PY" "$DEMO/scripts/narrative.py" --evidence-dir "$EVID" --natural-wait-s "${NARRATIVE_NATURAL_WAIT_S:-30}" --wake-source "${WAKE_SOURCE:-script}" --external-wake-wait-s "${EXTERNAL_WAKE_WAIT_S:-120}" | tee "$EVID/narrative.log" || rc=${PIPESTATUS[0]}

# -- evidence -----------------------------------------------------------------------------------
cp "$LOG_DIR"/*.log "$EVID/" 2>/dev/null || true
if [[ "$PROFILE" == "full" ]]; then
  docker logs ocp-auru-ics >"$EVID/ocp-auru-ics.log" 2>&1 || true
  docker logs ocp-auru-rapp-catalogue >"$EVID/ocp-auru-rapp-catalogue.log" 2>&1 || true
fi
echo "== evidence in $EVID_REL: $(ls "$EVID" | tr '\n' ' ')"
if (( rc != 0 )); then
  echo "== NARRATIVE FAILED (rc=$rc). Processes are still running for inspection; demo/scripts/status.sh, then demo/scripts/stop_demo.sh${PROFILE:+ }$([[ $PROFILE == full ]] && echo --down)" >&2
  exit "$rc"
fi
echo "== done. Processes keep running (demo/scripts/status.sh); stop with demo/scripts/stop_demo.sh$([[ $PROFILE == full ]] && echo ' --down')"
