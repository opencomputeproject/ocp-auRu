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

# Show the health of every demo process (and of NONRTRIC in the `full` profile).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_settings

probe() {   # probe NAME URL
  local name="$1" url="$2" body
  if body="$(curl -fsS -m 2 "$url" 2>/dev/null)"; then
    printf '  %-16s %-45s %s\n' "$name" "$url" "$(head -c 160 <<<"$body")"
  else
    printf '  %-16s %-45s DOWN\n' "$name" "$url"
  fi
}

echo "== demo status (profile=$OCP_AURU_PROFILE, VAR_DIR=$VAR_DIR)"
for name in twin "${PROCS[@]}"; do
  if pid_alive "$name"; then pidinfo="pid $(cat "$RUN_DIR/$name.pid")"; else pidinfo="no pid"; fi
  if [[ "$name" == twin ]]; then   # who actually holds the twin port (an orphan/foreign listener shows up here)
    owner="$(twin_port_owner)"
    if [[ -n "$owner" && "$pidinfo" != "pid $owner" ]]; then pidinfo="$pidinfo; :$OCP_AURU_TWIN_PORT held by pid $owner"; fi
  fi
  probe "$name ($pidinfo)" "$(health_url_of "$name")"
done
# the twin as the rApp sees it (enabled / status / cell / rung / checks / blocks) is inside the rApp's /health
if body="$(curl -fsS -m 2 "$OCP_AURU_RAPP_URL/health" 2>/dev/null)"; then
  printf '  %-16s %s\n' "rapp -> twin" "$("$PY" -c 'import json,sys; t=json.load(sys.stdin).get("twin",{}); print({k: t.get(k) for k in ("enabled","status","cell_id","rung","checks","blocks")})' <<<"$body")"
fi
if [[ "$OCP_AURU_PROFILE" == "full" ]]; then
  probe "ics" "$OCP_AURU_ICS_URL/status"
  probe "ics jobs" "$OCP_AURU_ICS_URL/data-consumer/v1/info-jobs"
  probe "rapp-catalogue" "$OCP_AURU_CATALOGUE_URL/services"
  docker ps --filter "name=ocp-auru-" --format '  container {{.Names}}: {{.Status}}' || true
fi
