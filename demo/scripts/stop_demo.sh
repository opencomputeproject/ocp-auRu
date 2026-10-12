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

# Stop the demo processes started by run_demo.sh. `--down` also stops the NONRTRIC containers.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

DOWN=0
for arg in "$@"; do
  case "$arg" in
    --down) DOWN=1 ;;
    -h|--help) echo "usage: $0 [--down]"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

echo "== stopping demo processes"
mkdir -p "$RUN_DIR"
for name in mock_ru rapp smo twin; do stop_proc "$name"; done
# A twin still listening on the twin port after its pid was stopped (orphan of an older start_twin that
# recorded a wrapper pid) is stopped as well; the port must be free for the next run_demo.sh.
load_settings
stop_twin_orphan || true
rm -f "$RUN_DIR/profile"

if (( DOWN )); then
  echo "== docker compose -p ocp-auru down"
  "${COMPOSE[@]}" --profile a1 down --remove-orphans || echo "  (compose down failed or nothing to stop)"
fi
echo "done."
