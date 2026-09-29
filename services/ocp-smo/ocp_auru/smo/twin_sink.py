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
"""Twin ingest sink (ADR-7): every PM report and notification appended as one JSON line.

Line shape: `{"ts": <ISO-8601 UTC>, "kind": "measurement"|"notification"|"heartbeat",
"ru_id": <sourceName>, "payload": <VES event>}`. The physics-twin engine is out of scope;
this file is what it would ingest (`settings.twin_ingest_path`).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ocp_auru.common.config import display_path


class TwinSink:
    def __init__(self, path: Path, var_dir: Optional[Path] = None):
        self.path = Path(path)
        # What `stats()` and logs show: relative to the demo root / VAR_DIR, never the absolute
        # checkout path (evidence directories are meant to be shared).
        self.display_path = display_path(self.path, var_dir)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lines = self._count_existing()

    def _count_existing(self) -> int:
        try:
            with self.path.open("rb") as fh:
                return sum(1 for _ in fh)
        except FileNotFoundError:
            return 0

    def append(self, kind: str, ru_id: str, payload: dict[str, Any]) -> None:
        line = json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            "kind": kind,
            "ru_id": ru_id,
            "payload": payload,
        }, separators=(",", ":"))
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self._lines += 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"path": self.display_path, "lines": self._lines}
