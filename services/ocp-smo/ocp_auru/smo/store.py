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
"""O-RU north-bound PM/notification store: latest measurement per RU and a bounded ring of notifications with SMO sequence numbers."""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Optional

NOTIFICATION_RING = 1000


class EventStore:
    def __init__(self, ring: int = NOTIFICATION_RING):
        self._lock = threading.Lock()
        self._latest_pm: dict[str, dict[str, Any]] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=ring)
        self._seq = 0
        self.counts: dict[str, int] = {"measurement": 0, "notification": 0, "heartbeat": 0}

    def ingest(self, envelope: dict[str, Any]) -> tuple[str, str, Optional[int]]:
        """Store one VES envelope; returns (domain, sourceName, seq-if-notification)."""
        hdr = envelope["event"]["commonEventHeader"]
        domain, source = hdr["domain"], hdr["sourceName"]
        now = time.time()
        with self._lock:
            self.counts[domain] = self.counts.get(domain, 0) + 1
            if domain == "measurement":
                self._latest_pm[source] = {"received_ts": now, "event": envelope["event"]}
                return domain, source, None
            self._seq += 1
            self._events.append({"seq": self._seq, "received_ts": now, "event": envelope["event"]})
            return domain, source, self._seq

    def latest_pm(self, ru_id: Optional[str] = None) -> dict[str, Any]:
        with self._lock:
            if ru_id is not None:
                return {ru_id: self._latest_pm[ru_id]} if ru_id in self._latest_pm else {}
            return dict(self._latest_pm)

    def events_since(self, since: int = 0, limit: int = NOTIFICATION_RING) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self._events if e["seq"] > since][:limit]

    @property
    def last_seq(self) -> int:
        return self._seq
