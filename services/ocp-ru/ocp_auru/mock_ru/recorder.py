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
"""Ring buffer of RU-internal events with a monotonically increasing sequence number.

Backs `GET /events?since=` on the management API and the `POST /sim/prach` wait-for-the
ruApp-decision path. Subscribes inline to every bus event; appends are O(1) and cheap so
they do not disturb the wake latency measurement.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import BaseModel

from .bus import Bus


@dataclass(frozen=True)
class Recorded:
    seq: int
    kind: str
    event: Any  # the pydantic event instance

    def as_json(self) -> dict:
        payload = self.event.model_dump(mode="json") if isinstance(self.event, BaseModel) else dict(self.event)
        return {"seq": self.seq, "kind": self.kind, "event": payload}


class EventRecorder:
    def __init__(self, bus: Bus, maxlen: int = 500, kinds: Optional[tuple[type, ...]] = None) -> None:
        self._cond = threading.Condition()
        self._buf: deque[Recorded] = deque(maxlen=maxlen)
        self._seq = 0
        self._kinds = kinds
        bus.subscribe(object, self.record, inline=True, name="recorder")

    @property
    def seq(self) -> int:
        with self._cond:
            return self._seq

    def record(self, event: Any) -> None:
        if self._kinds is not None and not isinstance(event, self._kinds):
            return
        with self._cond:
            self._seq += 1
            self._buf.append(Recorded(self._seq, type(event).__name__, event))
            self._cond.notify_all()

    def since(self, seq: int) -> list[Recorded]:
        with self._cond:
            return [r for r in self._buf if r.seq > seq]

    def wait_for(self, predicate: Callable[[Recorded], bool], timeout: float, after_seq: int = 0) -> Optional[Recorded]:
        """Return the first recorded event with seq > after_seq matching predicate, waiting up to timeout s."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                for r in self._buf:
                    if r.seq > after_seq and predicate(r):
                        return r
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
