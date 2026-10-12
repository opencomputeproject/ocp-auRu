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
"""Minimal in-process publish/subscribe bus (the RU's internal event fabric).

O-RAN concept: the ruApp lives *inside* the O-RU (HLD ADR-4), so the PRACH detector,
the ruApp and the M-Plane datastore talk over this bus, not over a network. Two delivery
modes:

* ``inline=True`` (default): the handler runs synchronously in the publisher's thread.
  This is what the ruApp wake path uses, so the measured wake latency contains no
  scheduler hop, queue or thread hand-off.
* ``inline=False``: the handler gets its own worker thread fed by a queue, so slow
  consumers (NETCONF notification push, VES HTTP POST) never sit on the wake path.

Handlers are looked up by the event's concrete class; subscribing with ``object`` receives
every event. Delivery follows subscription order (so a recorder subscribed first sees the
cause before the effect). Handler exceptions are logged and never propagate into the
publisher.
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

Handler = Callable[[Any], None]


@dataclass
class Subscription:
    event_type: type
    handler: Handler
    inline: bool
    name: str
    _queue: Optional["queue.Queue[Any]"] = field(default=None, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)


class Bus:
    """Thread-safe pub/sub. `publish` is safe from any thread."""

    _STOP = object()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: list[Subscription] = []  # in subscription order
        self._closed = False

    # ---- subscribe ----------------------------------------------------------------
    def subscribe(self, event_type: type, handler: Handler, *, inline: bool = True, name: str = "") -> Subscription:
        sub = Subscription(event_type, handler, inline, name or getattr(handler, "__qualname__", "handler"))
        if not inline:
            sub._queue = queue.Queue()
            sub._thread = threading.Thread(target=self._worker, args=(sub,), name=f"bus-{sub.name}", daemon=True)
            sub._thread.start()
        with self._lock:
            self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)
        if sub._queue is not None:
            sub._queue.put(self._STOP)

    # ---- publish ------------------------------------------------------------------
    def publish(self, event: Any) -> None:
        et = type(event)
        with self._lock:
            targets = [s for s in self._subs if s.event_type is et or s.event_type is object]
        for sub in targets:
            if sub.inline:
                try:
                    sub.handler(event)
                except Exception:  # noqa: BLE001 - a bad subscriber must not break the publisher
                    log.exception("bus: inline handler %s failed for %s", sub.name, type(event).__name__)
            else:
                assert sub._queue is not None
                sub._queue.put(event)

    # ---- lifecycle ----------------------------------------------------------------
    def _worker(self, sub: Subscription) -> None:
        assert sub._queue is not None
        while True:
            item = sub._queue.get()
            if item is self._STOP:
                return
            try:
                sub.handler(item)
            except Exception:  # noqa: BLE001
                log.exception("bus: worker handler %s failed for %s", sub.name, type(item).__name__)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            subs = list(self._subs)
            self._subs.clear()
        for s in subs:
            if s._queue is not None:
                s._queue.put(self._STOP)
        for s in subs:
            if s._thread is not None:
                s._thread.join(timeout=1.0)
