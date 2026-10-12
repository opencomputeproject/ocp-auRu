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
"""Physics-twin guardrail before a SLEEP actuation (HLD ADR-7).

When `TWIN_URL` is set the rApp asks the RF twin to evaluate the cell switched off
(`POST $TWIN_URL/v1/evaluate`, `cell_configs=[{cell_id, on_off:false}]`) before every SLEEP
and blocks the SLEEP if the reply's `delta_vs_baseline.coverage_rate` drops below
`TWIN_COVERAGE_DROP_MAX` (default -0.05, i.e. more than 5 points of coverage lost).

Cell to evaluate: `TWIN_CELL_ID` when set; otherwise (`TWIN_AUTO_CELL=1`, default) the first
cell of `GET $TWIN_URL/v1/cells` (logged, shown in `/health.twin`); as a last resort the
carrier name (the twin refuses unknown cells with 422, which counts as an error below).

The twin is advisory on transport/parse errors: the SLEEP is allowed, the record says
`status="unreachable"` (or `"error"`) and `/health.twin.status` shows it. Every verdict is
labelled with the twin's own `rung` string — in the public repo that is the generic estate,
"simulation, generic 3GPP defaults (uncalibrated)"; the calibrated rung only exists after a
calibration on locally supplied data. The twin consumes topology/config (which cell is off), not PM: the SMO's twin
ingest JSONL is the NDT feed, this guard is the RF question.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import httpx

from ocp_auru.common.config import Settings, settings

log = logging.getLogger(__name__)

GENERIC_RUNG_MARKER = "generic"


class TwinGuard:
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        self._lock = threading.Lock()
        self.cell_id: Optional[str] = cfg.twin_cell_id or None
        self.cell_source: str = "TWIN_CELL_ID" if cfg.twin_cell_id else "unresolved"
        self.cells_seen: int = 0
        self.status: str = "disabled" if not cfg.twin_url else "unknown"   # unknown|ok|unreachable|error
        self.rung: Optional[str] = None
        self.checks = 0
        self.blocks = 0
        self.errors = 0
        self.last_verdict: Optional[dict[str, Any]] = None
        self.last_error: Optional[str] = None

    # ---- config ---------------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(self.cfg.twin_url)

    @property
    def base_url(self) -> str:
        return self.cfg.twin_url.rstrip("/")

    # ---- discovery ------------------------------------------------------------------------
    def discover_cell(self) -> Optional[str]:
        """Resolve the cell the guardrail evaluates. `TWIN_CELL_ID` wins; else the first cell
        listed by the twin (`GET /v1/cells`). Cached once found; a failure is retried on the
        next call (the twin may still be loading its estate)."""
        if not self.enabled:
            return None
        if self.cfg.twin_cell_id:
            self.cell_id, self.cell_source = self.cfg.twin_cell_id, "TWIN_CELL_ID"
            return self.cell_id
        if self.cell_id and self.cell_source.startswith("auto"):
            return self.cell_id
        if not self.cfg.twin_auto_cell:
            self.cell_source = "carrier-name (TWIN_AUTO_CELL=0)"
            return None
        url = f"{self.base_url}/v1/cells"
        try:
            r = httpx.get(url, timeout=self.cfg.twin_timeout_s)
            r.raise_for_status()
            data = r.json()
            cells = data.get("cells") if isinstance(data, dict) else data
            ids = [c.get("cell_id") if isinstance(c, dict) else c for c in (cells or [])]
            ids = [str(i) for i in ids if i]
            if not ids:
                raise ValueError("twin listed no cells")
            with self._lock:
                self.cell_id, self.cell_source, self.cells_seen = ids[0], "auto:/v1/cells[0]", len(ids)
                self.status = "ok" if self.status in ("unknown", "unreachable") else self.status
            log.info("twin guardrail: evaluating cell %s (first of %d from %s)", ids[0], len(ids), url)
            return self.cell_id
        except Exception as exc:  # noqa: BLE001 - advisory: fall back, retry next time
            with self._lock:
                self.last_error = f"cell discovery {url}: {exc}"
                if self.status == "unknown":
                    self.status = "unreachable" if isinstance(exc, httpx.TransportError) else "error"
            log.warning("twin guardrail: cell discovery failed (%s); will use the carrier name until the twin lists cells", exc)
            return None

    def probe(self) -> dict[str, Any]:
        """`GET /health` on the twin (its rung + whether an estate is loaded) and cell discovery.
        Called once in the background at rApp start and lazily before an evaluation."""
        if not self.enabled:
            return {"enabled": False}
        url = f"{self.base_url}/health"
        try:
            r = httpx.get(url, timeout=self.cfg.twin_timeout_s)
            r.raise_for_status()
            h = r.json()
            with self._lock:
                # The rung is the rung of a MODEL. A twin that reports `model: null` (nothing loaded)
                # has no rung to adopt, whatever string it puts in the field; older builds without
                # the `model` key are taken at their word.
                if h.get("rung") and h.get("model", "unknown") is not None:
                    self.rung = h["rung"]
                self.status = "ok"
                self.last_error = None
            self.discover_cell()
            return h
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.status = "unreachable" if isinstance(exc, httpx.TransportError) else "error"
                self.last_error = f"{url}: {exc}"
            log.warning("twin guardrail: %s unreachable (%s); SLEEP decisions stay allowed, logged", url, exc)
            return {"enabled": True, "error": str(exc)}

    def start_background(self) -> None:
        if self.enabled:
            threading.Thread(target=self.probe, name="twin-probe", daemon=True).start()

    # ---- evaluation -----------------------------------------------------------------------
    def request_body(self, carrier: str) -> dict[str, Any]:
        cell = self.cfg.twin_cell_id or self.cell_id or carrier
        return {"cell_configs": [{"cell_id": cell, "on_off": False}]}

    def evaluate_sleep(self, carrier: str) -> tuple[bool, dict[str, Any]]:
        """`(allowed, detail)`; `allowed=False` only on a well-formed veto from the twin.
        `detail` is the verdict recorded on the decision: cell, deltas, rung, latency, status."""
        if not self.enabled:
            return True, {"enabled": False}
        if not self.cell_id and not self.cfg.twin_cell_id:
            self.discover_cell()
        body = self.request_body(carrier)
        cell = body["cell_configs"][0]["cell_id"]
        url = f"{self.base_url}/v1/evaluate"
        detail: dict[str, Any] = {"enabled": True, "url": url, "cell_id": cell, "cell_source": self.cell_source,
                                  "carrier": carrier, "threshold": self.cfg.twin_coverage_drop_max}
        t0 = time.perf_counter()
        try:
            r = httpx.post(url, json=body, timeout=self.cfg.twin_timeout_s)
            r.raise_for_status()
            data = r.json()
            delta_all = data.get("delta_vs_baseline") or {}
            delta = delta_all.get("coverage_rate")
            rung = data.get("rung")
            detail.update({"latency_ms": round((time.perf_counter() - t0) * 1000.0, 3), "status": "ok", "rung": rung,
                           "delta_vs_baseline": delta_all, "coverage_rate_delta": delta,
                           "guardrail_kpis": data.get("guardrail_kpis"), "baseline_kpis": data.get("baseline_kpis"),
                           "serving_share": data.get("serving_share")})
            blocked = delta is not None and float(delta) < self.cfg.twin_coverage_drop_max
            detail["blocked"] = blocked
            detail["verdict"] = "blocked" if blocked else "allowed"
            if delta is None:
                detail["note"] = "reply carried no delta_vs_baseline.coverage_rate; allowed (advisory)"
            if blocked:
                log.warning("twin vetoed SLEEP for %s (cell %s): coverage_rate delta %.3f < %.3f [%s]", carrier, cell, float(delta),
                            self.cfg.twin_coverage_drop_max, rung)
            else:
                log.info("twin allowed SLEEP for %s (cell %s): delta_vs_baseline=%s [%s] in %.1f ms", carrier, cell, delta_all, rung,
                         detail["latency_ms"])
            self._commit(detail, rung, status="ok")
            return (not blocked), detail
        except Exception as exc:  # noqa: BLE001 - advisory: allow, but say so everywhere
            status = "unreachable" if isinstance(exc, httpx.TransportError) else "error"
            detail.update({"latency_ms": round((time.perf_counter() - t0) * 1000.0, 3), "status": status, "error": str(exc),
                           "blocked": False, "verdict": "allowed (twin %s)" % status, "rung": None})
            log.warning("twin guardrail %s (%s); allowing SLEEP for %s (advisory)", status, exc, carrier)
            self._commit(detail, None, status=status)
            return True, detail

    def _commit(self, detail: dict[str, Any], rung: Optional[str], *, status: str) -> None:
        with self._lock:
            self.checks += 1
            if detail.get("blocked"):
                self.blocks += 1
            if status != "ok":
                self.errors += 1
                self.last_error = detail.get("error")
            else:
                self.last_error = None
            if rung:
                self.rung = rung
            self.status = status
            self.last_verdict = {k: detail.get(k) for k in ("verdict", "blocked", "cell_id", "delta_vs_baseline", "rung", "latency_ms", "status", "error")}

    # ---- views ----------------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        """`/health.twin`: switch, URL, reachability, the cell under evaluation and the rung."""
        if not self.enabled:
            return {"enabled": False, "status": "disabled", "note": "TWIN_URL empty: no RF guardrail; the SMO's twin ingest JSONL still runs"}
        with self._lock:
            out = {"enabled": True, "url": self.cfg.twin_url, "status": self.status, "cell_id": self.cell_id,
                   "cell_source": self.cell_source, "cells_listed": self.cells_seen, "rung": self.rung,
                   "expected_rung": "generic 3GPP defaults (uncalibrated)" if self.cfg.twin_generic else "calibrated (locally supplied data)",
                   "coverage_drop_max": self.cfg.twin_coverage_drop_max, "checks": self.checks, "blocks": self.blocks,
                   "errors": self.errors, "last_verdict": self.last_verdict, "last_error": self.last_error,
                   "consumes": "topology/config (which cell is off) - not PM; the SMO twin-ingest JSONL is the NDT feed"}
        if self.rung:
            is_generic = GENERIC_RUNG_MARKER in self.rung.lower()
            if self.cfg.twin_generic and not is_generic:
                out["rung_warning"] = f"twin reports rung {self.rung!r} but TWIN_GENERIC expects a generic (uncalibrated) rung"
            elif not self.cfg.twin_generic and is_generic:
                out["rung_warning"] = f"twin reports rung {self.rung!r} but TWIN_GENERIC=0 expects a calibrated (locally supplied data) rung"
        return out

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            return {"twin_checks": self.checks, "twin_blocks": self.blocks, "twin_errors": self.errors,
                    "twin_status": self.status, "last_twin_verdict": self.last_verdict}
