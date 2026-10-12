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
"""Helpers for the mock O-RU / ruApp tests: ephemeral-port settings so tests never touch the default demo ports."""
from __future__ import annotations

import dataclasses
import socket
from pathlib import Path

from ocp_auru.common.config import Settings


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ephemeral_settings(tmp_path: Path, **overrides) -> Settings:
    """Settings with port 0 (kernel-assigned) for the RU servers, an unused SMO port, tmp host key/var dir."""
    base = dict(
        ru_netconf_host="127.0.0.1", ru_netconf_port=0, ru_mgmt_port=0,
        ru_host_key=tmp_path / "ru-host-key", var_dir=tmp_path,
        ru_settle_ms=2.0, ru_pm_interval_s=60.0,
        smo_host="127.0.0.1", smo_port=free_port(),  # nothing listens: VES push fails fast and is ignored
    )
    base.update(overrides)
    return dataclasses.replace(Settings(), **base)
