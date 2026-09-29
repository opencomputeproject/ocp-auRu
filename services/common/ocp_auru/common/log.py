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
"""One logging setup for every process: `OCP_AURU_LOG_LEVEL`, ISO timestamps, component tag."""
from __future__ import annotations

import logging
import os
import sys


def setup_logging(component: str) -> logging.Logger:
    level = os.environ.get("OCP_AURU_LOG_LEVEL", "INFO").upper()
    fmt = f"%(asctime)s %(levelname)-5s [{component}] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stdout, force=True)
    logging.getLogger("paramiko").setLevel(logging.WARNING)
    logging.getLogger("sshutil").setLevel(logging.WARNING)
    logging.getLogger("ncclient").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger(component)
