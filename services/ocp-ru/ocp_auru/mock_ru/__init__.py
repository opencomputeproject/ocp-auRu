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
"""Mock O-RAN O-RU: WG4 M-Plane (NETCONF/YANG) server, radio simulation, ruApp host.

Implements HLD ADR-4: one Python process that is the O-RU's management plane
(`o-ran-uplane-conf` running datastore over NETCONF/SSH), a simulated radio (diurnal
load, PRACH arrivals, illustrative energy model), the in-process ruApp event bus, and the
VES-style PM/notification push north to the SMO.

Importing this package installs :mod:`ocp_auru.mock_ru._compat` first: the `netconf` server
library needs a paramiko module that paramiko 5 removed.
"""
from . import _compat  # noqa: F401  (paramiko 5 compatibility patch for the `netconf` library)
