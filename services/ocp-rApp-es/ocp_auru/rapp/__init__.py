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
"""es-rApp: energy-saving rApp on the Non-RT RIC (HLD §3.7).

The slow loop of the demo. It consumes O1 PM delivered through the R1 data-management
job (`ocp_auru.rapp.r1`), forecasts "next window is low-load" with a small supervised model
(`ocp_auru.rapp.model`), and — when the policy engine (`ocp_auru.rapp.policy_engine`) agrees —
writes the policy envelope `tx-array-carriers/active = SLEEP` (ADR-3) through an R1 CM
request to the SMO's CM route. The ruApp in the O-RU then wakes the carrier locally
on PRACH energy (the fast loop). Everything here is standard-interface glue: R1 to the
platform services, a CM request to the SMO; no direct M-Plane access from the rApp.
"""
