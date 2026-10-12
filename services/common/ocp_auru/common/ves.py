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
"""Builders for VES-shaped events the RU pushes north to the SMO (hybrid M-Plane north-bound,
the `o-ran-ves-subscribed-notifications` pattern; not O1, which terminates on CU/DU/RIC).

Shape note: the `commonEventHeader` follows VES 7.2.1; the `measurementFields` /
`notificationFields` bodies are project-specific (a flat list of `CarrierPm` objects in
`additionalObjects`, nested values in `additionalFields`) and would not pass a strict VES
collector's schema validation. See README "What is mocked and what is real".
"""
from __future__ import annotations

import itertools
import uuid

from .events import CommonEventHeader, PmReport, VesEnvelope, VesEvent

_seq = itertools.count(1)


def measurement_event(report: PmReport) -> VesEnvelope:
    hdr = CommonEventHeader(
        domain="measurement",
        eventName="Measurement_O-RU_Carriers",
        eventId=f"meas-{uuid.uuid4().hex[:12]}",
        sourceName=report.ru_id,
        reportingEntityName=report.ru_id,
        sequence=next(_seq),
        startEpochMicrosec=report.epoch_us - int(report.interval_s * 1_000_000),
        lastEpochMicrosec=report.epoch_us,
    )
    return VesEnvelope(event=VesEvent(commonEventHeader=hdr, measurementFields={
        "measurementFieldsVersion": "4.0",
        "measurementInterval": report.interval_s,
        "simTime": report.sim_time_iso,
        # Marker carried on the wire (twin ingest, SMO views): power_w_avg/energy_wh come from the
        # illustrative power model in ocp_auru/mock_ru/radio.py, not from measurements.
        "powerModel": "illustrative",
        "additionalObjects": [c.model_dump(mode="json") for c in report.carriers],
    }))


def notification_event(ru_id: str, name: str, fields: dict, epoch_us: int) -> VesEnvelope:
    hdr = CommonEventHeader(
        domain="notification",
        eventName=name,
        eventId=f"notif-{uuid.uuid4().hex[:12]}",
        sourceName=ru_id,
        reportingEntityName=ru_id,
        sequence=next(_seq),
        startEpochMicrosec=epoch_us,
        lastEpochMicrosec=epoch_us,
    )
    return VesEnvelope(event=VesEvent(commonEventHeader=hdr, notificationFields={
        "notificationFieldsVersion": "2.0",
        "changeType": name,
        "changeIdentifier": fields.get("carrier", ru_id),
        "additionalFields": fields,
    }))
