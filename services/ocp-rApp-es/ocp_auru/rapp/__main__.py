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
"""`python -m ocp_auru.rapp` — start the es-rApp on `settings.rapp_host:settings.rapp_port`."""
from __future__ import annotations

import uvicorn

from ocp_auru.common.config import settings
from ocp_auru.common.log import setup_logging

from .app import create_app


def main() -> None:
    log = setup_logging("es-rapp")
    log.info("starting es-rApp on http://%s:%d (catalogue=%s ics=%s smo=%s)", settings.rapp_host, settings.rapp_port,
             settings.catalogue_url, settings.ics_url, settings.smo_url)
    uvicorn.run(create_app(settings), host=settings.rapp_host, port=settings.rapp_port, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
