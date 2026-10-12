# Contributing

Thank you for your interest in the OCP Autonomous AI-RU reference implementation. This is a
reference implementation. A good change makes the closed loop easier to reproduce, easier to
verify, or more faithful to the O-RAN interfaces it uses.

## License of contributions

The project is licensed under the Apache License 2.0 (`LICENSE`, `NOTICE`). Contributions are
accepted under the same license.

## Sign your commits off (DCO)

The project uses the [Developer Certificate of Origin](https://developercertificate.org/) 1.1.
There is no contributor license agreement. By signing off you state that you wrote the
contribution, or otherwise have the right to submit it under the license of the file.

```sh
git commit -s -m "Add carrier state assertion to the end-to-end test"
```

`-s` appends the trailer:

```
Signed-off-by: Your Name <your.email@example.com>
```

Use your real name and a working address. Every commit in a pull request needs the trailer.
`git rebase --signoff <base>` fixes a branch that is missing them.

## Repository layout

| Path | Content |
|---|---|
| `services/common/` | Shared settings, event contracts, YANG helpers (`ocp_auru.common`). |
| `services/ocp-rApp-es/` | Energy-saving rApp (`ocp_auru.rapp`). |
| `services/ocp-smo/` | SMO services, CM adapter seam, M-Plane client (`ocp_auru.smo`). |
| `services/ocp-ru/` | Mock O-RU (`ocp_auru.mock_ru`) and the YANG extension module. |
| `services/ocp-ruApp/` | The ruApp (`ocp_auru.ruapp`). |
| `services/ocp-physics-digitalTwin/` | Physics twin, a standalone service with its own environment. |
| `demo/` | Demo scripts, NONRTRIC compose file, end-to-end test. |
| `artifacts/design/` | Design documents. |

`ocp_auru` is a PEP 420 namespace package. Do not add an `ocp_auru/__init__.py`. Each
subpackage (`ocp_auru/common`, `ocp_auru/rapp` and so on) has its own `__init__.py`.

## Build and test

Python 3.11 or newer. From a fresh clone:

```sh
python3 -m venv .venv
.venv/bin/pip install \
  -e "services/common[dev]" \
  -e "services/ocp-ruApp[dev]" \
  -e "services/ocp-ru[dev]" \
  -e "services/ocp-smo[dev]" \
  -e "services/ocp-rApp-es[dev]"
```

For every change, run the tests from the repository root:

```sh
.venv/bin/python -m pytest -q
```

This runs the unit, component and end-to-end tests of all Python services on free ports. It
takes a few minutes. One service at a time:

```sh
.venv/bin/python -m pytest -q services/ocp-smo/tests
```

The physics twin has its own environment and test run:

```sh
cd services/ocp-physics-digitalTwin
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest tests -q
```

Run the loop itself before you state that it still works:

```sh
demo/scripts/run_demo.sh lite
demo/scripts/stop_demo.sh
```

A change that touches process startup, ports or a cross-process contract is not covered by
`pytest` alone. Run the demo and read the narrative it prints.

Test files are collected with `--import-mode=importlib` (root `pyproject.toml`). Shared test
helpers live in `services/ocp-ru/tests/_mock_ru_helpers.py` and
`services/ocp-smo/tests/_smo_helpers.py` and are imported by module name.

## License header

Every source file authored by this project carries the following header, adapted to the
comment syntax of the file: `#` for `.py`, `.sh`, `.toml`, `.yaml`, `.yml` and `Dockerfile`,
`//` for `.yang`, an HTML comment block for `.html`. A `#!` shebang stays the very first line.
In Python the header goes above the module docstring.

```
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
```

Do not add the header to:

- `.md` files (covered by `LICENSE` and `NOTICE`),
- `.json` files (no comment syntax),
- data files, notebooks, and generated files,
- any file that carries the copyright of someone else.
  `demo/nonrtric/policy-service/config/application-policyagent.yaml` comes from the O-RAN SC
  NONRTRIC project and keeps its own header unchanged.

Do not change `LICENSE`.

## Honesty rules

The project shows numbers for latency and energy saving. They are only useful when their
origin is stated.

- Label every number. Demo latencies are laptop simulation figures. The power model of the
  mock O-RU is illustrative. The physics twin reports
  `simulation, generic 3GPP defaults (uncalibrated)` on every response in generic mode. The
  rApp shows the rung it receives and never rewrites it.
- Describe only what ships: synthetic data and generic 3GPP defaults.
- Never commit input data from a network, site identifiers, cell names, coordinates of real
  sites, fitted calibration parameters, or absolute KPI or energy values. This applies to code,
  tests, fixtures, documentation and logs pasted into issues.
- No secrets, credentials or `.env` values in git.

## Interfaces and contracts

- R1 is the interface between the rApp and the Non-RT RIC platform.
- M-Plane is NETCONF and YANG to the O-RU.
- A1 is not on the actuation path.
- The policy envelope is the standard `o-ran-uplane-conf` leaf `active`. `SLEEP` permits a
  local wake by the ruApp. `INACTIVE` forbids it.

Ports and URLs have one source of truth: `services/common/ocp_auru/common/config.py`.
Cross-process payloads live in `ocp_auru/common/events.py` and `ocp_auru/common/yang.py`.
Change them together with section 3 of `artifacts/design/HLD_closed-loop.md`.

A new south-bound path for the SMO is a new adapter. Implement the `CmAdapter` interface in
`services/ocp-smo/ocp_auru/smo/o1_adapter.py`, register it, and add tests next to
`services/ocp-smo/tests/test_smo_o1_adapter.py`.

## Commits and pull requests

- Write commit messages in the imperative mood: `Add build script`.
- One logical change per commit. Explain the reason in the body when the diff does not.
- `Signed-off-by:` is the trailer this project requires.
- Keep a change scoped to its purpose. Match the style of the surrounding code.
- In the pull request description, paste the output that shows the change works: the `pytest`
  summary line and, where relevant, the summary block of the demo.
- Report partial work as partial.

## Code of conduct and security

Participation is governed by `CODE_OF_CONDUCT.md`. Report security issues privately as
described in `SECURITY.md`, and read its list of demo defaults first.
