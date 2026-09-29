Open reference implementation of the Autonomous AI-RU closed loop, from Non-RT RIC policy to radio. It contains ocp-rApp-es (energy-saving rApp, R1), ocp-physics-digitalTwin (3GPP TR 38.901-based propagation twin), ocp-ru (mock O-RU, M-plane NETCONF server), ocp-ruApp (on-RU app for PRACH wake-on-demand), ocp-smo, and supporting modules.

This repository is contributed to the OCP AI Wireless / Autonomous AI-RU program. It is the reference code for OCP Global Summit 2026 session 10120, "The Autonomous AI-RU: Closing the RIC intelligence gap to the radio through the ruApp"

## What is here

| Path | Service | Role |
|---|---|---|
| `services/ocp-rApp-es/` | `ocp-rApp-es` | Energy-saving rApp. Forecasts low load, decides the policy, registers and receives PM over R1, asks the twin before a `SLEEP`. |
| `services/ocp-smo/` | `ocp-smo` | SMO services. Collects PM and notifications from the O-RU, serves the CM routes, produces data for R1, reaches the radio through a pluggable adapter (reference: direct M-Plane). |
| `services/ocp-ru/` | `ocp-ru` | Mock O-RU. M-Plane NETCONF server on a subset of `o-ran-uplane-conf`, simulated radio, host of the ruApp. |
| `services/ocp-ruApp/` | `ocp-ruApp` | The on-RU app. Wakes a sleeping carrier locally when it detects PRACH energy. |
| `services/ocp-physics-digitalTwin/` | `ocp-physics-digitalTwin` | RF propagation twin (3GPP TR 38.901). Coverage guardrail for the rApp. Standalone FastAPI service with its own environment. |
| `services/common/` | supporting module | Shared settings, event contracts, YANG and XML helpers. |
| `demo/` | supporting module | Scripts that run the loop end to end, the NONRTRIC compose file, the end-to-end test. |
| `artifacts/design/` | documentation | High-level design: `HLD_closed-loop.md`. |

The Python services share one import package, `ocp_auru`, as a namespace package. Each service
folder has its own `pyproject.toml` and contributes one subpackage.

## Architecture

```text
 es-rApp  --R1-->  Non-RT RIC platform services (catalogue, data jobs)
    |  ^
    |  | PM through the R1 data job
    |  +--- physics twin: coverage check before every SLEEP
    v
   SMO  --M-Plane (NETCONF edit-config, o-ran-uplane-conf)-->  O-RU
    ^                                                           |  hosts the ruApp
    +------- PM and notifications (VES style) ------------------+  PRACH energy -> local wake
```

Two loops run together.

- **Slow loop.** The rApp forecasts a low-load window and sets the policy. The request travels
  rApp, R1, SMO, M-Plane NETCONF, O-RU. The carrier goes to `SLEEP`.
- **Fast loop.** The ruApp on the O-RU detects PRACH energy and wakes the carrier locally,
  `SLEEP` to `ACTIVE`, with no network hop on the path. The O-RU then reports the transition.
- **Guardrail.** Before every `SLEEP` the rApp asks the physics twin for the coverage change
  with that cell switched off. A drop beyond the threshold blocks the `SLEEP`.

The demo prints two numbers: the local wake latency in microseconds and the management plane
round trip in milliseconds.

The SMO reaches the radio through an adapter seam (`services/ocp-smo/ocp_auru/smo/o1_adapter.py`).
The reference adapter follows the O-RAN hybrid M-Plane model: the SMO performs the NETCONF
`edit-config` on the O-RU itself. A hierarchical path through an O-DU can be plugged in through
the same seam by registering another adapter and selecting it with `CM_ADAPTER`.

## Quick start

Requirements: Python 3.11 or newer, `curl`, and `lsof`. Docker is needed for the `full` profile
only.

```bash
python3 -m venv .venv
.venv/bin/pip install \
  -e "services/common[dev]" \
  -e "services/ocp-ruApp[dev]" \
  -e "services/ocp-ru[dev]" \
  -e "services/ocp-smo[dev]" \
  -e "services/ocp-rApp-es[dev]"

.venv/bin/python -m pytest -q          # a few minutes

demo/scripts/run_demo.sh lite          # the whole loop, no Docker
demo/scripts/stop_demo.sh
```

Install the services in one command as shown. `ocp-ru` and `ocp-ruapp` depend on each other.

The first demo run creates a second virtual environment for the physics twin in
`services/ocp-physics-digitalTwin/.venv`. `TWIN=0 demo/scripts/run_demo.sh lite` runs without
the twin. `demo/README.md` describes the narrative and the switches.

The physics twin has its own tests:

```bash
cd services/ocp-physics-digitalTwin
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest tests -q
```

## Interface naming

| Name | Between | In this repository |
|---|---|---|
| R1 | rApp and the Non-RT RIC platform | REST: service catalogue, data jobs, and the CM request to the SMO. |
| M-Plane | SMO and O-RU | NETCONF and YANG (O-RAN WG4), model `o-ran-uplane-conf`. |
| A1 | Non-RT RIC and Near-RT RIC | Not on the actuation path. An optional compose profile starts the A1 services to show the platform is complete. |

## The policy envelope

The envelope is the standard `o-ran-uplane-conf` leaf `tx-array-carriers/active`.

| Value | Meaning for the ruApp |
|---|---|
| `SLEEP` | Energy saving. The ruApp may wake the carrier locally. |
| `INACTIVE` | Off. The ruApp must not wake the carrier. |
| `ACTIVE` | Normal service. |

The tunables of the local decision (wake on PRACH, energy threshold, hold after wake, report
interval) travel in the same `edit-config` in a small extension module,
`services/ocp-ru/yang/ocp-ruapp-policy.yang`.

## Honesty labels

- Simulated numbers are labelled simulated. The wake latency and the round trip are laptop
  simulation figures.
- The energy constants are illustrative. The mock O-RU has a simple power model and no energy
  counters behind it.
- The physics twin runs on a generic estate with uncalibrated 3GPP defaults. Every response
  carries the rung `simulation, generic 3GPP defaults (uncalibrated)`.
- The training data that ships is synthetic.

This is a reference implementation for study and demonstration. It is not production software.
`SECURITY.md` lists the demo defaults that are insecure by design.

## Documentation

- `artifacts/design/HLD_closed-loop.md`: design, decisions and cross-process contracts.
- `demo/README.md`: how to run the loop and what it shows.
- `services/*/README.md`: purpose, ports, configuration and tests of each service.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `MAINTAINERS.md`.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
