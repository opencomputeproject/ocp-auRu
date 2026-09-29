# Demo

Scripts that run the closed loop of the OCP Autonomous AI-RU reference implementation on one
machine, plus the tests that cross service boundaries.

| Path | Content |
|---|---|
| `scripts/run_demo.sh` | Starts the processes, runs the narrative, writes evidence. |
| `scripts/stop_demo.sh` | Stops the processes. `--down` also stops the NONRTRIC containers. |
| `scripts/status.sh` | Health of every process. |
| `scripts/lib.sh` | Shared shell helpers. Sourced by the other scripts. |
| `scripts/narrative.py` | The demo narrative, driven over the services' HTTP APIs. |
| `scripts/eval_energy.py` | Offline evaluation of the energy-saving policy. |
| `nonrtric/` | Docker Compose file for the O-RAN SC NONRTRIC services (`full` profile). |
| `tests/` | End-to-end test of the `lite` profile. |
| `var/` | Runtime state, logs and evidence. Created at run time, gitignored. |

## Run

Set up the environment first (see the Quick start in the top-level `README.md`). Then, from
the repository root:

```bash
demo/scripts/run_demo.sh lite      # no Docker
demo/scripts/run_demo.sh full      # with the NONRTRIC containers
demo/scripts/status.sh
demo/scripts/stop_demo.sh          # add --down after a full run
```

`run_demo.sh` starts four host processes and waits for the health check of each before it
starts the next:

| Order | Process | Started as | Port |
|---|---|---|---|
| 1 | Physics twin, generic mode | `uvicorn app.main:app` in `services/ocp-physics-digitalTwin`, own venv | 8093 |
| 2 | SMO | `python -m ocp_auru.smo` | 8103 |
| 3 | es-rApp | `python -m ocp_auru.rapp` | 8101 |
| 4 | Mock O-RU with the ruApp | `python -m ocp_auru.mock_ru` | NETCONF 1830, HTTP 8102 |

The twin's venv is created on the first run (`services/ocp-physics-digitalTwin/.venv`). The
twin's state is written under `demo/var/twin`, never into the twin's source folder.

## What the narrative shows

1. The carrier on the O-RU is `ACTIVE` and `READY`.
2. The rApp has a trained model, is registered over R1, and receives PM through the R1 job.
3. Slow loop: the rApp decides `SLEEP` and sends a CM request to the SMO. The SMO performs the
   NETCONF `edit-config` on the O-RU. The carrier becomes `DISABLED`. If the model does not
   decide `SLEEP` within the wait window, the script forces it and says so.
4. The verdict of the physics twin for that `SLEEP`, with its rung.
5. The management plane round trip: `cm_rtt_ms` (whole SMO-side round trip) and
   `netconf_rtt_ms` (`edit-config` only).
6. Fast loop: PRACH energy is injected and the ruApp wakes the carrier locally. The wake
   latency is printed in microseconds.
7. The wake arrives at the SMO as a notification and in PM, and at the rApp through the R1 job.
8. The R1 platform view: job list and service catalogue.
9. A summary block and `summary.json` in `demo/var/evidence/<UTC timestamp>/`.

All numbers are laptop simulation figures. The power and energy values come from an
illustrative model. The twin verdict carries the rung
`simulation, generic 3GPP defaults (uncalibrated)`.

## Switches

| Variable | Default | Meaning |
|---|---|---|
| `TWIN` | `1` | `0` runs without the physics twin. The rApp then reports the guardrail as disabled. |
| `TWIN_GENERIC` | `1` | Load the generic estate in the twin. |
| `TWIN_DIR` | `services/ocp-physics-digitalTwin` | Location of the twin service. |
| `TWIN_PORT` | `8093` | Port of the twin. |
| `RU_EXTERNAL` | `0` | `1`: the O-RU runs on another host. Set `RU_NETCONF_HOST` and `RU_MGMT_URL`. The scripts do not start or stop it. |
| `RU_LOAD_PROFILE` | empty | JSON file with a day profile of DL PRB utilisation for the mock O-RU. |
| `RU_SIM_PRB_SIGMA` | `3` | Noise on the simulated PRB utilisation. |
| `WAKE_SOURCE` | `script` | `external`: the narrative sends no PRACH trigger and waits for a wake that is triggered elsewhere. |
| `EXTERNAL_WAKE_WAIT_S` | `120` | How long the narrative waits for an external wake. |
| `NARRATIVE_NATURAL_WAIT_S` | `30` | How long the narrative waits for a natural `SLEEP` decision before it forces one. |
| `VAR_DIR` | `demo/var` | Runtime state, logs and evidence. |
| `OCP_AURU_PY` | `.venv/bin/python` | Python interpreter for the demo processes. |
| `OCP_AURU_TWIN_VENV` | `$TWIN_DIR/.venv` | Virtual environment of the twin. |

Every port and URL comes from `ocp_auru.common.config`. The service READMEs list the settings
of each process.

## Offline evaluation

```bash
.venv/bin/python demo/scripts/eval_energy.py --data-kind synthetic --days 14 --cells 3
```

The script replays the policy over a data set, with and without the ruApp, and writes a
report, a summary and plots to `demo/var/eval/<UTC timestamp>/`. `--help` lists the options.
The results are simulation results on the data set given, priced with the illustrative power
model.

## Tests

```bash
.venv/bin/python -m pytest -q demo/tests
```

`tests/test_e2e_lite.py` starts the SMO, the rApp and the mock O-RU as subprocesses on free
ports and runs the same narrative. It includes the physics twin when the twin's venv exists
and skips the twin assertions, with that reason, when it does not.
