# ocp-physics-digitalTwin

The physics twin of the OCP Autonomous AI-RU reference implementation: an RF propagation twin
served as a standalone FastAPI service. It uses the 3GPP TR 38.901 Urban Macro (UMa) path loss
and line of sight models and the TR 38.901 section 7.3 sector antenna pattern.

In the closed loop it is the coverage guardrail. Before the energy-saving rApp puts a carrier
to `SLEEP`, it asks the twin what coverage looks like with that cell switched off.

The service has its own environment (`requirements.txt`) and does not import the `ocp_auru`
package.

## Honesty labels

Every response carries a `rung` string that says where the model parameters came from.

| Rung | Loaded by | Parameters | What the numbers mean |
|---|---|---|---|
| `simulation, generic 3GPP defaults (uncalibrated)` | `TWIN_GENERIC=1` or `POST /v1/load-generic` | Public defaults from TR 38.901 and TR 38.913 (`app/generic.py`) | Illustrative. Relative comparisons under identical assumptions (tilt A against tilt B, cell on against cell off) are meaningful. Absolute KPIs are not a statement about any network. |
| `simulation, calibrated to supplied measurement data` | `POST /v1/train` on input files you supply locally | Fitted to the RSRP and SINR quantiles of your measurement report | Exists only after you have run a calibration on your own data. |

This repository ships and runs the generic rung. With no model loaded the rung is `null`.
Quote a number only together with its rung.

## Run in generic mode

Generic mode needs no input data. It builds a fictional estate: three sites on a hexagonal
grid (500 m inter-site distance), three sectors each, one cell per band (1800 and 3500 MHz).
That gives 18 cells with ids like `g-s1-a-n78`, placed around latitude 0, longitude 0, a point
in the open ocean.

```bash
cd services/ocp-physics-digitalTwin
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
TWIN_GENERIC=1 VAR_DIR=/tmp/physics-twin-generic PYTHONPATH=$(pwd) \
  .venv/bin/python -m uvicorn app.main:app --port 8093

curl -s localhost:8093/health
curl -s localhost:8093/v1/cells | head -c 300
curl -s -X POST localhost:8093/v1/evaluate -H 'Content-Type: application/json' \
  -d '{"cell_configs":[{"cell_id":"g-s1-a-n78","on_off":false}]}'
curl -s -X POST localhost:8093/v1/tilt-sweep -H 'Content-Type: application/json' \
  -d '{"cell_id":"g-s1-a-n3","delta_range_deg":6,"step_deg":2}'
```

With Docker (no data mount, nothing persists):

```bash
docker compose --profile generic up -d --build     # http://localhost:8093
```

Without `TWIN_GENERIC=1` the service starts empty. The same estate can then be loaded on
demand, with any of the generator's options:

```bash
curl -s -X POST localhost:8093/v1/load-generic -H 'Content-Type: application/json' \
  -d '{"n_sites":7,"isd_m":500,"bands_mhz":[700,1800,3500],"h_tx_m":30,"tilt_deg":6,"beam_width_deg":65}'
```

The demo scripts start the twin in generic mode for you, see `demo/README.md`.

## Run the calibration fit on your own data

The calibration mode fits the model parameters to the quantiles of a measurement report. You
supply three files in a local `data/` folder, which is gitignored and mounted read-only:

| File (default name) | Content |
|---|---|
| `data/topology.csv` | One row per cell: `cell_id`, `cell_lat`, `cell_lon`, `cell_az_deg`, `cell_carrier_freq_mhz`, `hTx`, and optionally `beam_width_deg`, `technology`. |
| `data/config.csv` | `cell_id`, `cell_el_deg` (electrical tilt). |
| `data/mr_report.xlsx` | One row per device: `rsrp_dbm`, `sinr_db`, `sample_count` (average RSRP, average SINR, and how many reports that row stands for). Rename your export's headers to these, or override them through `MR_COLUMNS` in `app/physics/calibrate.py`. |

```bash
docker compose up -d --build          # http://localhost:8093, ./data mounted read-only
curl -X POST localhost:8093/v1/load-local -H 'Content-Type: application/json' -d '{}'
curl -X POST localhost:8093/v1/train                  # differential evolution fit, takes minutes
curl -X POST localhost:8093/v1/evaluate -H 'Content-Type: application/json' \
  -d '{"cell_configs":[{"cell_id":"<id>","cell_el_deg":6.0}]}'
```

Without Docker:

```bash
DATA_DIR=$(pwd)/data VAR_DIR=$(pwd)/var PYTHONPATH=$(pwd) \
  .venv/bin/python -m uvicorn app.main:app --port 8093
```

The fit records an acceptance gate in the model (`app/physics/calibrate.py`). Hard checks:
RSRP median within 2 dB, SINR 5th percentile within 1.5 dB, fitted EIRP inside the physical
window. Soft checks on the distribution tails are reported alongside. `POST /v1/train` accepts
`{"holdout_frac": 0.3}` to validate against a device split the fit never saw.

Your input files and the fitted state in `var/` stay on your machine. Both folders are
gitignored. Generic mode never overwrites persisted calibrated state unless
`"overwrite_calibrated": true` is passed.

## Ports and routes

Default port: `8093` (the container listens on 8090, published as 8093).

| Route | Purpose |
|---|---|
| `GET /health` | Status, whether an estate and a model are loaded, the rung. |
| `POST /v1/load-generic` | Build the fictional estate with uncalibrated defaults. Unknown keys are rejected with 422. |
| `POST /v1/upload`, `POST /v1/load-local` | Load your own topology, config and measurement report. |
| `POST /v1/train` | Calibration fit. Not available in generic mode. |
| `GET /v1/cells` | The cells of the loaded estate. |
| `POST /v1/evaluate` | KPIs for a configuration (tilt, cell on or off) and the delta against the baseline. Unknown cells are rejected with 422. |
| `POST /v1/tilt-sweep` | KPIs over a range of tilts for one cell. |
| `POST /v1/attach` | Per-point RSRP, SINR and serving cell for coordinates supplied by the caller. |

`/v1/evaluate` accepts `"aggregate": false` for the point-sample view (the physical field,
unclamped). The default `true` is the device-mean view that the calibration gate is judged on.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `TWIN_GENERIC` | unset | `1` loads the generic estate at startup. |
| `VAR_DIR` | `./var` | Persisted inputs and model (plain CSV and JSON). |
| `DATA_DIR` | `/data` | Root for the paths of `POST /v1/load-local`. |

## Generic defaults

`app/generic.py::default_params`, all inside the bounds of the calibration search:

| Parameter | Value | Reasoning |
|---|---|---|
| `p0_low`, `p0_mid` | 12.2 dBm/RE | 43 dBm macro radio over 1200 resource elements (20 MHz at 15 kHz SCS). |
| `p0_hi` | 14.8 dBm/RE | 53 dBm active antenna over about 6600 resource elements (100 MHz). |
| `sigma_extra` | 0 dB | The TR 38.901 UMa shadowing of 4 and 6 dB is already in the fields. |
| `p_in` | 0.80 | TR 38.901 Table 7.2-1: 80 percent of UMa users are indoor. |
| `ltw_low` | 12 dB | TR 38.901 Table 7.4.3-2, low-loss wall model at 1.8 GHz. |
| `ltw_hi` | 27 dB | TR 38.901 Table 7.4.3-2, high-loss wall model at 3.5 GHz. |
| `load` | 0.50 | Mid-load interferer activity. |
| `agg` | 1 | Point samples. There is no measurement report to emulate. |
| `bias_mid`, `bias_hi` | 0 dB | Plain max-RSRP serving. |
| `w_tier` | 1.0 | The wrap-around tier is fully visible. |

## Notebook

`notebooks/twin_walkthrough.ipynb` walks through the engine cell by cell on the generic rung:
estate build, propagation and antenna equations, the calibration search against targets that
the engine generates itself, a tilt sweep and a cell switch-off. Its example data in
`notebooks/data/example_*.csv` is the fictional estate.

```bash
.venv/bin/pip install -r requirements.txt -r notebooks/requirements-notebook.txt
.venv/bin/jupyter notebook notebooks/twin_walkthrough.ipynb
```

## Tests

```bash
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest tests -q
```

`tests/test_physics.py` covers the physics and the calibration parsing.
`tests/test_generic.py` covers generic mode and runs against a temporary `VAR_DIR`.

## Properties

- Deterministic: every random draw is seeded from a hash of a name, so the same inputs give
  the same outputs.
- The response vocabulary is `cell_id`, `cell_el_deg`, `on_off`, `guardrail_kpis`.
