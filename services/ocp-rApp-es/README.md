# ocp-rApp-es

The energy-saving rApp of the OCP Autonomous AI-RU reference implementation. It is the slow
loop.

1. It registers with the Non-RT RIC platform over R1 (service catalogue, data job).
2. It receives PM through the R1 data job.
3. A small supervised model forecasts whether the next window is low load.
4. The policy engine decides `SLEEP`, `ACTIVE` or no action, with rate limits and a daily
   actuation budget.
5. Before a `SLEEP` it asks the physics twin whether coverage holds with the cell switched
   off. A coverage drop beyond the threshold blocks the `SLEEP`.
6. It sends a CM request to the SMO. The rApp never talks M-Plane itself.

The training data that ships is synthetic. A CSV loader for 30 minute KPI reports is included
for local data, which stays in the gitignored `demo/data/private/`.

## Run

```
pip install -e services/common -e services/ocp-ruApp -e services/ocp-ru -e services/ocp-rApp-es
python -m ocp_auru.rapp
```

The rApp uses the power model of `ocp-ru` for its offline evaluation, which is why `ocp-ru` is
a dependency.

## Ports and routes

Default listen address: `127.0.0.1:8101`.

| Route | Purpose |
|---|---|
| `POST /r1/pm-delivery` | PM delivery from the R1 data job. |
| `POST /cycle` | Run one decision cycle now. `{"force": "SLEEP", "carrier": "..."}` forces an action. |
| `GET /policy`, `GET /decisions`, `GET /pm`, `GET /metrics` | Read views. |
| `GET /health` | Model, R1 registration and twin status. |

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `RAPP_HOST`, `RAPP_PORT` | `127.0.0.1`, `8101` | Listen address. |
| `RAPP_DATA` | `synthetic` | `synthetic` or `kpi_report_csv`. |
| `RAPP_DATA_DIR` | `demo/data/private` | Directory of CSV files for `kpi_report_csv`. |
| `RAPP_CYCLE_S`, `RAPP_AUTO_CYCLE` | `10`, `1` | Decision cycle. |
| `RAPP_SLEEP_THRESHOLD`, `RAPP_WAKE_THRESHOLD` | `0.7`, `0.3` | Probability thresholds for `SLEEP` and for the pre-emptive wake. |
| `RAPP_LOW_THRESHOLD` | empty | Low load threshold in PRB percent. Empty uses the 25th percentile of the training set. |
| `RAPP_MIN_HOLD_S` | `20` | Wall seconds between actuations per carrier. |
| `RAPP_MIN_SLEEP_DWELL_MIN` | `60` | Minimum sleep dwell in simulated minutes. |
| `RAPP_POST_LOCAL_WAKE_HOLD_MIN` | `30` | No new `SLEEP` for this long after a local wake. |
| `RAPP_PROFILE_SCALE` | `1.0` | Weight of the carrier's own load level in the forecast made while asleep. |
| `RAPP_MAX_ACTUATIONS_PER_DAY`, `RAPP_ACTUATION_ALARM_PER_DAY` | `6`, `4` | Daily actuation budget and alarm level per carrier. |
| `RAPP_ALLOW_INACTIVE` | `0` | Permit a forced `INACTIVE`. |
| `RAPP_RETRAIN` | `0` | `1` ignores the persisted model. |
| `SMO_URL` | `http://127.0.0.1:8103` | SMO base URL for the CM request. |
| `TWIN_URL` | empty | Physics twin base URL. Empty disables the guardrail. |
| `TWIN_CELL_ID`, `TWIN_AUTO_CELL` | empty, `1` | Cell to evaluate. With auto cell the first cell of `GET /v1/cells` is used. |
| `TWIN_COVERAGE_DROP_MAX` | `-0.05` | Block `SLEEP` when the coverage rate delta is below this. |
| `TWIN_TIMEOUT_S` | `3` | Timeout of the guardrail call. |

The guardrail is advisory when the twin cannot answer: the rApp records the twin status and
allows the `SLEEP`.

## Offline evaluation

`demo/scripts/eval_energy.py` replays the policy over a data set and writes a report. Energy
figures use the illustrative power model and are labelled as such. Plots need the `plots`
extra (`pip install -e "services/ocp-rApp-es[plots]"`).

## Tests

```
.venv/bin/python -m pytest -q services/ocp-rApp-es/tests
```
