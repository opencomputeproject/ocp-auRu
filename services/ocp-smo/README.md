# ocp-smo

SMO services of the OCP Autonomous AI-RU reference implementation, in one FastAPI process.

- VES-style collector for the PM reports and notifications the O-RU pushes north.
- CM routes that the rApp calls to set a carrier's state, with a pluggable south-bound adapter.
- R1 data producer. In the `lite` profile it also serves routes compatible with the
  Information Coordinator Service (ICS) and the rApp Catalogue, so no Docker is needed.
- Twin ingest sink: every PM report and notification is appended to a JSON lines file.

## The CM adapter seam

The CM routes do not know how the radio is reached. They call the interface defined in
`ocp_auru/smo/o1_adapter.py`:

| Method | Purpose |
|---|---|
| `set_carrier_state(req, include_policy)` | Write the policy envelope (`active`) of one carrier, plus the ruApp policy tunables. |
| `get_carriers()` | Read the carriers' `active` and `state`. |
| `describe()` | Adapter name and target, shown in `/health`. |

The reference adapter is `mplane-direct`. The SMO performs the NETCONF `edit-config` on the
O-RU's `o-ran-uplane-conf` itself, using `ocp_auru/smo/mplane_client.py`. This is the O-RAN
hybrid M-Plane model, in which the SMO manages the O-RU directly.

A hierarchical path, in which the SMO asks an O-DU and the O-DU runs the M-Plane session
towards its O-RU, plugs in through the same seam. Implement the `CmAdapter` interface for
that O-DU's management interface, call `o1_adapter.register("<name>", factory)`, and select it
with `CM_ADAPTER=<name>`. The CM routes, the rApp and the response contract stay unchanged.

## Run

```
pip install -e services/common -e services/ocp-smo
python -m ocp_auru.smo
```

The demo scripts start it for you, see `demo/README.md`.

## Ports and routes

Default listen address: `127.0.0.1:8103`.

| Route | Purpose |
|---|---|
| `POST /eventListener/v7` | VES-style collector. Answers 202. |
| `GET /o1/pm/latest`, `GET /o1/events?since=<seq>` | Latest PM per RU, notification ring. |
| `POST /o1/cm/carrier` | CM request from the rApp. Always HTTP 200 with `ok`, `cm_rtt_ms`, `netconf_rtt_ms`, `detail`. |
| `GET /o1/cm/carriers`, `GET /o1/cm/last` | Carrier state as read through the adapter, last CM response. |
| `GET /health`, `GET /twin/ingest/stats` | Health and ingest counters. |
| `/data-producer/v1/*`, `/data-consumer/v1/*`, `/services` | `lite` profile only: ICS and catalogue compatible routes. |
| `/r1/producer/*` | R1 producer callbacks, both profiles. |

`cm_rtt_ms` is the whole SMO-side round trip. `netconf_rtt_ms` is the `edit-config` alone.

The routes carry no authentication. Keep the SMO on loopback or behind an authenticating proxy.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SMO_HOST`, `SMO_PORT` | `127.0.0.1`, `8103` | Listen address. |
| `CM_ADAPTER` | `mplane-direct` | Name of the CM adapter. |
| `CM_RUAPP_POLICY` | `1` | `0` sends the standard `active` leaf only, for M-Plane servers without the extension module. |
| `RU_NETCONF_HOST`, `RU_NETCONF_PORT` | `127.0.0.1`, `1830` | M-Plane server of the O-RU. |
| `RU_NETCONF_USER`, `RU_NETCONF_PASSWORD` | `oru`, `oru-demo` | Demo credentials of the mock O-RU. Override for anything else. |
| `MPLANE_TIMEOUT_S` | `10` | NETCONF connect and RPC timeout. |
| `MPLANE_HOSTKEY_VERIFY` | `0` | `1` verifies the O-RU's SSH host key. Required for a real O-RU. |
| `ICS_URL`, `CATALOGUE_URL` | `http://127.0.0.1:18083`, `http://127.0.0.1:18680` | NONRTRIC services, `full` profile. |
| `TWIN_INGEST_PATH` | `$VAR_DIR/twin-ingest.jsonl` | Ingest sink file. |

## Tests

```
.venv/bin/python -m pytest -q services/ocp-smo/tests
```
