# ocp-ru

Mock O-RU of the OCP Autonomous AI-RU reference implementation. One Python process that holds:

- an O-RAN WG4 M-Plane server (NETCONF over SSH) on a subset of `o-ran-uplane-conf`,
- a simulated radio (diurnal load, PRACH arrivals, an illustrative power model),
- the in-process event bus that hosts the ruApp (`services/ocp-ruApp`),
- a VES-style PM and notification push north to the SMO,
- a small HTTP management API for the demo.

All radio numbers are simulated and labelled as such. The power and energy constants are
illustrative.

## M-Plane subset

| Operation | Behaviour |
|---|---|
| `get`, `get-config` | `user-plane-configuration` (tx and rx array carriers), the `ocp-ruapp-policy` container, `ietf-interfaces`. `state` appears in `get` only. |
| `edit-config` (merge) | `tx-array-carriers/active` plus the ruApp policy. Rejected with `operation-failed` inside the hold window after a local wake. |
| `create-subscription` | `tx-array-carriers-state-change` notifications. |
| `supervision-watchdog-reset` | Replies `next-update-at`. |

The served subtree is a subset and is not schema validated.

## The policy envelope

The envelope is the standard `o-ran-uplane-conf` `active` leaf. `SLEEP` permits a local wake by
the ruApp. `INACTIVE` forbids it. The extension module `yang/ocp-ruapp-policy.yang`
(namespace `urn:ocp:auru:ruapp-policy:1.0`, prefix `orp`) only carries the tunables of the local
decision.

## Run

`ocp-ru` and `ocp-ruapp` depend on each other, so install them in one command:

```
pip install -e services/common -e services/ocp-ruApp -e services/ocp-ru
python -m ocp_auru.mock_ru
```

## Ports and routes

| Port | Default | Purpose |
|---|---|---|
| NETCONF over SSH | `127.0.0.1:1830` | M-Plane server. |
| HTTP | `127.0.0.1:8102` | Management API. |

| Route | Purpose |
|---|---|
| `GET /health`, `GET /state` | Health, carrier state and simulated radio values. |
| `POST /sim/prach` | Inject PRACH energy on a carrier (the demo trigger of the fast loop). |
| `GET /events?since=<seq>` | Recorded bus events. |
| `GET /pm/latest` | Latest PM report. |

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `RU_ID` | `oru-0001` | Name of the RU in reports. |
| `RU_NETCONF_HOST`, `RU_NETCONF_PORT` | `127.0.0.1`, `1830` | M-Plane listen address. |
| `RU_NETCONF_USER`, `RU_NETCONF_PASSWORD` | `oru`, `oru-demo` | Demo credentials. |
| `RU_HOST_KEY` | `$VAR_DIR/ru-host-key` | SSH host key, generated on first start. |
| `RU_MGMT_PORT` | `8102` | Management API port. |
| `RU_CARRIERS` | `txcarrier0` | Comma separated carrier names. |
| `RU_INITIAL_ACTIVE` | `ACTIVE` | Initial value of `active`. |
| `RU_PM_INTERVAL_S` | `5` | PM reporting interval. |
| `RU_SIM_SPEEDUP` | `720` | Simulated seconds per wall second. |
| `RU_SIM_START_HOUR` | `2` | Hour of day at which the simulated clock starts. |
| `RU_SETTLE_MS` | `2` | Time a carrier stays `BUSY` after a write. |
| `RU_LOAD_PROFILE` | empty | JSON file with a day profile of DL PRB utilisation (24 or 48 values). Empty uses the generic diurnal curve. |
| `RU_SIM_PRB_SIGMA` | `3` | Noise on the simulated DL PRB utilisation, in PRB percent. |
| `SMO_URL` | `http://127.0.0.1:8103` | Where PM and notifications are pushed. |

## Tests

```
.venv/bin/python -m pytest -q services/ocp-ru/tests
```
