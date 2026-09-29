# ocp-ruApp

The ruApp of the OCP Autonomous AI-RU reference implementation: the application hosted on the
O-RU that closes the fast loop.

It listens for PRACH energy on the O-RU's in-process event bus. When a carrier is in `SLEEP`,
the policy permits a wake, and the detected energy reaches the threshold, the ruApp sets the
carrier to `ACTIVE` locally. No network hop is on that path. The time from detection to
`ACTIVE` applied is reported as `wake_latency_us`.

## Decisions

| Decision | When |
|---|---|
| `wake` | Carrier in `SLEEP`, `wake-on-prach` true, energy at or above the threshold. |
| `ignore-inactive` | Carrier in `INACTIVE`. The envelope forbids a local wake. |
| `ignore-disabled-policy` | `wake-on-prach` is false. |
| `ignore-threshold` | Energy below `prach-energy-threshold-db`. |
| `ignore-active` | Carrier already `ACTIVE`. |

After a wake the O-RU rejects a new `SLEEP` for `min-hold-after-wake-s` seconds.

## Run

The ruApp has no process of its own. The mock O-RU hosts it:

```
pip install -e services/common -e services/ocp-ruApp -e services/ocp-ru
python -m ocp_auru.mock_ru
```

`ocp-ruapp` uses the event bus and datastore of `ocp-ru`, and `ocp-ru` starts the ruApp, so
the two packages depend on each other and are installed together.

## Ports

None. The ruApp is reached through the O-RU: the policy arrives over M-Plane, the decisions
are visible at `GET /events` and `GET /state` of the O-RU's management API.

## Configuration

The tunables come from the policy written by the rApp, per carrier:

| Leaf (`ocp-ruapp-policy`) | Default | Meaning |
|---|---|---|
| `wake-on-prach` | `true` | Permit the local wake. |
| `prach-energy-threshold-db` | `10` | Minimum PRACH energy that counts as demand. |
| `min-hold-after-wake-s` | `30` | Hold after a local wake. |
| `report-interval-s` | `5` | PM reporting interval for the carrier. |

## Tests

```
.venv/bin/python -m pytest -q services/ocp-ruApp/tests
```
