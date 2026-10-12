# High-level design: Autonomous AI-RU closed loop

Scope: the design of the OCP Autonomous AI-RU reference implementation. It covers the
architecture, the decisions behind it (section 2), the components and their cross-process
contracts (section 3), and how the result is verified (section 4).

## 1. Objective

Show the closed loop from Non-RT RIC policy to the radio, runnable by anyone on one laptop.

```text
+--------------------------- Non-RT RIC / SMO ---------------------------+
|  es-rApp (energy saving, AI/ML) --- R1 --- platform services           |
|     |  R1 service management: rApp Catalogue    R1 data: ICS           |
|     |  R1 CM request ---> SMO CM routes ---> CM adapter                |
+-----|--------------------------------------------------^---------------+
      |  Open Fronthaul M-Plane (NETCONF/YANG)           |  PM and notifications
      |  edit-config o-ran-uplane-conf                   |  (VES style, hybrid M-Plane
      |  tx-array-carriers/active = SLEEP                |   north-bound)
      v                                                  |
   O-RU M-Plane server  ------------------------------->-+
      hosts the ruApp:  PRACH energy ---> local wake (microseconds)
```

Two loops, one demo.

- The **slow loop**: the rApp's policy sets a carrier to `SLEEP`.
- The **fast loop**: the ruApp wakes the carrier, `SLEEP` to `ACTIVE`, on PRACH energy. It
  does so locally and then reports.

The demo outputs two numbers: the local wake latency in microseconds and the management plane
round trip in milliseconds. Both are laptop simulation figures and are labelled as such.

## 2. Decisions (ADRs)

### ADR-1: Non-RT RIC platform is O-RAN SC NONRTRIC (Docker Compose)

NONRTRIC is the open-source Non-RT RIC of the O-RAN Software Community. The demo runs the
subset that provides the R1 platform services without ONAP or Kubernetes:

- **rApp Catalogue** for R1 service management,
- **ICS**, the Information Coordinator Service, for R1 data management and exposure,
- optionally the **A1 Policy Management Service** and an A1 simulator. They are not on the
  actuation path, see ADR-2.

Image tags come from the upstream `docker-compose/.env` on `master` (PMS 2.3.1, ICS 1.2.1,
rApp Catalogue 1.0.2, A1 simulator 2.2.0), which upstream tests together.

Rejected alternatives: a full ONAP SMO (days of setup and no gain for what the demo shows), and
a home-grown stand-in for the RIC (the demo would then not show a real R1 platform).

**Profiles.** `full` uses the NONRTRIC containers. `lite` has the SMO process serve the same
ICS and catalogue REST subset itself, so the loop runs with no Docker at all. The rApp code is
identical in both.

### ADR-2: Interfaces, named precisely

- **R1**: rApp to the Non-RT RIC platform services. REST: catalogue, ICS, and a CM request to
  the SMO's CM routes.
- **Open Fronthaul M-Plane** (O-RAN WG4, NETCONF and YANG): the management interface of the
  O-RU. Carrier control uses the standard `o-ran-uplane-conf` model, leaf
  `tx-array-carriers/active`.
- **A1 is not on the path.** There is no Near-RT RIC in an energy-saving carrier sleep loop.
  A1 appears only if the optional compose profile is enabled, to show the platform is complete.

The PM reports and notifications that the O-RU pushes north follow the VES event shape
(`o-ran-ves-subscribed-notifications` pattern of the hybrid M-Plane model).

The HTTP routes of the SMO keep the prefix `/o1` (`/o1/cm/...`, `/o1/pm/...`). The prefix names
the SMO's management surface as the rApp sees it. It makes no statement about the south-bound
transport, which is the adapter's business (ADR-5).

### ADR-3: The policy envelope is the standard `active` leaf, plus a small ruApp module

`o-ran-uplane-conf` defines `active` with the values `INACTIVE`, `SLEEP` and `ACTIVE`. `SLEEP`
means fully configured, energy saving mode, and the server rejects all operations except a
change to `ACTIVE`. That is the envelope the ruApp needs:

- The rApp sets **`active=SLEEP`**: the ruApp may bring the carrier to `ACTIVE` on its own
  (wake on demand inside the envelope).
- The rApp sets **`active=INACTIVE`**: off. The ruApp must not wake the carrier.
- The rApp sets **`active=ACTIVE`**: normal service.

The ruApp tunables (wake on PRACH, PRACH energy threshold, minimum hold after wake, report
interval) travel in the same `edit-config` in a minimal extension module, `ocp-ruapp-policy`
(namespace `urn:ocp:auru:ruapp-policy:1.0`, prefix `orp`). The module is a candidate input to
a community contract for policy to M-Plane to ruApp.

### ADR-4: Mock O-RU and ruApp as one Python process, M-Plane server included

For a demo that anyone can reproduce on a laptop, the O-RU is a Python NETCONF over SSH server
(`netconf` library on paramiko). It implements:

- `get`, `get-config` and `edit-config` (merge) on the `o-ran-uplane-conf` running datastore,
- `create-subscription` and `<notification>` push (`tx-array-carriers-state-change`),
- `supervision-watchdog-reset`,
- VES style HTTP event push for PM and notifications.

The ruApp is an in-process agent on the O-RU's event bus. Its decision latency is therefore a
local measurement taken with a nanosecond clock and contains no network time.

### ADR-5: The SMO reaches the radio through an adapter seam; the reference is direct M-Plane

The SMO's CM routes do not know how the radio is reached. They call an interface,
`CmAdapter` in `ocp_auru/smo/o1_adapter.py`:

| Method | Purpose |
|---|---|
| `set_carrier_state(req, include_policy)` | Write `active` for one carrier, plus the ruApp policy tunables. Failures are returned as `ok=false` with a reason. |
| `get_carriers()` | Read the carriers' `active` and `state`. |
| `describe()` | Adapter name and target for `/health`. |

Adapters are registered by name and selected with the setting `CM_ADAPTER`.

The reference adapter, `mplane-direct`, follows the O-RAN hybrid M-Plane model, in which the
SMO manages the O-RU directly. The SMO itself performs the NETCONF `edit-config` on the O-RU's
`o-ran-uplane-conf` (`ocp_auru/smo/mplane_client.py`, ncclient). One NETCONF session is opened
per operation and closed afterwards.

A hierarchical path, in which the SMO asks an O-DU and the O-DU runs the M-Plane session
towards its O-RU, plugs in through the same seam. It needs an implementation of `CmAdapter`
for that O-DU's management interface, a call to `o1_adapter.register()`, and
`CM_ADAPTER=<name>`. The CM routes, the rApp and the response contract stay unchanged.

### ADR-6: Data uses a canonical KPI schema and is synthetic by default

The rApp trains on a canonical KPI table with one row per cell and 30 minute window: `ts`,
`cell_id`, `prb_util_dl_pct`, `prb_util_ul_pct`, `thr_dl_mbps`, `thr_ul_mbps`, `vol_dl_mib`,
`active_users`.

Two loaders exist:

- `synthetic`: a generic diurnal profile. This is what ships and what the demo uses.
- `kpi_report_csv`: reads 30 minute KPI report CSV files from a local directory, with the
  column map in a YAML file. The default directory is the gitignored `demo/data/private/`.

The model is a small supervised classifier (scikit-learn) that predicts whether the next
window is low load. Model files stay in the gitignored `demo/var/`.

Publication rule: no input data, site identifiers or absolute KPI values in committed files or
public logs. The CSV loader labels cells `cell-<i>` by file order and never derives a label
from a file name or from file content.

### ADR-7: Twin as RF guardrail before every SLEEP, ingest sink as a separate feed

Two different things are called "twin" in the loop. They are kept apart on purpose.

1. **The physics twin** (`services/ocp-physics-digitalTwin`, a FastAPI service with TR 38.901
   UMa propagation and section 7.3 sector antennas) is a **guardrail**.

   Before every `SLEEP` actuation, natural or forced, the rApp asks it what the estate looks
   like with the cell switched off: `POST $TWIN_URL/v1/evaluate` with
   `cell_configs=[{cell_id, on_off:false}]`. The verdict is recorded on the decision
   (`/decisions[i].twin`: `verdict`, `delta_vs_baseline`, `rung`, `cell_id`, `latency_ms`,
   `status`) and in `/metrics` (`twin_checks`, `twin_blocks`, `last_twin_verdict`).

   **Semantics.** A policy `SLEEP` is blocked when `delta_vs_baseline.coverage_rate` is below
   `TWIN_COVERAGE_DROP_MAX` (default -0.05). A forced `SLEEP` (`POST /cycle {"force":"SLEEP"}`)
   is an explicit override. It still consults the twin, a block is recorded as
   `overridden: true`, and the CM request goes out. If the twin is unreachable or answers with
   something malformed, the `SLEEP` is allowed, the record says `status: unreachable` or
   `status: error`, and `/health` of the rApp shows it. The guardrail is advisory and never
   invents a verdict.

   **Cell under evaluation.** `TWIN_CELL_ID` if set. Otherwise the rApp discovers it at start
   (`GET $TWIN_URL/v1/cells`, first cell) and shows it in `/health`. The mock O-RU's carrier
   has no coordinates, so the RF question is asked of the twin's own estate, with one cell
   standing in for the carrier.

   **Rungs, labelled in every output.** The twin has two rungs. The demo runs the twin in
   generic mode (`TWIN_GENERIC=1`): a synthetic hexagonal estate with cell ids like
   `g-s1-a-n78` and uncalibrated 3GPP default parameters, rung
   `"simulation, generic 3GPP defaults (uncalibrated)"`. The narrative, `summary.json`, the
   rApp's `/health` and every decision carry that string. The calibrated rung,
   `"simulation, calibrated to supplied measurement data"`, exists only after a user has run
   the calibration fit on input files of their own. The rApp warns (`rung_warning`) if a twin
   reports a rung other than the generic one while `TWIN_GENERIC=1`. Deltas on the generic
   rung show the mechanism, the relative change in coverage and SINR when a cell sleeps. They
   are not the numbers of any network.

   **What the physics twin consumes.** Topology and configuration (which cell is off, tilts).
   It does not consume PM. It is an RF twin and has no load model.

2. **The twin ingest sink** (SMO, `demo/var/evidence/<timestamp>/twin-ingest.jsonl`) appends
   every PM report and state change notification as one JSON line. It is the feed that a
   network digital twin with a load model would ingest. The physics twin does not read it.

Process model: `demo/scripts/run_demo.sh` starts the twin as a host process in its own venv,
with `VAR_DIR` and `DATA_DIR` under the gitignored `demo/var/`, on port 8093. `TWIN=0` disables
it. Both profiles use it. The twin's Docker setup is optional and the demo does not use it.

## 3. Components and contracts

The Python services share the import package `ocp_auru`, a namespace package split across the
service folders. One virtual environment at the repository root holds them. The physics twin
has its own. Ports are chosen to stay clear of common local stacks. Every port and URL comes
from `ocp_auru.common.config`.

| Component | Folder and package | Process and port | Role |
|---|---|---|---|
| Mock O-RU and ruApp | `services/ocp-ru` (`ocp_auru.mock_ru`), `services/ocp-ruApp` (`ocp_auru.ruapp`) | `python -m ocp_auru.mock_ru`, NETCONF over SSH **1830**, HTTP **8102** | M-Plane server, radio simulation, PRACH detector, ruApp, PM and notifications |
| SMO services | `services/ocp-smo` (`ocp_auru.smo`) | `python -m ocp_auru.smo`, HTTP **8103** | VES style collector, CM routes and adapter, M-Plane client, R1 data producer. In `lite` it also serves ICS and catalogue compatible routes |
| es-rApp | `services/ocp-rApp-es` (`ocp_auru.rapp`) | `python -m ocp_auru.rapp`, HTTP **8101** | data, model, policy engine, R1 client, twin guardrail |
| Physics twin | `services/ocp-physics-digitalTwin` | `uvicorn app.main:app` from its own venv, HTTP **8093** | RF guardrail: `GET /v1/cells`, `POST /v1/evaluate` |
| Shared contracts | `services/common` (`ocp_auru.common`) | none | settings, events, YANG helpers, VES builders |
| NONRTRIC (`full`) | `demo/nonrtric/` | Docker Compose project `ocp-auru`, ICS **18083**, catalogue **18680**, PMS **18081** (optional) | Non-RT RIC platform services |

Cross-process contracts live in `ocp_auru/common/events.py` and `ocp_auru/common/yang.py`.
Change them together with this section.

### 3.1 Mock O-RU datastore (running), `o-ran-uplane-conf` subset

Namespace `urn:o-ran:uplane-conf:1.0`.

Config: one or more `tx-array-carriers` with `name`, `absolute-frequency-center` (deprecated
in the module, kept), `center-of-channel-bandwidth`, `channel-bandwidth`, `active`, `type=NR`,
`gain`, and the module's mandatory timing leaves with illustrative constants. Matching
`rx-array-carriers`.

State (`config false`): `state` with the values `DISABLED`, `BUSY`, `READY`, and
`duplex-scheme=TDD`. Both are served in `get` only.

The subtree is a subset and is not schema validated.

Rules:

- The module's SLEEP state rule is enforced. A carrier in `SLEEP` accepts only `active=ACTIVE`
  on `tx-array-carriers`. Anything else is answered with an rpc-error.
- When `active` is written, `state` becomes `BUSY` for a short simulated settle time
  (`RU_SETTLE_MS`, default 2 ms), then `READY` (for `ACTIVE`) or `DISABLED` (for `INACTIVE` and
  `SLEEP`).
- Every change of `state` emits `tx-array-carriers-state-change` to NETCONF subscribers and a
  VES notification event.

Extension module `ocp-ruapp-policy` (`urn:ocp:auru:ruapp-policy:1.0`):
`ruapp-policy/carrier[name]` with the leaves `wake-on-prach` (boolean),
`prach-energy-threshold-db`, `min-hold-after-wake-s` and `report-interval-s` (integers). The
module file is `services/ocp-ru/yang/ocp-ruapp-policy.yang`.

### 3.2 ruApp (in the O-RU)

The ruApp subscribes to `PrachDetected(carrier, energy_db, t_ns)`.

If the carrier's `active` is `SLEEP`, the policy leaf `wake-on-prach` is true, and `energy_db`
is at or above the threshold, it sets `active=ACTIVE` locally, records
`wake_latency_ns = t_applied - t_detected`, and emits `RuAppDecision`.

It never wakes a carrier from `INACTIVE`. After a wake the O-RU enforces
`min-hold-after-wake-s` before it accepts a new `SLEEP`.

All decisions are reported in the PM counters `wake_events` and `wake_latency_us`
(`last`, `p50`, `max`), so the SMO sees every transition.

### 3.3 Radio simulation (in the O-RU)

An accelerated diurnal load (`RU_SIM_SPEEDUP`, default 720, which is one simulated hour per
5 s) drives `prb_util_dl_pct`, `rrc_conn`, and a Poisson PRACH arrival process whose rate
follows the load. `RU_LOAD_PROFILE` replaces the generic diurnal curve with a day profile from
a JSON file. `POST /sim/prach` on the management API forces a PRACH burst. This is the demo
trigger.

Power model, with illustrative constants: `ACTIVE` is 250 W + 1.5 W per percent of DL PRB
utilisation, `SLEEP` is 60 W, `INACTIVE` is 20 W. `energy_wh` is integrated per PM interval.
These are not measurements.

### 3.4 Events north-bound (O-RU to SMO), VES shaped

`POST $SMO_URL/eventListener/v7` with the body

```text
{"event": {"commonEventHeader": {"domain": "notification" | "measurement", "eventName",
                                 "sourceName": RU_ID, "startEpochMicrosec",
                                 "lastEpochMicrosec", "version": "4.1", ...},
           "notificationFields" | "measurementFields": {...}}}
```

A measurement is sent every `RU_PM_INTERVAL_S` (default 5). It carries per carrier: `name`,
`active`, `state`, `tx_power_dbm`, `prb_util_dl_pct`, `rrc_conn`, `power_w_avg`, `energy_wh`,
`prach_detections`, `wake_events`, `wake_latency_us`.

The `commonEventHeader` follows VES 7.2.1. The bodies are specific to this project and would
not pass the schema validation of a strict VES collector.

### 3.5 CM adapter and M-Plane client (in the SMO)

The adapter seam is described in ADR-5. The contract between the CM routes and an adapter:

```text
set_carrier_state(CmCarrierRequest, include_policy) -> AdapterResult
    AdapterResult = {ok: bool, netconf_rtt_ms: float | null, detail: str}
get_carriers() -> {ok: bool, carriers: [{name, active, state}], netconf: {...}}
                  or {ok: false, detail: str, carriers: []}
```

The reference adapter builds the `<config>` payload with `yang.build_carrier_edit()` and calls
the M-Plane client:

```text
MplaneClient.edit_config(config_xml) -> NetconfResult
    NetconfResult = {ok: true, rtt_ms, connect_ms, target: "host:port", operation}
MplaneClient.get_carrier_status()  -> {carriers: [...], netconf: NetconfResult}
```

`rtt_ms` is the round trip of the RPC alone. `connect_ms` is the NETCONF session setup. Any
transport, authentication or `rpc-error` failure raises `MplaneError`, which the adapter turns
into `ok=false`.

`CM_RUAPP_POLICY=0` leaves the `ocp-ruapp-policy` container out of the `edit-config`, for
M-Plane servers that do not have the extension module installed. The standard `active` leaf is
then the only content.

The demo skips SSH host key verification, because the mock O-RU generates its key on first
start. A real O-RU needs `MPLANE_HOSTKEY_VERIFY=1` and credentials per device.

### 3.6 SMO services (port 8103)

- `POST /eventListener/v7`: VES collector. Stores the latest PM and the notifications,
  forwards PM to R1 consumers (job delivery) and appends to the twin ingest sink.
- `POST /o1/cm/carrier` with `{ru_id, carrier, active, ruapp_policy, reason}`: calls the CM
  adapter and returns `CmCarrierResponse`:

  | Field | Meaning |
  |---|---|
  | `ok` | Whether the write was applied. |
  | `cm_rtt_ms` | The whole SMO-side round trip, measured at the SMO. For the reference adapter: session setup, `edit-config`, session close. |
  | `netconf_rtt_ms` | The `edit-config` alone, measured by the M-Plane client. |
  | `detail` | Target on success, reason on failure. |

  The route always answers HTTP 200. A failure is `ok=false`, so the rApp reads one shape and
  still gets the round trip time.
- `GET /o1/cm/carriers`, `GET /o1/cm/last`: carrier state as read through the adapter, last
  CM response.
- `GET /o1/pm/latest`, `GET /o1/events?since=`: inspection.
- `lite` profile only: ICS compatible `/data-producer/v1/info-types/{id}`,
  `/data-producer/v1/info-producers/{id}`, `/data-consumer/v1/info-jobs/{id}` and catalogue
  compatible `/services/{name}`. The request and response shapes are those of NONRTRIC ICS
  1.2.x and rApp Catalogue 1.0.x, so the rApp code is identical across profiles.
- `full` profile: the SMO registers info type `ocp-auru-ru-pm` and producer `ocp-auru-smo-pm`
  in ICS (`ICS_URL`) with its own job callback. Deliveries go to each job's `job_result_uri`.
- `GET /health` includes `cm_adapter`, the adapter in use and its target.

### 3.7 es-rApp (port 8101)

On start the rApp registers in the catalogue (`PUT /services/es-rapp`), creates the ICS job
`es-rapp-pm` (type `ocp-auru-ru-pm`, `job_result_uri = http://<rapp>/r1/pm-delivery`), and
loads or trains the model (`RAPP_DATA`, `RAPP_DATA_DIR`).

**Cycle.** Every `RAPP_CYCLE_S` (default 10 s) the rApp takes features from the last PM window
and computes `p_low`, the probability that the next window is low load.

- Carrier `ACTIVE`, `p_low` at or above `RAPP_SLEEP_THRESHOLD` (0.7), and the holds satisfied:
  CM request `active=SLEEP` with the ruApp policy.
- Carrier `SLEEP` and the forecast says load is coming: CM request `active=ACTIVE`, a
  pre-emptive wake.
- `INACTIVE` is never written unless `RAPP_ALLOW_INACTIVE=1`.
- Before every `SLEEP`: the twin guardrail (ADR-7). A block turns a policy `SLEEP` into no
  action. A forced `SLEEP` overrides the block, and the record says so.

**API.** `GET /health`, `GET /policy`, `GET /decisions`, `GET /pm`, `GET /metrics`,
`POST /r1/pm-delivery`, `POST /cycle`. Every decision record has the parts `decision`, `twin`
and `cm`.

**Forecast for a sleeping carrier.** A carrier in `SLEEP` reports zero PRB utilisation, so its
own PM cannot forecast the return of load. For a sleeping carrier `p_low` comes from the
hour-of-day load profile learned at training time. That profile is pooled over all training
cells. A carrier that is quieter or busier than the pool would be forecast from the pooled
level while asleep and from its own PM while awake, and the two can disagree at the edge of
the low-load window. That disagreement causes repeated sleep and wake transitions (chatter).
`RAPP_PROFILE_SCALE` (default 1.0, 0 is off) scales the profile by the carrier's own level
ratio, taken from its PM history while awake. Both estimates are recorded on every decision
(`p_low_pm_window`, `p_low_profile`, `forecast_divergence`).

**Hold mechanisms.** Five separate holds touch a carrier. Each names itself in the decision
reason.

| Mechanism | Clock | Guards | Where |
|---|---|---|---|
| `min-hold-after-wake-s` (ruApp policy, 30 s) | wall seconds on the O-RU | a new `SLEEP` or `INACTIVE` after a local wake. The O-RU rejects it | `mock_ru/datastore.py` |
| `RAPP_MIN_HOLD_S` (20 s) | wall seconds since the rApp's own last actuation | `ACTIVE` to `SLEEP` | `policy_engine.decide` |
| `RAPP_POST_LOCAL_WAKE_HOLD_MIN` (default 30, 0 is off) | simulated minutes since a local wake seen in PM | `ACTIVE` to `SLEEP` | `policy_engine.decide` |
| `RAPP_MIN_SLEEP_DWELL_MIN` (default 60, 0 is off), `RAPP_WAKE_CONFIRM_WINDOWS` (default 1, off), `RAPP_WAKE_MARGIN` (default 0, off) | simulated minutes since the rApp's own applied `SLEEP` | `SLEEP` to `ACTIVE`, pre-emptive | `policy_engine.decide` |
| `RAPP_MAX_ACTUATIONS_PER_DAY` (default 6, 0 is off), `RAPP_ACTUATION_ALARM_PER_DAY` (default 4) | applied CM writes of the rApp per carrier per simulated calendar day | any new `SLEEP` and any new pre-emptive `ACTIVE` | `policy_engine.decide` |

Restart semantics: every clock and the budget counter live in the rApp's memory. After a
restart the dwell and the hold after a local wake do not apply until the next observed
transition, and the day's count starts at 0. The result is at most one extra transition, in
favour of service.

Demo cadence: with `RU_SIM_SPEEDUP` 720, one PM interval of 5 s is one simulated hour. A dwell
of up to 120 simulated minutes is therefore satisfied by the next cycle, and the narrative
cannot show it. The offline evaluation (`demo/scripts/eval_energy.py`), which works at the
30 minute cadence of the data, is where the dwell has an effect.

### 3.8 Scripts and evidence

`demo/scripts/run_demo.sh [lite|full]` starts everything, the physics twin first. It waits for
the health checks, runs the narrative (model ready, sleep decision, twin verdict, PRACH, wake,
reports), prints the two numbers, and writes evidence to `demo/var/evidence/<timestamp>/`:
logs, `summary.json` with a `twin` block, and the twin ingest file.

`demo/scripts/stop_demo.sh` stops the processes.

Shared outputs show paths relative to the demo folder and never the absolute path of the
checkout.

## 4. Verification

1. **Unit.** Datastore merge semantics. The ruApp wakes from `SLEEP` and refuses to wake from
   `INACTIVE`. Policy engine decisions. Column mapping of the CSV loader. M-Plane client timing
   and error mapping. Adapter registry and selection.
2. **Component.** ncclient against the mock O-RU's NETCONF server. The SMO's CM routes with a
   recording adapter and with the reference adapter on a fake NETCONF client. Notifications
   received by the SMO. PM delivered to the rApp through the R1 job.
3. **End to end** (`demo/tests/test_e2e_lite.py`, `lite` profile). The SMO, the rApp and the
   mock O-RU run as subprocesses on free ports. The test asserts the `SLEEP` through the
   reference adapter, the carrier state as read back over M-Plane, the local wake, the reports,
   and the twin verdict with the generic rung.
4. **Narrative** (`lite`, then `full` with NONRTRIC). The summary shows `wake_latency_us` and
   `cm_rtt_ms`. ICS lists the job. The catalogue lists `es-rapp`.
5. **Publication check.** A scan of the tree for identifiers that must not be published
   returns nothing.

Run the tests from the repository root with `.venv/bin/python -m pytest -q`. The physics twin
has its own test run, see its README.

## 5. Out of scope

- A hierarchical CM path through an O-DU. The seam for it exists (ADR-5). No such adapter
  ships.
- An M-Plane server on radio hardware that hosts the ruApp.
- Any load model or network digital twin that would consume the ingest sink.
- A1 and a Near-RT RIC.
- Real RF.
- Energy figures beyond the illustrative model.
- Authentication, authorisation and TLS on the HTTP services. See `SECURITY.md`.
