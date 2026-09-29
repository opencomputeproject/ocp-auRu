# NONRTRIC platform services for the `full` profile

The `full` profile of the demo runs the O-RAN SC **NONRTRIC** platform services in Docker
(HLD ADR-1). The es-rApp then talks R1 to genuine Non-RT RIC components. In the `lite` profile
the SMO process serves the same REST subset itself. The rApp code is the same in both
profiles. Only `settings.ics_url` and `settings.catalogue_url` change.

```text
docker compose -p ocp-auru -f demo/nonrtric/docker-compose.yaml up -d                 # ICS + rApp Catalogue
docker compose -p ocp-auru -f demo/nonrtric/docker-compose.yaml --profile a1 up -d    # + A1 PMS + A1 simulator (optional)
docker compose -p ocp-auru -f demo/nonrtric/docker-compose.yaml --profile a1 down     # stop everything (demo/scripts/stop_demo.sh --down does this)
```

The compose project name is `ocp-auru` (top-level `name:` in the file), containers are
prefixed `ocp-auru-`, and everything is on its own bridge network `ocp-auru-net`, so it coexists
with other NONRTRIC stacks on the same host.

## Services

| Service | Container | Image (tag) | Host port -> container | Role in the demo |
|---|---|---|---|---|
| `ics` | `ocp-auru-ics` | `nexus3.o-ran-sc.org:10002/o-ran-sc/nonrtric-information-coordinator-service:1.2.1` | **18083** -> 8083 | **R1 data management and exposure.** The SMO registers info type `ocp-auru-ru-pm` and producer `ocp-auru-smo-pm` (`/data-producer/v1`). The es-rApp creates info job `es-rapp-pm` (`/data-consumer/v1`). ICS pushes the job to the SMO's producer callback. The SMO then delivers every PM report to the job's `job_result_uri`. |
| `rapp-catalogue` | `ocp-auru-rapp-catalogue` | `nexus3.o-ran-sc.org:10002/o-ran-sc/nonrtric-r-app-catalogue:1.0.2` | **18680** -> 8680 | **R1 service management.** The es-rApp registers itself with `PUT /services/es-rapp`. |
| `a1pms` (profile `a1`) | `ocp-auru-a1pms` | `nexus3.o-ran-sc.org:10002/o-ran-sc/nonrtric-a1-policy-management-service:2.3.1` | **18081** -> 8081 | A1 Policy Management Service. **Not on the actuation path** (HLD ADR-2: there is no Near-RT RIC in a carrier sleep loop). It shows the platform is complete. |
| `a1-sim-STD-v2` (profile `a1`) | `ocp-auru-a1-sim-STD-v2` | `nexus3.o-ran-sc.org:10002/o-ran-sc/a1-simulator:2.2.0` | **18085** -> 8085 | A1 simulator (STD 2.0.0) standing in for a Near-RT RIC for the PMS. |

Image tags are the ones upstream's `docker-compose/.env` on `master` is tested with.
`.env.example` carries the same tags and host ports. Copy it to `.env` to change them. The
compose file inlines identical defaults, so it works without `.env`. The images are
`linux/amd64`. On Apple silicon Docker Desktop runs them emulated (`platform:` is set
explicitly in the compose file).

## How the demo uses ICS

Registration chain in `full` (all automatic, retried until the containers are up):

1. SMO to ICS: `PUT /data-producer/v1/info-types/ocp-auru-ru-pm` and
   `PUT /data-producer/v1/info-producers/ocp-auru-smo-pm` with
   `info_job_callback_url = http://host.docker.internal:8103/r1/producer/job` (the SMO on the host).
2. rApp to ICS: `PUT /data-consumer/v1/info-jobs/es-rapp-pm` (`info_type_id=ocp-auru-ru-pm`,
   `job_result_uri=http://127.0.0.1:8101/r1/pm-delivery`).
3. ICS to SMO: `POST http://host.docker.internal:8103/r1/producer/job` (ProducerJobInfo). The
   job lands in the SMO's registry.
4. SMO to rApp: every VES measurement is posted to the job's `job_result_uri`.

`job_result_uri` is consumed by the SMO (a host process), not by a container, so
`settings.rapp_public_url` defaults to `settings.rapp_url` (`http://127.0.0.1:8101`) in both
profiles. Set `RAPP_PUBLIC_URL` only for a remote SMO.

All four host ports are published on `BIND_IP` (default `127.0.0.1`, see `.env.example`).
ICS and the rApp Catalogue have no authentication, so they stay on loopback unless you widen
`BIND_IP` deliberately.

## Checks

```bash
curl -s http://127.0.0.1:18083/status                                  # ICS: "hunky dory"
curl -s http://127.0.0.1:18083/data-producer/v1/info-producers         # ["ocp-auru-smo-pm"]
curl -s http://127.0.0.1:18083/data-producer/v1/info-producers/ocp-auru-smo-pm/status   # {"operational_state":"ENABLED"}
curl -s http://127.0.0.1:18083/data-consumer/v1/info-jobs              # ["es-rapp-pm"]
curl -s http://127.0.0.1:18083/data-consumer/v1/info-jobs/es-rapp-pm/status         # {"info_job_status":"ENABLED","producers":["ocp-auru-smo-pm"]}
curl -s http://127.0.0.1:18680/services                                # [{"name":"es-rapp",...}]
curl -s http://127.0.0.1:18081/a1-policy/v2/rics                       # profile a1 only
docker logs ocp-auru-ics --tail 50
```

`demo/scripts/status.sh` prints the same probes together with the demo processes.

## A1 PMS configuration (`policy-service/config/`)

- `application-policyagent.yaml`: upstream PMS application config (http 8081, https 8433). It
  is a third-party file and keeps its own license header, see `NOTICE`.
- `application_configuration.json`: one Near-RT RIC entry, `ric1`, with `baseUrl`
  `http://ocp-auru-a1-sim-STD-v2:8085/` (the A1 simulator over plain HTTP, `ALLOW_HTTP=true`).
  The managed element ids are generic placeholders.
- Known limitation on arm64 hosts (Apple silicon): the `a1-simulator:2.2.0` image's nginx
  front fails under amd64 emulation (`failed to initialize Lua VM`), so the simulator never
  answers on 8085 and PMS reports `ric1` as `UNAVAILABLE`. PMS itself is fine. On an amd64
  host the simulator is expected to work. The profile is optional either way.

## Troubleshooting

- `run_demo.sh full` waits up to 90 s for `:18083/status` and `:18680/services`. The JVMs take
  about 10 s natively and longer under emulation. If a wait fails, the script prints the
  container log tail.
- Ports 18083, 18680, 18081 or 18085 in use: change `*_HOST_PORT` in `.env` and export
  `ICS_URL` and `CATALOGUE_URL` accordingly for the demo processes.
- ICS says the job is `DISABLED` or the producer is missing: the SMO could not be reached from
  the container. Check `docker logs ocp-auru-ics` and `curl http://127.0.0.1:8103/health` (the
  `r1.registration.state` must be `registered`). On Linux, `host.docker.internal` is provided
  through `extra_hosts: host-gateway` in the compose file. The SMO must then listen on an
  address the bridge can reach. Prefer the bridge gateway address, because the SMO has no
  authentication and `POST /o1/cm/carrier` actuates carriers:
  `GW=$(docker network inspect ocp-auru-net -f '{{(index .IPAM.Config 0).Gateway}}')`, then
  `SMO_HOST=$GW SMO_PUBLIC_URL=http://$GW:8103 demo/scripts/run_demo.sh full`. `SMO_HOST=0.0.0.0`
  is a last resort on a firewalled host. On Docker Desktop (macOS, Windows) the default
  `127.0.0.1` binding works.
