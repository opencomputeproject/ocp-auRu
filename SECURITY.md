# Security Policy

## Scope and supported versions

The OCP Autonomous AI-RU reference implementation is a reference implementation and a
demonstration. It is not production software. There are no released versions and therefore no
releases with security support. Fixes land on the default branch.

| Version | Security support |
|---|---|
| default branch | Best effort. Fixes land here. |
| any tag, commit or fork | None |

Treat the code as a prototype. It is not hardened and not audited. It is not intended to be
exposed to a network or connected to production equipment.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability.

Report it through GitHub's private vulnerability reporting on this repository: open the
**Security** tab and choose **Report a vulnerability**. The report stays confidential until a
fix is published.

Please include:

- what you found, and which file or endpoint it is in,
- how to reproduce it, ideally the smallest command sequence or request that shows the problem,
- what an attacker gains and under which preconditions, in particular whether the service must
  be reachable from outside `127.0.0.1`,
- a suggested fix, if you have one.

Never include input data from a network, site identifiers, credentials or absolute KPI values
in a report. Describe the shape of the data instead.

The maintainers acknowledge a report as promptly as they can, say whether they consider it in
scope, and credit the reporter when a fix lands unless asked not to. Please allow reasonable
time before public disclosure.

## Known demo defaults that are insecure by design

The demo is deliberately simple so that a reader can follow the closed loop end to end. The
following properties are known, intentional and documented. Each is scoped to loopback and
names the setting that hardens it. Reports about them are closed as working as intended. A
report that one of them is reachable in an unexpected way, for example a service that binds
beyond loopback when it should not, is in scope.

1. **The mock O-RU ships a demo NETCONF password.** `ocp_auru/common/config.py` defaults to
   `RU_NETCONF_USER=oru` and `RU_NETCONF_PASSWORD=oru-demo` for the NETCONF over SSH server of
   the mock device (port 1830). Override both for any use beyond the demo. The server binds
   `RU_NETCONF_HOST` (default `127.0.0.1`) only.

2. **The M-Plane client does not verify the SSH host key.**
   `ocp_auru/smo/mplane_client.py` connects with host key verification off
   (`MPLANE_HOSTKEY_VERIFY`, default `0`), because the mock O-RU generates its host key on
   first start. This removes the defence against a man in the middle on the M-Plane session.
   A real O-RU needs `MPLANE_HOSTKEY_VERIFY=1` and a pinned key.

3. **The local HTTP services are unauthenticated.** The management API of the mock O-RU
   (8102), the SMO (8103), the es-rApp (8101) and the physics twin (8093) have no
   authentication, no authorisation and no TLS. Several endpoints actuate the loop, for
   example `POST /sim/prach`, `POST /cycle` and `POST /o1/cm/carrier`. All bind `127.0.0.1` by
   default. The NONRTRIC containers of the `full` profile (ICS 18083, rApp Catalogue 18680,
   optional A1 PMS 18081) have no authentication either and are published on `BIND_IP`, default
   `127.0.0.1`. Widening any of these puts unauthenticated control of the demo on your network.

4. **The delivery target of an R1 job is not validated.** When a job is registered with the
   SMO, `ocp_auru/smo/dme.py` stores the target URI and posts each measurement to it as given,
   with no allow list. Anyone who can reach the SMO can make it issue outbound HTTP requests to
   an address of their choice. The demo accepts this because the platform supplies the callback
   URL. Any real deployment needs an allow list here.

Related and equally intentional: the twin guardrail fails open. If the physics twin is
unreachable or answers with something malformed, a `SLEEP` is still allowed and the decision is
marked `status=unreachable` or `status=error`. See ADR-7 in
`artifacts/design/HLD_closed-loop.md`. This is an availability choice for an advisory
guardrail and has no security function.

## What to report

Please report, privately:

- a service that listens beyond `127.0.0.1` when its configuration says it should not,
- a credential, key, token, `.env` value, site identifier or absolute KPI value that has been
  committed to the repository or written into a log, an evidence file or a test fixture,
- an absolute file system path with a user account name in shared output (the evidence
  `summary.json`, the narrative, `GET /twin/ingest/stats`). These are rendered relative to the
  demo folder on purpose, and a leak is a bug,
- a path traversal, deserialisation, template injection or command injection in the code,
- a dependency with a known vulnerability,
- anything that lets a NETCONF client bypass the policy envelope, for example a carrier in
  `INACTIVE` being woken locally, or the SLEEP state rule of the mock O-RU being evaded.
