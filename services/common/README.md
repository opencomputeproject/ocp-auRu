# common

Shared contracts for the OCP Autonomous AI-RU reference implementation. Every service imports
this package. It holds no process of its own.

| Module | Role |
|---|---|
| `ocp_auru.common.config` | Environment driven settings. The single source of truth for every port and URL. |
| `ocp_auru.common.events` | Typed cross-process contracts (pydantic v2): carrier state, PM, VES envelope, CM request and response. |
| `ocp_auru.common.yang` | XML builders and parsers for the M-Plane subset (`o-ran-uplane-conf`, `ietf-interfaces`, `ocp-ruapp-policy`). |
| `ocp_auru.common.ves` | Builders for the VES-shaped events the O-RU pushes north. |
| `ocp_auru.common.log` | One logging setup for every process (`OCP_AURU_LOG_LEVEL`). |

`ocp_auru` is a PEP 420 namespace package. There is no `ocp_auru/__init__.py`. Each service
contributes one subpackage.

## Install

```
pip install -e services/common
```

## Configuration

All settings are read from the environment by `Settings` in `config.py`. The service READMEs
list the knobs each process uses. These settings apply everywhere:

| Variable | Default | Meaning |
|---|---|---|
| `PROFILE` | `lite` | `lite` runs without Docker. `full` uses the O-RAN SC NONRTRIC containers. |
| `VAR_DIR` | `demo/var` | Runtime state, logs and evidence. Gitignored. |
| `OCP_AURU_LOG_LEVEL` | `INFO` | Log level of every process. |
| `OCP_AURU_REPO_ROOT` | derived from the checkout | Repository root, for installs that are not editable. |

## Changing a contract

`events.py` and `yang.py` are cross-process contracts. Change them together with section 3 of
`artifacts/design/HLD_closed-loop.md`.

## Tests

The contracts are exercised by the tests of the services that use them. Run the whole suite
from the repository root:

```
.venv/bin/python -m pytest -q
```
