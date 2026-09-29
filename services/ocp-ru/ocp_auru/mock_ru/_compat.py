# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 UGECO Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compatibility patch so the `netconf`/`sshutil` server libraries import on paramiko 5.

`sshutil.server` (a dependency of `netconf`) does `import paramiko.dsskey` at import
time. paramiko 5.0 removed DSA host keys entirely, so the import fails. The mock RU only
ever uses an RSA host key; this module registers a stub `paramiko.dsskey` whose
`DSSKey.from_private_key_file` always fails with `SSHException`, which is exactly the
branch `sshutil.server.from_private_key_file` already handles (it tries the next key
class). Nothing else in the demo touches DSS.
"""
from __future__ import annotations

import sys
import types

import paramiko

if "paramiko.dsskey" not in sys.modules:
    try:
        import paramiko.dsskey  # type: ignore[import-not-found]  # noqa: F401
    except ModuleNotFoundError:
        _stub = types.ModuleType("paramiko.dsskey")

        class DSSKey:  # pragma: no cover - never instantiated
            """Stand-in for the removed paramiko DSA key class (unsupported)."""

            @classmethod
            def from_private_key_file(cls, *_args, **_kwargs):
                raise paramiko.SSHException("DSS keys are not supported by paramiko >= 5")

            @classmethod
            def from_private_key(cls, *_args, **_kwargs):
                raise paramiko.SSHException("DSS keys are not supported by paramiko >= 5")

        _stub.DSSKey = DSSKey  # type: ignore[attr-defined]
        sys.modules["paramiko.dsskey"] = _stub
        paramiko.dsskey = _stub  # type: ignore[attr-defined]
