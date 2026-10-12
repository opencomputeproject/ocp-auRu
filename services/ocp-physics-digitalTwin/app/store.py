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

"""Persist the loaded estate inputs + fitted params under VAR_DIR so a container restart keeps the
calibrated model. Plain JSON/CSV — inspectable, no pickle."""
from __future__ import annotations

import json
import os
import pathlib

import pandas as pd

VAR = pathlib.Path(os.getenv("VAR_DIR", pathlib.Path(__file__).resolve().parent.parent / "var"))
VAR.mkdir(parents=True, exist_ok=True)


def save_inputs(topology: pd.DataFrame, config: pd.DataFrame, targets: dict) -> None:
    topology.to_csv(VAR / "topology.csv", index=False)
    config.to_csv(VAR / "config.csv", index=False)
    (VAR / "targets.json").write_text(json.dumps(targets, indent=2))


def load_inputs():
    t, c, g = VAR / "topology.csv", VAR / "config.csv", VAR / "targets.json"
    if not (t.exists() and c.exists()):
        return None, None, None
    targets = json.loads(g.read_text()) if g.exists() else None
    return pd.read_csv(t), pd.read_csv(c), targets


def save_model(report: dict) -> None:
    (VAR / "model.json").write_text(json.dumps(report, indent=2))


def load_model() -> dict | None:
    p = VAR / "model.json"
    return json.loads(p.read_text()) if p.exists() else None


def delete_model() -> bool:
    """Remove the persisted model (used when a generic placeholder must not survive a switch to
    non-generic estate inputs). Returns whether a file was removed."""
    p = VAR / "model.json"
    if p.exists():
        p.unlink()
        return True
    return False
