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
"""O-RAN WG4 M-Plane server: NETCONF over SSH on the `o-ran-uplane-conf` running datastore.

Implements the subset in HLD ADR-4 with the `netconf` library (paramiko SSH subsystem):

* `get` / `get-config` - `<data>` with `user-plane-configuration` (tx/rx array carriers,
  `state` only in `get`), the `ocp-ruapp-policy` container and `ietf-interfaces`.
  A subtree filter selects top-level containers by tag (enough for the demo clients).
* `edit-config` (merge) - `tx-array-carriers/active` + ruApp policy, or an
  `ietf-interfaces/interface/description`. A request inside the ruApp hold window, or a
  write other than `active=ACTIVE` to a carrier in SLEEP (the module's SLEEP-state rule),
  is answered with `rpc-error` / `operation-failed` and an explanatory `error-message`.
* `create-subscription` (RFC 5277) - the session receives `tx-array-carriers-state-change`
  `<notification>`s for every `state` change.
* `supervision-watchdog-reset` (`o-ran-supervision`) - replies `next-update-at`.

Binding: the `sshutil` server binds every interface; the demo binds only
`settings.ru_netconf_host` (see `_BoundNetconfSSHServer`). Host key: RSA, generated on
first start into `settings.ru_host_key`.
"""
from __future__ import annotations

import logging
import socket
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import paramiko
from lxml import etree

from . import _compat  # noqa: F401  (must precede the netconf import on paramiko 5)

import sshutil.server  # noqa: E402
from netconf import error as ncerror  # noqa: E402
from netconf import server as ncserver  # noqa: E402
from netconf import util as ncutil  # noqa: E402

from ocp_auru.common import yang  # noqa: E402
from ocp_auru.common.config import Settings, display_path  # noqa: E402
from ocp_auru.common.events import CarrierStateChanged  # noqa: E402

from .bus import Bus  # noqa: E402
from .datastore import Datastore, HoldViolation, SleepStateViolation, UnknownCarrier  # noqa: E402

log = logging.getLogger(__name__)

_BASE_CAPS = {"urn:ietf:params:netconf:base:1.0", "urn:ietf:params:netconf:base:1.1"}


def ensure_host_key(path: Path) -> Path:
    """Generate an RSA host key at `path` if missing (parents created)."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        key = paramiko.RSAKey.generate(2048)
        key.write_private_key_file(str(path))
        log.info("generated RSA host key %s", display_path(path))
    return path


class RuNetconfMethods(ncserver.NetconfMethods):
    """The rpc_* handlers the `netconf` dispatcher calls as `rpc_<name>(session, rpc, *children)`."""

    def __init__(self, settings: Settings, datastore: Datastore) -> None:
        self._settings = settings
        self._ds = datastore
        self._subs_lock = threading.Lock()
        self._subscribers: list[ncserver.NetconfServerSession] = []

    # ---- hello --------------------------------------------------------------------------
    def nc_append_capabilities(self, capabilities: etree._Element) -> None:
        for cap in yang.CAPABILITIES:
            if cap in _BASE_CAPS:
                continue  # the library already advertises base:1.0 and base:1.1
            ncutil.subelm(capabilities, "capability").text = cap

    # ---- data ---------------------------------------------------------------------------
    def _data(self, include_state: bool) -> etree._Element:
        data = ncutil.elm("nc:data")
        data.append(yang.carriers_to_xml(self._ds.carriers(), self._ds.statuses(), include_state=include_state))
        data.append(yang.ruapp_policy_to_xml(self._ds.policies()))
        ifs = etree.SubElement(data, f"{{{yang.NS_IETF_IF}}}interfaces", nsmap={None: yang.NS_IETF_IF})
        for name, desc in self._ds.interfaces().items():
            itf = etree.SubElement(ifs, f"{{{yang.NS_IETF_IF}}}interface")
            etree.SubElement(itf, f"{{{yang.NS_IETF_IF}}}name").text = name
            etree.SubElement(itf, f"{{{yang.NS_IETF_IF}}}description").text = desc
        return data

    @staticmethod
    def _apply_filter(data: etree._Element, filter_or_none: Optional[etree._Element]) -> etree._Element:
        """Top-level subtree selection: keep the containers named in the filter (all when empty)."""
        if filter_or_none is None:
            return data
        wanted = {child.tag for child in filter_or_none}
        if not wanted:
            return data
        for child in list(data):
            if child.tag not in wanted:
                data.remove(child)
        return data

    def rpc_get(self, session, rpc, filter_or_none):  # noqa: D401
        return self._apply_filter(self._data(include_state=True), filter_or_none)

    def rpc_get_config(self, session, rpc, source_elm, filter_or_none):
        return self._apply_filter(self._data(include_state=False), filter_or_none)

    # ---- edit-config ---------------------------------------------------------------------
    def rpc_edit_config(self, session, rpc, *params):
        config = rpc.find(f".//{{{yang.NS_NC}}}config")
        if config is None:
            raise ncerror.MissingElementProtoError(rpc, ncutil.qname("nc:config"))
        carrier_edits = yang.parse_carrier_edits(config)
        interface_edits = yang.parse_interface_edits(config)
        try:
            self._ds.apply_edit(carrier_edits, cause="netconf-edit-config")
        except HoldViolation as exc:
            log.warning("edit-config rejected: %s", exc)
            raise ncerror.RPCServerError(
                rpc, ncerror.RPCERR_TYPE_APPLICATION, ncerror.RPCERR_TAG_OPERATION_FAILED,
                message=str(exc), info={"carrier": exc.carrier, "hold-remaining-s": f"{exc.remaining_s:.1f}"},
            ) from exc
        except SleepStateViolation as exc:
            log.warning("edit-config rejected: %s", exc)
            raise ncerror.RPCServerError(
                rpc, ncerror.RPCERR_TYPE_APPLICATION, ncerror.RPCERR_TAG_OPERATION_FAILED,
                message=str(exc), info={"carrier": exc.carrier, "requested-active": exc.requested.value},
            ) from exc
        except UnknownCarrier as exc:
            raise ncerror.DataMissingAppError(rpc, message=f"unknown tx-array-carriers name {exc.args[0]}") from exc
        for name, desc in interface_edits.items():
            self._ds.set_interface_description(name, desc)
            log.info("ietf-interfaces %s description=%r", name, desc)
        return ncutil.elm("nc:ok")

    # ---- notifications (RFC 5277) --------------------------------------------------------
    def rpc_create_subscription(self, session, rpc, *params):
        with self._subs_lock:
            if session not in self._subscribers:
                self._subscribers.append(session)
        log.info("notification subscription from %s", session)
        return ncutil.elm("nc:ok")

    def on_state_change(self, ev: CarrierStateChanged) -> None:
        """Push `tx-array-carriers-state-change` to every subscribed session (bus worker thread)."""
        when = datetime.fromtimestamp(ev.wall_us / 1_000_000, tz=timezone.utc)
        xml = yang.state_change_notification(ev.carrier, ev.state, when=when)
        with self._subs_lock:
            sessions = list(self._subscribers)
        for s in sessions:
            try:
                if not s.session_open:
                    raise ConnectionError("session closed")
                s.send_message(xml)
            except Exception as exc:  # noqa: BLE001 - drop dead sessions
                log.info("dropping notification subscriber %s: %s", s, exc)
                with self._subs_lock:
                    if s in self._subscribers:
                        self._subscribers.remove(s)

    def subscriber_count(self) -> int:
        with self._subs_lock:
            return len(self._subscribers)

    # ---- o-ran-supervision ---------------------------------------------------------------
    def rpc_supervision_watchdog_reset(self, session, rpc, *params):
        interval_s = 60
        for p in params:
            if etree.QName(p).localname == "supervision-notification-interval" and p.text:
                try:
                    interval_s = int(p.text)
                except ValueError:
                    pass
        nxt = datetime.now(timezone.utc) + timedelta(seconds=interval_s)
        el = etree.Element(f"{{{yang.NS_SUPERVISION}}}next-update-at", nsmap={None: yang.NS_SUPERVISION})
        el.text = nxt.isoformat(timespec="microseconds").replace("+00:00", "Z")
        return el


class _BoundNetconfSSHServer(ncserver.NetconfSSHServer):
    """`NetconfSSHServer` that binds one (host, port) instead of every interface.

    Re-implements the constructor of `sshutil.server.SSHServer` (which hard-codes '' / '::')
    with the same attributes so the inherited accept loop and session classes work unchanged.
    """

    def __init__(self, server_ctl, server_methods, host: str, port: int, host_key: Path, debug: bool = False) -> None:  # noqa: PLR0913
        # netconf.server.NetconfSSHServer attributes
        self.server_methods = server_methods
        self.session_id = 1
        self.session_locks_lock = threading.Lock()
        self.session_locks = {"running": 0, "candidate": 0}
        # sshutil.server.SSHServer attributes
        self.server_ctl = server_ctl
        self.server_socket_class = sshutil.server.SSHServerSocket
        self.server_session_class = ncserver.NetconfServerSession
        self.extra_args = None
        self.debug = debug
        self.ports: dict = {}
        self.host_key = sshutil.server.from_private_key_file(str(host_key))
        if self.host_key is None:
            raise RuntimeError(f"cannot load host key {host_key}")
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        self.port = sock.getsockname()[1]
        self.ports[family] = self.port
        sock.listen(100)
        self.close_wsocket, self.close_rsocket = socket.socketpair()
        self.lock = threading.Lock()
        self.sockets: list = []
        self.thread = threading.Thread(None, self._accept_socket_thread, name="SSHAcceptThread", args=[sock], daemon=True)
        self.thread.start()

    def __del__(self) -> None:  # the library logs an ERROR from __del__; nothing to clean up here
        return


class RuNetconfServer:
    """Lifecycle wrapper: start/stop the M-Plane server and wire notifications to the bus."""

    def __init__(self, settings: Settings, datastore: Datastore, bus: Bus,
                 host: Optional[str] = None, port: Optional[int] = None) -> None:
        self._settings = settings
        self._bus = bus
        self._host = host if host is not None else settings.ru_netconf_host
        self._port = port if port is not None else settings.ru_netconf_port
        self.methods = RuNetconfMethods(settings, datastore)
        self._server: Optional[_BoundNetconfSSHServer] = None
        self._sub = None

    @property
    def port(self) -> int:
        return self._server.port if self._server is not None else self._port

    @property
    def host(self) -> str:
        return self._host

    def start(self) -> "RuNetconfServer":
        key = ensure_host_key(self._settings.ru_host_key)
        ctl = ncserver.SSHUserPassController(self._settings.ru_netconf_user, self._settings.ru_netconf_password)
        self._server = _BoundNetconfSSHServer(ctl, self.methods, self._host, self._port, key)
        self._sub = self._bus.subscribe(CarrierStateChanged, self.methods.on_state_change, inline=False, name="netconf-notify")
        # paramiko logs an ERROR ("Connection reset by peer") every time a client closes its SSH
        # transport after <close-session>; that is normal client behaviour, not a fault.
        logging.getLogger("paramiko.transport").setLevel(logging.CRITICAL)
        log.info("NETCONF/SSH M-Plane server listening on %s:%d (user %s)", self._host, self.port, self._settings.ru_netconf_user)
        return self

    def stop(self) -> None:
        if self._sub is not None:
            self._bus.unsubscribe(self._sub)
            self._sub = None
        if self._server is not None:
            try:
                self._server.close()
                self._server.join()
            except Exception as exc:  # noqa: BLE001
                log.debug("netconf server close: %s", exc)
            self._server = None
