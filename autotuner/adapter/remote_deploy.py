"""Concrete paramiko SSH for deploy.execute() — the real transport behind the gated executor.

deploy.py keeps the SSH dependency injected (so the planner stays pure and execute is unit-tested
against a fake). This is the production implementation of that `SSH` protocol: `exec(cmd)->(out,rc)`
and `put(local, remote)` over a persistent paramiko connection, credentials from config/ssh.json.

Outward-facing: only used when deploy.execute(confirm=True) is called. Constructing it does NOT
connect (lazy) — connection opens on first exec/put.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple


class ParamikoSSH:
    """Implements deploy.SSH (exec + put) over a lazy, reused paramiko connection."""

    def __init__(self, host: str, port: int, user: str, password: Optional[str] = None,
                 known_hosts: str = "", auto_add_host_key: bool = False):
        self.host, self.port, self.user, self.password = host, int(port), user, password
        self.known_hosts = known_hosts
        self.auto_add_host_key = bool(auto_add_host_key)
        self._client = None
        self._sftp = None

    def _conn(self):
        if self._client is not None:
            tr = self._client.get_transport()
            if tr is not None and tr.is_active():
                return self._client
        import paramiko
        from autotuner.training.remote import _AcceptNewHostKeyPolicy, _DEFAULT_KNOWN_HOSTS
        c = paramiko.SSHClient()
        # Host-key verification: accept-new (TOFU) by default — usable on first connect, detects a
        # later key swap (MITM). Pinned known_hosts → RejectPolicy; blind AutoAdd is opt-in.
        kh_path = self.known_hosts or _DEFAULT_KNOWN_HOSTS
        try:
            c.load_system_host_keys()
        except Exception:
            pass
        if os.path.exists(kh_path):
            try:
                c.load_host_keys(kh_path)
            except Exception:
                pass
        if self.auto_add_host_key:
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        elif self.known_hosts:
            c.set_missing_host_key_policy(paramiko.RejectPolicy())
        else:
            c.set_missing_host_key_policy(_AcceptNewHostKeyPolicy(kh_path))
        c.connect(hostname=self.host, port=self.port, username=self.user,
                  password=self.password, timeout=15, banner_timeout=20)
        self._client = c
        self._sftp = None
        return c

    def exec(self, cmd: str, timeout: int = 30) -> Tuple[str, int]:
        c = self._conn()
        _, out, err = c.exec_command(cmd, timeout=timeout)
        # recv_exit_status() blocks until the channel closes regardless of the exec_command
        # timeout, so a hung remote command would wedge the deploy forever. Bound the wait.
        chan = out.channel
        deadline = time.monotonic() + timeout
        while not chan.exit_status_ready():
            if time.monotonic() > deadline:
                try:
                    chan.close()
                except Exception:
                    pass
                raise TimeoutError(f"remote command timed out after {timeout}s: {cmd[:80]!r}")
            time.sleep(0.05)
        rc = chan.recv_exit_status()
        text = out.read().decode("utf-8", "replace")
        if rc != 0:
            text += err.read().decode("utf-8", "replace")
        return text, rc

    def put(self, local: str, remote: str) -> None:
        c = self._conn()
        if self._sftp is None:
            self._sftp = c.open_sftp()
        self._sftp.put(local, remote)

    def close(self) -> None:
        if self._sftp is not None:
            try:
                self._sftp.close()
            except Exception:
                pass
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = self._sftp = None


def from_ssh_json(path: str = "config/ssh.json") -> ParamikoSSH:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    host = cfg.get("ssh_host") or cfg.get("host")
    port = int(cfg.get("ssh_port") or cfg.get("port") or 22)
    user = cfg.get("ssh_user") or cfg.get("user") or "root"
    pwd = cfg.get("ssh_pass") or cfg.get("password") or cfg.get("pass")
    if not host:
        raise ValueError(f"no ssh_host in {path}")
    return ParamikoSSH(host, port, user, pwd,
                       known_hosts=cfg.get("known_hosts", ""),
                       auto_add_host_key=bool(cfg.get("ssh_auto_add_host_key", False)))
