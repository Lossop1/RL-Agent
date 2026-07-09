"""Persistent SSH + tmux helpers.

Key invariants:
  - Single SSH client reused across calls; auto-reconnect on EOF/SSHException.
  - tmux pipe-pane uses the `-o` flag (existing tools/training_exec.py behavior)
    which is **toggle**: a second `-o` call with a different path will close
    the pipe instead of switching. So we track per-session pipe state and
    refuse to call pipe_pane twice within the same tmux session lifetime.
    The supported flow is: session_kill -> session_create -> pipe_pane (once).
"""
from __future__ import annotations
import os
import re
import time
from typing import Optional, Tuple

import paramiko


_DEFAULT_KNOWN_HOSTS = os.path.expanduser("~/.locomotion_console/known_hosts")


class _AcceptNewHostKeyPolicy(paramiko.MissingHostKeyPolicy):
    """Trust-on-first-use (OpenSSH `accept-new`): accept an UNKNOWN host key and PERSIST it, so
    every later connection verifies against it. paramiko rejects a CHANGED key (BadHostKeyException)
    on its own — the persisted key is checked before this policy is consulted — so a MITM that
    swaps the key after first contact is caught. Strictly safer than AutoAddPolicy (which never
    persists, so never detects a change) and usable unlike RejectPolicy (which blocks first connect)."""

    def __init__(self, save_path: str):
        self._save_path = save_path

    def missing_host_key(self, client, hostname, key):
        client.get_host_keys().add(hostname, key.get_name(), key)
        try:
            os.makedirs(os.path.dirname(self._save_path), exist_ok=True)
            client.save_host_keys(self._save_path)
        except Exception:
            pass


class _SSHConfig:
    def __init__(self, cfg: dict):
        self.host = cfg["ssh_host"]
        self.port = int(cfg.get("ssh_port", 22))
        self.user = cfg.get("ssh_user", "root")
        self.password = cfg.get("ssh_pass", "")
        self.work_dir = cfg.get("work_dir", "/root/robot_lab")
        self.conda_sh = cfg.get("conda_sh_path", "/opt/conda/etc/profile.d/conda.sh")
        self.conda_env = cfg.get("conda_env", "isaaclab")
        self.shell_init = cfg.get("shell_init", "")
        # Optional pinned known_hosts path; when set, only hosts in it are accepted.
        self.known_hosts = cfg.get("known_hosts", "")
        # Opt-in escape hatch for first-connect trust-on-first-use (default off = verify).
        self.host_key_auto_add = bool(cfg.get("ssh_auto_add_host_key", False))


class RemoteSSH:
    """Persistent paramiko client with reconnect + tmux helpers.

    Usage:
      remote = RemoteSSH(cfg_dict)
      remote.exec("ls /root")
      remote.tmux_reset()           # session_kill + session_create + pipe_pane
      remote.tmux_send("python train.py ...")
      remote.tmux_capture(lines=200)
    """

    TMUX_SESSION = "rl_train"
    PIPE_FILE = "/tmp/rl_train_capture.log"

    def __init__(self, ssh_cfg: dict):
        self.cfg = _SSHConfig(ssh_cfg)
        self._client: Optional[paramiko.SSHClient] = None
        self._pipe_set = False   # whether pipe_pane was called for the current tmux session
        self.default_retries = 5
        self.connect_timeout = 20
        self.banner_timeout = 30
        self.auth_timeout = 20

    # ── connection ──
    def _connect(self):
        client = paramiko.SSHClient()
        # Host-key verification: default to accept-new (TOFU) — usable on first connect to a fresh
        # box AND detects a later key swap (MITM). A pinned known_hosts + RejectPolicy is used when
        # `known_hosts` is configured; blind AutoAdd is opt-in via `ssh_auto_add_host_key`.
        kh_path = self.cfg.known_hosts or _DEFAULT_KNOWN_HOSTS
        try:
            client.load_system_host_keys()
        except Exception:
            pass
        if os.path.exists(kh_path):
            try:
                client.load_host_keys(kh_path)
            except Exception:
                pass
        if self.cfg.host_key_auto_add:
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        elif self.cfg.known_hosts:
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
        else:
            client.set_missing_host_key_policy(_AcceptNewHostKeyPolicy(kh_path))
        client.connect(
            hostname=self.cfg.host,
            port=self.cfg.port,
            username=self.cfg.user,
            password=self.cfg.password,
            timeout=self.connect_timeout,
            banner_timeout=self.banner_timeout,
            auth_timeout=self.auth_timeout,
            allow_agent=False,
            look_for_keys=False,
        )
        self._client = client

    def _ensure_connected(self):
        if self._client is None:
            self._connect()
            return
        try:
            transport = self._client.get_transport()
            if transport is None or not transport.is_active():
                self._connect()
        except Exception:
            self._connect()

    def close(self):
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    # ── raw exec ──
    def exec(self, cmd: str, timeout: int = 30, retries: Optional[int] = None) -> Tuple[str, str]:
        """Execute remote command. Returns (stdout, stderr) decoded as text.

        Reconnects on EOF/SSHException with EXPONENTIAL BACKOFF:
        attempt N waits min(2**N, 30) seconds before retry. Without this,
        rapid retries hammer an overloaded SSH server and cascade
        "Error reading SSH protocol banner" errors.
        """
        retry_count = self.default_retries if retries is None else int(retries)
        last_err = None
        for attempt in range(retry_count + 1):
            try:
                self._ensure_connected()
                _, stdout, stderr = self._client.exec_command(cmd, timeout=timeout)
                out = stdout.read().decode("utf-8", errors="replace")
                err = stderr.read().decode("utf-8", errors="replace")
                return out, err
            except (EOFError, paramiko.SSHException, OSError) as e:
                last_err = e
                self.close()
                if attempt < retry_count:
                    backoff = min(2 ** attempt, 30)
                    time.sleep(backoff)
        raise RuntimeError(f"ssh exec failed after {retry_count+1} attempts: {last_err}")

    def exec_out(self, cmd: str, timeout: int = 30, retries: Optional[int] = None) -> str:
        """Convenience: return stdout stripped."""
        out, _ = self.exec(cmd, timeout=timeout, retries=retries)
        return out.strip()

    # ── file upload (binary-safe SFTP) ──
    def put(self, local_path: str, remote_path: str, retries: int = 3) -> None:
        """Upload a local file to the remote via SFTP (binary-safe; for .npz/.py).

        Reconnects with backoff like exec(). Caller should backup remote target first
        (deploy layer does) — this does not back up.
        """
        last_err = None
        for attempt in range(retries + 1):
            try:
                self._ensure_connected()
                sftp = self._client.open_sftp()
                try:
                    sftp.put(local_path, remote_path)
                finally:
                    sftp.close()
                return
            except (EOFError, paramiko.SSHException, OSError) as e:
                last_err = e
                self.close()
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"sftp put failed after {retries+1} attempts: {last_err}")

    def get(self, remote_path: str, local_path: str, retries: int = 3) -> None:
        """Download a remote file via SFTP."""
        last_err = None
        local_dir = os.path.dirname(os.path.abspath(local_path))
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
        for attempt in range(retries + 1):
            try:
                self._ensure_connected()
                sftp = self._client.open_sftp()
                try:
                    sftp.get(remote_path, local_path)
                finally:
                    sftp.close()
                return
            except (EOFError, paramiko.SSHException, OSError) as e:
                last_err = e
                self.close()
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"sftp get failed after {retries+1} attempts: {last_err}")

    # ── tmux ──
    @property
    def _shell_prefix(self) -> str:
        return f"{self.cfg.shell_init} && " if self.cfg.shell_init else ""

    def tmux_kill(self):
        self.exec(
            f"{self._shell_prefix}tmux kill-session -t {self.TMUX_SESSION} 2>/dev/null; true",
            timeout=10,
        )
        self._pipe_set = False

    def tmux_create(self):
        out, _ = self.exec(
            f"{self._shell_prefix}tmux new-session -d -s {self.TMUX_SESSION} && echo CREATED",
            timeout=10,
        )
        if "CREATED" not in out:
            raise RuntimeError(f"tmux session creation failed: {out!r}")
        self._pipe_set = False

    def tmux_reset(self, pipe_file: Optional[str] = None):
        """kill + create + (optionally) pipe_pane → single-call pattern.

        This is the SUPPORTED way to start a new training capture cycle.
        Don't call pipe_pane directly multiple times — the `-o` toggle bug bites.
        """
        self.tmux_kill()
        time.sleep(0.5)
        self.tmux_create()
        target = pipe_file or self.PIPE_FILE
        self.tmux_pipe(target)

    def tmux_pipe(self, log_path: str):
        """Set pipe ONCE for the current tmux session. Refuses second call."""
        if self._pipe_set:
            raise RuntimeError(
                "tmux_pipe called twice for the same session — "
                "the `-o` flag is a toggle and would close the pipe. "
                "Use tmux_reset() to start a new capture cycle."
            )
        import shlex
        q = shlex.quote(log_path)
        # truncate target file first (avoid stale content from prior runs)
        self.exec(f"> {q}", timeout=10)
        # auto-create parent dir
        parent = log_path.rsplit("/", 1)[0] if "/" in log_path else "."
        self.exec(f"mkdir -p {shlex.quote(parent)}", timeout=10)
        out, _ = self.exec(
            f"{self._shell_prefix}tmux pipe-pane -t {self.TMUX_SESSION} -o "
            f"'cat > {q}' && echo PIPE_OK",
            timeout=10,
        )
        if "PIPE_OK" not in out:
            raise RuntimeError(f"tmux_pipe failed: {out!r}")
        self._pipe_set = True

    def tmux_send(self, command: str, wait: float = 0.5):
        """Send a command to the tmux session as if typed at prompt."""
        escaped = command.replace("'", "'\\''")
        out, _ = self.exec(
            f"{self._shell_prefix}tmux send-keys -t {self.TMUX_SESSION} "
            f"'{escaped}' Enter && echo SENT",
            timeout=10,
        )
        if "SENT" not in out:
            raise RuntimeError(f"tmux_send failed: {out!r}")
        time.sleep(wait)

    def tmux_capture(self, lines: int = 200) -> str:
        out, _ = self.exec(
            f"tmux capture-pane -t {self.TMUX_SESSION} -p -S -{lines}",
            timeout=15,
        )
        return out

    def tmux_has_session(self) -> bool:
        out, _ = self.exec(
            f"tmux has-session -t {self.TMUX_SESSION} 2>/dev/null && echo YES || echo NO",
            timeout=10,
        )
        return "YES" in out

    # ── training helpers ──
    def kill_train_procs(self, task_filter: str = "train.py"):
        """SIGKILL any running train.py processes."""
        self.exec(f"pkill -9 -f '{task_filter}' 2>/dev/null; true", timeout=10)

    def find_latest_iter(self, log_path: str) -> Optional[int]:
        """Grep last 'Learning iteration N' from a remote log file. Returns None if not found."""
        try:
            out = self.exec_out(
                f"grep -oP 'Learning iteration\\s+\\K\\d+' {log_path} 2>/dev/null | tail -1",
                timeout=10,
            )
            if out:
                return int(out)
        except Exception:
            return None
        return None

    def find_checkpoint(self, run_dir: str, target_iter: int) -> Optional[str]:
        """Return path to model_<i>.pt with the largest i <= target_iter."""
        out = self.exec_out(
            f"ls {run_dir}/model_*.pt 2>/dev/null", timeout=15,
        )
        if not out:
            return None
        best_i = -1
        best = None
        for line in out.splitlines():
            line = line.strip()
            m = re.search(r"model_(\d+)\.pt", line)
            if not m:
                continue
            it = int(m.group(1))
            if it <= target_iter and it > best_i:
                best_i = it
                best = line
        return best

    def list_recent_run_dirs(self, framework: str = "rsl_rl",
                              experiment_name: str = "taili_amp",
                              limit: int = 5) -> list:
        """Return paths to N most recently modified run dirs under logs/<framework>/<experiment_name>/."""
        base = f"{self.cfg.work_dir}/logs/{framework}/{experiment_name}"
        # ls -t sorts by mtime desc
        out = self.exec_out(f"ls -1t {base} 2>/dev/null | head -{limit}", timeout=15)
        if not out:
            return []
        return [f"{base}/{name.strip()}" for name in out.splitlines() if name.strip()]

    def remote_time(self) -> float:
        """Remote epoch seconds, useful for diffing against directory mtimes."""
        try:
            out = self.exec_out("date +%s", timeout=10)
            return float(out)
        except Exception:
            return time.time()


# ── module-level singleton helper ──
_GLOBAL_REMOTE: Optional[RemoteSSH] = None


def get_default_remote() -> RemoteSSH:
    """Return process-wide RemoteSSH instance built from runtime_state config."""
    global _GLOBAL_REMOTE
    if _GLOBAL_REMOTE is None:
        import runtime_state
        _GLOBAL_REMOTE = RemoteSSH(runtime_state.get_config())
    return _GLOBAL_REMOTE
