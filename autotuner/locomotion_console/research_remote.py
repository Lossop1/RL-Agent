"""Controlled SSH backend for approved research experiments.

Only argv arrays persisted in an approved ``ExperimentPlan`` cross this
boundary. The backend creates its own runner scripts, quotes every argument,
uploads the compiled mechanism artifact, and mirrors remote logs locally.
"""
from __future__ import annotations

import json
import re
import shlex
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from autotuner.infrastructure.remote import RemoteSSH

from autotuner.research.research_ledger import ExperimentPlan
from autotuner.research.research_supervisor import BackendHandle, BackendStatus


_ENV_EXACT = frozenset({
    "CUDA_VISIBLE_DEVICES",
    "RL_RESEARCH_EXPERIMENT_REF",
    "RL_RESEARCH_CANDIDATE_ROOT",
})
_ENV_PREFIXES = ("TAILI_",)
_SAFE_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")[:100] or "experiment"


def _remote_root(value: str) -> str:
    if not value.startswith("/") or "\n" in value or "\r" in value or "\0" in value:
        raise ValueError("remote research root must be a safe absolute POSIX path")
    normalized = str(PurePosixPath(value))
    if normalized == "/":
        raise ValueError("remote research root cannot be filesystem root")
    if ".." in PurePosixPath(value).parts:
        raise ValueError("remote research root cannot contain parent traversal")
    return normalized


def _argv(plan: ExperimentPlan, section: str) -> list[str]:
    source = plan.training_window if section == "training" else plan.evaluation_plan
    value = source.get("command")
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"approved experiment requires {section} command as a non-empty argv list")
    return list(value)


@dataclass
class _RemoteRun:
    remote: Any
    remote_workspace: str
    local_workspace: Path
    environment: dict[str, str]
    terminal_state: str = ""


class SSHExperimentBackend:
    """Run a registered experiment on one SSH target without accepting shell text."""

    def __init__(
        self,
        ssh_config: Mapping[str, Any],
        *,
        remote_root: str = "/root/gpufree-data/research_experiments",
        remote_factory=RemoteSSH,
    ) -> None:
        self.ssh_config = dict(ssh_config)
        self.remote_root = _remote_root(remote_root)
        self.remote_factory = remote_factory
        self._runs: dict[str, _RemoteRun] = {}
        self._workspace_handles: dict[str, str] = {}

    @staticmethod
    def _checked(remote: Any, command: str, *, timeout: int = 30) -> tuple[str, str]:
        marker = f"__RL_RESEARCH_RC_{uuid.uuid4().hex}__"
        script = f"{command}\nrc=$?\nprintf '\\n{marker}%s\\n' \"$rc\""
        stdout, stderr = remote.exec("bash -lc " + shlex.quote(script), timeout=timeout)
        position = stdout.rfind(marker)
        if position < 0:
            raise RuntimeError(f"remote command did not return a status marker: {stderr[-1000:]}")
        status_text = stdout[position + len(marker):].strip().splitlines()[0]
        try:
            status = int(status_text)
        except ValueError as exc:
            raise RuntimeError("remote command returned an invalid status marker") from exc
        clean_stdout = stdout[:position].rstrip()
        if status != 0:
            detail = (stderr or clean_stdout)[-1500:]
            raise RuntimeError(f"remote command failed with exit code {status}: {detail}")
        return clean_stdout, stderr

    @staticmethod
    def _remote_environment(environment: Mapping[str, str], candidate_root: str) -> dict[str, str]:
        selected: dict[str, str] = {}
        for key, value in environment.items():
            name = str(key)
            if not _SAFE_ENV_NAME.fullmatch(name):
                continue
            if name in _ENV_EXACT or name.startswith(_ENV_PREFIXES):
                selected[name] = str(value)
        selected["RL_RESEARCH_CANDIDATE_ROOT"] = candidate_root
        selected["TAILI_MECHANISM_BUNDLE"] = f"{candidate_root}/mechanisms.json"
        return selected

    @staticmethod
    def _runner_script(
        command: list[str],
        environment: Mapping[str, str],
        remote_workspace: str,
        exit_file: str,
        *,
        stdout_file: str = "",
        stderr_file: str = "",
    ) -> str:
        assignments = " ".join(
            f"{key}={shlex.quote(str(value))}" for key, value in sorted(environment.items())
        )
        invocation = f"env {assignments} {shlex.join(command)}" if assignments else shlex.join(command)
        if stdout_file:
            invocation += f" > {shlex.quote(stdout_file)}"
        if stderr_file:
            invocation += f" 2> {shlex.quote(stderr_file)}"
        return (
            "#!/usr/bin/env bash\n"
            "set +e\n"
            f"cd {shlex.quote(remote_workspace)}\n"
            f"{invocation}\n"
            "rc=$?\n"
            f"printf '%s\\n' \"$rc\" > {shlex.quote(exit_file + '.tmp')}\n"
            f"mv {shlex.quote(exit_file + '.tmp')} {shlex.quote(exit_file)}\n"
            "exit \"$rc\"\n"
        )

    def _upload_tree(self, remote: Any, local_root: Path, remote_root: str) -> None:
        self._checked(remote, f"mkdir -p {shlex.quote(remote_root)}")
        for source in sorted(local_root.rglob("*")):
            if source.is_symlink():
                raise ValueError(f"candidate artifact contains a symlink: {source}")
            relative = source.relative_to(local_root).as_posix()
            destination = f"{remote_root}/{relative}"
            if source.is_dir():
                self._checked(remote, f"mkdir -p {shlex.quote(destination)}")
                continue
            self._checked(remote, f"mkdir -p {shlex.quote(str(PurePosixPath(destination).parent))}")
            remote.put(str(source), destination)

    @staticmethod
    def _write_local(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Runner scripts are executed by Bash on the remote Linux host.  Write
        # bytes so Windows cannot translate LF into CRLF during staging.
        path.write_bytes(content.encode("utf-8"))

    def _sync_file(self, run: _RemoteRun, remote_name: str, local_name: str) -> str:
        local = run.local_workspace / local_name
        remote_path = f"{run.remote_workspace}/{remote_name}"
        try:
            run.remote.get(remote_path, str(local))
            return local.read_text(encoding="utf-8", errors="replace")
        except Exception:
            output, _ = self._checked(
                run.remote,
                f"test -f {shlex.quote(remote_path)} && tail -c 8000 {shlex.quote(remote_path)} || true",
            )
            self._write_local(local, output)
            return output

    def start(
        self,
        plan: ExperimentPlan,
        workspace: Path,
        environment: Mapping[str, str],
    ) -> BackendHandle:
        command = _argv(plan, "training")
        remote = self.remote_factory(self.ssh_config)
        handle = BackendHandle(id=f"ssh:{uuid.uuid4().hex}", started_at=time.time())
        remote_workspace = (
            f"{self.remote_root}/{_safe_name(plan.id)}-{handle.id.rsplit(':', 1)[-1][:12]}"
        )
        candidate_remote = f"{remote_workspace}/candidate"
        try:
            self._checked(remote, f"mkdir -p {shlex.quote(remote_workspace)}")
            self._upload_tree(remote, workspace / "candidate", candidate_remote)
            remote_env = self._remote_environment(environment, candidate_remote)
            runner = self._runner_script(
                command,
                remote_env,
                remote_workspace,
                "training.exit",
            )
            local_meta = workspace / ".remote"
            self._write_local(local_meta / "run.sh", runner)
            self._write_local(
                local_meta / "plan.json",
                json.dumps(plan.model_dump(mode="json"), indent=2) + "\n",
            )
            remote.put(str(local_meta / "run.sh"), f"{remote_workspace}/run.sh")
            remote.put(str(local_meta / "plan.json"), f"{remote_workspace}/plan.json")
            launch = (
                f"cd {shlex.quote(remote_workspace)} && "
                "rm -f training.exit training.pid && "
                "(nohup setsid bash ./run.sh > training.log 2>&1 < /dev/null & "
                "printf '%s\\n' \"$!\" > training.pid) && test -s training.pid"
            )
            self._checked(remote, launch)
        except Exception:
            remote.close()
            raise
        run = _RemoteRun(
            remote=remote,
            remote_workspace=remote_workspace,
            local_workspace=workspace,
            environment=dict(environment),
        )
        self._runs[handle.id] = run
        self._workspace_handles[str(workspace.resolve())] = handle.id
        self._write_local(
            workspace / "remote_execution.json",
            json.dumps({"handle_id": handle.id, "remote_workspace": remote_workspace}, indent=2) + "\n",
        )
        return handle

    def poll(self, handle: BackendHandle) -> BackendStatus:
        run = self._runs[handle.id]
        if run.terminal_state:
            return BackendStatus(run.terminal_state)  # type: ignore[arg-type]
        probe = (
            f"cd {shlex.quote(run.remote_workspace)} && "
            "if test -f training.exit; then printf 'EXIT:%s\\n' \"$(cat training.exit)\"; "
            "elif test -s training.pid && kill -0 \"$(cat training.pid)\" 2>/dev/null; then echo RUNNING; "
            "else echo LOST; fi"
        )
        output, _ = self._checked(run.remote, probe)
        state = output.strip().splitlines()[-1] if output.strip() else "LOST"
        if state == "RUNNING":
            return BackendStatus("running")
        log = self._sync_file(run, "training.log", "training.log")[-2000:]
        if state.startswith("EXIT:"):
            try:
                code = int(state.split(":", 1)[1])
            except ValueError:
                code = -1
            run.terminal_state = "succeeded" if code == 0 else "failed"
        else:
            run.terminal_state = "failed"
            log = (log + "\nremote process disappeared without an exit record").strip()
        if run.terminal_state == "failed":
            run.remote.close()
        return BackendStatus(run.terminal_state, message=log)  # type: ignore[arg-type]

    def stop(self, handle: BackendHandle, reason: str) -> None:
        run = self._runs.get(handle.id)
        if run is None:
            return
        command = (
            f"cd {shlex.quote(run.remote_workspace)} && "
            "if test -s training.pid; then pid=$(cat training.pid); "
            "kill -TERM -- \"-$pid\" 2>/dev/null || kill -TERM \"$pid\" 2>/dev/null || true; "
            "sleep 2; kill -KILL -- \"-$pid\" 2>/dev/null || kill -KILL \"$pid\" 2>/dev/null || true; fi; "
            "test -f training.exit || printf '143\\n' > training.exit"
        )
        try:
            self._checked(run.remote, command, timeout=15)
            self._sync_file(run, "training.log", "training.log")
        finally:
            run.terminal_state = "stopped"
            run.remote.close()

    def evaluate(self, plan: ExperimentPlan, workspace: Path) -> Mapping[str, Any]:
        handle_id = self._workspace_handles.get(str(workspace.resolve()))
        if not handle_id or handle_id not in self._runs:
            return {"success": False, "reason": "remote_run_not_found"}
        run = self._runs[handle_id]
        try:
            command = _argv(plan, "evaluation")
            candidate_remote = f"{run.remote_workspace}/candidate"
            environment = self._remote_environment(run.environment, candidate_remote)
            environment["RL_RESEARCH_EXPERIMENT_REF"] = plan.id
            runner = self._runner_script(
                command,
                environment,
                run.remote_workspace,
                "evaluation.exit",
                stdout_file="evaluation.stdout",
                stderr_file="evaluation.stderr",
            )
            local_runner = workspace / ".remote" / "evaluate.sh"
            self._write_local(local_runner, runner)
            run.remote.put(str(local_runner), f"{run.remote_workspace}/evaluate.sh")
            timeout = int(plan.resource_budget.get("evaluation_timeout_s", 900))
            self._checked(
                run.remote,
                f"cd {shlex.quote(run.remote_workspace)} && rm -f evaluation.exit && bash ./evaluate.sh || true",
                timeout=max(timeout, 1),
            )
            exit_text, _ = self._checked(
                run.remote,
                f"cat {shlex.quote(run.remote_workspace + '/evaluation.exit')}",
            )
            stdout = self._sync_file(run, "evaluation.stdout", "evaluation.stdout")
            stderr = self._sync_file(run, "evaluation.stderr", "evaluation.stderr")
            try:
                exit_code = int(exit_text.strip().splitlines()[-1])
            except (ValueError, IndexError):
                exit_code = -1
            if exit_code != 0:
                return {
                    "success": False,
                    "reason": "evaluator_failed",
                    "exit_code": exit_code,
                    "stderr": stderr[-2000:],
                }
            try:
                result = json.loads(stdout)
            except json.JSONDecodeError as exc:
                return {
                    "success": False,
                    "reason": "evaluator_output_not_json",
                    "stdout": stdout[-2000:],
                    "error": str(exc),
                }
            return result if isinstance(result, Mapping) else {
                "success": False,
                "reason": "evaluator_output_not_object",
            }
        finally:
            run.remote.close()
