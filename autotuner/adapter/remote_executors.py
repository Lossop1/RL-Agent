"""Real orchestrator executors — bridge the deterministic backbone (orchestrator.py) to the remote.

The orchestrator sequences offline → deploy → train → eval with INJECTED executors so its control
flow is unit-tested with mocks. This provides the REAL deploy executor: it runs deploy.execute over
a paramiko connection (backup-then-put + sha256 verify), returning a StepResult the orchestrator
consumes. OUTWARD-FACING — only reached when the orchestrator is called with allow_deploy=True.

train / evaluate are intentionally NOT synchronous executors here: they are long-running launch-and-
monitor jobs (tmux + cron polling), not a blocking call — so the orchestrator handles deploy
synchronously and training/eval are launched + watched separately. A LaunchExecutor that fires the
tmux job (without blocking for hours) is provided for that.
"""
from __future__ import annotations

import re
import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autotuner.adapter.orchestrator import StepResult


class RemoteSSHTransportAdapter:
    """将控制台的 RemoteSSH 适配为执行层的最小传输协议。

    控制台历史接口返回 ``(stdout, stderr)``，执行层接口返回
    ``(stdout, return_code)``。适配只存在于边界处，避免执行层知道
    控制台的重连和认证细节；新的产品部署因此可以复用同一套校验、
    staging 和原子激活逻辑。

    ``RemoteSSH`` 为了兼容旧调用者仍然只返回 stderr，不提供退出码。
    对这种传输，适配器在远端子 shell 中追加一个不可混淆的退出码标记；
    stderr 是否为空不再参与成功判定。若底层已经提供 ``exec_status``，
    则直接使用它，避免额外的 shell 包装。
    """

    _RETURN_CODE_MARKER = "__RL_AGENT_RC__"

    def __init__(self, remote, *, legacy_stderr: bool = True):
        self.remote = remote
        self.legacy_stderr = bool(legacy_stderr)

    def exec(self, cmd: str, timeout: int = 30) -> tuple[str, int]:
        if not self.legacy_stderr:
            result = self.remote.exec(cmd, timeout=timeout)
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("remote transport must return (output, return_code)")
            output, return_code = result
            if not isinstance(return_code, int):
                raise RuntimeError("return-code transport returned a non-integer status")
            return str(output), return_code

        exec_status = getattr(self.remote, "exec_status", None)
        if callable(exec_status):
            output, return_code = exec_status(cmd, timeout=timeout)
            return str(output), int(return_code)

        wrapped = (
            "set +e; "
            f"( {cmd} ); "
            "rc=$?; "
            f"printf '\\n{self._RETURN_CODE_MARKER}%s\\n' \"$rc\"; "
            "exit 0"
        )
        stdout, stderr = self.remote.exec("bash -lc " + shlex.quote(wrapped), timeout=timeout)
        output = str(stdout or "")
        match = re.search(
            rf"(?:^|\n){re.escape(self._RETURN_CODE_MARKER)}(-?\d+)\s*$",
            output,
        )
        if match is None:
            detail = str(stderr or "").strip()
            suffix = f"; stderr={detail!r}" if detail else ""
            raise RuntimeError(f"legacy remote transport did not return an exit marker{suffix}")
        return output[: match.start()].rstrip("\n"), int(match.group(1))

    def put(self, local: str, remote: str) -> None:
        self.remote.put(local, remote)

    def get(self, remote: str, local: str) -> None:
        """Download file from remote to local."""
        self.remote.get(remote, local)


class RemoteDeployExecutor:
    """Legacy file-plan deployer kept for existing ConfigSet callers.

    New payload deployments use ``VersionedPayloadDeployExecutor`` below.
    Keeping this bridge separate prevents the old source-tree path from
    becoming an implicit dependency of the execution layer.
    """

    def __init__(self, ssh_json: str = "config/ssh.json", do_launch: bool = False):
        self.ssh_json = ssh_json
        self.do_launch = do_launch

    def deploy(self, plan, stamp: str) -> StepResult:
        from autotuner.adapter.orchestrator import StepResult

        from autotuner.adapter.deploy import execute, restore_cmds
        from autotuner.adapter.remote_deploy import from_ssh_json
        try:
            ssh = from_ssh_json(self.ssh_json)
        except Exception as e:  # noqa: BLE001
            return StepResult("deploy", False, f"ssh setup failed: {e}")
        try:
            res = execute(plan, ssh, confirm=True, do_launch=self.do_launch, logger=lambda *_: None)
        except Exception as e:  # noqa: BLE001
            return StepResult("deploy", False, f"execute raised: {e}")
        finally:
            try:
                ssh.close()
            except Exception:
                pass
        n_ok = sum(1 for it in res.items if it.verified)
        detail = f"{n_ok}/{len(res.items)} verified" + ("" if res.ok else "; FAILED — rollback available")
        return StepResult("deploy", res.ok, detail,
                          {"items": [(it.remote, it.verified) for it in res.items],
                           "launched": res.launched,
                           "rollback": restore_cmds(plan) if not res.ok else []})


class VersionedPayloadDeployExecutor:
    """Bridge a product-produced ``DeploymentSpec`` to the execution layer."""

    def __init__(self, ssh_json: str = "config/ssh.json", layout=None):
        self.ssh_json = ssh_json
        self.layout = layout

    def deploy(self, spec) -> StepResult:
        from autotuner.adapter.orchestrator import StepResult

        from autotuner.execution.deployment import VersionedRemoteDeployer
        from autotuner.adapter.remote_deploy import from_ssh_json

        try:
            ssh = from_ssh_json(self.ssh_json)
        except Exception as exc:  # noqa: BLE001
            return StepResult("versioned-deploy", False, f"ssh setup failed: {exc}")
        try:
            deployer = VersionedRemoteDeployer(ssh, layout=self.layout)
            result = deployer.deploy_spec(spec)
            return StepResult(
                "versioned-deploy",
                result.status == "activated",
                result.status,
                result.to_dict(),
            )
        except Exception as exc:  # noqa: BLE001
            return StepResult("versioned-deploy", False, f"deployment raised: {exc}")
        finally:
            try:
                ssh.close()
            except Exception:
                pass


class VersionedTrainingStartExecutor:
    """把产品生成的启动计划桥接到安全的远程训练启动器。

    该适配器只负责建立 SSH、转换传输协议和关闭连接；会话冲突、marker、
    进程可见性及禁止杀死既有训练等规则全部由执行层启动器负责。
    """

    def __init__(self, ssh_json: str = "config/ssh.json"):
        self.ssh_json = ssh_json

    @staticmethod
    def _failure(plan, error: str):
        from autotuner.execution import TrainingStartResult

        run_id = str(getattr(plan, "run_id", ""))
        run_dir = str(getattr(plan, "run_dir", ""))
        session = str(getattr(plan, "tmux_session", ""))
        return TrainingStartResult(
            status="failed",
            run_id=run_id,
            run_dir=run_dir,
            handle_ref=f"tmux:{session}" if session else "",
            process_pattern=str(getattr(plan, "process_pattern", "")),
            error=error,
        )

    def start(self, plan):
        """启动一次已物料化的计划，并将连接或远端异常转为失败回执。"""
        from autotuner.adapter.remote_deploy import from_ssh_json
        from autotuner.execution import VersionedRemoteTrainingStarter

        try:
            ssh = from_ssh_json(self.ssh_json)
        except Exception as exc:  # noqa: BLE001
            return self._failure(plan, f"ssh setup failed: {type(exc).__name__}: {exc}")
        try:
            # ParamikoSSH 已经提供正式退出码；显式走适配器以固定执行层边界。
            transport = RemoteSSHTransportAdapter(ssh, legacy_stderr=False)
            return VersionedRemoteTrainingStarter(transport).start(plan)
        except Exception as exc:  # noqa: BLE001
            return self._failure(plan, f"training start raised: {type(exc).__name__}: {exc}")
        finally:
            try:
                ssh.close()
            except Exception:
                pass


def launch_training(ssh, train_cmd: str, session: str, log: str) -> StepResult:
    """Fire a long-running training job in a fresh tmux session (non-blocking). The orchestrator does
    NOT wait for it — monitoring is via the poll cron. Returns immediately with launch status."""
    from autotuner.adapter.orchestrator import StepResult

    visible_cmd = f"set -o pipefail; {train_cmd} 2>&1 | tee -a {shlex.quote(log)}"
    cmd = (
        f"rm -f {shlex.quote(log)}; "
        f"tmux kill-session -t {shlex.quote(session)} 2>/dev/null; "
        f"tmux new-session -d -s {shlex.quote(session)} {shlex.quote(visible_cmd)}"
    )
    out, rc = ssh.exec(cmd)
    alive, _ = ssh.exec(f"tmux has-session -t {shlex.quote(session)} 2>&1 && echo YES || echo NO")
    ok = "YES" in alive
    return StepResult("launch_training", ok, f"tmux {session} {'started' if ok else 'failed'}",
                      {"session": session, "log": log, "attach": f"tmux attach -t {session}"})


__all__ = [
    "RemoteDeployExecutor",
    "RemoteSSHTransportAdapter",
    "VersionedPayloadDeployExecutor",
    "VersionedTrainingStartExecutor",
    "launch_training",
]
