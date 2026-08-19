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

from autotuner.adapter.orchestrator import StepResult


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


def launch_training(ssh, train_cmd: str, session: str, log: str) -> StepResult:
    """Fire a long-running training job in a fresh tmux session (non-blocking). The orchestrator does
    NOT wait for it — monitoring is via the poll cron. Returns immediately with launch status."""
    import shlex
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
