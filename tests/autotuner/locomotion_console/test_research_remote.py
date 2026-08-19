"""Controlled SSH research backend tests; no network connection is made."""
from __future__ import annotations

import re
from pathlib import Path

from autotuner.research.research_ledger import ExperimentPlan
from autotuner.locomotion_console.research_remote import SSHExperimentBackend


class FakeRemote:
    def __init__(self, config):
        self.config = dict(config)
        self.files: dict[str, bytes] = {}
        self.commands: list[str] = []
        self.closed = False

    def exec(self, command, timeout=30):
        self.commands.append(command)
        marker = re.search(r"__RL_RESEARCH_RC_[0-9a-f]+__", command).group(0)
        payload = ""
        if "if test -f training.exit" in command:
            payload = "EXIT:0\n"
        elif "cat /remote/research/" in command and "evaluation.exit" in command:
            payload = "0\n"
        return f"{payload}\n{marker}0\n", ""

    def put(self, local_path, remote_path):
        self.files[remote_path] = Path(local_path).read_bytes()

    def get(self, remote_path, local_path):
        if remote_path.endswith("training.log"):
            content = b"training complete\n"
        elif remote_path.endswith("evaluation.stdout"):
            content = b'{"success": true, "evidence_complete": true, "protected": {"flat": {"passed": true}}}'
        elif remote_path.endswith("evaluation.stderr"):
            content = b""
        else:
            content = self.files[remote_path]
        Path(local_path).write_bytes(content)

    def close(self):
        self.closed = True


def _plan() -> ExperimentPlan:
    return ExperimentPlan(
        id="experiment:ssh",
        problem_statement="bounded remote experiment",
        baseline_ref="bundle:base",
        intervention_diff={"patch_ref": "patch:1"},
        unchanged_fields=["contract"],
        protected_capabilities=["flat"],
        training_window={"command": ["python", "train.py", "--note", "a value; not shell"]},
        evaluation_plan={"command": ["python", "evaluate.py"]},
        success_condition="independent evaluator passes",
        rollback_condition="flat regresses",
        resource_budget={"resources": ["gpu:0"], "evaluation_timeout_s": 30},
        authorization_ref="approval:test",
        status="approved",
    )


def test_ssh_backend_uploads_artifact_quotes_argv_and_returns_evaluation(tmp_path: Path):
    created: list[FakeRemote] = []

    def factory(config):
        remote = FakeRemote(config)
        created.append(remote)
        return remote

    workspace = tmp_path / "workspace"
    candidate = workspace / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "mechanisms.json").write_text("{}", encoding="utf-8")
    backend = SSHExperimentBackend(
        {"ssh_host": "example", "ssh_user": "root"},
        remote_root="/remote/research",
        remote_factory=factory,
    )
    handle = backend.start(
        _plan(),
        workspace,
        {
            "PATH": "/must/not/be-forwarded",
            "SECRET_TOKEN": "must-not-leak",
            "CUDA_VISIBLE_DEVICES": "0",
            "RL_MODE": "research",
        },
    )
    remote = created[0]
    runner_path = next(path for path in remote.files if path.endswith("/run.sh"))
    runner_bytes = remote.files[runner_path]
    runner = runner_bytes.decode("utf-8")
    assert b"\r\n" not in runner_bytes
    assert "'a value; not shell'" in runner
    assert "SECRET_TOKEN" not in runner
    assert "must/not/be-forwarded" not in runner
    assert "CUDA_VISIBLE_DEVICES=0" in runner
    assert "RL_MECHANISM_BUNDLE=" in runner
    assert "/candidate/mechanisms.json" in runner
    assert any(path.endswith("/candidate/mechanisms.json") for path in remote.files)

    assert backend.poll(handle).state == "succeeded"
    evaluation = backend.evaluate(_plan(), workspace)
    assert evaluation["success"] is True
    assert evaluation["protected"]["flat"]["passed"] is True
    assert remote.closed is True
    assert (workspace / "training.log").is_file()
    assert (workspace / "evaluation.stdout").is_file()
