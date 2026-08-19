from autotuner.locomotion_console.config import LocomotionConsoleSettings
from autotuner.locomotion_console.datasource import RealDataSource


class _Remote:
    def __init__(self, metadata=None):
        self.metadata = metadata or {}
        self.commands = []

    def exec_out(self, command, timeout=None):
        self.commands.append(command)
        for path, raw in self.metadata.items():
            if path in command and command.startswith("cat "):
                return raw
        if "printf payload_ok" in command:
            return "payload_ok"
        return ""


def _source():
    return RealDataSource(LocomotionConsoleSettings(source="real"))


def test_payload_root_from_run_metadata_accepts_launcher_contracts():
    source = _source()
    assert source._payload_root_from_run_metadata(
        '{"payload_root":"/root/gpufree-data/training_payloads/source_a"}'
    ) == "/root/gpufree-data/training_payloads/source_a"
    assert source._payload_root_from_run_metadata(
        '{"payload":"/root/gpufree-data/training_payloads/source_b"}'
    ) == "/root/gpufree-data/training_payloads/source_b"
    assert source._payload_root_from_run_metadata("not-json") == ""


def test_recorded_payload_root_prefers_run_contract():
    run = "/root/gpufree-data/taili_runs/source_run"
    payload = "/root/gpufree-data/training_payloads/source_payload"
    remote = _Remote({f"{run}/run.json": f'{{"payload_root":"{payload}"}}'})

    assert _source()._recorded_payload_root(remote, run) == payload
    assert any("train_taili.py" in command for command in remote.commands)


def test_recorded_payload_root_falls_back_to_console_marker():
    run = "/root/gpufree-data/taili_runs/source_run"
    payload = "/root/gpufree-data/training_payloads/source_payload"
    remote = _Remote({f"{run}/console_start.json": f'{{"payload":"{payload}"}}'})

    assert _source()._recorded_payload_root(remote, run) == payload


def test_active_payload_root_resolves_content_addressed_pointer():
    source = _source()
    digest = "a" * 64
    active = source._remote_payload_root() + "/active.json"
    remote = _Remote({active: f'{{"payload_digest":"{digest}"}}'})

    assert source._active_payload_root(remote) == source._remote_payload_root() + f"/payloads/{digest}"


def test_active_payload_root_rejects_untrusted_digest():
    source = _source()
    active = source._remote_payload_root() + "/active.json"
    remote = _Remote({active: '{"payload_digest":"../../escape"}'})

    assert source._active_payload_root(remote) == ""


def test_resume_refuses_checkpoint_without_payload_provenance(monkeypatch):
    source = _source()
    remote = _Remote()
    source_run = "/root/gpufree-data/taili_runs/source_run"
    checkpoint = f"{source_run}/checkpoints/agent_64000.pt"

    monkeypatch.setattr(source, "_is_running", lambda _remote: False)
    monkeypatch.setattr(source, "_newest_run", lambda _remote: source_run)
    monkeypatch.setattr(source, "_resolve_checkpoint", lambda _remote, _run: checkpoint)
    monkeypatch.setattr(source, "_recorded_payload_root", lambda _remote, _run: "")

    try:
        source._start_payload_training(remote, resume=True)
    except RuntimeError as error:
        assert "refusing to combine it with the newest payload" in str(error)
    else:
        raise AssertionError("resume should reject a checkpoint without payload provenance")


def test_resume_uses_checkpoint_source_payload_not_latest(monkeypatch):
    source = _source()
    remote = _Remote()
    source_run = "/root/gpufree-data/taili_runs/source_run"
    payload = "/root/gpufree-data/training_payloads/source_payload"
    checkpoint = f"{source_run}/checkpoints/agent_64000.pt"
    launched = {}

    monkeypatch.setattr(source, "_is_running", lambda _remote: False)
    monkeypatch.setattr(source, "_newest_run", lambda _remote: source_run)
    monkeypatch.setattr(source, "_resolve_checkpoint", lambda _remote, _run: checkpoint)
    monkeypatch.setattr(source, "_recorded_payload_root", lambda _remote, _run: payload)
    monkeypatch.setattr(source, "_infer_resume_phase", lambda _remote, _run, _checkpoint: 3)
    monkeypatch.setattr(source, "_remote_timestamp", lambda _remote: "20260719_150000")
    monkeypatch.setattr(source, "_run_boot_id", lambda _remote: "boot-id")
    monkeypatch.setattr(
        source,
        "_latest_payload_root",
        lambda _remote: (_ for _ in ()).throw(AssertionError("latest payload must not be used")),
    )
    monkeypatch.setattr(
        source,
        "_launch_tmux",
        lambda _remote, session, command: launched.update(session=session, command=command),
    )

    run_id, _ = source._start_payload_training(remote, resume=True)

    assert run_id == "taili_train_20260719_150000_resume"
    assert launched["session"] == "rl_train"
    assert f"cd {payload}" in launched["command"]
    assert checkpoint in launched["command"]
    marker_command = remote.commands[-1]
    assert source_run in marker_command
    assert checkpoint in marker_command
