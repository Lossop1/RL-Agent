"""Safety behavior of the acceptance-measurement CLI: it must refuse to contend with an active
training run, and must reject a run id that could reach the remote shell."""
import pytest

from autotuner.training import acceptance_run as AR


class _FakeRemote:
    def __init__(self, busy: bool):
        self._busy = busy
        self.closed = False

    def exec_out(self, cmd, *a, **k):
        if "pgrep -f train_taili" in cmd:
            return "55123\n" if self._busy else ""
        return ""

    def exec(self, cmd, *a, **k):
        return ("", "")

    def close(self):
        self.closed = True


def test_refuses_when_training_active(monkeypatch):
    fake = _FakeRemote(busy=True)
    monkeypatch.setattr(AR, "_ssh_from_json", lambda _p: fake)
    with pytest.raises(SystemExit) as ei:
        AR.run_acceptance("newest", ["flat"], "agent_50000.pt", 64, "config/ssh.json",
                          force=False, python="py", data_root="/root/gpufree-data")
    assert "ACTIVE" in str(ei.value)
    assert fake.closed, "the SSH connection must be closed even on the guard path"


def test_force_bypasses_busy_guard(monkeypatch):
    # with force=True it proceeds past the busy check; run resolution then fails cleanly (empty dir),
    # proving the guard itself did not fire.
    fake = _FakeRemote(busy=True)
    monkeypatch.setattr(AR, "_ssh_from_json", lambda _p: fake)
    with pytest.raises(SystemExit) as ei:
        AR.run_acceptance("newest", ["flat"], "agent.pt", 64, "config/ssh.json",
                          force=True, python="py", data_root="/root/gpufree-data")
    assert "ACTIVE" not in str(ei.value)


@pytest.mark.parametrize("bad", ["a; rm -rf /", "$(whoami)", "a && b", "../etc", "a|b"])
def test_resolve_run_rejects_injection(bad):
    with pytest.raises(SystemExit) as ei:
        AR._resolve_run(_FakeRemote(busy=False), bad, "/root/gpufree-data")
    assert "invalid run id" in str(ei.value)


def test_resolve_run_accepts_normal_id():
    got = AR._resolve_run(_FakeRemote(busy=False), "tune1_20260705_010817", "/root/gpufree-data")
    assert got == "/root/gpufree-data/taili_runs/tune1_20260705_010817"


def test_best_registry_updates_only_on_improvement():
    """BEST_CHECKPOINT.json must move only when the measured score improves — 'best', not 'newest'."""
    class _Reg:
        def __init__(self, prev):
            import json
            self._raw = json.dumps({"score": prev}) if prev is not None else ""
            self.wrote = None
        def exec_out(self, cmd, *a, **k):
            if cmd.startswith("cat "):
                return self._raw
            self.wrote = cmd
            return ""
    verdict = {"run": "/data/taili_runs/r1", "checkpoint": "agent_15000.pt",
               "gates": {f"G{i}": {"ok": True, "detail": ""} for i in range(10)}}
    worse = _Reg(prev=12)
    assert AR._maybe_update_best_registry(worse, "/data", verdict) is False
    assert worse.wrote is None                              # 10 <= 12: no write
    better = _Reg(prev=7)
    assert AR._maybe_update_best_registry(better, "/data", verdict) is True
    assert "agent_15000.pt" in better.wrote                 # 10 > 7: registry updated
    empty = _Reg(prev=None)
    assert AR._maybe_update_best_registry(empty, "/data", verdict) is True  # first-ever write


def test_action_run_acceptance_allowlists_inputs():
    """The copilot's measure action reaches a subprocess argv — inputs must be allowlisted and no
    shell metacharacters can survive into the spawned command."""
    import asyncio
    import os
    from unittest.mock import patch

    os.environ.setdefault("LOCOMOTION_CONSOLE_SOURCE", "fake")
    from autotuner.locomotion_console.config import get_settings
    from autotuner.locomotion_console.datasource import RealDataSource

    src = RealDataSource(get_settings())

    async def go():
        with patch("subprocess.Popen") as P:
            bad = await src.action_run_acceptance(checkpoint="a; rm -rf /")
            assert bad.ok is False and not P.called          # rejected before any spawn
            P.reset_mock()
            good = await src.action_run_acceptance(run="tune1c_x", terrains="flat rough",
                                                   checkpoint="agent_20000.pt")
            assert good.ok is True and P.call_count == 1
            argv = P.call_args[0][0]
            assert "flat" in argv and "rough" in argv        # allowlisted terrains threaded through
            assert all(";" not in a and "$" not in a and "|" not in a for a in argv)

    asyncio.run(go())
