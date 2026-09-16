"""P4.2 checkpoint_hook 的安装判据测试。

这里钉住的是一个**静默失效**：环境在 ``__init__`` 里就把
``self._checkpoint_integration`` 置成 ``None``（``blind_tp_env.py:179``），
而安装函数原先用 ``if not hasattr(env, "_checkpoint_integration")`` 做判据。
属性存在、值是 None，于是 hasattr 为真、集成器永远不建，
注册/清理/清单导出全部失效，连 "installed" 那行日志都不打——
2026-09-16 在 3060 容器里跑到 complete 的 fixstep_n4 日志里一条 CheckpointHook 都没有。

所以这些用例里替身环境的 ``_checkpoint_integration`` 必须**预置为 None**，
而不是用裸类：裸类会掩盖这个缺陷（第一版探针就是这么假阳性的）。
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from products.taili.blind_locomotion.checkpoint_hook import (
    finalize_checkpoint_management,
    install_checkpoint_hook,
)


class _Agent:
    """够用的 skrl agent 替身：save / write_checkpoint / timestep / experiment_dir。

    ``write_checkpoint`` 是 skrl 1.4.3 训练循环真正调用的那个
    （``agents/torch/base.py:677``），文件名照抄它的 ``agent_<timestep>.pt``。
    ``checkpoint_store_separately=False`` 时就是这一个文件。
    """

    timestep = 7

    def __init__(self, experiment_dir=None, write_dir=None) -> None:
        self.saved: list[str] = []
        self.experiment_dir = str(experiment_dir) if experiment_dir else ""
        # 这个替身直接写进 write_dir（真实 skrl 写的是 experiment_dir/checkpoints）。
        # 单独可调，是为了构造"agent 没报告 experiment_dir"时钩子退回安装目录的情形。
        self.write_dir = Path(write_dir) if write_dir else Path(experiment_dir or ".") / "checkpoints"
        self.write_checkpoint_calls: list[tuple[int, int]] = []

    def write_checkpoint(self, timestep, timesteps):
        self.write_checkpoint_calls.append((timestep, timesteps))
        self.write_dir.mkdir(parents=True, exist_ok=True)
        (self.write_dir / f"agent_{timestep}.pt").write_text("skrl-checkpoint", encoding="utf-8")

    def save(self, path, *args, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("fake-checkpoint", encoding="utf-8")
        self.saved.append(str(path))
        return "saved-ok"


class _Env:
    """对齐真实环境：属性**存在但为 None**。"""

    def __init__(self) -> None:
        self._checkpoint_integration = None


def _snapshot() -> dict:
    return {
        "reward_mean": 1.0,
        "terminal_rate": 0.1,
        "episode_length_mean": 20.0,
        "curriculum_phase": 0,
        "checkpoint_mtime": time.time(),
    }


def test_install_creates_the_integration_when_the_attribute_exists_but_is_none(capsys):
    """回归：环境把该属性置成 None 时，仍然要把集成器建起来。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent, env = _Agent(), _Env()
        install_checkpoint_hook(agent, env, tmpdir)

        assert env._checkpoint_integration is not None
        assert "installed checkpoint management" in capsys.readouterr().out


def test_a_saved_checkpoint_is_registered_through_the_hook(capsys):
    """钩子装好之后，agent.save() 要真的把检查点登记进 registry。

    "没报错"不算数：save 里的 except 会吞异常，所以直接读登记条数。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        agent, env = _Agent(), _Env()
        install_checkpoint_hook(agent, env, tmpdir)
        env._latest_performance_snapshot = _snapshot()

        assert agent.save(str(Path(tmpdir) / "agent_7.pt")) == "saved-ok"
        capsys.readouterr()

        registry = env._checkpoint_integration.registry
        assert len(registry._registry) == 1


def test_finalize_exports_a_manifest(capsys):
    """收尾要导出清单——原来这条路径也一并被静默跳过了。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent, env = _Agent(), _Env()
        install_checkpoint_hook(agent, env, tmpdir)
        env._latest_performance_snapshot = _snapshot()
        agent.save(str(Path(tmpdir) / "agent_7.pt"))
        capsys.readouterr()

        finalize_checkpoint_management(env)

        assert (Path(tmpdir) / "checkpoint_manifest.json").is_file()
        assert "finalized checkpoint management" in capsys.readouterr().out


def test_installing_twice_does_not_clobber_existing_registrations(capsys):
    """重复安装要保持幂等：不能把已经登记过的集成器换掉。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent, env = _Agent(), _Env()
        install_checkpoint_hook(agent, env, tmpdir)
        env._latest_performance_snapshot = _snapshot()
        agent.save(str(Path(tmpdir) / "agent_7.pt"))
        first = env._checkpoint_integration
        capsys.readouterr()

        install_checkpoint_hook(agent, env, tmpdir)

        assert env._checkpoint_integration is first
        assert "installed checkpoint management" not in capsys.readouterr().out
        assert len(env._checkpoint_integration.registry._registry) == 1


def test_write_checkpoint_registers_the_file_skrl_actually_wrote(capsys):
    """回归：拦的必须是 skrl 训练循环调用的 write_checkpoint。

    hooksave_n4（2026-09-16，576→240 步、间隔 200）磁盘上写出了
    46 MB 的 agent_200.pt，registry 里却是 0 条：因为钩子包的是 save()，
    而训练循环调的是 write_checkpoint()。这条用例钉的就是后者。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        agent, env = _Agent(experiment_dir=tmpdir), _Env()
        install_checkpoint_hook(agent, env, tmpdir)
        env._latest_performance_snapshot = _snapshot()
        capsys.readouterr()

        agent.write_checkpoint(200, 240)

        assert agent.write_checkpoint_calls == [(200, 240)]
        registry = env._checkpoint_integration.registry
        assert len(registry._registry) == 1
        registered = next(iter(registry._registry))
        assert registered.endswith("agent_200.pt")
        assert Path(registered).is_file()


def test_write_checkpoint_records_the_real_timestep(capsys):
    """步数取 write_checkpoint 的入参，不是 agent.timestep。

    agent.timestep 是上一轮交互的计数，和落盘文件名里的 timestep 差一，
    用它会让 registry 的 step 与文件名对不上。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        agent, env = _Agent(experiment_dir=tmpdir), _Env()
        agent.timestep = 999  # 故意和落盘步数不同
        install_checkpoint_hook(agent, env, tmpdir)
        env._latest_performance_snapshot = _snapshot()
        capsys.readouterr()

        agent.write_checkpoint(400, 480)

        entry = next(iter(env._checkpoint_integration.registry._registry.values()))
        assert entry.step == 400


def test_write_checkpoint_without_a_snapshot_registers_nothing(capsys):
    """环境还没跑出快照时不能登记——宁可为空，也不要一条假数据。

    两次落盘连写：第一次没快照（跳过），第二次有快照（登记）。
    断言只登记了第二条，既钉住"跳过"，也钉住"前后比目录不会把旧文件补登一遍"。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        agent, env = _Agent(experiment_dir=tmpdir), _Env()
        install_checkpoint_hook(agent, env, tmpdir)
        capsys.readouterr()

        agent.write_checkpoint(200, 240)
        env._latest_performance_snapshot = _snapshot()
        agent.write_checkpoint(400, 480)

        registered = [Path(p).name for p in env._checkpoint_integration.registry._registry]
        assert registered == ["agent_400.pt"]


class _AgentWithoutWriteCheckpoint:
    """模拟 skrl 版本不符：没有 write_checkpoint 的 agent。"""

    timestep = 0

    def save(self, path, *args, **kwargs):
        return "saved-ok"


def test_a_missing_write_checkpoint_warns_instead_of_raising(capsys):
    """不能抛：调用点会把异常吞成一行日志，等于又静默失效一次。

    训练可以照跑，但"检查点不会被登记"这件事必须打出来。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        env = _Env()
        install_checkpoint_hook(_AgentWithoutWriteCheckpoint(), env, tmpdir)

        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "write_checkpoint" in out
        # 集成器还是建起来了：save() 那一支仍然可用。
        assert env._checkpoint_integration is not None
    # 没抛异常 = 训练不会被打断。


def test_write_checkpoint_falls_back_to_the_installed_dir(capsys):
    """agent 没报告 experiment_dir 时，退回安装时给的目录，而不是静默不登记。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # experiment_dir 为空，落盘仍发生在 tmpdir —— 与安装目录一致。
        agent, env = _Agent(write_dir=tmpdir), _Env()
        install_checkpoint_hook(agent, env, tmpdir)
        env._latest_performance_snapshot = _snapshot()
        capsys.readouterr()

        agent.write_checkpoint(200, 240)

        assert len(env._checkpoint_integration.registry._registry) == 1
