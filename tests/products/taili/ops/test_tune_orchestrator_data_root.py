"""换机器只改配置：调参编排拼出来的远端命令里，数据根目录必须来自配置/环境。

`TuningDriver` 自己的命令都走 `self.data_root`，但 `produce_policy` 是模块级函数，
起步读远端 BEST_CHECKPOINT 登记表时曾经写死 `/root/gpufree-data`。这里断言这条命令
跟着传入/配置的根目录走。

没有覆盖的：真正连远端、真的把命令跑起来——这些用例全部用远端替身，只检查命令文本。
"""

import pytest

from products.taili.ops import tune_orchestrator as TO

CONFIGURED = "/home/chuan/robot_lab/taili_data"


class _Remote:
    """只记录命令的远端替身；`exec_out` 一律回空，把 `produce_policy` 顶到 SystemExit。"""

    def __init__(self):
        self.commands = []
        self.closed = False

    def exec_out(self, command, timeout=None):
        self.commands.append(command)
        return ""

    def close(self):
        self.closed = True


@pytest.fixture
def registry_probe(monkeypatch):
    """把 `_ssh_from_json` 换成替身，返回喂给 `produce_policy` 的记录器。"""
    remote = _Remote()
    monkeypatch.setattr(
        "products.taili.ops.acceptance_run._ssh_from_json", lambda _path: remote
    )
    return remote


def test_env_var_wins_over_the_configured_root(monkeypatch):
    monkeypatch.setenv("TAILI_DATA_ROOT", CONFIGURED)

    assert TO._default_data_root() == CONFIGURED


@pytest.mark.parametrize("declared", ["", "relative/root", "/root/../etc", " /root/../etc "])
def test_unsafe_override_falls_back_to_the_legacy_root(monkeypatch, declared):
    monkeypatch.setenv("TAILI_DATA_ROOT", declared)

    assert TO._default_data_root() == TO.FALLBACK_DATA_ROOT


def test_produce_policy_bootstraps_under_the_configured_root(monkeypatch, registry_probe):
    monkeypatch.setenv("TAILI_DATA_ROOT", CONFIGURED)

    with pytest.raises(SystemExit, match="no BEST_CHECKPOINT registry"):
        TO.produce_policy(ssh_json="ignored.json", log=lambda *_: None)

    assert registry_probe.commands == [
        f"cat {CONFIGURED}/taili_runs/BEST_CHECKPOINT.json 2>/dev/null"
    ]
    assert TO.FALLBACK_DATA_ROOT not in registry_probe.commands[0]
    assert registry_probe.closed


def test_explicit_data_root_wins_over_the_env(monkeypatch, registry_probe):
    monkeypatch.setenv("TAILI_DATA_ROOT", "/somewhere/else")
    explicit = "/home/chuan/robot_lab/other_data"

    with pytest.raises(SystemExit, match="no BEST_CHECKPOINT registry"):
        TO.produce_policy(ssh_json="ignored.json", data_root=explicit, log=lambda *_: None)

    assert f"cat {explicit}/taili_runs/BEST_CHECKPOINT.json 2>/dev/null" in registry_probe.commands


def test_driver_defaults_to_the_configured_root_and_takes_overrides(monkeypatch, registry_probe):
    monkeypatch.setenv("TAILI_DATA_ROOT", CONFIGURED)

    assert TO.TuningDriver(ssh_json="ignored.json", log=lambda *_: None).data_root == CONFIGURED
    assert (
        TO.TuningDriver(ssh_json="ignored.json", data_root="/tmp/scratch", log=lambda *_: None).data_root
        == "/tmp/scratch"
    )
