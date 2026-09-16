"""换机器只改配置：`df -h` 的挂载点必须来自部署声明，不能写死旧机器的路径。

本文件覆盖两段链路，两段都断言了：
    1. `_data_root()` 从 `_deployment_config()` 取 `data_root`，取值非法时退回旧路径；
    2. 远端真的收到的那条命令用的是这个值（探针命令与 `raw["commands"]` 里的记录是同一个字符串）。

没有覆盖的链路：`config/products/taili.yaml` → `_deployment_config()` 这一段由产品适配器的
合同测试保证，这里只做 `_deployment_config()` → 命令这一段。两段合起来才是"改 yaml 就换机器"。
"""

import pathlib
import re

import pytest

from autotuner.locomotion_console.config import LocomotionConsoleSettings
from autotuner.locomotion_console.datasource import RealDataSource

FALLBACK = "/root/gpufree-data"
PRODUCT_CONFIG = pathlib.Path(__file__).resolve().parents[3] / "config" / "products" / "taili.yaml"


class _Remote:
    """只记录命令、一律回空的远端替身（够驱动 `_fetch_remote_machine_status`）。"""

    def __init__(self):
        self.commands = []

    def exec_out(self, command, timeout=None):
        self.commands.append(command)
        return ""


def _source(monkeypatch, data_root):
    """造一个部署配置里 `data_root` 被改成 data_root 的数据源（None 表示这一项缺失）。"""
    source = RealDataSource(LocomotionConsoleSettings(source="real"))
    deployment = dict(source._deployment_config())
    if data_root is None:
        deployment.pop("data_root", None)
    else:
        deployment["data_root"] = data_root
    monkeypatch.setattr(source, "_deployment_config", lambda: deployment)
    return source


def _disk_commands(commands):
    return [command for command in commands if command.startswith("df -h ")]


def test_configured_data_root_drives_the_df_command(monkeypatch):
    source = _source(monkeypatch, "/home/chuan/robot_lab/taili_data")

    assert source._disk_probe_command() == "df -h /home/chuan/robot_lab/taili_data /root /tmp"


def test_the_sent_command_uses_the_configured_root(monkeypatch):
    source = _source(monkeypatch, "/home/chuan/robot_lab/taili_data")
    remote = _Remote()

    status = source._fetch_remote_machine_status(remote)

    sent = _disk_commands(remote.commands)
    assert len(sent) == 1
    assert sent[0].startswith("df -h /home/chuan/robot_lab/taili_data /root /tmp")
    assert FALLBACK not in sent[0]
    # 回报给前端的命令清单与实际发出的必须是同一条，否则前端展示的是另一台机器的路径。
    assert source._disk_probe_command() in status.raw["commands"]


def test_trailing_slash_is_normalised_once(monkeypatch):
    source = _source(monkeypatch, "/home/chuan/robot_lab/taili_data/")

    assert source._disk_probe_command() == "df -h /home/chuan/robot_lab/taili_data /root /tmp"


@pytest.mark.parametrize("declared", [None, "", "relative/root", "/root/../etc", " /root/../etc "])
def test_missing_or_unsafe_data_root_falls_back(monkeypatch, declared):
    source = _source(monkeypatch, declared)

    assert source._disk_probe_command() == f"df -h {FALLBACK} /root /tmp"


def test_shipped_config_declares_the_root_the_code_will_use(monkeypatch):
    """锚住"改这一行"：不改配置时，代码用的就是 yaml 里那一行的值。"""
    matches = re.findall(
        r"^\s+data_root:\s*(\S+)\s*$",
        PRODUCT_CONFIG.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )

    assert len(matches) == 1, f"部署配置里应当只有一处 data_root，实际 {len(matches)} 处"
    source = RealDataSource(LocomotionConsoleSettings(source="real"))
    assert source._data_root() == matches[0].rstrip("/")
