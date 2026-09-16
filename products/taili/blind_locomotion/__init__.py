"""Taili blind locomotion runtime package.

The package registers its own Gymnasium task entry points. It does not import or
overwrite robot_lab task modules; remote deployment only needs this package on
PYTHONPATH plus the IsaacLab/skrl/PyTorch environment and a training entry point.
"""
from __future__ import annotations

import os

from . import _torchvision_pair
from .taili_blind_config import CONFIG_FILENAME

TASK_IDS = (
    "RobotLab-Isaac-Taili-Blind-Direct-v0",
    "RobotLab-Isaac-Taili-AMP-Blind-Direct-v0",
)

try:  # sim-only dependencies are absent in local CPU tests
    import gymnasium as gym

    _HAVE_SIM = True
except Exception:
    _HAVE_SIM = False


def _register_blind_locomotion():
    try:
        # 必须先于 skrl：skrl 会 `import torch`，而 Isaac Sim 启动时又会把自带的
        # torchvision（不同 CUDA 构建）接进来，两边一混就报 `torchvision::nms does not exist`。
        # 详见 _torchvision_pair 的模块文档串。钉不住就按原样跑，不因此放弃注册。
        try:
            _torchvision_pair.pin_torchvision()
        except Exception:  # noqa: BLE001 - 钉不住不阻断，真出错由 skrl/训练侧暴露
            pass
        from skrl.utils.runner.torch import Runner
        from .terrain_perceiver_policy import terrain_perceiver_gaussian_model, terrain_perceiver_model
        from .terrain_perceiver_aux_patch import patch_amp_terrain_aux
    except Exception:
        return

    patch_amp_terrain_aux()
    if getattr(Runner, "_taili_blind_locomotion_registered", False):
        return
    prev = Runner._component

    def _component(self, name):
        n = str(name).lower()
        if n in {"terrainperceiverpolicy", "terrain_perceiver_policy"}:
            return terrain_perceiver_gaussian_model
        if n in {"terrainperceivermodel", "terrain_perceiver_model"}:
            return terrain_perceiver_model
        return prev(self, name)

    Runner._component = _component
    Runner._taili_blind_locomotion_registered = True


def _register_gym_tasks():
    package = __name__
    skrl_cfg_entry_point = os.environ.get("TAILI_SKRL_CFG_ENTRY_POINT") or f"{package}:{CONFIG_FILENAME}"
    kwargs = {
        "env_cfg_entry_point": f"{package}.taili_blind_env_cfg:TailiBlindEnvCfg",
        "skrl_amp_cfg_entry_point": skrl_cfg_entry_point,
    }
    for task_id in TASK_IDS:
        try:
            if task_id in gym.registry:
                del gym.registry[task_id]
        except Exception:
            pass
        gym.register(
            id=task_id,
            entry_point=f"{package}.blind_tp_env:TailiBlindTPEnv",
            disable_env_checker=True,
            kwargs=kwargs,
        )


if _HAVE_SIM:
    _register_blind_locomotion()
    _register_gym_tasks()
