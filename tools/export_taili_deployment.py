"""将 Taili SKRL 检查点导出为可审计的部署 TorchScript。"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml


BODY_DIM = 53
HISTORY_LEN = 25
TICK_DIM = 54
ACTOR_OBS_DIM = BODY_DIM + HISTORY_LEN * TICK_DIM
POLICY_OBS_DIM = 1600
ACTION_DIM = 12
COMMAND_SLICE = slice(6, 9)
GAIT_CLOCK_SLICE = slice(45, 53)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _role_values(value: dict[str, float]) -> list[float]:
    return (
        [float(value["hip"])] * 4
        + [float(value["thigh"])] * 4
        + [float(value["calf"])] * 4
    )


def _joint_role_values(value: object) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * ACTION_DIM
    if not isinstance(value, dict):
        raise TypeError(f"关节参数必须是标量或映射，实际为 {type(value).__name__}")
    result = []
    for role in ("hip", "thigh", "calf"):
        matches = [float(v) for k, v in value.items() if role in str(k)]
        if not matches:
            raise KeyError(f"关节参数缺少 {role} 项: {value}")
        result.extend([matches[0]] * 4)
    return result


def _read_asset_contract(payload: Path) -> dict[str, object]:
    """从资产源码读取 DCMotorCfg 常量，避免离线导出时启动 SimulationApp。"""
    asset_path = payload / "taili_blind_runtime" / "assets" / "taili.py"
    tree = ast.parse(asset_path.read_text(encoding="utf-8"), filename=str(asset_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "TAILI_DOG_CFG" for target in node.targets):
            continue
        if not isinstance(node.value, ast.Call):
            break
        actuators_kw = next((kw for kw in node.value.keywords if kw.arg == "actuators"), None)
        if actuators_kw is None or not isinstance(actuators_kw.value, ast.Dict):
            break
        for key, value in zip(actuators_kw.value.keys, actuators_kw.value.values):
            if ast.literal_eval(key) != "legs" or not isinstance(value, ast.Call):
                continue
            fields = {}
            for keyword in value.keywords:
                if keyword.arg in {"effort_limit", "velocity_limit", "saturation_effort"}:
                    fields[keyword.arg] = ast.literal_eval(keyword.value)
            if set(fields) == {"effort_limit", "velocity_limit", "saturation_effort"}:
                return fields
    raise ValueError(f"无法从资产源码读取完整电机契约: {asset_path}")


class DeploymentPolicy(nn.Module):
    """仅包含部署可见观测、训练归一化器、历史编码器和确定性 actor。"""

    def __init__(
        self,
        history_encoder: nn.Module,
        actor: nn.Module,
        history_scale: torch.Tensor,
        running_mean: torch.Tensor,
        running_variance: torch.Tensor,
        preserve_command_channels: bool = False,
        zero_command_history_encoder: nn.Module | None = None,
        zero_command_actor: nn.Module | None = None,
        exploration_clock_quiet_rms: float = 0.10,
        exploration_clock_moving_rms: float = 0.50,
    ) -> None:
        super().__init__()
        self.history_encoder = history_encoder
        self.actor = actor
        self.zero_command_history_encoder = zero_command_history_encoder
        self.zero_command_actor = zero_command_actor
        self.register_buffer("history_scale", history_scale.detach().float().reshape(()).clone())
        self.register_buffer("running_mean", running_mean[:ACTOR_OBS_DIM].detach().float().clone())
        self.register_buffer(
            "running_variance",
            running_variance[:ACTOR_OBS_DIM].detach().float().clone(),
        )
        self.preserve_command_channels = bool(preserve_command_channels)
        self.exploration_clock_quiet_rms = float(exploration_clock_quiet_rms)
        self.exploration_clock_moving_rms = float(exploration_clock_moving_rms)

    def _motion_gate(self, body: torch.Tensor) -> torch.Tensor:
        clock_rms = torch.sqrt(torch.mean(torch.square(body[:, GAIT_CLOCK_SLICE]), dim=-1))
        ratio = torch.clamp(
            (clock_rms - self.exploration_clock_quiet_rms)
            / (self.exploration_clock_moving_rms - self.exploration_clock_quiet_rms),
            0.0,
            1.0,
        )
        return ratio.square() * (3.0 - 2.0 * ratio)

    def forward(self, raw_observation: torch.Tensor) -> torch.Tensor:
        states = torch.clamp(
            (raw_observation - self.running_mean)
            / (torch.sqrt(self.running_variance) + 1.0e-8),
            min=-5.0,
            max=5.0,
        )
        if self.preserve_command_channels:
            states = states.clone()
            states[:, COMMAND_SLICE] = raw_observation[:, COMMAND_SLICE]
        body = states[:, :BODY_DIM]
        history = states[:, BODY_DIM:].reshape(-1, HISTORY_LEN, TICK_DIM)
        latent = self.history_encoder.encode(history) * self.history_scale
        moving_mean = self.actor.mean(body, latent)
        if self.zero_command_history_encoder is None or self.zero_command_actor is None:
            return moving_mean
        quiet_latent = self.zero_command_history_encoder.encode(history) * self.history_scale
        quiet_mean = self.zero_command_actor.mean(body, quiet_latent)
        gate = self._motion_gate(body).unsqueeze(-1)
        conditioned = quiet_mean + gate * (moving_mean - quiet_mean)
        conditioned = torch.where(gate >= 1.0, moving_mean, conditioned)
        return torch.where(gate <= 0.0, quiet_mean, conditioned)


def _load_runtime(payload: Path):
    sys.path.insert(0, str(payload))
    from taili_blind_runtime.history_actor_policy import HistoryActorPolicy
    from taili_blind_runtime.taili_core import taili_export

    return HistoryActorPolicy, taili_export


def _preserves_command_channels(config: dict) -> bool:
    name = str(
        config.get("skrl", {})
        .get("agent", {})
        .get("state_preprocessor", "")
    ).lower()
    return name in {
        "commandpreservingrunningstandardscaler",
        "command_preserving_running_standard_scaler",
    }


def _build_metadata(
    *,
    checkpoint: Path,
    output: Path,
    config: dict,
    manifest: dict,
    asset_contract: dict[str, object],
    eager_error: float,
    trace_error: float,
    zero_output: list[float],
) -> dict:
    env_cfg = config["env"]
    control = env_cfg["control"]
    actuator = env_cfg["actuator"]
    gait = env_cfg["gait"]
    commands = env_cfg["commands"]
    stiffness = _role_values(actuator["stiffness"])
    damping = _role_values(actuator["damping"])
    effort_limit = _joint_role_values(asset_contract["effort_limit"])
    velocity_limit = _joint_role_values(asset_contract["velocity_limit"])
    transition_enabled = bool(commands.get("transition_enable", False))
    preserve_command_channels = _preserves_command_channels(config)
    policy_cfg = config.get("skrl", {}).get("models", {}).get("policy", {})
    isolated_zero_command_mean = bool(policy_cfg.get("isolate_zero_command_mean", False))
    physics_dt = float(control["dt"])
    decimation = int(control["decimation"])
    return {
        "schema_version": "taili_deployment_contract_v2",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "torchscript_sha256": _sha256(output),
        "input_shape": [1, ACTOR_OBS_DIM],
        "output_shape": [1, ACTION_DIM],
        "normalization": {
            "type": (
                "Taili CommandPreservingRunningStandardScaler"
                if preserve_command_channels
                else "SKRL RunningStandardScaler"
            ),
            "slice": [0, ACTOR_OBS_DIM],
            "epsilon": 1.0e-8,
            "clip": 5.0,
            "passthrough_slices": [[6, 9]] if preserve_command_channels else [],
        },
        "observation": {
            "body_dim": BODY_DIM,
            "history_len": HISTORY_LEN,
            "history_tick_dim": TICK_DIM,
            "history_order": "newest_first",
            "history_layout": "[current_tick, t-1, ..., oldest_tick]",
            "history_update_interval_policy_steps": 1,
            "training_policy_tensor_dim": POLICY_OBS_DIM,
            "privileged_tail_dim": POLICY_OBS_DIM - ACTOR_OBS_DIM,
            "body_layout": manifest["layouts"]["body53"],
            "tick_layout": manifest["layouts"]["tick54"],
        },
        "timing": {
            "physics_dt": physics_dt,
            "decimation": decimation,
            "policy_dt": physics_dt * decimation,
            "action_delay_policy_steps": int(control["action_delay_steps"]),
            "action_delay_semantics": "previous_policy_action_for_full_control_period",
        },
        "command": {
            "frame": "body [vx_forward, vy_left, wz_yaw]",
            "transition_enabled": transition_enabled,
            "smoothing_active": transition_enabled,
            "configured_smooth_alpha": float(commands.get("smooth_alpha", 0.0)),
            "runtime_recipe": (
                "command_effective = command_target"
                if not transition_enabled
                else "legacy transition branch enabled"
            ),
            "isolated_zero_command_mean": isolated_zero_command_mean,
            "mean_gate": "standardized masked gait-clock RMS smoothstep",
        },
        "control": {
            "joint_order": manifest["joint_order"]["canonical"],
            "q_default": manifest["constants"]["q_default"],
            "action_scale": float(control["action_scale"]),
            "action_lower": [float(x) for x in control["action_lower_by_joint"]],
            "action_upper": [float(x) for x in control["action_upper_by_joint"]],
            "stiffness": stiffness,
            "damping": damping,
            "effort_limit": effort_limit,
            "velocity_limit": velocity_limit,
            "saturation_effort": float(asset_contract["saturation_effort"]),
            "target_recipe": "q_des = q_default + action_scale * clip(action_raw)",
        },
        "gait": {
            "period": float(gait["period"]),
            "period_min": float(gait["period_min"]),
            "period_slope": float(gait["period_slope"]),
            "yaw_speed_equiv": float(gait["yaw_speed_equiv"]),
            "offsets": [0.0, 0.5, 0.5, 0.0],
        },
        "frames": manifest["frames"],
        "validation": {
            "max_wrapper_vs_skrl_error": eager_error,
            "max_torchscript_vs_wrapper_error": trace_error,
            "zero_input_output": zero_output,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--effective-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = args.payload.resolve()
    checkpoint_path = args.checkpoint.resolve()
    config_path = args.effective_config.resolve()
    output_path = args.output.resolve()
    for path in (payload, checkpoint_path, config_path):
        if not path.exists():
            raise FileNotFoundError(path)

    os.environ.pop("TAILI_NO_EQUIV", None)
    HistoryActorPolicy, taili_export = _load_runtime(payload)
    asset_contract = _read_asset_contract(payload)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    preserve_command_channels = _preserves_command_channels(config)
    policy_cfg = config.get("skrl", {}).get("models", {}).get("policy", {})
    isolate_zero_command_mean = bool(policy_cfg.get("isolate_zero_command_mean", False))
    from gymnasium.spaces import Box

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy_state = checkpoint["policy"]
    scaler_state = checkpoint["state_preprocessor"]
    observation_space = Box(-np.inf, np.inf, shape=(POLICY_OBS_DIM,), dtype=np.float32)
    action_space = Box(-np.inf, np.inf, shape=(ACTION_DIM,), dtype=np.float32)
    policy = HistoryActorPolicy(
        observation_space,
        action_space,
        "cpu",
        clip_actions=False,
        actor_hidden=(1024, 512),
        history_ramp_steps=0,
        isolate_zero_command_mean=isolate_zero_command_mean,
        exploration_clock_quiet_rms=float(policy_cfg.get("exploration_clock_quiet_rms", 0.10)),
        exploration_clock_moving_rms=float(policy_cfg.get("exploration_clock_moving_rms", 0.50)),
    ).eval()
    policy.load_state_dict(policy_state, strict=True)

    wrapper = DeploymentPolicy(
        copy.deepcopy(policy.history_encoder),
        copy.deepcopy(policy.actor),
        policy.history_scale,
        scaler_state["running_mean"],
        scaler_state["running_variance"],
        preserve_command_channels=preserve_command_channels,
        zero_command_history_encoder=(
            copy.deepcopy(policy.zero_command_history_encoder)
            if isolate_zero_command_mean
            else None
        ),
        zero_command_actor=(
            copy.deepcopy(policy.zero_command_actor)
            if isolate_zero_command_mean
            else None
        ),
        exploration_clock_quiet_rms=policy.exploration_clock_quiet_rms,
        exploration_clock_moving_rms=policy.exploration_clock_moving_rms,
    ).eval()

    generator = torch.Generator(device="cpu").manual_seed(20260723)
    mean = scaler_state["running_mean"].float()
    std = torch.sqrt(scaler_state["running_variance"].float()).clamp_min(1.0e-4)
    raw_full = mean.unsqueeze(0) + torch.randn(
        (16, POLICY_OBS_DIM), generator=generator
    ) * std.unsqueeze(0) * 0.75
    scaled_full = torch.clamp((raw_full - mean) / (std + 1.0e-8), -5.0, 5.0)
    if preserve_command_channels:
        scaled_full = scaled_full.clone()
        scaled_full[:, COMMAND_SLICE] = raw_full[:, COMMAND_SLICE]
    with torch.inference_mode():
        expected, _, _ = policy.compute({"states": scaled_full}, role="policy")
        actual = wrapper(raw_full[:, :ACTOR_OBS_DIM])
    eager_error = float((expected - actual).abs().max())
    if eager_error > 1.0e-6:
        raise RuntimeError(f"部署包装与 SKRL eager 输出不一致: {eager_error:.9g}")

    example = torch.zeros((1, ACTOR_OBS_DIM), dtype=torch.float32)
    traced = torch.jit.trace(wrapper, example, check_trace=True, strict=True)
    traced = torch.jit.freeze(traced.eval())
    with torch.inference_mode():
        traced_actual = traced(raw_full[:, :ACTOR_OBS_DIM])
        zero_output = traced(example).squeeze(0).tolist()
    trace_error = float((actual - traced_actual).abs().max())
    if trace_error > 1.0e-6:
        raise RuntimeError(f"TorchScript 与 eager 输出不一致: {trace_error:.9g}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(output_path))
    metadata = _build_metadata(
        checkpoint=checkpoint_path,
        output=output_path,
        config=config,
        manifest=taili_export.build_export_manifest(),
        asset_contract=asset_contract,
        eager_error=eager_error,
        trace_error=trace_error,
        zero_output=zero_output,
    )
    # Export the same resolved contract consumed by sim2sim parity checks.
    from autotuner.research.policy_contract import contract_from_export_metadata

    metadata["policy_contract"] = contract_from_export_metadata(
        metadata,
        contract_id=f"deployment:{checkpoint_path.name}",
        version="taili_deployment_contract_v2",
    ).model_dump(mode="json")
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"policy={output_path}")
    print(f"metadata={metadata_path}")
    print(f"eager_error={eager_error:.9g} trace_error={trace_error:.9g}")


if __name__ == "__main__":
    main()
