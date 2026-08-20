"""Read-only reward and scalar-critic audit for a live Taili checkpoint.

The probe restores the checkpoint's terrain sidecar, runs the packaged task with
deterministic mean actions, and separates transitions into the sample buckets
that share one actor/critic during training. It never writes training state.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
from pathlib import Path
import traceback
from typing import Any


BUCKET_NAMES = (
    "flat_moving",
    "flat_stop",
    "stairs_down_frontier",
    "stairs_down_replay",
    "stairs_up_frontier",
    "stairs_up_replay",
)

EPISODE_OUTCOME_NAMES = (
    "stairs_down_success",
    "stairs_down_failure",
    "stairs_down_other",
    "stairs_up_success",
    "stairs_up_failure",
    "stairs_up_other",
)

COUNTERFACTUAL_OUTCOME_COMPONENT_NAMES = (
    "total",
    "tracking_lin",
    "tracking_lin_far",
    "linear_progress",
    "terrain_progress",
    "terrain_supported_progress",
    "terrain_post_contact_progress",
    "terrain_direction",
    "terrain_clearance",
    "terrain_body_height",
    "terrain_limb_collision",
    "terrain_foot_collision",
    "terrain_support_continuity_cost",
    "terrain_down_overspeed",
    "terrain_down_vz",
    "terrain_down_support",
    "terrain_down_touchdown",
    "terrain_capability_drive_scale",
    "terrain_overspeed_gate",
    "terrain_progress_overspeed_gate",
    "terrain_safe_progress_quality",
    "terrain_direction_quality",
    "terrain_course_quality",
    "terrain_post_contact_quality",
    "terrain_support_specific_quality",
    "terrain_supported_progress_quality",
    "terrain_support_load_window_quality",
    "terrain_support_continuity_quality",
    "terrain_support_total_load",
    "terrain_support_front_load",
    "terrain_support_rear_load",
    "terrain_support_left_load",
    "terrain_support_right_load",
    "terrain_effective_support_quality",
    "terrain_load_balance_quality",
    "terrain_touchdown_load_quality",
    "terrain_support_load_quality",
    "support_integrity",
    "stance_slip",
    "worst_leg_slip",
    "touchdown_slip",
    "landing_impact",
    "touchdown_force_rate",
    "clearance_under",
    "clearance_over",
    "base_vz",
    "base_wxy",
    "cycle_wxy_bias",
    "action_rate",
    "swing_high_frequency",
    "terminal_swing_velocity",
    "torque_margin",
    "torque_saturation",
)


def _reward_config_overrides(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Return only candidate reward fields that differ from the payload baseline."""
    return {
        key: value
        for key, value in candidate.items()
        if key not in baseline or baseline[key] != value
    }


def _counterfactual_cfg_from_runtime(
    config_type: type,
    runtime_cfg: Any,
    overrides: dict[str, Any],
) -> tuple[Any, list[str], list[str]]:
    """Clone effective runtime fields into the candidate config, then apply payload diffs."""
    candidate_cfg = config_type()
    copied: list[str] = []
    for field in dataclasses.fields(candidate_cfg):
        if hasattr(runtime_cfg, field.name):
            setattr(
                candidate_cfg,
                field.name,
                copy.deepcopy(getattr(runtime_cfg, field.name)),
            )
            copied.append(field.name)
    applied: list[str] = []
    for key, value in overrides.items():
        if hasattr(candidate_cfg, key):
            setattr(candidate_cfg, key, copy.deepcopy(value))
            applied.append(key)
    return candidate_cfg, copied, applied


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Taili rewards by training sample bucket")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--task", default="RobotLab-Isaac-Taili-AMP-Blind-Direct-v0")
    parser.add_argument("--agent-yaml", default="")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument(
        "--stochastic-actions",
        action="store_true",
        help="Use sampled policy actions instead of deterministic mean actions",
    )
    parser.add_argument("--recovery-horizon-s", type=float, default=1.0)
    parser.add_argument(
        "--continuous-quality-taus",
        default="0.50,0.65,0.80",
        help="Comma-separated history decay constants for the read-only supported-progress audit",
    )
    parser.add_argument(
        "--continuous-quality-floors",
        default="0.30,0.35,0.40",
        help="Comma-separated recovery floors for the read-only supported-progress audit",
    )
    parser.add_argument(
        "--force-stair-level",
        type=int,
        default=-1,
        help="Fix all stair environments at this level and disable probe-side curriculum/replay",
    )
    parser.add_argument(
        "--counterfactual-reward-module",
        default="",
        help="Optional taili_reward.py evaluated on the exact same physical transitions",
    )
    parser.add_argument(
        "--counterfactual-reward-config",
        default="",
        help="YAML whose reward section configures the counterfactual RewardConfig",
    )
    parser.add_argument(
        "--counterfactual-base-config",
        default="",
        help=(
            "Baseline payload YAML. When set, only reward fields that differ in the "
            "candidate YAML override the effective runtime RewardConfig"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    return parser


def _load_modules(agent: Any, checkpoint_path: str) -> list[str]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = (
        "policy",
        "value",
        "discriminator",
        "state_preprocessor",
        "value_preprocessor",
        "amp_state_preprocessor",
    )
    loaded: list[str] = []
    for name in required:
        if name not in checkpoint:
            raise KeyError(f"checkpoint is missing module: {name}")
        module = agent.checkpoint_modules.get(name)
        if module is None or not hasattr(module, "load_state_dict"):
            raise KeyError(f"agent is missing checkpoint module: {name}")
        module.load_state_dict(checkpoint[name])
        if hasattr(module, "eval"):
            module.eval()
        loaded.append(name)
    return loaded


def _state_tensor(observation: Any) -> Any:
    if isinstance(observation, dict):
        for key in ("states", "policy"):
            if key in observation:
                return observation[key]
    return observation


def _critic_value(agent: Any, observation: Any) -> Any:
    import torch

    states = _state_tensor(observation)
    with torch.inference_mode():
        values, _, _ = agent.value.act(
            {"states": agent._state_preprocessor(states)}, role="value"
        )
        values = agent._value_preprocessor(values, inverse=True)
    return values.reshape(values.shape[0], -1)


def _style_reward(agent: Any, amp_states: Any) -> Any:
    import torch

    with torch.inference_mode():
        logits, _, _ = agent.discriminator.act(
            {"states": agent._amp_state_preprocessor(amp_states)},
            role="discriminator",
        )
        return -torch.log(torch.clamp(1.0 - torch.sigmoid(logits), min=1e-4)).reshape(-1) * float(
            agent._discriminator_reward_scale
        )


class _BucketAccumulator:
    def __init__(self, names: list[str], device: Any) -> None:
        import torch

        width = len(names)
        self.names = names
        self.count = torch.zeros((), dtype=torch.float64, device=device)
        self.steps = torch.zeros((), dtype=torch.float64, device=device)
        self.total = torch.zeros(width, dtype=torch.float64, device=device)
        self.square = torch.zeros(width, dtype=torch.float64, device=device)
        self.minimum = torch.full((width,), float("inf"), dtype=torch.float64, device=device)
        self.maximum = torch.full((width,), float("-inf"), dtype=torch.float64, device=device)
        self.positive = torch.zeros(width, dtype=torch.float64, device=device)

    def add(self, matrix: Any, mask: Any) -> None:
        import torch

        selected = matrix[mask]
        if selected.numel() == 0:
            return
        selected = torch.nan_to_num(
            selected.to(dtype=torch.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        self.count += selected.shape[0]
        self.steps += 1.0
        self.total += selected.sum(dim=0)
        self.square += selected.square().sum(dim=0)
        self.minimum = torch.minimum(self.minimum, selected.min(dim=0).values)
        self.maximum = torch.maximum(self.maximum, selected.max(dim=0).values)
        self.positive += (selected > 0).to(torch.float64).sum(dim=0)

    def report(self) -> dict[str, Any]:
        import torch

        count = int(self.count.item())
        if count <= 0:
            return {"samples": 0, "steps_with_samples": 0, "metrics": {}}
        mean = self.total / self.count
        variance = torch.clamp(self.square / self.count - mean.square(), min=0.0)
        metrics: dict[str, Any] = {}
        for index, name in enumerate(self.names):
            metrics[name] = {
                "mean": float(mean[index]),
                "std": float(torch.sqrt(variance[index])),
                "min": float(self.minimum[index]),
                "max": float(self.maximum[index]),
                "positive_fraction": float(self.positive[index] / self.count),
            }
        return {
            "samples": count,
            "steps_with_samples": int(self.steps.item()),
            "metrics": metrics,
        }


def _float_grid(value: str, *, name: str, lower: float, upper: float) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    if any(not math.isfinite(item) or item < lower or item > upper for item in values):
        raise ValueError(f"{name} values must be finite and within [{lower}, {upper}]")
    return values


class _ContinuousQualityCounterfactualAudit:
    """Compare instantaneous and history-aware stair progress credit by episode outcome."""

    OUTCOME_NAMES = ("all", "success", "failure", "other")
    SUPPORT_FLOORS = (0.0, 0.25, 0.35, 0.50, 0.65)

    def __init__(
        self,
        base: Any,
        *,
        taus_s: tuple[float, ...],
        floors: tuple[float, ...],
    ) -> None:
        import torch

        self.device = base.device
        self.num_envs = int(base.num_envs)
        self.step_dt = float(getattr(base, "step_dt", 0.02))
        self.candidates = tuple(
            (float(tau), float(floor)) for tau in taus_s for floor in floors
        )
        if not self.candidates:
            raise ValueError("continuous-quality audit requires at least one candidate")

        shape = (self.num_envs,)
        self.risk = {
            self._candidate_key(tau, floor): torch.zeros(shape, device=self.device)
            for tau, floor in self.candidates
        }
        self.episode_steps = torch.zeros(shape, device=self.device)
        self.support_quality_sum = torch.zeros(shape, device=self.device)
        self.collision_risk_sum = torch.zeros(shape, device=self.device)
        self.current_credit_sum = torch.zeros(shape, device=self.device)
        self.diagnostic_metric_names = (
            "safe_progress_quality",
            "support_specific_quality",
            "support_structure",
            "contact_count",
            "base_height",
            "body_wxy",
            "course_quality",
            "direction_alignment",
            "speed_target_ratio",
            "touchdown_clean_quality",
            "action_clean_quality",
        )
        self.diagnostic_sums = {
            name: torch.zeros(shape, device=self.device)
            for name in self.diagnostic_metric_names
        }
        self.support_variant_names = (
            "safe_min_support",
            *(f"safe_times_support_floor_{floor:.2f}" for floor in self.SUPPORT_FLOORS),
        )
        self.support_variant_quality_sum = {
            name: torch.zeros(shape, device=self.device)
            for name in self.support_variant_names
        }
        self.support_variant_credit_sum = {
            name: torch.zeros(shape, device=self.device)
            for name in self.support_variant_names
        }
        self.history_quality_sum = {
            key: torch.zeros(shape, device=self.device) for key in self.risk
        }
        self.counterfactual_credit_sum = {
            key: torch.zeros(shape, device=self.device) for key in self.risk
        }

        self.frame_metric_names = [
            "instantaneous_support_quality",
            "collision_risk",
            "current_supported_progress_credit",
        ]
        self.frame_metric_names.extend(self.diagnostic_metric_names)
        for name in self.support_variant_names:
            self.frame_metric_names.extend((f"{name}/quality", f"{name}/credit"))
        for key in self.risk:
            self.frame_metric_names.extend(
                (f"{key}/history_quality", f"{key}/counterfactual_credit")
            )
        self.frame_accumulators = {
            name: _BucketAccumulator(self.frame_metric_names, self.device)
            for name in BUCKET_NAMES
        }

        self.episode_metric_names = [
            "episode_steps",
            "instantaneous_support_quality_mean",
            "collision_risk_mean",
            "current_supported_progress_credit_mean",
        ]
        self.episode_metric_names.extend(
            f"{name}_mean" for name in self.diagnostic_metric_names
        )
        for name in self.support_variant_names:
            self.episode_metric_names.extend(
                (f"{name}/quality_mean", f"{name}/credit_mean")
            )
        for key in self.risk:
            self.episode_metric_names.extend(
                (
                    f"{key}/history_quality_mean",
                    f"{key}/counterfactual_credit_mean",
                    f"{key}/credit_ratio_to_current",
                )
            )
        self.episode_accumulators = {
            family: {
                outcome: _BucketAccumulator(self.episode_metric_names, self.device)
                for outcome in self.OUTCOME_NAMES
            }
            for family in ("stairs_down", "stairs_up")
        }
        self.missing_outcome_episodes = 0

    @staticmethod
    def _candidate_key(tau: float, floor: float) -> str:
        return f"tau_{tau:.2f}_floor_{floor:.2f}"

    def _reset(self, mask: Any) -> None:
        if not bool(mask.any()):
            return
        self.episode_steps[mask] = 0.0
        self.support_quality_sum[mask] = 0.0
        self.collision_risk_sum[mask] = 0.0
        self.current_credit_sum[mask] = 0.0
        for values in self.diagnostic_sums.values():
            values[mask] = 0.0
        for values in self.support_variant_quality_sum.values():
            values[mask] = 0.0
        for values in self.support_variant_credit_sum.values():
            values[mask] = 0.0
        for key in self.risk:
            self.risk[key][mask] = 0.0
            self.history_quality_sum[key][mask] = 0.0
            self.counterfactual_credit_sum[key][mask] = 0.0

    def update(
        self,
        snapshot: dict[str, Any],
        masks: dict[str, Any],
        terminated: Any,
        truncated: Any,
        outcome: dict[str, Any] | None,
        *,
        record_frame: bool,
        record_episode: bool,
    ) -> None:
        import torch

        components = snapshot["components"]
        support_quality = torch.clamp(
            components["terrain_supported_progress_quality"].reshape(-1), 0.0, 1.0
        )
        current_credit = components["terrain_supported_progress"].reshape(-1)
        collision_risk = torch.clamp(
            snapshot["terrain_clearance_trace"].amax(dim=-1), 0.0, 1.0
        )
        stair_mask = snapshot["stairs_down"] | snapshot["stairs_up"]
        immediate_risk = torch.maximum(1.0 - support_quality, collision_risk)
        unmodulated_credit = torch.where(
            support_quality > 1e-6,
            current_credit / support_quality.clamp_min(1e-6),
            torch.zeros_like(current_credit),
        )

        safe_quality = torch.clamp(
            components.get("terrain_safe_progress_quality", support_quality).reshape(-1),
            0.0,
            1.0,
        )
        support_specific_quality = torch.clamp(
            components.get("terrain_support_specific_quality", support_quality).reshape(-1),
            0.0,
            1.0,
        )
        ones = torch.ones_like(support_quality)
        diagnostic_values = {
            "safe_progress_quality": safe_quality,
            "support_specific_quality": support_specific_quality,
            "support_structure": torch.clamp(
                components.get("support_structure_gate", ones).reshape(-1), 0.0, 1.0
            ),
            "contact_count": snapshot["foot_contact"].sum(dim=-1),
            "base_height": snapshot["base_height"].reshape(-1),
            "body_wxy": torch.linalg.vector_norm(
                snapshot["base_ang_vel"][:, :2], dim=-1
            ),
            "course_quality": torch.clamp(
                components.get("terrain_course_quality", ones).reshape(-1), 0.0, 1.0
            ),
            "direction_alignment": torch.clamp(
                components.get("terrain_direction_alignment", ones).reshape(-1),
                0.0,
                1.0,
            ),
            "speed_target_ratio": components.get(
                "terrain_speed_target_ratio", ones
            ).reshape(-1),
            "touchdown_clean_quality": torch.clamp(
                components.get("touchdown_clean_gate", ones).reshape(-1), 0.0, 1.0
            ),
            "action_clean_quality": torch.clamp(
                components.get("action_clean_gate", ones).reshape(-1), 0.0, 1.0
            ),
        }
        support_variant_quality = {
            "safe_min_support": torch.minimum(safe_quality, support_specific_quality)
        }
        for floor in self.SUPPORT_FLOORS:
            support_variant_quality[f"safe_times_support_floor_{floor:.2f}"] = (
                safe_quality * (floor + (1.0 - floor) * support_specific_quality)
            )
        support_variant_credit = {
            name: unmodulated_credit * quality
            for name, quality in support_variant_quality.items()
        }

        history_quality: dict[str, Any] = {}
        counterfactual_credit: dict[str, Any] = {}
        for tau, floor in self.candidates:
            key = self._candidate_key(tau, floor)
            decay = math.exp(-self.step_dt / max(tau, 1e-6))
            updated_risk = torch.maximum(self.risk[key] * decay, immediate_risk)
            updated_risk = torch.where(stair_mask, updated_risk, torch.zeros_like(updated_risk))
            self.risk[key].copy_(updated_risk)
            quality = floor + (1.0 - floor) * (1.0 - updated_risk)
            quality = torch.where(stair_mask, quality, torch.ones_like(quality))
            history_quality[key] = quality
            counterfactual_credit[key] = unmodulated_credit * quality

        if record_frame:
            frame_columns = [support_quality, collision_risk, current_credit]
            frame_columns.extend(
                diagnostic_values[name] for name in self.diagnostic_metric_names
            )
            for name in self.support_variant_names:
                frame_columns.extend(
                    (support_variant_quality[name], support_variant_credit[name])
                )
            for key in self.risk:
                frame_columns.extend((history_quality[key], counterfactual_credit[key]))
            frame_matrix = torch.stack(frame_columns, dim=-1)
            for name, mask in masks.items():
                self.frame_accumulators[name].add(frame_matrix, mask)

        stair_float = stair_mask.to(self.episode_steps.dtype)
        self.episode_steps += stair_float
        self.support_quality_sum += support_quality * stair_float
        self.collision_risk_sum += collision_risk * stair_float
        self.current_credit_sum += current_credit * stair_float
        for name, values in diagnostic_values.items():
            self.diagnostic_sums[name] += values * stair_float
        for name in self.support_variant_names:
            self.support_variant_quality_sum[name] += (
                support_variant_quality[name] * stair_float
            )
            self.support_variant_credit_sum[name] += (
                support_variant_credit[name] * stair_float
            )
        for key in self.risk:
            self.history_quality_sum[key] += history_quality[key] * stair_float
            self.counterfactual_credit_sum[key] += counterfactual_credit[key] * stair_float

        done = (terminated | truncated).reshape(-1).bool() & stair_mask
        if bool(done.any()) and record_episode:
            if outcome is None:
                self.missing_outcome_episodes += int(done.sum().item())
            else:
                outcome_ids = outcome.get("env_ids")
                if outcome_ids is None and outcome["valid"].numel() == self.num_envs:
                    outcome_ids = torch.arange(self.num_envs, device=self.device)
                if outcome_ids is None:
                    self.missing_outcome_episodes += int(done.sum().item())
                    self._reset(done)
                    return
                outcome_ids = outcome_ids.to(device=self.device, dtype=torch.long).reshape(-1)
                full_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                direction = torch.zeros(self.num_envs, device=self.device)
                success = torch.zeros_like(full_valid)
                failure = torch.zeros_like(full_valid)
                full_valid[outcome_ids] = outcome["valid"].reshape(-1).bool()
                direction[outcome_ids] = outcome["direction"].reshape(-1)
                success[outcome_ids] = outcome["move_up"].reshape(-1).bool()
                failure[outcome_ids] = outcome["failure_down"].reshape(-1).bool()
                valid = done & full_valid
                denominator = self.episode_steps.clamp_min(1.0)
                current_mean = self.current_credit_sum / denominator
                episode_columns = [
                    self.episode_steps,
                    self.support_quality_sum / denominator,
                    self.collision_risk_sum / denominator,
                    current_mean,
                ]
                episode_columns.extend(
                    self.diagnostic_sums[name] / denominator
                    for name in self.diagnostic_metric_names
                )
                for name in self.support_variant_names:
                    episode_columns.extend(
                        (
                            self.support_variant_quality_sum[name] / denominator,
                            self.support_variant_credit_sum[name] / denominator,
                        )
                    )
                for key in self.risk:
                    candidate_mean = self.counterfactual_credit_sum[key] / denominator
                    episode_columns.extend(
                        (
                            self.history_quality_sum[key] / denominator,
                            candidate_mean,
                            torch.where(
                                current_mean.abs() > 1e-6,
                                candidate_mean / current_mean.abs().clamp_min(1e-6),
                                torch.zeros_like(candidate_mean),
                            ),
                        )
                    )
                episode_matrix = torch.stack(episode_columns, dim=-1)
                for family, expected_direction in (("stairs_down", -1.0), ("stairs_up", 1.0)):
                    family_mask = valid & direction.eq(expected_direction)
                    outcome_masks = {
                        "all": family_mask,
                        "success": family_mask & success,
                        "failure": family_mask & failure,
                        "other": family_mask & ~success & ~failure,
                    }
                    for name, mask in outcome_masks.items():
                        self.episode_accumulators[family][name].add(episode_matrix, mask)
                self.missing_outcome_episodes += int((done & ~valid).sum().item())
        self._reset(done)

    def report(self) -> dict[str, Any]:
        return {
            "definition": {
                "risk": "max(previous_risk * exp(-dt/tau), 1-support_quality, collision_trace)",
                "history_quality": "floor + (1-floor) * (1-risk)",
                "scope": "read-only replacement audit for terrain_supported_progress quality only",
                "step_dt_s": self.step_dt,
                "candidates": [
                    {
                        "key": self._candidate_key(tau, floor),
                        "tau_s": tau,
                        "floor": floor,
                    }
                    for tau, floor in self.candidates
                ],
            },
            "missing_outcome_episodes": self.missing_outcome_episodes,
            "frames": {
                name: accumulator.report()
                for name, accumulator in self.frame_accumulators.items()
            },
            "episodes": {
                family: {
                    outcome: accumulator.report()
                    for outcome, accumulator in outcomes.items()
                }
                for family, outcomes in self.episode_accumulators.items()
            },
        }


def _snapshot_environment(base: Any, inp: Any, components: dict[str, Any]) -> dict[str, Any]:
    import torch

    replay = getattr(
        base,
        "_terrain_replay_mask",
        torch.zeros(base.num_envs, dtype=torch.bool, device=base.device),
    )
    levels = getattr(getattr(base, "_terrain", None), "terrain_levels", None)
    if levels is None:
        levels = torch.zeros(base.num_envs, dtype=torch.long, device=base.device)
    all_contact_forces = base._contact_sensor.data.net_forces_w
    limb_force_by_link = []
    limb_horizontal_force_by_link = []
    for body_ids in getattr(base, "_reward_probe_limb_contact_ids", ()):
        if body_ids:
            link_forces = all_contact_forces[:, body_ids, :]
            limb_force_by_link.append(
                torch.linalg.vector_norm(link_forces, dim=-1)
            )
            limb_horizontal_force_by_link.append(
                torch.linalg.vector_norm(link_forces[:, :, :2], dim=-1)
            )
        else:
            zeros = torch.zeros(
                (base.num_envs, 3),
                dtype=all_contact_forces.dtype,
                device=base.device,
            )
            limb_force_by_link.append(zeros)
            limb_horizontal_force_by_link.append(zeros)
    if limb_force_by_link:
        limb_force_by_link_tensor = torch.stack(limb_force_by_link, dim=1)
        limb_horizontal_force_by_link_tensor = torch.stack(
            limb_horizontal_force_by_link, dim=1
        )
    else:
        limb_force_by_link_tensor = torch.zeros(
            (base.num_envs, 4, 3),
            dtype=all_contact_forces.dtype,
            device=base.device,
        )
        limb_horizontal_force_by_link_tensor = torch.zeros_like(
            limb_force_by_link_tensor
        )
    return {
        "components": {
            key: value.detach().clone()
            for key, value in components.items()
            if torch.is_tensor(value)
        },
        "flat": base._flat_terrain_mask.detach().clone(),
        "stairs_down": base._stairs_down_terrain_mask.detach().clone(),
        "stairs_up": base._stairs_up_terrain_mask.detach().clone(),
        "replay": replay.detach().clone(),
        "levels": levels.detach().clone(),
        "commands": inp.cmd.detach().clone(),
        "base_lin_vel": inp.base_lin_vel.detach().clone(),
        "base_ang_vel": inp.base_ang_vel.detach().clone(),
        "base_height": inp.base_h_above_terrain.detach().clone(),
        "foot_contact": inp.foot_contact.detach().clone(),
        "foot_clearance": inp.foot_clearance.detach().clone(),
        "terrain_clearance_event": getattr(
            inp, "terrain_clearance_event", inp.foot_clearance
        ).detach().clone(),
        "terrain_clearance_trace": getattr(
            inp,
            "terrain_clearance_foot_mask",
            torch.zeros_like(inp.foot_clearance),
        ).detach().clone(),
        "terrain_clearance_anchor": getattr(
            inp,
            "terrain_clearance_anchor",
            torch.zeros_like(inp.foot_clearance),
        ).detach().clone(),
        "local_obstacle_h": getattr(
            inp,
            "local_obstacle_h",
            torch.zeros(base.num_envs, device=base.device),
        ).detach().clone(),
        "terrain_step_height": getattr(
            inp,
            "terrain_step_height",
            torch.zeros(base.num_envs, device=base.device),
        ).detach().clone(),
        "limb_collision_response": getattr(
            base,
            "_terrain_limb_collision_response",
            torch.zeros_like(inp.foot_clearance),
        ).detach().clone(),
        "support_height": getattr(
            base,
            "_passive_support_z",
            torch.zeros(base.num_envs, device=base.device),
        ).detach().clone(),
        "limb_force_by_link": limb_force_by_link_tensor.detach().clone(),
        "limb_horizontal_force_by_link": (
            limb_horizontal_force_by_link_tensor.detach().clone()
        ),
    }


class _CollisionRecoveryAudit:
    """Follow each limb collision through the next gait-cycle-sized window."""

    def __init__(self, base: Any, horizon_s: float) -> None:
        import torch

        self.device = base.device
        self.shape = (base.num_envs, 4)
        self.bucket_count = len(BUCKET_NAMES)
        self.step_dt = float(getattr(base, "step_dt", 0.02))
        self.horizon_steps = max(1, int(math.ceil(float(horizon_s) / self.step_dt)))
        self.event_threshold = float(
            getattr(base.cfg, "terrain_collision_event_threshold", 0.08)
        )
        self.trace_threshold = float(
            getattr(base.cfg, "terrain_clearance_event_threshold", 0.02)
        )
        self.flat_target = float(getattr(base.cfg, "flat_clearance_target", 0.08))
        self.margin = float(getattr(base.cfg, "terrain_clearance_margin", 0.04))
        self.band = float(getattr(base.cfg, "clearance_band", 0.02))

        self.active = torch.zeros(self.shape, dtype=torch.bool, device=self.device)
        self.previous_collision = torch.zeros_like(self.active)
        self.bucket = torch.full(self.shape, -1, dtype=torch.long, device=self.device)
        self.age = torch.zeros(self.shape, dtype=torch.long, device=self.device)
        self.target = torch.zeros(self.shape, device=self.device)
        self.anchor = torch.zeros(self.shape, device=self.device)
        self.initial_clearance = torch.zeros(self.shape, device=self.device)
        self.event_response = torch.zeros(self.shape, device=self.device)
        self.initial_support = torch.zeros(self.shape, device=self.device)
        self.step_height = torch.zeros(self.shape, device=self.device)
        self.max_clearance = torch.zeros(self.shape, device=self.device)
        self.max_target_progress = torch.zeros(self.shape, device=self.device)
        self.max_support_up = torch.zeros(self.shape, device=self.device)
        self.max_support_down = torch.zeros(self.shape, device=self.device)
        self.forward_progress = torch.zeros(self.shape, device=self.device)
        self.entered_swing = torch.zeros_like(self.active)
        self.trace_at_first_swing = torch.zeros_like(self.active)
        self.target_reached = torch.zeros_like(self.active)
        self.touchdown_after_swing = torch.zeros_like(self.active)
        self.recollision = torch.zeros_like(self.active)
        self.first_swing_delay = torch.zeros(self.shape, device=self.device)
        self.target_delay = torch.zeros(self.shape, device=self.device)
        self.last_contact = torch.ones(self.shape, dtype=torch.bool, device=self.device)

        def zeros():
            return torch.zeros(
                self.bucket_count, dtype=torch.float64, device=self.device
            )
        self.started = zeros()
        self.closed = zeros()
        self.censored = zeros()
        self.terminated = zeros()
        self.entered_swing_count = zeros()
        self.trace_at_first_swing_count = zeros()
        self.target_reached_count = zeros()
        self.touchdown_count = zeros()
        self.support_transition_count = zeros()
        self.recollision_count = zeros()
        self.clearance_gain_total = zeros()
        self.target_progress_total = zeros()
        self.forward_progress_total = zeros()
        self.first_swing_delay_total = zeros()
        self.target_delay_total = zeros()
        self.event_response_total = zeros()
        self.event_target_total = zeros()

    def _scatter(self, destination: Any, values: Any, mask: Any) -> None:
        import torch

        if not bool(mask.any()):
            return
        indexes = self.bucket[mask]
        destination.scatter_add_(0, indexes, values[mask].to(torch.float64))

    def _close(self, mask: Any, *, terminated: Any = None, censored: bool = False) -> None:
        import torch

        mask = mask & self.active & (self.bucket >= 0)
        if not bool(mask.any()):
            return
        ones = torch.ones(self.shape, device=self.device)
        if censored:
            self._scatter(self.censored, ones, mask)
            self.active[mask] = False
            return

        self._scatter(self.closed, ones, mask)
        if terminated is not None:
            self._scatter(self.terminated, terminated.to(torch.float32), mask)
        self._scatter(self.entered_swing_count, self.entered_swing.to(torch.float32), mask)
        self._scatter(
            self.trace_at_first_swing_count,
            self.trace_at_first_swing.to(torch.float32),
            mask,
        )
        self._scatter(self.target_reached_count, self.target_reached.to(torch.float32), mask)
        self._scatter(
            self.touchdown_count, self.touchdown_after_swing.to(torch.float32), mask
        )
        up_bucket = (self.bucket == BUCKET_NAMES.index("stairs_up_frontier")) | (
            self.bucket == BUCKET_NAMES.index("stairs_up_replay")
        )
        transition_threshold = torch.clamp(0.5 * self.step_height, min=0.03, max=0.12)
        support_transition = torch.where(
            up_bucket,
            self.max_support_up >= transition_threshold,
            self.max_support_down >= transition_threshold,
        )
        self._scatter(
            self.support_transition_count, support_transition.to(torch.float32), mask
        )
        self._scatter(self.recollision_count, self.recollision.to(torch.float32), mask)
        self._scatter(
            self.clearance_gain_total,
            torch.clamp(self.max_clearance - self.initial_clearance, min=0.0),
            mask,
        )
        self._scatter(self.target_progress_total, self.max_target_progress, mask)
        self._scatter(self.forward_progress_total, self.forward_progress, mask)
        self._scatter(
            self.first_swing_delay_total,
            self.first_swing_delay * self.entered_swing.to(self.first_swing_delay.dtype),
            mask,
        )
        self._scatter(
            self.target_delay_total,
            self.target_delay * self.target_reached.to(self.target_delay.dtype),
            mask,
        )
        self._scatter(self.event_response_total, self.event_response, mask)
        self._scatter(self.event_target_total, self.target, mask)
        self.active[mask] = False

    def update(
        self,
        snapshot: dict[str, Any],
        masks: dict[str, Any],
        terminated: Any,
        truncated: Any,
    ) -> None:
        import torch

        bucket_by_env = torch.full(
            (self.shape[0],), -1, dtype=torch.long, device=self.device
        )
        for index, name in enumerate(BUCKET_NAMES):
            bucket_by_env[masks[name]] = index

        raw = snapshot["limb_collision_response"]
        collision = raw > self.event_threshold
        onset = collision & ~self.previous_collision
        stair_onset = onset & (bucket_by_env[:, None] >= 2)

        repeated = stair_onset & self.active
        self.recollision |= repeated
        self._close(repeated)

        if bool(stair_onset.any()):
            bucket_matrix = bucket_by_env[:, None].expand_as(self.bucket)
            anchor = snapshot["terrain_clearance_anchor"]
            clearance = snapshot["terrain_clearance_event"]
            obstacle = snapshot["local_obstacle_h"][:, None].expand_as(clearance)
            target = torch.maximum(
                torch.maximum(
                    obstacle + self.margin,
                    torch.full_like(clearance, self.flat_target),
                ),
                anchor + 0.01,
            )
            support = snapshot["support_height"][:, None].expand_as(clearance)
            step_height = snapshot["terrain_step_height"][:, None].expand_as(clearance)
            contact = snapshot["foot_contact"] > 0.5

            self.active[stair_onset] = True
            self.bucket[stair_onset] = bucket_matrix[stair_onset]
            self.age[stair_onset] = 0
            self.target[stair_onset] = target[stair_onset]
            self.anchor[stair_onset] = anchor[stair_onset]
            self.initial_clearance[stair_onset] = clearance[stair_onset]
            self.event_response[stair_onset] = raw[stair_onset]
            self.initial_support[stair_onset] = support[stair_onset]
            self.step_height[stair_onset] = step_height[stair_onset]
            self.max_clearance[stair_onset] = clearance[stair_onset]
            self.max_target_progress[stair_onset] = 0.0
            self.max_support_up[stair_onset] = 0.0
            self.max_support_down[stair_onset] = 0.0
            self.forward_progress[stair_onset] = 0.0
            for value in (
                self.entered_swing,
                self.trace_at_first_swing,
                self.target_reached,
                self.touchdown_after_swing,
                self.recollision,
            ):
                value[stair_onset] = False
            self.first_swing_delay[stair_onset] = 0.0
            self.target_delay[stair_onset] = 0.0
            self.last_contact[stair_onset] = contact[stair_onset]
            self._scatter(self.started, torch.ones(self.shape, device=self.device), stair_onset)

        active = self.active
        if bool(active.any()):
            clearance = snapshot["terrain_clearance_event"]
            trace = snapshot["terrain_clearance_trace"]
            contact = snapshot["foot_contact"] > 0.5
            support = snapshot["support_height"][:, None].expand_as(clearance)
            self.age[active] += 1
            self.max_clearance = torch.where(
                active, torch.maximum(self.max_clearance, clearance), self.max_clearance
            )
            lift_span = torch.clamp(self.target - self.anchor, min=0.01)
            target_progress = torch.clamp((clearance - self.anchor) / lift_span, 0.0, 1.0)
            self.max_target_progress = torch.where(
                active,
                torch.maximum(self.max_target_progress, target_progress),
                self.max_target_progress,
            )
            support_delta = support - self.initial_support
            self.max_support_up = torch.where(
                active,
                torch.maximum(self.max_support_up, support_delta),
                self.max_support_up,
            )
            self.max_support_down = torch.where(
                active,
                torch.maximum(self.max_support_down, -support_delta),
                self.max_support_down,
            )

            cmd_xy = snapshot["commands"][:, :2]
            cmd_speed = torch.linalg.vector_norm(cmd_xy, dim=-1)
            cmd_unit = cmd_xy / torch.clamp(cmd_speed[:, None], min=1e-6)
            along = (snapshot["base_lin_vel"][:, :2] * cmd_unit).sum(dim=-1)
            along = torch.where(cmd_speed > 0.05, along, torch.zeros_like(along))
            self.forward_progress += active.to(along.dtype) * along[:, None] * self.step_dt

            swing = ~contact
            first_swing = active & swing & ~self.entered_swing
            self.first_swing_delay[first_swing] = (
                self.age[first_swing].to(self.first_swing_delay.dtype) * self.step_dt
            )
            self.trace_at_first_swing[first_swing] = trace[first_swing] > self.trace_threshold
            self.entered_swing |= active & swing

            reached = active & ~self.target_reached & (clearance >= self.target - self.band)
            self.target_delay[reached] = self.age[reached].to(self.target_delay.dtype) * self.step_dt
            self.target_reached |= reached

            touchdown = active & self.entered_swing & contact & ~self.last_contact
            self.touchdown_after_swing |= touchdown
            self.last_contact = torch.where(active, contact, self.last_contact)
            self.recollision |= active & onset & (self.age > 1)

        done = (terminated | truncated).reshape(-1)
        done_matrix = done[:, None].expand_as(self.active)
        terminated_matrix = terminated.reshape(-1, 1).expand_as(self.active)
        self._close(done_matrix, terminated=terminated_matrix)
        self._close(self.active & (self.age >= self.horizon_steps))
        self.previous_collision = torch.where(
            done_matrix, torch.zeros_like(collision), collision
        )

    @staticmethod
    def _ratio(numerator: Any, denominator: Any, index: int) -> float | None:
        numerator_value = numerator if getattr(numerator, "ndim", 0) == 0 else numerator[index]
        denominator_value = (
            denominator if getattr(denominator, "ndim", 0) == 0 else denominator[index]
        )
        den = float(denominator_value)
        return float(numerator_value / den) if den > 0.0 else None

    def report(self) -> dict[str, Any]:
        self._close(self.active, censored=True)
        result: dict[str, Any] = {
            "horizon_s": self.horizon_steps * self.step_dt,
            "event_threshold": self.event_threshold,
            "trace_threshold": self.trace_threshold,
            "buckets": {},
        }
        for index, name in enumerate(BUCKET_NAMES):
            started = int(self.started[index])
            closed = int(self.closed[index])
            swing_count = self.entered_swing_count[index]
            reached_count = self.target_reached_count[index]
            result["buckets"][name] = {
                "events_started": started,
                "windows_closed": closed,
                "windows_censored": int(self.censored[index]),
                "terminated_rate": self._ratio(self.terminated, self.closed, index),
                "entered_swing_rate": self._ratio(
                    self.entered_swing_count, self.closed, index
                ),
                "trace_survived_to_first_swing_rate": self._ratio(
                    self.trace_at_first_swing_count, swing_count, index
                ),
                "target_reached_rate": self._ratio(
                    self.target_reached_count, self.closed, index
                ),
                "touchdown_after_swing_rate": self._ratio(
                    self.touchdown_count, self.closed, index
                ),
                "support_transition_rate": self._ratio(
                    self.support_transition_count, self.closed, index
                ),
                "recollision_rate": self._ratio(self.recollision_count, self.closed, index),
                "mean_clearance_gain_m": self._ratio(
                    self.clearance_gain_total, self.closed, index
                ),
                "mean_max_target_progress": self._ratio(
                    self.target_progress_total, self.closed, index
                ),
                "mean_forward_progress_m": self._ratio(
                    self.forward_progress_total, self.closed, index
                ),
                "mean_first_swing_delay_s": self._ratio(
                    self.first_swing_delay_total, swing_count, index
                ),
                "mean_target_delay_s": self._ratio(
                    self.target_delay_total, reached_count, index
                ),
                "mean_event_response": self._ratio(
                    self.event_response_total, self.closed, index
                ),
                "mean_event_target_m": self._ratio(
                    self.event_target_total, self.closed, index
                ),
            }
        return result


class _CurriculumOutcomeAudit:
    """Aggregate the exact episode-end stair curriculum predicates by direction."""

    FAMILY_DIRECTIONS = {
        "stairs_down": -1.0,
        "stairs_up": 1.0,
    }
    BOOLEAN_METRICS = (
        "height_measurement_valid",
        "height_ok",
        "forward_ok",
        "nonterminal",
        "stable_height_ok",
        "stable_upright_ok",
        "stable_contact_ok",
        "stable_wxy_ok",
        "stable_end",
        "speed_ok",
        "height_and_forward_ok",
        "height_forward_stable_ok",
        "controlled_height",
        "move_up",
        "low_progress_down",
        "failure_down",
        "catastrophic",
    )
    VALUE_METRICS = (
        "signed_height_m",
        "forward_dist_m",
        "base_height_m",
        "upright_score",
        "contact_count",
        "body_wxy_rad_s",
        "along_speed_abs_m_s",
        "command_speed_m_s",
        "required_height_m",
        "stair_step_height_m",
        "lateral_drift_m",
        "heading_error_rad",
    )

    def __init__(self) -> None:
        self.thresholds: dict[str, float] = {}
        self.latest: dict[str, Any] | None = None
        self.families = {
            name: {**self._empty_stats(), "by_step_height": {}}
            for name in self.FAMILY_DIRECTIONS
        }

    @classmethod
    def _empty_stats(cls) -> dict[str, Any]:
        return {
            "episodes": 0,
            "boolean_counts": {metric: 0 for metric in cls.BOOLEAN_METRICS},
            "value_sums": {metric: 0.0 for metric in cls.VALUE_METRICS},
            "value_counts": {metric: 0 for metric in cls.VALUE_METRICS},
        }

    @staticmethod
    def _selected_sum(value: Any, mask: Any) -> float:
        return float(value[mask].to(dtype=value.dtype).sum().item())

    def _accumulate(
        self,
        target: dict[str, Any],
        mask: Any,
        boolean_values: dict[str, Any],
        value_values: dict[str, Any],
    ) -> None:
        import torch

        count = int(mask.sum().item())
        if count <= 0:
            return
        target["episodes"] += count
        for metric, values in boolean_values.items():
            target["boolean_counts"][metric] += int((values & mask).sum().item())
        for metric, values in value_values.items():
            finite_mask = mask & torch.isfinite(values)
            finite_count = int(finite_mask.sum().item())
            if finite_count <= 0:
                continue
            target["value_sums"][metric] += self._selected_sum(values, finite_mask)
            target["value_counts"][metric] += finite_count

    def capture(self, original: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        if kwargs:
            self.update(kwargs, result)
        return result

    def consume_latest(self) -> dict[str, Any] | None:
        latest = self.latest
        self.latest = None
        return latest

    def attach_env_ids(self, env_ids: Any) -> None:
        if self.latest is None:
            return
        ids = env_ids.detach().clone().reshape(-1)
        if ids.numel() != self.latest["valid"].numel():
            raise RuntimeError("curriculum outcome and reset environment ids have different sizes")
        self.latest["env_ids"] = ids

    def update(self, inputs: dict[str, Any], result: dict[str, Any]) -> None:
        import torch

        valid = inputs["valid_episode"].bool() & result["evaluation_eligible"].bool()
        eligible = inputs.get("eligible_mask")
        if eligible is not None:
            valid &= eligible.bool()
        direction = torch.sign(inputs["expected_height_direction"])
        terminal = inputs["terminal_now"].bool()

        height_delta = inputs.get("height_delta", inputs.get("support_height_delta"))
        if height_delta is None:
            return
        stable_h = float(inputs.get("stable_h", 0.42))
        stable_upright = float(inputs.get("stable_upright", 0.85))
        stable_contact_min = float(inputs.get("stable_contact_min", 2.0))
        stable_wxy_max = float(inputs.get("stable_wxy_max", 1.50))
        stair_forward_min = float(inputs.get("forward_min", inputs.get("stair_forward_min", 0.75)))
        speed_cap_ratio = float(inputs.get("speed_cap_ratio", 1.60))
        speed_cap_min = float(inputs.get("speed_cap_min", 0.75))
        stair_height_min = float(inputs.get("height_required_min", inputs.get("stair_height_min", 0.08)))
        self.thresholds = {
            "height_required_fraction": float(inputs.get("height_required_fraction", 0.0)),
            "height_required_min_m": stair_height_min,
            "stair_forward_min_m": stair_forward_min,
            "stable_height_min_m": stable_h,
            "stable_upright_min": stable_upright,
            "stable_contact_min": stable_contact_min,
            "stable_wxy_max_rad_s": stable_wxy_max,
            "speed_cap_ratio": speed_cap_ratio,
            "speed_cap_min_m_s": speed_cap_min,
        }

        height_ok = result["height_ok"].bool()
        stair_forward_dist = inputs.get("stair_forward_dist", inputs["forward_dist"])
        forward_ok = result.get(
            "stair_forward_ok", stair_forward_dist >= stair_forward_min
        ).bool()
        stable_height_ok = inputs["base_h_local"] >= stable_h
        stable_upright_ok = inputs["upright_score"] >= stable_upright
        stable_contact_ok = inputs["contact_count"] >= stable_contact_min
        stable_wxy_ok = inputs["body_wxy"] <= stable_wxy_max
        stable_end = result["stable_end"].bool()
        speed_ok = result["speed_controlled"].bool()
        height_and_forward_ok = height_ok & forward_ok
        height_forward_stable_ok = height_and_forward_ok & stable_end
        controlled_height = height_forward_stable_ok & speed_ok
        self.latest = {
            "valid": valid.detach().clone(),
            "direction": direction.detach().clone(),
            "move_up": result["move_up"].bool().detach().clone(),
            "failure_down": result["failure_down"].bool().detach().clone(),
            "stable_end": stable_end.detach().clone(),
            "controlled_height": controlled_height.detach().clone(),
        }
        boolean_values = {
            "height_measurement_valid": inputs.get(
                "support_height_valid", torch.isfinite(height_delta)
            ).bool(),
            "height_ok": height_ok,
            "forward_ok": forward_ok,
            "nonterminal": ~terminal,
            "stable_height_ok": stable_height_ok,
            "stable_upright_ok": stable_upright_ok,
            "stable_contact_ok": stable_contact_ok,
            "stable_wxy_ok": stable_wxy_ok,
            "stable_end": stable_end,
            "speed_ok": speed_ok,
            "height_and_forward_ok": height_and_forward_ok,
            "height_forward_stable_ok": height_forward_stable_ok,
            "controlled_height": controlled_height,
            "move_up": result["move_up"].bool(),
            "low_progress_down": result["low_progress_down"].bool(),
            "failure_down": result["failure_down"].bool(),
            "catastrophic": result["catastrophic"].bool(),
        }
        value_values = {
            "signed_height_m": result["signed_height"],
            "forward_dist_m": stair_forward_dist,
            "base_height_m": inputs["base_h_local"],
            "upright_score": inputs["upright_score"],
            "contact_count": inputs["contact_count"],
            "body_wxy_rad_s": inputs["body_wxy"],
            "along_speed_abs_m_s": inputs["v_along"].abs(),
            "command_speed_m_s": inputs.get("height_task_cmd_mag", inputs["cmd_mag"]),
            "required_height_m": result.get(
                "required_height", torch.full_like(height_delta, stair_height_min)
            ),
            "stair_step_height_m": inputs.get(
                "stair_step_height", torch.zeros_like(height_delta)
            ),
            "lateral_drift_m": inputs.get(
                "stair_lateral_drift", torch.zeros_like(height_delta)
            ).abs(),
            "heading_error_rad": inputs.get(
                "stair_heading_error", torch.zeros_like(height_delta)
            ).abs(),
        }

        for family_name, family_direction in self.FAMILY_DIRECTIONS.items():
            mask = valid & direction.eq(family_direction)
            if not bool(mask.any()):
                continue
            family = self.families[family_name]
            self._accumulate(family, mask, boolean_values, value_values)
            step_height_mm = torch.round(
                value_values["stair_step_height_m"] * 1000.0
            ).long()
            for height_mm in torch.unique(step_height_mm[mask]).tolist():
                height_mask = mask & step_height_mm.eq(int(height_mm))
                key = f"{int(height_mm)}mm"
                height_stats = family["by_step_height"].setdefault(
                    key, self._empty_stats()
                )
                self._accumulate(
                    height_stats, height_mask, boolean_values, value_values
                )

    def _summarize_stats(self, stats: dict[str, Any]) -> dict[str, Any]:
        episodes = int(stats["episodes"])
        rates = {
            metric: (count / episodes if episodes > 0 else None)
            for metric, count in stats["boolean_counts"].items()
        }
        means = {
            metric: (
                stats["value_sums"][metric] / stats["value_counts"][metric]
                if stats["value_counts"][metric] > 0
                else None
            )
            for metric in self.VALUE_METRICS
        }
        return {"episodes": episodes, "rates": rates, "means": means}

    def report(self) -> dict[str, Any]:
        report: dict[str, Any] = {"thresholds": self.thresholds, "families": {}}
        for family_name, family in self.families.items():
            summary = self._summarize_stats(family)
            summary["by_step_height"] = {
                key: self._summarize_stats(stats)
                for key, stats in sorted(
                    family["by_step_height"].items(),
                    key=lambda item: int(item[0][:-2]),
                )
            }
            report["families"][family_name] = summary
        return report


def _bucket_masks(snapshot: dict[str, Any]) -> dict[str, Any]:
    import torch

    commands = snapshot["commands"]
    command_magnitude = torch.linalg.vector_norm(commands, dim=-1)
    flat_stop = snapshot["flat"] & (command_magnitude <= 0.05)
    flat_moving = snapshot["flat"] & ~flat_stop
    replay = snapshot["replay"]
    return {
        "flat_moving": flat_moving,
        "flat_stop": flat_stop,
        "stairs_down_frontier": snapshot["stairs_down"] & ~replay,
        "stairs_down_replay": snapshot["stairs_down"] & replay,
        "stairs_up_frontier": snapshot["stairs_up"] & ~replay,
        "stairs_up_replay": snapshot["stairs_up"] & replay,
    }


def _physical_metrics(snapshot: dict[str, Any], terminated: Any, truncated: Any) -> dict[str, Any]:
    import torch

    commands = snapshot["commands"]
    velocity = snapshot["base_lin_vel"]
    cmd_xy = commands[:, :2]
    cmd_speed = torch.linalg.vector_norm(cmd_xy, dim=-1)
    cmd_unit = cmd_xy / torch.clamp(cmd_speed[:, None], min=1e-6)
    along = (velocity[:, :2] * cmd_unit).sum(dim=-1)
    along = torch.where(cmd_speed > 0.05, along, torch.zeros_like(along))
    limb_force_by_link = snapshot["limb_force_by_link"]
    limb_horizontal_force_by_link = snapshot["limb_horizontal_force_by_link"]
    limb_force_by_leg = limb_force_by_link.max(dim=-1).values
    limb_horizontal_force_by_leg = limb_horizontal_force_by_link.max(dim=-1).values
    metrics = {
        "physics/cmd_speed": cmd_speed,
        "physics/actual_speed": torch.linalg.vector_norm(velocity[:, :2], dim=-1),
        "physics/progress_along": along,
        "physics/lin_error": torch.linalg.vector_norm(velocity[:, :2] - cmd_xy, dim=-1),
        "physics/yaw_error": (snapshot["base_ang_vel"][:, 2] - commands[:, 2]).abs(),
        "physics/wxy": torch.linalg.vector_norm(snapshot["base_ang_vel"][:, :2], dim=-1),
        "physics/vz_abs": velocity[:, 2].abs(),
        "physics/base_height": snapshot["base_height"].reshape(-1),
        "physics/contact_count": snapshot["foot_contact"].sum(dim=-1),
        "physics/limb_contact_force_max": limb_force_by_leg.max(dim=-1).values,
        "physics/limb_contact_over_5n": (
            limb_force_by_leg.max(dim=-1).values > 5.0
        ).to(torch.float32),
        "physics/limb_contact_legs_over_5n": (
            limb_force_by_leg > 5.0
        ).to(torch.float32).sum(dim=-1),
        "physics/limb_horizontal_force_max": (
            limb_horizontal_force_by_leg.max(dim=-1).values
        ),
        "physics/limb_horizontal_over_5n": (
            limb_horizontal_force_by_leg.max(dim=-1).values > 5.0
        ).to(torch.float32),
        "physics/terrain_level": snapshot["levels"].to(torch.float32),
        "physics/terminated": terminated.reshape(-1).to(torch.float32),
        "physics/truncated": truncated.reshape(-1).to(torch.float32),
    }
    for link_index, link_name in enumerate(("hip", "thigh", "calf")):
        link_force = limb_force_by_link[:, :, link_index].max(dim=-1).values
        link_horizontal_force = limb_horizontal_force_by_link[
            :, :, link_index
        ].max(dim=-1).values
        metrics[f"physics/{link_name}_contact_over_5n"] = (
            link_force > 5.0
        ).to(torch.float32)
        metrics[f"physics/{link_name}_horizontal_over_5n"] = (
            link_horizontal_force > 5.0
        ).to(torch.float32)
    return metrics


def _scalar(value: Any, *, like: Any) -> Any:
    import torch

    tensor = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if tensor.numel() == 1:
        tensor = tensor.expand_as(like)
    return tensor.reshape(-1)


def _restore_probe_curriculum(base: Any, checkpoint_path: str) -> dict[str, Any]:
    """Restore a larger training sidecar into a smaller read-only probe.

    The production environment requires an exact environment-count match. A
    diagnostic probe intentionally uses fewer environments, so preserve each
    semantic terrain family's joint level/peak distribution with deterministic
    quantile sampling before replay slots are rebuilt.
    """
    import torch

    source_run = Path(checkpoint_path).expanduser().parent.parent
    state_path = source_run / "curriculum_state.json"
    if not state_path.is_file():
        return {"mode": "missing", "state_path": str(state_path)}

    state = json.loads(state_path.read_text(encoding="utf-8"))
    levels = state.get("terrain_levels")
    peaks = state.get("terrain_level_peak", levels)
    streaks = state.get("terrain_move_down_streak", [0] * len(levels or ()))
    semantic = state.get("terrain_semantic_types")
    source_names = state.get("terrain_type_names")
    if not all(isinstance(value, list) for value in (levels, peaks, streaks, semantic)):
        raise ValueError("probe curriculum sidecar is missing level vectors")
    if not isinstance(source_names, list) or not source_names:
        raise ValueError("probe curriculum sidecar is missing terrain type names")
    source_count = len(levels)
    if not (
        source_count == len(peaks) == len(streaks) == len(semantic)
    ):
        raise ValueError("probe curriculum sidecar vectors have inconsistent lengths")

    if source_count == int(base.num_envs):
        return {
            "mode": "environment-exact",
            "state_path": str(state_path),
            "source_envs": source_count,
            "probe_envs": int(base.num_envs),
            "step": int(state.get("step", 0)),
        }

    device = base.device
    level_dtype = base._terrain.terrain_levels.dtype
    source_levels = torch.as_tensor(levels, dtype=level_dtype, device=device)
    source_peaks = torch.as_tensor(peaks, dtype=level_dtype, device=device)
    source_streaks = torch.as_tensor(streaks, dtype=level_dtype, device=device)
    source_semantic = torch.as_tensor(semantic, dtype=torch.long, device=device)
    current_semantic = base._col_type[base._terrain.terrain_types].long()

    restored_levels = base._terrain.terrain_levels.clone()
    restored_peaks = restored_levels.clone()
    restored_streaks = torch.zeros_like(restored_levels)
    families: dict[str, Any] = {}
    for current_id, name in enumerate(base._type_names):
        current_ids = torch.nonzero(current_semantic == current_id, as_tuple=False).flatten()
        if current_ids.numel() == 0 or name not in source_names:
            continue
        source_id = source_names.index(name)
        source_ids = torch.nonzero(source_semantic == source_id, as_tuple=False).flatten()
        if source_ids.numel() == 0:
            raise ValueError(f"probe curriculum has no source samples for terrain {name!r}")

        # Sort paired state by current level and then peak so quantile sampling
        # preserves their correlation instead of constructing impossible pairs.
        order_key = source_levels[source_ids] * 1024 + source_peaks[source_ids]
        ordered = source_ids[torch.argsort(order_key)]
        if current_ids.numel() == 1:
            quantile_ids = ordered[ordered.numel() // 2].reshape(1)
        else:
            positions = torch.linspace(
                0,
                ordered.numel() - 1,
                steps=current_ids.numel(),
                device=device,
            ).round().long()
            quantile_ids = ordered[positions]
        restored_levels[current_ids] = source_levels[quantile_ids]
        restored_peaks[current_ids] = torch.maximum(
            source_peaks[quantile_ids], source_levels[quantile_ids]
        )
        restored_streaks[current_ids] = source_streaks[quantile_ids]
        families[name] = {
            "source_envs": int(source_ids.numel()),
            "probe_envs": int(current_ids.numel()),
            "level_mean": float(restored_levels[current_ids].float().mean().item()),
            "level_min": int(restored_levels[current_ids].min().item()),
            "level_max": int(restored_levels[current_ids].max().item()),
            "peak_mean": float(restored_peaks[current_ids].float().mean().item()),
        }

    base._terrain.terrain_levels.copy_(restored_levels)
    base._terrain_level_peak = restored_peaks
    base._terrain_move_down_streak = restored_streaks
    base._phase = max(int(base._phase), int(state.get("phase", base._phase)))
    base._penalty_gate = max(
        float(base._penalty_gate), float(state.get("penalty_gate", 0.0))
    )
    base._penalty_ramp_unlocked = bool(
        state.get("penalty_ramp_unlocked", base._penalty_ramp_unlocked)
    )
    base._clearance_gate = max(
        float(base._clearance_gate), float(state.get("clearance_gate", 0.0))
    )
    base._dr_level = int(state.get("dr_level", base._dr_level))
    base._dr_gate_count = int(state.get("dr_gate_count", base._dr_gate_count))
    base._sync_terrain_origins_from_levels()
    base._configure_stair_replay_curriculum()
    return {
        "mode": "semantic-quantile-remapped",
        "state_path": str(state_path),
        "source_envs": source_count,
        "probe_envs": int(base.num_envs),
        "step": int(state.get("step", 0)),
        "phase": int(state.get("phase", base._phase)),
        "families": families,
    }


def _force_stair_level(base: Any, requested_level: int) -> dict[str, int] | None:
    """Pin probe-only stair samples to one level without changing training state."""
    if requested_level < 0:
        return None


    terrain = base._terrain
    max_level = int(terrain.terrain_origins.shape[0]) - 1
    level = max(0, min(int(requested_level), max_level))
    stair_mask = base._stairs_down_terrain_mask | base._stairs_up_terrain_mask
    terrain.terrain_levels[stair_mask] = level
    terrain.env_origins[stair_mask] = terrain.terrain_origins[
        terrain.terrain_levels[stair_mask].long(),
        terrain.terrain_types[stair_mask].long(),
    ]
    if hasattr(base, "_terrain_level_peak"):
        base._terrain_level_peak[stair_mask] = level
    if hasattr(base, "_terrain_move_down_streak"):
        base._terrain_move_down_streak[stair_mask] = 0

    # A fixed-level probe measures the actor, not replay reshuffling. Keep curriculum
    # outcome computation active for the audit, but block its level write-back.
    for name in ("_terrain_replay_candidate_mask", "_terrain_replay_mask"):
        value = getattr(base, name, None)
        if value is not None:
            value[stair_mask] = False
    if hasattr(base, "_terrain_frontier_mask"):
        base._terrain_frontier_mask[stair_mask] = True
    base._reward_probe_update_env_origins = terrain.update_env_origins
    terrain.update_env_origins = lambda *_args, **_kwargs: None
    return {
        "requested": int(requested_level),
        "applied": level,
        "stairs_down": int(base._stairs_down_terrain_mask.sum().item()),
        "stairs_up": int(base._stairs_up_terrain_mask.sum().item()),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:  # pragma: no cover - requires IsaacLab
    import importlib
    import random
    import sys

    import gymnasium as gym
    import numpy as np
    import torch
    import yaml
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg
    from skrl.utils.runner.torch import Runner

    import taili_blind_runtime  # noqa: F401 - register task and AMP patch
    from taili_blind_runtime import diagnose_taili
    from taili_blind_runtime.taili_core import taili_reward

    counterfactual_module = None
    counterfactual_cfg = None
    counterfactual_overrides: dict[str, Any] = {}
    counterfactual_metadata: dict[str, Any] | None = None
    if args.counterfactual_reward_module:
        module_path = Path(args.counterfactual_reward_module).resolve()
        config_path = Path(args.counterfactual_reward_config).resolve()
        if not module_path.is_file():
            raise FileNotFoundError(module_path)
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        spec = importlib.util.spec_from_file_location(
            "taili_reward_counterfactual_probe", module_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load counterfactual reward module: {module_path}")
        counterfactual_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = counterfactual_module
        spec.loader.exec_module(counterfactual_module)
        reward_values = yaml.safe_load(config_path.read_text(encoding="utf-8"))["reward"]
        base_config_path = None
        if args.counterfactual_base_config:
            base_config_path = Path(args.counterfactual_base_config).resolve()
            if not base_config_path.is_file():
                raise FileNotFoundError(base_config_path)
            baseline_reward_values = yaml.safe_load(
                base_config_path.read_text(encoding="utf-8")
            )["reward"]
            counterfactual_overrides = _reward_config_overrides(
                baseline_reward_values, reward_values
            )
        else:
            counterfactual_overrides = dict(reward_values)
        counterfactual_metadata = {
            "module": str(module_path),
            "config": str(config_path),
            "base_config": str(base_config_path) if base_config_path else None,
            "override_fields": sorted(counterfactual_overrides),
            "override_field_count": len(counterfactual_overrides),
        }

    captured: dict[str, Any] = {}
    base_holder: dict[str, Any] = {}
    original_compute = taili_reward.compute_reward_components
    amp_env_module = importlib.import_module("taili_blind_runtime.taili_amp_env")
    original_curriculum = amp_env_module.compute_terrain_curriculum_moves
    curriculum_outcomes = _CurriculumOutcomeAudit()

    def capture_components(inp: Any, cfg: Any) -> dict[str, Any]:
        nonlocal counterfactual_cfg
        components = original_compute(inp, cfg)
        if counterfactual_module is not None:
            if counterfactual_cfg is None:
                counterfactual_cfg, copied_fields, applied_fields = (
                    _counterfactual_cfg_from_runtime(
                        counterfactual_module.RewardConfig,
                        cfg,
                        counterfactual_overrides,
                    )
                )
                if counterfactual_metadata is not None:
                    counterfactual_metadata["runtime_fields_copied"] = len(copied_fields)
                    counterfactual_metadata["applied_reward_fields"] = applied_fields
                    unapplied = sorted(set(counterfactual_overrides) - set(applied_fields))
                    counterfactual_metadata["unapplied_reward_fields"] = unapplied
                    if unapplied:
                        raise ValueError(
                            "counterfactual reward fields are not present in RewardConfig: "
                            + ", ".join(unapplied)
                        )
            captured["counterfactual_components"] = (
                counterfactual_module.compute_reward_components(inp, counterfactual_cfg)
            )
        base = base_holder.get("base")
        if base is not None:
            captured["transition"] = _snapshot_environment(base, inp, components)
        return components

    taili_reward.compute_reward_components = capture_components
    amp_env_module.compute_terrain_curriculum_moves = (
        lambda *call_args, **call_kwargs: curriculum_outcomes.capture(
            original_curriculum, *call_args, **call_kwargs
        )
    )
    experiment_cfg = diagnose_taili._load_experiment_cfg(args)
    experiment_cfg.setdefault("trainer", {})["close_environment_at_exit"] = False
    experiment_cfg.setdefault("agent", {}).setdefault("experiment", {})["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    original_terrain_snapshot = None
    try:
        env = SkrlVecEnvWrapper(env, ml_framework="torch")
        base = env.unwrapped
        base_holder["base"] = base
        original_terrain_snapshot = base._terrain_curriculum_snapshot

        def capture_terrain_snapshot(env_ids: Any, *, include_replay: bool = False) -> Any:
            result = original_terrain_snapshot(env_ids, include_replay=include_replay)
            curriculum_outcomes.attach_env_ids(env_ids)
            return result

        base._terrain_curriculum_snapshot = capture_terrain_snapshot
        base._reward_probe_limb_contact_ids = []
        for leg in ("FL", "FR", "RL", "RR"):
            leg_body_ids = []
            for link in ("hip", "thigh", "calf"):
                body_ids, _ = base._contact_sensor.find_bodies(f"{leg}_{link}")
                if len(body_ids) != 1:
                    raise RuntimeError(
                        f"expected one contact body for {leg}_{link}, got {len(body_ids)}"
                    )
                leg_body_ids.append(int(body_ids[0]))
            base._reward_probe_limb_contact_ids.append(leg_body_ids)
        print(
            "[REWARD_BUCKET_PROBE] limb contact bodies="
            + ",".join(str(len(body_ids)) for body_ids in base._reward_probe_limb_contact_ids),
            flush=True,
        )
        runner = Runner(env, experiment_cfg)
        loaded_modules = _load_modules(runner.agent, args.checkpoint)
        runner.agent.set_running_mode("eval")

        curriculum_restore = _restore_probe_curriculum(base, args.checkpoint)
        forced_stair_level = _force_stair_level(base, args.force_stair_level)
        obs, _ = env.reset()
        previous_action = torch.zeros(
            (base.num_envs, int(getattr(base.cfg, "action_space", 12))),
            device=base.device,
        )
        task_weight = float(runner.agent._task_reward_weight)
        style_weight = float(runner.agent._style_reward_weight)
        gamma = float(runner.agent._discount_factor)
        gae_lambda = float(runner.agent._lambda)
        time_limit_bootstrap = bool(runner.agent._time_limit_bootstrap)
        group_names = list(taili_reward.REWARD_GROUP_NAMES)

        metric_names: list[str] | None = None
        accumulators: dict[str, _BucketAccumulator] = {}
        rewards_history: list[Any] = []
        counterfactual_rewards_history: list[Any] = []
        counterfactual_component_history: list[Any] = []
        baseline_component_history: list[Any] = []
        counterfactual_outcome_component_names: tuple[str, ...] | None = None
        values_history: list[Any] = []
        next_values_history: list[Any] = []
        dones_history: list[Any] = []
        bucket_history: list[Any] = []
        episode_outcome_history: list[Any] = []
        episode_start = torch.zeros(base.num_envs, dtype=torch.long)
        episode_outcome_counts = [0 for _ in EPISODE_OUTCOME_NAMES]
        bucket_counts_initial: dict[str, int] | None = None
        bucket_counts_final: dict[str, int] | None = None
        collision_recovery: _CollisionRecoveryAudit | None = None
        continuous_quality = _ContinuousQualityCounterfactualAudit(
            base,
            taus_s=_float_grid(
                args.continuous_quality_taus,
                name="continuous-quality-taus",
                lower=1e-3,
                upper=10.0,
            ),
            floors=_float_grid(
                args.continuous_quality_floors,
                name="continuous-quality-floors",
                lower=0.0,
                upper=1.0,
            ),
        )

        total_steps = max(1, args.warmup_steps) + max(1, args.steps)
        for step_index in range(total_steps):
            value = _critic_value(runner.agent, obs)
            with torch.inference_mode():
                action_out = runner.agent.act(obs, timestep=0, timesteps=0)
                action = (
                    action_out[0]
                    if args.stochastic_actions
                    else action_out[-1].get("mean_actions", action_out[0])
                )
            action_delta = torch.linalg.vector_norm(action - previous_action, dim=-1) / math.sqrt(
                action.shape[-1]
            )
            next_obs, task_reward, terminated, truncated, _info = env.step(action)
            next_value = _critic_value(runner.agent, next_obs)
            previous_action = action.detach().clone()

            snapshot = captured.get("transition")
            if snapshot is None:
                raise RuntimeError("reward component capture did not observe an environment transition")
            masks = _bucket_masks(snapshot)
            if bucket_counts_initial is None:
                bucket_counts_initial = {name: int(mask.sum()) for name, mask in masks.items()}
            bucket_counts_final = {name: int(mask.sum()) for name, mask in masks.items()}
            latest_outcome = curriculum_outcomes.consume_latest()
            continuous_quality.update(
                snapshot,
                masks,
                terminated,
                truncated,
                latest_outcome,
                record_frame=step_index >= args.warmup_steps,
                record_episode=step_index >= args.warmup_steps,
            )

            amp_states = base.extras["amp_obs"]
            style_raw = _style_reward(runner.agent, amp_states)
            style_scale = _scalar(base.extras.get("amp_style_scale", 1.0), like=style_raw)
            style_effective = style_raw * style_scale
            task_flat = task_reward.reshape(-1)
            value_flat = value[:, 0]
            next_value_flat = next_value[:, 0] * ~terminated.reshape(-1)
            task_for_training = task_flat
            if time_limit_bootstrap:
                task_for_training = task_for_training + gamma * value_flat * truncated.reshape(-1)
            combined = task_weight * task_for_training + style_weight * style_effective
            counterfactual_components = captured.get("counterfactual_components")
            counterfactual_combined = None
            if counterfactual_components is not None:
                # 保留环境可能注入到 task reward 的外部分量，只替换核心 total。
                counterfactual_task = task_flat + (
                    counterfactual_components["total"].reshape(-1)
                    - snapshot["components"]["total"].reshape(-1)
                )
                counterfactual_task_for_training = counterfactual_task
                if time_limit_bootstrap:
                    counterfactual_task_for_training = (
                        counterfactual_task_for_training
                        + gamma * value_flat * truncated.reshape(-1)
                    )
                counterfactual_combined = (
                    task_weight * counterfactual_task_for_training
                    + style_weight * style_effective
                )

            if step_index < args.warmup_steps:
                obs = next_obs
                continue

            if collision_recovery is None:
                collision_recovery = _CollisionRecoveryAudit(
                    base, horizon_s=args.recovery_horizon_s
                )
                collision_recovery.previous_collision.copy_(
                    snapshot["limb_collision_response"]
                    > collision_recovery.event_threshold
                )
            else:
                collision_recovery.update(snapshot, masks, terminated, truncated)

            component_values = snapshot["components"]
            component_names = sorted(
                key
                for key, tensor in component_values.items()
                if tensor.ndim == 1 and tensor.shape[0] == base.num_envs
            )
            component_matrix = torch.stack([component_values[key] for key in component_names], dim=-1)
            counterfactual_component_names: list[str] = []
            counterfactual_component_matrix = torch.empty(
                (base.num_envs, 0), device=base.device
            )
            if counterfactual_components is not None:
                counterfactual_component_names = sorted(
                    key
                    for key, tensor in counterfactual_components.items()
                    if tensor.ndim == 1 and tensor.shape[0] == base.num_envs
                )
                counterfactual_component_matrix = torch.stack(
                    [counterfactual_components[key] for key in counterfactual_component_names],
                    dim=-1,
                )
                selected_names = tuple(
                    name
                    for name in COUNTERFACTUAL_OUTCOME_COMPONENT_NAMES
                    if name in counterfactual_components
                )
                if counterfactual_outcome_component_names is None:
                    counterfactual_outcome_component_names = selected_names
                elif selected_names != counterfactual_outcome_component_names:
                    raise RuntimeError(
                        "counterfactual outcome component schema changed during the probe"
                    )
                counterfactual_component_history.append(
                    torch.stack(
                        [counterfactual_components[name] for name in selected_names],
                        dim=-1,
                    ).detach().cpu()
                )
                baseline_component_history.append(
                    torch.stack(
                        [
                            component_values[name]
                            if name in component_values
                            else torch.full_like(
                                counterfactual_components[name], float("nan")
                            )
                            for name in selected_names
                        ],
                        dim=-1,
                    ).detach().cpu()
                )
            reward_groups = getattr(base, "_reward_groups", None)
            if reward_groups is None:
                reward_groups = taili_reward.group_reward_vector(component_values)
            reward_groups = reward_groups.reshape(base.num_envs, len(group_names))
            physical = _physical_metrics(snapshot, terminated, truncated)
            physical["physics/action_delta_rms"] = action_delta
            extra = {
                "reward/task_raw": task_flat,
                "reward/style_raw": style_raw,
                "reward/style_scale": style_scale,
                "reward/style_effective": style_effective,
                "reward/combined": combined,
                "reward/critic_value": value_flat,
                "reward/next_critic_value": next_value_flat,
                "reward/group_sum_error": reward_groups.sum(dim=-1) - task_flat,
                **physical,
            }
            if counterfactual_combined is not None:
                extra.update(
                    {
                        "counterfactual/reward/task_raw": counterfactual_task,
                        "counterfactual/reward/combined": counterfactual_combined,
                        "counterfactual/reward/task_delta": counterfactual_task - task_flat,
                        "counterfactual/reward/combined_delta": counterfactual_combined - combined,
                    }
                )
            extra_names = list(extra)
            extra_matrix = torch.stack([extra[name] for name in extra_names], dim=-1)
            weighted_groups = task_weight * reward_groups
            matrix = torch.cat(
                (
                    component_matrix,
                    counterfactual_component_matrix,
                    reward_groups,
                    weighted_groups,
                    extra_matrix,
                ),
                dim=-1,
            )
            names = (
                [f"component/{name}" for name in component_names]
                + [
                    f"counterfactual/component/{name}"
                    for name in counterfactual_component_names
                ]
                + [f"group_raw/{name}" for name in group_names]
                + [f"group_weighted/{name}" for name in group_names]
                + extra_names
            )
            if metric_names is None:
                metric_names = names
                accumulators = {
                    name: _BucketAccumulator(metric_names, base.device) for name in BUCKET_NAMES
                }
            elif names != metric_names:
                raise RuntimeError("reward component schema changed during the probe")
            for name, mask in masks.items():
                accumulators[name].add(matrix, mask)

            bucket_codes = torch.full(
                (base.num_envs,), -1, dtype=torch.int8, device=base.device
            )
            for code, name in enumerate(BUCKET_NAMES):
                bucket_codes[masks[name]] = code
            rewards_history.append(combined.detach().cpu())
            if counterfactual_combined is not None:
                counterfactual_rewards_history.append(
                    counterfactual_combined.detach().cpu()
                )
            values_history.append(value_flat.detach().cpu())
            next_values_history.append(next_value_flat.detach().cpu())
            dones_history.append((terminated | truncated).reshape(-1).detach().cpu())
            bucket_history.append(bucket_codes.detach().cpu())
            sample_index = len(episode_outcome_history)
            episode_outcome_history.append(
                torch.full((base.num_envs,), -1, dtype=torch.int8)
            )
            done_cpu = (terminated | truncated).reshape(-1).detach().cpu().bool()
            if latest_outcome is not None and latest_outcome.get("env_ids") is not None:
                outcome_ids = latest_outcome["env_ids"].detach().cpu().long().reshape(-1)
                outcome_valid = latest_outcome["valid"].detach().cpu().bool().reshape(-1)
                outcome_direction = latest_outcome["direction"].detach().cpu().reshape(-1)
                outcome_success = latest_outcome["controlled_height"].detach().cpu().bool().reshape(-1)
                outcome_failure = latest_outcome["failure_down"].detach().cpu().bool().reshape(-1)
                for local_index, env_id_tensor in enumerate(outcome_ids):
                    env_id = int(env_id_tensor)
                    if not bool(outcome_valid[local_index]):
                        continue
                    direction_offset = 3 if float(outcome_direction[local_index]) > 0.0 else 0
                    if bool(outcome_success[local_index]):
                        outcome_code = direction_offset
                    elif bool(outcome_failure[local_index]):
                        outcome_code = direction_offset + 1
                    else:
                        outcome_code = direction_offset + 2
                    episode_outcome_counts[outcome_code] += 1
                    first_index = int(episode_start[env_id])
                    for history_index in range(first_index, sample_index + 1):
                        episode_outcome_history[history_index][env_id] = outcome_code
            episode_start[done_cpu] = sample_index + 1
            obs = next_obs

        if not rewards_history:
            raise RuntimeError("probe collected no post-warmup transitions")
        if collision_recovery is None:
            raise RuntimeError("collision recovery audit was not initialized")
        rewards_tensor = torch.stack(rewards_history)
        values_tensor = torch.stack(values_history)
        next_values_tensor = torch.stack(next_values_history)
        dones_tensor = torch.stack(dones_history)
        bucket_tensor = torch.stack(bucket_history)
        episode_outcome_tensor = torch.stack(episode_outcome_history)
        td = rewards_tensor - values_tensor + gamma * next_values_tensor
        advantages = torch.zeros_like(td)
        carry = torch.zeros_like(td[0])
        for index in reversed(range(td.shape[0])):
            carry = td[index] + gamma * gae_lambda * (~dones_tensor[index]) * carry
            advantages[index] = carry
        valid = bucket_tensor >= 0
        global_advantage_mean = advantages[valid].mean()
        global_advantage_std = advantages[valid].std().clamp_min(1e-8)
        normalized_advantages = (advantages - global_advantage_mean) / global_advantage_std

        advantage_report: dict[str, Any] = {}
        for code, name in enumerate(BUCKET_NAMES):
            mask = bucket_tensor == code
            if not bool(mask.any()):
                advantage_report[name] = {"samples": 0}
                continue
            bucket_td = td[mask]
            bucket_adv = advantages[mask]
            bucket_norm = normalized_advantages[mask]
            advantage_report[name] = {
                "samples": int(mask.sum()),
                "td_mean": float(bucket_td.mean()),
                "td_std": float(bucket_td.std()),
                "gae_mean": float(bucket_adv.mean()),
                "gae_std": float(bucket_adv.std()),
                "normalized_gae_mean": float(bucket_norm.mean()),
                "normalized_gae_positive_fraction": float((bucket_norm > 0).float().mean()),
            }

        outcome_advantage_report: dict[str, Any] = {}
        for code, name in enumerate(EPISODE_OUTCOME_NAMES):
            mask = episode_outcome_tensor == code
            if not bool(mask.any()):
                outcome_advantage_report[name] = {"samples": 0, "episodes": 0}
                continue
            outcome_td = td[mask]
            outcome_adv = advantages[mask]
            outcome_norm = normalized_advantages[mask]
            outcome_advantage_report[name] = {
                "samples": int(mask.sum()),
                "episodes": episode_outcome_counts[code],
                "td_mean": float(outcome_td.mean()),
                "td_std": float(outcome_td.std()),
                "gae_mean": float(outcome_adv.mean()),
                "gae_std": float(outcome_adv.std()),
                "normalized_gae_mean": float(outcome_norm.mean()),
                "normalized_gae_positive_fraction": float((outcome_norm > 0).float().mean()),
            }

        counterfactual_report: dict[str, Any] | None = None
        if counterfactual_rewards_history:
            counterfactual_rewards = torch.stack(counterfactual_rewards_history)
            if counterfactual_rewards.shape != rewards_tensor.shape:
                raise RuntimeError("counterfactual reward history does not match baseline")
            counterfactual_td = (
                counterfactual_rewards - values_tensor + gamma * next_values_tensor
            )
            counterfactual_advantages = torch.zeros_like(counterfactual_td)
            counterfactual_carry = torch.zeros_like(counterfactual_td[0])
            for index in reversed(range(counterfactual_td.shape[0])):
                counterfactual_carry = (
                    counterfactual_td[index]
                    + gamma
                    * gae_lambda
                    * (~dones_tensor[index])
                    * counterfactual_carry
                )
                counterfactual_advantages[index] = counterfactual_carry
            counterfactual_global_mean = counterfactual_advantages[valid].mean()
            counterfactual_global_std = counterfactual_advantages[valid].std().clamp_min(1e-8)
            counterfactual_normalized = (
                counterfactual_advantages - counterfactual_global_mean
            ) / counterfactual_global_std
            delta_td = counterfactual_td - td
            delta_advantages = counterfactual_advantages - advantages

            counterfactual_bucket_report: dict[str, Any] = {}
            for code, name in enumerate(BUCKET_NAMES):
                mask = bucket_tensor == code
                if not bool(mask.any()):
                    counterfactual_bucket_report[name] = {"samples": 0}
                    continue
                counterfactual_bucket_report[name] = {
                    "samples": int(mask.sum()),
                    "td_mean": float(counterfactual_td[mask].mean()),
                    "gae_mean": float(counterfactual_advantages[mask].mean()),
                    "normalized_gae_mean": float(counterfactual_normalized[mask].mean()),
                    "td_delta_mean": float(delta_td[mask].mean()),
                    "gae_delta_mean": float(delta_advantages[mask].mean()),
                }

            counterfactual_outcome_report: dict[str, Any] = {}
            for code, name in enumerate(EPISODE_OUTCOME_NAMES):
                mask = episode_outcome_tensor == code
                if not bool(mask.any()):
                    counterfactual_outcome_report[name] = {
                        "samples": 0,
                        "episodes": 0,
                    }
                    continue
                counterfactual_outcome_report[name] = {
                    "samples": int(mask.sum()),
                    "episodes": episode_outcome_counts[code],
                    "td_mean": float(counterfactual_td[mask].mean()),
                    "gae_mean": float(counterfactual_advantages[mask].mean()),
                    "normalized_gae_mean": float(counterfactual_normalized[mask].mean()),
                    "td_delta_mean": float(delta_td[mask].mean()),
                    "gae_delta_mean": float(delta_advantages[mask].mean()),
                }

            component_outcome_report: dict[str, Any] = {}
            if counterfactual_component_history:
                if counterfactual_outcome_component_names is None:
                    raise RuntimeError("counterfactual component names were not captured")
                candidate_components = torch.stack(counterfactual_component_history)
                baseline_components = torch.stack(baseline_component_history)
                if candidate_components.shape[:2] != episode_outcome_tensor.shape:
                    raise RuntimeError(
                        "counterfactual component history does not match episode outcomes"
                    )
                for code, outcome_name in enumerate(EPISODE_OUTCOME_NAMES):
                    mask = episode_outcome_tensor == code
                    if not bool(mask.any()):
                        component_outcome_report[outcome_name] = {
                            "samples": 0,
                            "episodes": 0,
                            "components": {},
                        }
                        continue
                    component_metrics: dict[str, Any] = {}
                    for component_index, component_name in enumerate(
                        counterfactual_outcome_component_names
                    ):
                        candidate_values = candidate_components[..., component_index][mask]
                        baseline_values = baseline_components[..., component_index][mask]
                        finite_baseline = torch.isfinite(baseline_values)
                        metric = {
                            "counterfactual_mean": float(candidate_values.mean()),
                        }
                        if bool(finite_baseline.any()):
                            paired_candidate = candidate_values[finite_baseline]
                            paired_baseline = baseline_values[finite_baseline]
                            metric.update(
                                {
                                    "baseline_mean": float(paired_baseline.mean()),
                                    "delta_mean": float(
                                        (paired_candidate - paired_baseline).mean()
                                    ),
                                }
                            )
                        component_metrics[component_name] = metric
                    component_outcome_report[outcome_name] = {
                        "samples": int(mask.sum()),
                        "episodes": episode_outcome_counts[code],
                        "components": component_metrics,
                    }
            counterfactual_report = {
                "definition": counterfactual_metadata,
                "global_gae_mean": float(counterfactual_global_mean),
                "global_gae_std": float(counterfactual_global_std),
                "buckets": counterfactual_bucket_report,
                "episode_outcomes": counterfactual_outcome_report,
                "components_by_episode_outcome": component_outcome_report,
            }

        value_output_dim = int(value.shape[-1])
        return {
            "checkpoint": args.checkpoint,
            "agent_yaml": args.agent_yaml,
            "seed": int(args.seed),
            "num_envs": int(base.num_envs),
            "warmup_steps": int(args.warmup_steps),
            "sample_steps": int(args.steps),
            "action_mode": "stochastic" if args.stochastic_actions else "mean",
            "loaded_modules": loaded_modules,
            "critic": {
                "output_dim": value_output_dim,
                "architecture": "scalar" if value_output_dim == 1 else "vector",
                "reward_groups_exposed_for_probe": True,
                "reward_groups_used_by_current_checkpoint": value_output_dim == len(group_names),
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "global_gae_mean": float(global_advantage_mean),
                "global_gae_std": float(global_advantage_std),
            },
            "weights": {
                "task": task_weight,
                "style": style_weight,
                "discriminator": float(runner.agent._discriminator_reward_scale),
            },
            "terrain": {
                "phase": int(getattr(base, "_phase", -1)),
                "probe_curriculum_restore": curriculum_restore,
                "level_stats_after_reset": base._terrain_level_stats(),
                "forced_stair_level": forced_stair_level,
                "replay_active": int(getattr(base, "_terrain_replay_mask").sum()),
                "frontier_active": int(getattr(base, "_terrain_frontier_mask").sum()),
                "bucket_counts_initial": bucket_counts_initial,
                "bucket_counts_final": bucket_counts_final,
            },
            "curriculum_outcomes": curriculum_outcomes.report(),
            "collision_recovery": collision_recovery.report(),
            "continuous_quality_counterfactual": continuous_quality.report(),
            "exact_reward_counterfactual": counterfactual_report,
            "buckets": {name: accumulator.report() for name, accumulator in accumulators.items()},
            "advantages": advantage_report,
            "advantages_by_episode_outcome": outcome_advantage_report,
        }
    finally:
        taili_reward.compute_reward_components = original_compute
        amp_env_module.compute_terrain_curriculum_moves = original_curriculum
        if original_terrain_snapshot is not None:
            base_holder["base"]._terrain_curriculum_snapshot = original_terrain_snapshot
        try:
            env.close()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - requires IsaacLab
    args = _parser().parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.pop("TAILI_DIAGNOSTIC", None)
    os.environ["TAILI_RESUME_CHECKPOINT"] = args.checkpoint
    os.environ.setdefault("TAILI_MULTI_CRITIC", "1")
    os.environ.setdefault("TAILI_INIT_PHASE", "2")
    os.environ.setdefault("TAILI_RESTORE_PHASE_MAX", "2")
    os.environ.setdefault("TAILI_RESTORE_DR_MAX", "0")
    os.environ.setdefault("TAILI_TELEMETRY_INTERVAL", "1000000")
    os.environ.setdefault("TAILI_RUN_DIR", str(out_path.parent))
    os.environ.setdefault("TAILI_RUN_ID", out_path.parent.name)
    try:
        from isaaclab.app import AppLauncher
    except Exception as exc:
        raise RuntimeError("this probe must run in the IsaacLab Python environment") from exc

    launch_args = argparse.Namespace(**vars(args))
    launch_args.headless = bool(args.headless)
    launch_args.enable_cameras = False
    app_launcher = AppLauncher(launch_args)
    app = app_launcher.app
    try:
        result = _run(args)
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(f"[REWARD_BUCKET_PROBE] complete: {out_path}", flush=True)
    except BaseException:
        out_path.with_suffix(".error.json").write_text(
            json.dumps({"traceback": traceback.format_exc()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
