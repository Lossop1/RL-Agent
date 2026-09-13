"""验证IsaacLab适配器等价性工具。

P6.1步骤4：验证适配器与原始环境的等价性（MAE < 1e-6）。

使用方法：
    python tools/verify_isaaclab_adapter_equivalence.py --steps 100 --seed 42

验证内容：
- 观测值差异（逐tensor比较）
- 奖励值差异
- 终止标志差异
- info字典内容对应
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import numpy as np


def compute_mae(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """计算两个张量的平均绝对误差。"""
    return float(torch.mean(torch.abs(tensor1 - tensor2)).item())


def compute_max_ae(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    """计算两个张量的最大绝对误差。"""
    return float(torch.max(torch.abs(tensor1 - tensor2)).item())


def compare_observations(obs1: Dict[str, torch.Tensor], obs2: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """比较两个观测字典的差异。"""
    results = {}
    all_keys = set(obs1.keys()) | set(obs2.keys())

    for key in all_keys:
        if key not in obs1:
            results[key] = {"status": "missing_in_original"}
        elif key not in obs2:
            results[key] = {"status": "missing_in_adapter"}
        else:
            mae = compute_mae(obs1[key], obs2[key])
            max_ae = compute_max_ae(obs1[key], obs2[key])
            results[key] = {
                "status": "ok",
                "mae": mae,
                "max_ae": max_ae,
                "shape": list(obs1[key].shape),
            }

    return results


def verify_equivalence(
    task: str,
    num_steps: int = 100,
    seed: int = 42,
    num_envs: int = 4,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """验证IsaacLab适配器与原始环境的等价性。

    Args:
        task: 环境任务名称
        num_steps: 验证步数
        seed: 随机种子
        num_envs: 并行环境数
        device: 设备

    Returns:
        验证报告字典，包含所有差异统计
    """
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from autotuner.simulation.isaaclab_adapter import IsaacLabAdapter

    # 创建环境配置
    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    env_cfg.seed = seed

    # 创建原始环境和适配器环境
    print(f"创建原始环境: {task}")
    env_original = gym.make(task, cfg=env_cfg, render_mode=None)

    print(f"创建适配器环境: {task}")
    env_cfg_copy = parse_env_cfg(task, device=device, num_envs=num_envs)
    env_cfg_copy.seed = seed
    env_inner = gym.make(task, cfg=env_cfg_copy, render_mode=None)
    env_adapter = IsaacLabAdapter(env_inner)

    # 验证记录
    obs_diffs: List[Dict[str, Any]] = []
    reward_diffs: List[float] = []
    done_mismatches: List[int] = []

    # 重置环境
    print("重置环境...")
    obs_orig, _ = env_original.reset()
    obs_adapter = env_adapter.reset()

    reset_diff = compare_observations(obs_orig, obs_adapter)
    print(f"Reset观测差异: {reset_diff}")

    # 运行仿真
    print(f"运行{num_steps}步仿真...")
    for step in range(num_steps):
        # 生成相同的随机动作
        torch.manual_seed(seed + step)
        actions = torch.randn(num_envs, 12, device=torch.device(device)) * 0.1

        # 原始环境step
        obs_orig, reward_orig, term_orig, trunc_orig, info_orig = env_original.step(actions)
        done_orig = term_orig | trunc_orig

        # 适配器环境step
        obs_adapter, reward_adapter, done_adapter, info_adapter = env_adapter.step(actions)

        # 比较观测
        obs_diff = compare_observations(obs_orig, obs_adapter)
        obs_diffs.append(obs_diff)

        # 比较奖励
        reward_diff = compute_mae(reward_orig, reward_adapter)
        reward_diffs.append(reward_diff)

        # 比较终止标志
        done_mismatch = int(torch.sum(done_orig != done_adapter).item())
        done_mismatches.append(done_mismatch)

        if (step + 1) % 10 == 0:
            avg_obs_mae = np.mean([
                obs_diff.get("policy", {}).get("mae", 0.0)
                for obs_diff in obs_diffs[-10:]
            ])
            avg_reward_mae = np.mean(reward_diffs[-10:])
            print(f"Step {step + 1}/{num_steps}: "
                  f"obs_mae={avg_obs_mae:.2e}, "
                  f"reward_mae={avg_reward_mae:.2e}, "
                  f"done_mismatch={done_mismatch}")

    # 关闭环境
    env_original.close()
    env_adapter.close()

    # 生成报告
    report = {
        "task": task,
        "num_steps": num_steps,
        "seed": seed,
        "num_envs": num_envs,
        "device": device,
        "reset_observation_diff": reset_diff,
        "step_statistics": {
            "observation": {
                "policy_mae_mean": float(np.mean([d.get("policy", {}).get("mae", 0.0) for d in obs_diffs])),
                "policy_mae_max": float(np.max([d.get("policy", {}).get("mae", 0.0) for d in obs_diffs])),
                "policy_max_ae_mean": float(np.mean([d.get("policy", {}).get("max_ae", 0.0) for d in obs_diffs])),
                "policy_max_ae_max": float(np.max([d.get("policy", {}).get("max_ae", 0.0) for d in obs_diffs])),
            },
            "reward": {
                "mae_mean": float(np.mean(reward_diffs)),
                "mae_max": float(np.max(reward_diffs)),
            },
            "done": {
                "total_mismatches": int(np.sum(done_mismatches)),
                "steps_with_mismatch": int(np.sum(np.array(done_mismatches) > 0)),
            },
        },
        "acceptance_criteria": {
            "obs_mae_threshold": 1e-6,
            "reward_mae_threshold": 1e-6,
            "done_mismatch_threshold": 0,
        },
        "verdict": "UNKNOWN",
    }

    # 判断是否通过验收标准
    obs_mae = report["step_statistics"]["observation"]["policy_mae_max"]
    reward_mae = report["step_statistics"]["reward"]["mae_max"]
    done_mismatch = report["step_statistics"]["done"]["total_mismatches"]

    if (obs_mae < 1e-6 and reward_mae < 1e-6 and done_mismatch == 0):
        report["verdict"] = "PASS"
    else:
        report["verdict"] = "FAIL"
        report["failure_reasons"] = []
        if obs_mae >= 1e-6:
            report["failure_reasons"].append(f"观测MAE={obs_mae:.2e} >= 1e-6")
        if reward_mae >= 1e-6:
            report["failure_reasons"].append(f"奖励MAE={reward_mae:.2e} >= 1e-6")
        if done_mismatch > 0:
            report["failure_reasons"].append(f"终止标志不匹配次数={done_mismatch} > 0")

    return report


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="验证IsaacLab适配器等价性")
    parser.add_argument(
        "--task",
        default="RobotLab-Isaac-Taili-AMP-Blind-Direct-v0",
        help="环境任务名称"
    )
    parser.add_argument("--steps", type=int, default=100, help="验证步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-envs", type=int, default=4, help="并行环境数")
    parser.add_argument("--device", default="cuda:0", help="设备")
    parser.add_argument(
        "--output",
        default="equivalence_report.json",
        help="输出报告文件路径"
    )
    args = parser.parse_args(argv)

    print("=" * 70)
    print("IsaacLab适配器等价性验证工具 - P6.1步骤4")
    print("=" * 70)
    print(f"任务: {args.task}")
    print(f"步数: {args.steps}")
    print(f"种子: {args.seed}")
    print(f"环境数: {args.num_envs}")
    print(f"设备: {args.device}")
    print("=" * 70)

    try:
        report = verify_equivalence(
            task=args.task,
            num_steps=args.steps,
            seed=args.seed,
            num_envs=args.num_envs,
            device=args.device,
        )

        # 保存报告
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("=" * 70)
        print("验证完成")
        print("=" * 70)
        print(f"判定: {report['verdict']}")
        print(f"观测MAE最大值: {report['step_statistics']['observation']['policy_mae_max']:.2e}")
        print(f"奖励MAE最大值: {report['step_statistics']['reward']['mae_max']:.2e}")
        print(f"终止标志不匹配总数: {report['step_statistics']['done']['total_mismatches']}")

        if report["verdict"] == "FAIL":
            print("\n失败原因:")
            for reason in report.get("failure_reasons", []):
                print(f"  - {reason}")

        print(f"\n报告已保存到: {output_path.absolute()}")
        print("=" * 70)

        return 0 if report["verdict"] == "PASS" else 1

    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
