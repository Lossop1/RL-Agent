#!/usr/bin/env python3
"""直接比较检查点里的权重，判断"恢复"到底有没有把状态读进去。

用法（在**容器内**跑，因为要 torch）：
    /workspace/isaaclab/_isaac_sim/python.sh tools/verify_resume_weights.py 文件A.pt 文件B.pt [更多.pt ...]

为什么要有这个脚本：P4.3 原本的验收标准是「恢复后曲线连续，无性能跳变」，
但 2026-09-16 在 3060 上实测发现这条标准**不可测量**——一条没被打断的跑，
相邻采样点相对差的中位数就有 0.180，85% 的相邻点差超过 10%。奖励的噪声
压过了要判定的效应。真正能失败的判据是权重本身，所以有了这个脚本。

推荐的配对（成对是必须的，缺了对照就什么都证明不了）：

    # 父运行自己走 N 步后的样子，用来当"正常位移"的尺子
    A1000.pt  从 1000 步处恢复后跑 200 步写出的 agent_200.pt
    A1200.pt  父运行一路跑到 1200 步写出的 agent_1200.pt
    # 对照：同样的命令、同样的步数，只是不给 --checkpoint
    E200.pt   全新跑 200 步写出的 agent_200.pt

判读：
  * 父运行 vs 恢复跑：相对 L2 应**远小于**同长度的正常位移，才算恢复成功。
    2026-09-16 实测的分离度是 0.0011（复现误差）对 0.6665（正常位移）。
  * 父运行 vs 对照：应接近 2.0 量级（完全不相干），否则对照没起作用。
  * 另有一条**不依赖阈值**的离散判据：`optimizer.state` 的条目数与 step 计数器。
    新 agent 在第一次 `optimizer.step()` 之前 `state_dict()` 里根本没有 state，
    所以「恢复跑有 99 个非零的 optimizer 张量、对照有 0 个」本身就是证据。

注意：value 与 discriminator 两组在本方法下常常**分不出来**——它们初值的范数
压过了训练带来的位移，无关参照的距离可能比正常位移还小。那两项要看"逐位相等"
（即不训练直接比对，相对 L2 应为 0），别用这个表的距离下结论。
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


def leaves(obj, prefix: str = "") -> dict:
    """把任意嵌套结构拍平成 {路径: 张量}。"""
    out: dict = {}
    if torch.is_tensor(obj):
        out[prefix or "<root>"] = obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            out.update(leaves(value, f"{prefix}/{key}" if prefix else str(key)))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            out.update(leaves(value, f"{prefix}/{index}"))
    return out


def optimizer_state(obj) -> tuple[int, list[int]]:
    """返回 (state 条目数, step 计数器的取值集合)。"""
    state = ((obj.get("optimizer") or {}).get("state") or {}) if isinstance(obj, dict) else {}
    steps = sorted(
        {int(item["step"]) for item in state.values() if torch.is_tensor(item.get("step"))}
    )
    return len(state), steps


def compare(a: dict, b: dict) -> tuple[float, float, float, int] | None:
    """返回 (余弦, 相对L2, 最大绝对差, 共有张量数)。"""
    shared = sorted(set(a) & set(b))
    if not shared:
        return None
    flat_a = torch.cat([a[key].float().reshape(-1) for key in shared])
    flat_b = torch.cat([b[key].float().reshape(-1) for key in shared])
    norm_a, norm_b = flat_a.norm(), flat_b.norm()
    denom = float(norm_a + norm_b)
    cosine = float(torch.dot(flat_a, flat_b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else float("nan")
    relative = float((flat_a - flat_b).norm() / denom) * 2 if denom > 0 else float("nan")
    max_abs = max(float((a[key].float() - b[key].float()).abs().max()) for key in shared)
    return cosine, relative, max_abs, len(shared)


def group_by_top(flat: dict) -> dict[str, dict]:
    grouped: dict[str, dict] = {}
    for key, value in flat.items():
        grouped.setdefault(key.split("/")[0], {})[key] = value
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description="比较检查点权重，判断恢复是否真的发生了")
    parser.add_argument("checkpoints", nargs="+", help="两个或更多 .pt 检查点路径")
    parser.add_argument(
        "--label",
        nargs="*",
        default=None,
        help="给每个检查点一个短标签，顺序与位置参数一致；省略则用 父目录/文件名",
    )
    args = parser.parse_args()

    if len(args.checkpoints) < 2:
        print("至少要给两个检查点：", file=sys.stderr)
        return 2
    if args.label and len(args.label) != len(args.checkpoints):
        print("--label 的个数要和检查点个数一致", file=sys.stderr)
        return 2

    objs: dict[str, object] = {}
    flats: dict[str, dict] = {}
    for index, path in enumerate(args.checkpoints):
        if not os.path.isfile(path):
            print(f"!! 文件不存在: {path}", file=sys.stderr)
            continue
        tag = args.label[index] if args.label else f"{os.path.basename(os.path.dirname(os.path.dirname(path)))}/{os.path.basename(path)[:-3]}"
        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            print(f"!! {tag} 读取失败: {exc}", file=sys.stderr)
            continue
        objs[tag] = obj
        flats[tag] = leaves(obj)
        count, steps = optimizer_state(obj)
        print(
            f"{tag:<28} {os.path.getsize(path):>10} 字节  "
            f"顶层键 {len(obj) if isinstance(obj, dict) else 1:>2}  "
            f"optimizer state 条目 {count:>3}  step {steps}"
        )

    names = list(flats)
    if len(names) < 2:
        print("可比较的检查点不足两个", file=sys.stderr)
        return 2

    print()
    print(f"{'比较':<40} {'余弦':>10} {'相对L2':>10} {'最大绝对差':>12} {'张量数':>7}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            result = compare(flats[names[i]], flats[names[j]])
            if result is None:
                print(f"{names[i] + ' vs ' + names[j]:<40} {'（无共有张量）':>10}")
                continue
            cosine, relative, max_abs, count = result
            print(
                f"{names[i] + ' vs ' + names[j]:<40} {cosine:>10.6f} {relative:>10.6f} "
                f"{max_abs:>12.3e} {count:>7}"
            )

    keys = sorted(set().union(*[set(group_by_top(flat)) for flat in flats.values()]))
    print()
    print("=== 按顶层键分组（value / discriminator 常因初值范数占优而不可用，见文首说明）===")
    for key in keys:
        print(f"\n-- {key} --")
        print(f"  {'比较':<38} {'余弦':>10} {'相对L2':>10} {'最大绝对差':>12} {'张量数':>7}")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                left = group_by_top(flats[names[i]]).get(key, {})
                right = group_by_top(flats[names[j]]).get(key, {})
                result = compare(left, right)
                if result is None:
                    continue
                cosine, relative, max_abs, count = result
                print(
                    f"  {names[i] + ' vs ' + names[j]:<38} {cosine:>10.6f} {relative:>10.6f} "
                    f"{max_abs:>12.3e} {count:>7}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
