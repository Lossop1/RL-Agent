"""Inspect optimizer state and floating-point parameter deltas in Taili checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _load(path: Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint root must be a mapping: {path}")
    return value


def _describe(name: str, checkpoint: dict) -> None:
    print(f"=== {name} ===")
    print("keys", sorted(checkpoint))
    for key, value in checkpoint.items():
        if not isinstance(value, dict):
            continue
        if "param_groups" in value:
            lrs = [group.get("lr") for group in value["param_groups"]]
            print("optimizer", key, "lrs", lrs, "states", len(value.get("state", {})))
            continue
        tensors = [item for item in value.values() if torch.is_tensor(item)]
        if tensors:
            print(
                "state_dict",
                key,
                "tensors",
                len(tensors),
                "numel",
                sum(item.numel() for item in tensors),
            )


def _compare(name: str, source: dict, candidate: dict) -> None:
    print(f"=== delta {name} vs source ===")
    for key in sorted(source.keys() & candidate.keys()):
        left = source[key]
        right = candidate[key]
        if not isinstance(left, dict) or not isinstance(right, dict):
            continue
        pairs = [
            (left[item].float(), right[item].float())
            for item in left.keys() & right.keys()
            if torch.is_tensor(left[item])
            and torch.is_tensor(right[item])
            and left[item].shape == right[item].shape
            and left[item].is_floating_point()
        ]
        if not pairs:
            continue
        squared_delta = sum((a - b).square().sum().item() for a, b in pairs)
        squared_source = sum(a.square().sum().item() for a, _ in pairs)
        max_abs = max((a - b).abs().max().item() for a, b in pairs)
        relative_l2 = (squared_delta / max(squared_source, 1e-30)) ** 0.5
        print(key, "rel_l2", relative_l2, "max_abs", max_abs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("candidates", nargs="+", type=Path)
    args = parser.parse_args()

    source = _load(args.source)
    _describe("source", source)
    for path in args.candidates:
        candidate = _load(path)
        _describe(path.name, candidate)
        _compare(path.name, source, candidate)


if __name__ == "__main__":
    main()
