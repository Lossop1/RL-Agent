"""Data-disk launcher for the Taili blind runtime payload.

This module is intentionally small and explicit. It does not know about a
remote RobotLab source tree. The caller supplies the IsaacLab/skrl training
entry point, and this launcher supplies the Taili runtime package plus a stable
run-directory contract:

    /root/gpufree-data/taili_runs/<run_id>/
      agent.skrl.yaml
      train.log                 semantic [TP*] telemetry emitted by the env
      train.telemetry.jsonl     structured telemetry for the console/LLM
      console.log               complete stdout/stderr from the train process
      checkpoints/              skrl checkpoints
      events.out.tfevents.*     TensorBoard events, written by skrl

The launcher is fail-fast on setup errors but does not interpret training
results. It just makes the runtime observable and relocatable.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml

from .taili_blind_config import (
    CONFIG_FILENAME,
    build_skrl_config,
    effective_config_with_overrides,
    get_config_value,
    load_taili_blind_config,
    resolve_config_path,
    write_taili_blind_config,
)

_DEFAULT_CFG = load_taili_blind_config()
DEFAULT_TASK = str(_DEFAULT_CFG["task"]["default_id"])
DEFAULT_DATA_ROOT = str(_DEFAULT_CFG["runtime"]["data_root"])
DEFAULT_TELEMETRY_INTERVAL = int(_DEFAULT_CFG["runtime"]["telemetry_interval"])


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _clean_remainder(items: list[str]) -> list[str]:
    if items and items[0] == "--":
        return items[1:]
    return items


def _has_flag(items: list[str], *names: str) -> bool:
    names_set = set(names)
    for item in items:
        if item in names_set:
            return True
        if any(item.startswith(name + "=") for name in names_set):
            return True
    return False


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def _patched_agent_cfg(
    *,
    package_dir: Path,
    config_path: Path,
    run_root: Path,
    run_id: str,
    run_dir: Path,
    total_steps: int,
) -> tuple[Path, Path]:
    source_cfg = load_taili_blind_config(config_path)
    write_interval = os.environ.get("TAILI_WRITE_INTERVAL")
    checkpoint_interval = (
        int(os.environ["TAILI_CHECKPOINT_INTERVAL"])
        if os.environ.get("TAILI_CHECKPOINT_INTERVAL")
        else None
    )
    effective = effective_config_with_overrides(
        source_cfg,
        total_steps=total_steps,
        write_interval=write_interval,
        checkpoint_interval=checkpoint_interval,
    )
    cfg = build_skrl_config(effective)
    agent = cfg.setdefault("agent", {})
    if not isinstance(agent, dict):
        raise ValueError("agent section in skrl cfg must be a mapping")
    exp = agent.setdefault("experiment", {})
    if not isinstance(exp, dict):
        raise ValueError("agent.experiment section in skrl cfg must be a mapping")

    # skrl treats `directory` as the experiment parent directory and
    # `experiment_name` as the run directory name. Setting both makes the final
    # experiment directory deterministic: <run_root>/<run_id>.
    exp["directory"] = str(run_root)
    exp["experiment_name"] = run_id

    trainer = cfg.setdefault("trainer", {})
    if not isinstance(trainer, dict):
        raise ValueError("trainer section in skrl cfg must be a mapping")
    if total_steps > 0:
        trainer["timesteps"] = total_steps

    target = run_dir / "agent.skrl.yaml"
    _write_yaml(target, cfg)
    effective.setdefault("skrl", {})["agent"] = agent
    effective["skrl"]["trainer"] = trainer
    effective.setdefault("runtime", {})["resolved"] = {
        "config_path": str(config_path),
        "agent_cfg": str(target),
        "run_root": str(run_root),
        "run_id": run_id,
        "run_dir": str(run_dir),
    }
    effective_path = run_dir / "effective_config.yaml"
    write_taili_blind_config(effective_path, effective)
    return target, effective_path


def _build_env(
    *,
    payload_root: Path,
    run_root: Path,
    run_id: str,
    run_dir: Path,
    agent_cfg: Path,
    total_steps: int,
    telemetry_interval: int,
) -> dict[str, str]:
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(payload_root) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    env["TAILI_DATA_ROOT"] = env.get("TAILI_DATA_ROOT", str(run_root.parent))
    env["TAILI_RUN_ROOT"] = str(run_root)
    env["TAILI_RUN_ID"] = run_id
    env["TAILI_RUN_DIR"] = str(run_dir)
    env["TAILI_TRAIN_LOG"] = str(run_dir / "train.log")
    env["TAILI_TELEMETRY_JSONL"] = str(run_dir / "train.telemetry.jsonl")
    env["TAILI_CONSOLE_LOG"] = str(run_dir / "console.log")
    env["TAILI_CHECKPOINT_DIR"] = str(run_dir / "checkpoints")
    env["TAILI_SKRL_CFG_ENTRY_POINT"] = str(agent_cfg)
    env["TAILI_CONFIG_PATH"] = str(agent_cfg.parent / CONFIG_FILENAME)
    env["TAILI_TELEMETRY_INTERVAL"] = str(max(1, int(telemetry_interval)))
    existing_cuda_alloc = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    alloc_parts = [part for part in existing_cuda_alloc.split(",") if part]
    if not any(part.startswith("expandable_segments:") for part in alloc_parts):
        alloc_parts.append("expandable_segments:True")
    env["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(alloc_parts)
    if total_steps > 0:
        env["TAILI_TOTAL_STEPS"] = str(total_steps)
    return env


def _write_run_metadata(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch Taili blind training from the self-contained payload.")
    parser.add_argument(
        "--train-entry",
        default=os.environ.get("TAILI_TRAIN_ENTRY", ""),
        help="Optional external train.py. Omit to use payload-owned taili_blind_runtime.train_taili.",
    )
    parser.add_argument("--python", default=os.environ.get("TAILI_PYTHON", sys.executable))
    parser.add_argument("--task", default=os.environ.get("TAILI_TASK", DEFAULT_TASK))
    parser.add_argument(
        "--config",
        default=os.environ.get("TAILI_CONFIG_PATH", ""),
        help="Single source Taili config YAML. Defaults to taili_blind_config.yaml in the runtime package.",
    )
    parser.add_argument("--data-root", default=os.environ.get("TAILI_DATA_ROOT", DEFAULT_DATA_ROOT))
    parser.add_argument("--run-root", default=os.environ.get("TAILI_RUN_ROOT", ""))
    parser.add_argument("--run-id", default=os.environ.get("TAILI_RUN_ID", ""))
    parser.add_argument("--total-steps", type=int, default=int(os.environ.get("TAILI_TOTAL_STEPS", "0") or 0))
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("TAILI_RESUME_CHECKPOINT", ""),
        help="Optional skrl checkpoint to resume from; passed to payload-owned train_taili.",
    )
    parser.add_argument(
        "--telemetry-interval",
        type=int,
        default=int(os.environ.get("TAILI_TELEMETRY_INTERVAL", str(DEFAULT_TELEMETRY_INTERVAL)) or DEFAULT_TELEMETRY_INTERVAL),
        help="Print structured [TP*] training telemetry every N env steps; also prints first and final step.",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    package_dir = Path(__file__).resolve().parent
    payload_root = package_dir.parent
    data_root = Path(args.data_root)
    config_path = resolve_config_path(args.config)
    run_root = Path(args.run_root) if args.run_root else data_root / "taili_runs"
    run_id = args.run_id or f"taili_blind_{_stamp()}"
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    config_copy = run_dir / CONFIG_FILENAME
    write_taili_blind_config(config_copy, load_taili_blind_config(config_path))
    agent_cfg, effective_cfg = _patched_agent_cfg(
        package_dir=package_dir,
        config_path=config_copy,
        run_root=run_root,
        run_id=run_id,
        run_dir=run_dir,
        total_steps=args.total_steps,
    )
    train_args = _clean_remainder(list(args.train_args))
    if not _has_flag(train_args, "--task"):
        train_args = ["--task", args.task, *train_args]
    if args.checkpoint and not _has_flag(train_args, "--checkpoint"):
        train_args.extend(["--checkpoint", args.checkpoint])
    if args.headless and not _has_flag(train_args, "--headless"):
        train_args.append("--headless")

    if args.train_entry:
        cmd = [args.python, args.train_entry, *train_args]
    else:
        cmd = [args.python, "-m", "taili_blind_runtime.train_taili", "--agent-yaml", str(agent_cfg), *train_args]
    env = _build_env(
        payload_root=payload_root,
        run_root=run_root,
        run_id=run_id,
        run_dir=run_dir,
        agent_cfg=agent_cfg,
        total_steps=args.total_steps,
        telemetry_interval=args.telemetry_interval,
    )
    run_config = load_taili_blind_config(config_copy)
    metadata = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "payload_root": str(payload_root),
        "package_dir": str(package_dir),
        "agent_cfg": str(agent_cfg),
        "source_config": str(config_copy),
        "effective_config": str(effective_cfg),
        "task": args.task,
        "resume_checkpoint": args.checkpoint,
        "training_recipe": get_config_value(run_config, "training_recipe", {}),
        "command": cmd,
        "paths": {
            "train_log": env["TAILI_TRAIN_LOG"],
            "telemetry_jsonl": env["TAILI_TELEMETRY_JSONL"],
            "console_log": env["TAILI_CONSOLE_LOG"],
            "checkpoint_dir": env["TAILI_CHECKPOINT_DIR"],
            "tensorboard_dir": str(run_dir),
            "effective_config": str(effective_cfg),
        },
        "observability": {
            "telemetry_interval_steps": int(env["TAILI_TELEMETRY_INTERVAL"]),
            "terminal_lines": "[TPPATH] once, then [TPSTAT]/[TPREW]/[TPCURR] at first/every interval/final step",
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_run_metadata(run_dir / "run.json", metadata)

    print(f"[TAILI_LAUNCH] run_id={run_id}", flush=True)
    print(f"[TAILI_LAUNCH] run_dir={run_dir}", flush=True)
    print(f"[TAILI_LAUNCH] agent_cfg={agent_cfg}", flush=True)
    print(f"[TAILI_LAUNCH] console_log={env['TAILI_CONSOLE_LOG']}", flush=True)
    print(f"[TAILI_LAUNCH] telemetry_interval={env['TAILI_TELEMETRY_INTERVAL']}", flush=True)
    print("[TAILI_LAUNCH] command=" + " ".join(cmd), flush=True)
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
        return 0

    console_log = Path(env["TAILI_CONSOLE_LOG"])
    console_log.parent.mkdir(parents=True, exist_ok=True)
    with console_log.open("a", encoding="utf-8", errors="replace") as log:
        log.write("[TAILI_LAUNCH] command=" + " ".join(cmd) + "\n")
        log.write("[TAILI_LAUNCH] PYTORCH_CUDA_ALLOC_CONF=" + env.get("PYTORCH_CUDA_ALLOC_CONF", "") + "\n")
        log.flush()
        saw_traceback = False
        saw_cuda_oom = False
        proc = subprocess.Popen(
            cmd,
            cwd=str(run_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if "Traceback (most recent call last)" in line:
                saw_traceback = True
            if "torch.OutOfMemoryError" in line or "CUDA out of memory" in line:
                saw_cuda_oom = True
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        rc = int(proc.wait())
        if rc == 0 and (saw_traceback or saw_cuda_oom):
            rc = 2
        log.write(f"[TAILI_LAUNCH] train process rc={rc} traceback={saw_traceback} cuda_oom={saw_cuda_oom}\n")
        log.flush()
        return rc


if __name__ == "__main__":
    raise SystemExit(main())
