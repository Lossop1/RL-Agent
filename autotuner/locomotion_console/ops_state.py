"""向研究代理提供产品无关的系统自我状态。

这里仅做只读状态汇总。运行目录、训练进程、阶段环境变量和自动化入口都由产品
合同声明；缺少声明时返回不确定状态，绝不根据机器人目录名猜测。
"""
from __future__ import annotations

import glob
import os
import re
import shlex
from typing import Any, Mapping

from autotuner.product import resolve_product_runtime


_SAFE_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _runtime_declaration(view: Any) -> Mapping[str, Any]:
    """读取产品声明的运行字段，不改变 RuntimeIdentity 的外部结构。"""
    runtime = _mapping(getattr(view, "runtime", {}))
    return _mapping(runtime.get("declaration"))


def _local_automations(src: Any) -> list[dict[str, str]]:
    """读取合同声明的本地自动化标记，不维护系统级产品名单。"""
    settings = getattr(src, "settings", None)
    try:
        runtime = resolve_product_runtime(getattr(settings, "product_id", "") or None)
        markers = _runtime_declaration(runtime).get("automation_markers", {})
    except Exception:
        markers = {}
    if not isinstance(markers, Mapping):
        return []

    found: list[dict[str, str]] = []
    for path in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            command = open(path, "rb").read().replace(b"\x00", b" ").decode("utf-8", "ignore")
        except Exception:  # noqa: BLE001
            continue
        for marker, label in markers.items():
            if str(marker) in command:
                found.append({
                    "pid": path.split("/")[2],
                    "what": str(label),
                    "cmd": command.strip()[:160],
                })
                break
    return found


def _declarations(src: Any) -> dict[str, Any]:
    settings = getattr(src, "settings", None)
    product_id = getattr(settings, "product_id", "") or None
    try:
        view = resolve_product_runtime(product_id)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    deployment = dict(view.deployment)
    launch = _mapping(deployment.get("training_launch"))
    state_environment = _mapping(launch.get("resume_state_environment"))
    declared_environment = _mapping(launch.get("environment"))
    environment_names = {str(key).strip() for key in declared_environment}
    environment_names.update(str(value).strip() for value in state_environment.values() if str(value).strip())
    environment_names.update({"PYTHONPATH"})
    invalid_names = sorted(name for name in environment_names if not _SAFE_ENVIRONMENT_NAME.fullmatch(name))
    if invalid_names:
        return {"error": "invalid environment names in product contract: " + ", ".join(invalid_names)}

    raw_prefixes = launch.get("environment_prefixes", ())
    if raw_prefixes is None:
        raw_prefixes = ()
    if not isinstance(raw_prefixes, (list, tuple)):
        return {"error": "deployment.training_launch.environment_prefixes must be a list"}
    prefixes = tuple(sorted({str(item).strip() for item in raw_prefixes if str(item).strip()}))
    invalid_prefixes = sorted(
        prefix
        for prefix in prefixes
        if not _SAFE_ENVIRONMENT_NAME.fullmatch(prefix) or not prefix.endswith("_")
    )
    if invalid_prefixes:
        return {
            "error": "environment prefixes must be safe names ending in '_': "
            + ", ".join(invalid_prefixes)
        }

    process_pattern = ""
    resolver = getattr(src, "_training_process_pattern", None)
    if callable(resolver):
        try:
            process_pattern = str(resolver() or "").strip()
        except Exception:  # noqa: BLE001
            process_pattern = ""
    if not process_pattern:
        process_pattern = str(_runtime_declaration(view).get("training_process_pattern") or "").strip()

    run_root = str(deployment.get("runs_root") or "").strip().rstrip("/")
    run_glob = f"{run_root}/*/" if run_root.startswith("/") else ""
    if not run_glob:
        # 部署未给出单一运行根目录时，使用当前框架档案声明的首个监控 glob。
        try:
            profiles = view.framework_profiles()
            profile = _mapping(profiles.get(view.default_framework_id()))
            run_globs = profile.get("run_globs", ())
            if isinstance(run_globs, (list, tuple)):
                run_glob = next((str(item).strip() for item in run_globs if str(item).strip()), "")
        except Exception:  # noqa: BLE001
            run_glob = ""
    log_files = deployment.get("monitor_log_files", ("train.log", "console.log"))
    if not isinstance(log_files, (list, tuple)):
        log_files = ("train.log", "console.log")
    safe_log_files = tuple(
        str(item).strip().lstrip("/")
        for item in log_files
        if re.fullmatch(r"[A-Za-z0-9_.-]+", str(item).strip().lstrip("/"))
    )
    return {
        "product_id": view.product_id,
        "run_glob": run_glob,
        "process_pattern": process_pattern,
        "log_files": safe_log_files,
        "environment_names": tuple(sorted(environment_names)),
        "environment_prefixes": prefixes,
        "phase_environment": str(state_environment.get("phase") or "").strip(),
    }


def _build_remote_probe_script(declarations: Mapping[str, Any]) -> str:
    """根据产品声明构造一次性只读 shell 探针。"""
    run_glob = str(declarations.get("run_glob") or "")
    process_pattern = str(declarations.get("process_pattern") or "")
    if not run_glob or not process_pattern:
        return "printf '__ERROR__product runtime declarations are incomplete\\n'"

    log_files = tuple(str(item) for item in declarations.get("log_files", ()))
    log_candidates = " ".join(f'"$run/{item}"' for item in log_files)
    environment_names = tuple(str(item) for item in declarations.get("environment_names", ()))
    prefixes = tuple(str(item) for item in declarations.get("environment_prefixes", ()))
    # 精确名称避免漏掉离散配置；显式前缀用于产品自行声明的一族动态开关。
    env_alternatives = [re.escape(item) for item in environment_names]
    env_alternatives.extend(re.escape(item) + "[A-Za-z0-9_]*" for item in prefixes)
    env_pattern = "^(" + "|".join(env_alternatives) + ")=" if env_alternatives else "^$"
    phase_name = str(declarations.get("phase_environment") or "")
    phase_tag = shlex.quote(phase_name) if phase_name else "''"
    return (
        "run=$(ls -1td " + run_glob + " 2>/dev/null | head -1 | sed 's:/*$::'); "
        "printf '__RUN__%s\\n' \"$run\"; "
        "pid=$(pgrep -f " + shlex.quote(process_pattern) + " | head -1); "
        "printf '__PID__%s\\n' \"$pid\"; "
        "if [ -n \"$run\" ]; then "
        "  log=''; "
        "  for candidate in " + log_candidates + "; do "
        "    if [ -s \"$candidate\" ]; then log=\"$candidate\"; break; fi; "
        "  done; "
        "  printf '__STEP__%s\\n' \"$(grep -oE 'step=[0-9]+' \"$log\" 2>/dev/null | tail -1)\"; "
        "  printf '__AGE__%s\\n' \"$(( $(date +%s) - $(stat -c %Y \"$log\" 2>/dev/null || echo 0) ))\"; "
        "  printf '__TPCURR__%s\\n' \"$(grep 'TPCURR' \"$log\" 2>/dev/null | tail -1)\"; "
        "fi; "
        "printf '__GPU__%s\\n' \"$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)\"; "
        "if [ -n \"$pid\" ]; then "
        "  printf '__ENV__%s\\n' \"$(tr '\\0' '\\n' < /proc/$pid/environ 2>/dev/null | grep -E '" + env_pattern + "' | tr '\\n' ';')\"; "
        "  printf '__CMD__%s\\n' \"$(tr '\\0' ' ' < /proc/$pid/cmdline 2>/dev/null)\"; "
        "fi; "
        "printf '__PHASE_ENV__%s\\n' " + phase_tag
    )


def build_operations_state(src: Any) -> dict[str, Any]:
    """汇总本地自动化和远程训练状态；所有失败均降级为可解释的只读结果。"""
    out: dict[str, Any] = {
        "automations_running": _local_automations(src),
        "product_id": getattr(getattr(src, "settings", None), "product_id", "") or "",
    }
    declarations = _declarations(src)
    if declarations.get("error"):
        out["diagnosis"] = f"UNKNOWN：产品运行合同不可用：{declarations['error']}"
        out["instruction"] = "先修复或选择有效的产品合同，再解释训练状态。"
        return out
    out["product_id"] = declarations.get("product_id", out["product_id"])

    try:
        remote = src._get_remote()
        script = _build_remote_probe_script(declarations)
        blob = remote.exec_out("bash -lc " + shlex.quote(script)) or ""
        sections: dict[str, str] = {}
        for line in blob.splitlines():
            for tag in (
                "__RUN__", "__STEP__", "__AGE__", "__PID__", "__GPU__",
                "__ENV__", "__CMD__", "__TPCURR__", "__PHASE_ENV__", "__ERROR__",
            ):
                if line.startswith(tag):
                    sections[tag] = line[len(tag):]
                    break
        if sections.get("__ERROR__"):
            out["diagnosis"] = f"UNKNOWN：{sections['__ERROR__'].strip()}"
            return out

        run_dir = (sections.get("__RUN__", "") or "").strip().rstrip("/")
        out["latest_run"] = run_dir.rsplit("/", 1)[-1] if run_dir else ""
        step_match = re.search(r"(\d+)", sections.get("__STEP__", "") or "")
        step = int(step_match.group(1)) if step_match else None
        out["last_step"] = step
        try:
            out["step_age_s"] = int((sections.get("__AGE__", "") or "").strip())
        except ValueError:
            out["step_age_s"] = None
        pid = (sections.get("__PID__", "") or "").strip()
        out["training_process_alive"] = bool(pid)
        out["gpu"] = (sections.get("__GPU__", "") or "").strip()

        progress: dict[str, Any] = {
            "is_resume": False,
            "resumed_from_checkpoint": None,
            "phase_environment": (sections.get("__PHASE_ENV__", "") or "").strip() or None,
        }
        if pid:
            command = sections.get("__CMD__", "") or ""
            checkpoint = re.search(r"--checkpoint\s+(\S+)", command)
            progress["resumed_from_checkpoint"] = checkpoint.group(1) if checkpoint else None
            progress["is_resume"] = bool(checkpoint)
            flags: dict[str, str] = {}
            for item in (sections.get("__ENV__", "") or "").split(";"):
                if "=" in item:
                    key, value = item.split("=", 1)
                    if key and key != "PYTHONPATH":
                        flags[key] = value.strip()
            if flags:
                progress["environment_overrides"] = flags
        if run_dir:
            current = sections.get("__TPCURR__", "") or ""
            phase = re.search(r"phase=([A-Za-z0-9_.-]+)", current)
            blocked = re.search(r"blocked_by=([A-Za-z0-9_.-]+)", current)
            progress["current_phase"] = phase.group(1) if phase else None
            progress["blocked_by"] = blocked.group(1) if blocked else None
        out["training_progress"] = progress

        age = out.get("step_age_s")
        alive = bool(out["training_process_alive"])
        if not alive:
            out["diagnosis"] = "STOPPED：未发现合同声明的训练进程。"
        elif age is not None and age < 90:
            out["diagnosis"] = f"HEALTHY：训练正在写入日志（step={step}，{age}s 前更新）。"
        elif age is not None and age >= 180:
            out["diagnosis"] = f"STALE：进程仍存在，但日志已 {age}s 未更新；需要交叉检查 GPU 和进程状态。"
        else:
            out["diagnosis"] = f"UNCERTAIN：日志已 {age}s 未更新，继续观察后再判断。"
    except Exception as exc:  # noqa: BLE001
        out["remote_error"] = f"{type(exc).__name__}: {exc}"
        out["diagnosis"] = f"UNKNOWN：远程状态暂不可达：{type(exc).__name__}: {exc}"
    out["instruction"] = (
        "这是系统自我状态的只读摘要。先依据 training_progress、diagnosis 和实际证据判断，"
        "不要把缺失字段当成产品事实，也不要从旧产品名称推断运行状态。"
    )
    return out
