"""System SELF-AWARENESS for the LLM brain: what the system itself is doing right now, plus a
codified stall classifier so "卡住了吗" gets the OPERATOR's answer (the known GPU-kernel-stall
pattern, auto-recovery expectations), not an outsider's guess from raw telemetry.

Read-only; best-effort; never raises into the request path.
"""
from __future__ import annotations

import glob
import re
import shlex as _shlex
from typing import Any

# local automation entrypoints worth reporting (the "who is driving the box" inventory)
_AUTOMATION_MARKERS = {
    "-m autotuner.training.tune_orchestrator": "campaign/produce（自主调参·产出管线）",
    "auto_drive.py": "auto_drive（自愈驾驶：停滞检测+自动重启）",
    "-m autotuner.training.acceptance_run": "acceptance_run（验收测评）",
}


def _local_automations() -> list[dict[str, str]]:
    found = []
    for p in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            cmd = open(p, "rb").read().replace(b"\x00", b" ").decode("utf-8", "ignore")
        except Exception:  # noqa: BLE001
            continue
        for marker, label in _AUTOMATION_MARKERS.items():
            if marker in cmd:
                found.append({"pid": p.split("/")[2], "what": label, "cmd": cmd.strip()[:160]})
                break
    return found


def build_operations_state(src: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"automations_running": _local_automations()}
    try:
        remote = src._get_remote()
        # BATCHED READ (0707 audit#6): the "卡住了/what state" hot path used to fan out ~8 SERIAL SSH
        # round-trips (run dir, step, log age, pid, gpu, environ, cmdline, TPCURR) — pure latency tax on the
        # exact moment an operator needs a fast answer. Collapse them into ONE labeled remote shell read and
        # parse the blob. A missing section just yields empty → the same graceful degradation as before.
        _script = (
            "run=$(ls -1td /root/gpufree-data/taili_runs/*/ 2>/dev/null | head -1 | sed 's:/*$::'); "
            "printf '__RUN__%s\\n' \"$run\"; "
            "printf '__STEP__%s\\n' \"$(grep -oE 'step=[0-9]+' \"$run/train.log\" 2>/dev/null | tail -1)\"; "
            "printf '__AGE__%s\\n' \"$(( $(date +%s) - $(stat -c %Y \"$run/train.log\" 2>/dev/null || echo 0) ))\"; "
            "pid=$(pgrep -f 'taili_blind_runtime.train_taili' | grep -v pgrep | head -1); "
            "printf '__PID__%s\\n' \"$pid\"; "
            "printf '__GPU__%s\\n' \"$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | head -1)\"; "
            "if [ -n \"$pid\" ]; then "
            "  printf '__ENV__%s\\n' \"$(tr '\\0' '\\n' < /proc/$pid/environ 2>/dev/null | grep -E '^(PYTHONPATH|TAILI_)' | tr '\\n' ';')\"; "
            "  printf '__CMD__%s\\n' \"$(tr '\\0' ' ' < /proc/$pid/cmdline 2>/dev/null)\"; "
            "fi; "
            "printf '__TPCURR__%s\\n' \"$(grep 'TPCURR' \"$run/train.log\" 2>/dev/null | tail -1)\""
        )
        blob = remote.exec_out("bash -lc " + _shlex.quote(_script)) or ""
        _sec: dict[str, str] = {}
        for line in blob.splitlines():
            for tag in ("__RUN__", "__STEP__", "__AGE__", "__PID__", "__GPU__", "__ENV__", "__CMD__", "__TPCURR__"):
                if line.startswith(tag):
                    _sec[tag] = line[len(tag):]
                    break
        run_dir = (_sec.get("__RUN__", "") or "").strip().rstrip("/")
        out["latest_run"] = run_dir.split("/")[-1] if run_dir else ""
        m = re.search(r"(\d+)", _sec.get("__STEP__", "") or "")
        step = int(m.group(1)) if m else None
        out["last_step"] = step
        try:
            out["step_age_s"] = int((_sec.get("__AGE__", "") or "").strip())
        except Exception:  # noqa: BLE001
            out["step_age_s"] = None
        pid = (_sec.get("__PID__", "") or "").strip()
        out["training_process_alive"] = bool(pid)
        gpu = (_sec.get("__GPU__", "") or "").strip()
        out["gpu"] = gpu
        # ── TRAINING-PROGRESS FACTS (0706): the LLM was guessing phase/target from raw step numbers and
        # calling a resume "barely started, run tens of thousands more steps" (which would overtrain). Give
        # it the actual regime so it reasons from facts, not the 1.5M launch ceiling. ────────────────────
        tp: dict[str, Any] = {}
        if pid:
            cmdline = _sec.get("__CMD__", "") or ""
            ck = re.search(r"--checkpoint\s+(\S+)", cmdline)
            tp["resumed_from_checkpoint"] = ck.group(1).split("/")[-3] + "/" + ck.group(1).split("/")[-1] if ck else None
            tp["is_resume"] = bool(ck)
            # capture the ARCHITECTURE flags the run is actually using (0707): the arch stack (multi-critic,
            # dynamics latent, amp-stride, hard-equiv, yaw-fullrange...) is toggled by TAILI_* env vars, and a
            # flag being UNSET silently disables that upgrade (e.g. an unset TAILI_MULTI_CRITIC makes the K-group
            # reward split byte-inert). Surface them so the operator/LLM SEES what architecture is live, not just
            # the payload name.
            arch_flags = {}
            for line in (_sec.get("__ENV__", "") or "").split(";"):
                if line.startswith("PYTHONPATH="):
                    tp["loaded_payload"] = line.split("=", 1)[1].split(":")[0].split("/")[-1]
                elif line.startswith("TAILI_INIT_PHASE="):
                    tp["init_phase_env"] = line.split("=", 1)[1].strip()
                elif line.startswith("TAILI_") and "=" in line:
                    _k, _v = line.split("=", 1)
                    if _k not in ("TAILI_INIT_PHASE",):
                        arch_flags[_k] = _v.strip()
            if arch_flags:
                tp["arch_flags"] = arch_flags
        if run_dir:
            curr = _sec.get("__TPCURR__", "") or ""
            ph = re.search(r"phase=(phi\d)", curr)
            blk = re.search(r"blocked_by=(\w+)", curr)
            tp["current_phase"] = ph.group(1) if ph else None
            tp["blocked_by"] = blk.group(1) if blk else None
        tp["regime"] = (
            "1.5M --total-steps 是启动上限,不是目标。每一轮训练都是从一个已会走的 checkpoint 续训,"
            "在 ~18k 续训步停下做全覆盖复测(练更久会过训退化 B4/A2——已成文规律)。所以'才几百/几千步、"
            "离 1.5M 还远'是错误心智模型;看的是'离 18k 续训步还有多少'。刚改过奖励/curriculum 或从 phi0 "
            "重启后,前几千步 progress/slip 等指标会先降后升(重新预热),不要据此判定结构瓶颈。")
        out["training_progress"] = tp
        # ── the codified stall classifier (ops runbook pattern #1) ──────────────────────────────
        age = out.get("step_age_s")
        alive = out["training_process_alive"]
        if not alive:
            out["diagnosis"] = "STOPPED：训练进程不存在（正常停止、OOM 或已被自动化重启中——看 automations_running）"
        elif age is not None and age < 90:
            out["diagnosis"] = f"HEALTHY：训练正常步进（step={step}，{age}s 前仍在写日志）"
        elif age is not None and age >= 180:
            out["diagnosis"] = (
                f"KNOWN_GPU_STALL：已知 GPU 内核挂起模式——步数冻结在 {step}（{age}s 未写日志）、进程仍存活、"
                f"GPU={gpu}。这是租用盒子的透明故障（本项目已发生 10+ 次），与代码无关。"
                f"{'auto_drive 在运行,预计 ~10 分钟内自动杀掉并从最新 checkpoint 重启' if any('auto_drive' in a['what'] for a in out['automations_running']) else '当前无自愈驾驶在运行——需要手动杀掉进程并从最新 checkpoint 重启'}。"
                "不要猜测日志线程死锁/磁盘IO——从未发生过（见 taili_ops_runbook.md）。")
        else:
            out["diagnosis"] = f"UNCERTAIN：{age}s 未更新，再观察一个周期（<180s 不足以判停滞）"
    except Exception as e:  # noqa: BLE001
        out["remote_error"] = f"{type(e).__name__}: {e}"
        out["diagnosis"] = "UNKNOWN：训练机暂不可达（可能 SSH 瞬断，运营手册模式#3——重试即可）"
    out["instruction"] = (
        "这是系统的自我状态：automations_running 告诉你系统自己正在做什么（run 换名+从 checkpoint 续起="
        "自愈重启，不是异常）；diagnosis 是运营手册模式匹配的结论，直接引用它回答'卡住了吗'类问题，"
        "不要脱离它自行猜测新故障机制。training_progress 给你训练进度的事实（是否续训、来源 checkpoint、"
        "当前 phase、加载的代码 payload、目标区间 regime）——回答'训练怎么样/还要多久/进展如何'时必须先读它，"
        "尤其 regime：不要把 1.5M 当目标、不要建议'再跑几万步'(会过训)、不要把续训当从零开始。")
    return out
