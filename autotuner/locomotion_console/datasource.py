"""Run-data sources behind one interface.

`FakeDataSource` synthesizes a plausible training stream so the whole spine runs with zero
GPU / SSH dependency - this is what Loop 0 closes against first.

`RealDataSource` wraps the existing RemoteSSH + parse_log + physeval. Its heavy imports are
lazy (inside methods) so importing this module never drags in paramiko / torch on the fake
path. Loop 0 ships it as a thin, honest stub; the V step wires it to the live box.

Both implement the same async-friendly surface:
    get_status()                  -> RunStatus
    stream(stop)                  -> async generator of MetricPoint
    action_resume/kill()          -> ActionResult
    action_physeval()             -> PhysevalResult
"""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from .config import LocomotionConsoleSettings
from .schemas import (
    ActionResult,
    CommandEval,
    MetricPoint,
    PhysevalResult,
    RemoteDiskInfo,
    RemoteGPUInfo,
    RemoteMachineStatus,
    RemoteTmuxSessionInfo,
    RunStatus,
    TensorboardScalarCatalog,
    TensorboardScalarPoint,
    TensorboardScalarSeries,
    TensorboardSeriesResponse,
    TensorboardTagInfo,
    TrainingTelemetry,
)
from .telemetry import build_telemetry


class RunDataSource:
    """Interface. Subclasses must implement all methods."""

    async def get_status(self) -> RunStatus:  # pragma: no cover - interface
        raise NotImplementedError

    def get_acceptance(self) -> dict:
        """Spec-verdict for the newest run (best-effort, read-only). Default: unavailable."""
        return {"available": False, "reason": "acceptance scoring not supported by this source"}

    async def stream(self, stop: asyncio.Event) -> AsyncIterator[MetricPoint]:  # pragma: no cover
        raise NotImplementedError
        yield  # make this an async generator for type-checkers

    async def action_resume(self) -> ActionResult:  # pragma: no cover - interface
        raise NotImplementedError

    async def action_start(self) -> ActionResult:  # pragma: no cover - interface
        raise NotImplementedError

    async def action_deploy_payload(self) -> ActionResult:  # pragma: no cover - interface
        raise NotImplementedError

    async def action_kill(self) -> ActionResult:  # pragma: no cover - interface
        raise NotImplementedError

    async def action_physeval(self) -> PhysevalResult:  # pragma: no cover - interface
        raise NotImplementedError

    async def tensorboard_scalar_catalog(self) -> TensorboardScalarCatalog:  # pragma: no cover - interface
        raise NotImplementedError

    async def tensorboard_scalar_series(
        self,
        tags: list[str],
        max_points: int = 1200,
    ) -> TensorboardSeriesResponse:  # pragma: no cover - interface
        raise NotImplementedError

    async def training_telemetry(self) -> TrainingTelemetry:  # pragma: no cover - interface
        raise NotImplementedError

    async def remote_machine_status(self) -> RemoteMachineStatus:  # pragma: no cover - interface
        raise NotImplementedError


def _scalar_group(tag: str) -> str:
    head = (tag.split("/", 1)[0] or "Other").strip().lower()
    if "reward" in head:
        return "Reward"
    if "episode" in head or "length" in head or "timestep" in head:
        return "Episode"
    if "termination" in head or "done" in head or "reset" in head:
        return "Termination"
    if "curriculum" in head or "terrain" in head:
        return "Curriculum/Terrain"
    if "loss" in head:
        return "Loss"
    if "policy" in head or "actor" in head:
        return "Policy"
    if "value" in head or "critic" in head:
        return "Value"
    if "command" in head or "tracking" in head:
        return "Command tracking"
    return "Other"


def _scalar_display_name(tag: str) -> str:
    return tag.split("/")[-1].strip() or tag


def _scalar_formula(tag: str, event_file: str = "") -> str:
    source = "TensorBoard scalar event"
    if event_file:
        source += f" ({event_file})"
    return f"tbparse.SummaryReader(event_file).scalars where tag == {tag!r}; x=step, y=value"


def _scalar_description(tag: str) -> str:
    group = _scalar_group(tag)
    if group == "Reward":
        return "训练写入 TensorBoard 的 reward 标量。它不是从终端日志猜出来的。"
    if group == "Episode":
        return "episode 级统计，通常用于判断是否更接近 timeout 或更早终止。"
    if group == "Loss":
        return "优化器/网络损失曲线；主要看趋势和突变，不看单点。"
    if group == "Policy":
        return "策略分布或 actor 相关标量，用来观察策略是否坍缩或过度随机。"
    if group == "Value":
        return "critic/value 相关标量，用来判断值函数学习是否异常。"
    if group == "Command tracking":
        return "命令跟踪相关标量，优先和诊断中的 tracking_error 对照。"
    if group == "Curriculum/Terrain":
        return "课程/地形进度上下文，解释 reward 波动时很关键。"
    return "原始 TensorBoard scalar；具体语义由训练脚本写入该 tag 时决定。"


@dataclass
class _ScalarCache:
    run: str = ""
    remote_event: str = ""
    local_event: str = ""
    remote_size: int = 0
    remote_mtime: int = 0
    fetched_at: float = 0.0
    dataframe: object | None = None
    error: str = ""


@dataclass(frozen=True)
class _TelemetryPaths:
    run: str = ""
    log_path: str = ""
    telemetry_path: str = ""
    console_log_path: str = ""
    checkpoint_dir: str = ""
    event_file: str = ""
    effective_config_path: str = ""
    evidence: tuple[str, ...] = ()


class _RemoteCooldownError(RuntimeError):
    """Raised when a recent SSH failure is still inside the console cooldown window."""


# -- Fake source ------------------------------------------------------------

class FakeDataSource(RunDataSource):
    """Synthetic but believable: reward climbs with noise, terrain steps up, phase advances.

    Mirrors the shape of a real Taili AMP run so the frontend can be built and judged
    against realistic curves without touching the GPU.
    """

    def __init__(self, settings: LocomotionConsoleSettings):
        self.s = settings
        self._iter = 4500          # pretend we resumed mid-run
        self._t0 = time.time()
        self._running = True

    async def get_status(self) -> RunStatus:
        return RunStatus(
            run_id="fake_2026-06-22_taili_amp",
            task="RobotLab-Isaac-Taili-AMP-Direct-v0",
            robot_id="taili_dog_39kg",
            latest_iter=self._iter,
            total_iter=15000,
            running=self._running,
            source="fake",
            note="synthetic stream - no GPU/SSH",
        )

    async def stream(self, stop: asyncio.Event) -> AsyncIterator[MetricPoint]:
        while not stop.is_set():
            self._iter += 5
            elapsed = time.time() - self._t0
            # reward: saturating climb + slow ripple + small noise
            base = 3400 + 900 * (1 - math.exp(-elapsed / 40.0))
            ripple = 60 * math.sin(elapsed / 7.0)
            noise = ((self._iter * 2654435761) % 1000 / 1000.0 - 0.5) * 80
            reward = base + ripple + noise
            ep_len = min(900.0, 500 + elapsed * 6)
            terrain = min(6.0, 3.5 + elapsed / 30.0)
            phase = 2 if terrain > 4.0 else 1
            yield MetricPoint(
                iter=self._iter,
                ts=time.time(),
                reward=round(reward, 1),
                ep_len=round(ep_len, 1),
                terrain=round(terrain, 2),
                phase=phase,
                extra={"dr_level": 2 if phase == 2 else 0},
            )
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.s.poll_interval_s)
            except asyncio.TimeoutError:
                pass

    async def action_resume(self) -> ActionResult:
        self._running = True
        return ActionResult(action="resume", ok=True, message="fake: training resumed")

    async def action_start(self) -> ActionResult:
        self._running = True
        self._iter = 0
        return ActionResult(action="start", ok=True, message="fake: training started")

    async def action_deploy_payload(self) -> ActionResult:
        return ActionResult(action="deploy_payload", ok=True, message="fake: payload deployed")

    async def action_kill(self) -> ActionResult:
        self._running = False
        return ActionResult(action="kill", ok=True, message="fake: training killed")

    async def action_physeval(self) -> PhysevalResult:
        await asyncio.sleep(1.0)  # pretend a rollout took a moment
        cmds = [
            CommandEval(name="stand", vx=0, vy=0, wz=0, tracked=True,
                        foot_clearance_cm=0.2, duty=1.0, base_h_m=0.56),
            CommandEval(name="forward", vx=0.7, vy=0, wz=0, tracked=True,
                        foot_clearance_cm=14.8, duty=0.51, base_h_m=0.55),
            CommandEval(name="back", vx=-0.4, vy=0, wz=0, tracked=True,
                        foot_clearance_cm=12.1, duty=0.49, base_h_m=0.55),
            CommandEval(name="lateral", vx=0, vy=0.45, wz=0, tracked=True,
                        foot_clearance_cm=11.5, duty=0.50, base_h_m=0.55),
            CommandEval(name="yaw", vx=0, vy=0, wz=0.5, tracked=True,
                        foot_clearance_cm=10.9, duty=0.50, base_h_m=0.56),
        ]
        return PhysevalResult(
            ok=True,
            summary="fake: all 5 commands tracked, ~15cm lift, ~0.5 duty, L/R symmetric",
            commands=cmds,
            checkpoint="fake/agent_validated.pt",
        )

    def _fake_scalar_points(self, tag: str) -> list[tuple[int, float]]:
        points = []
        for i in range(180):
            step = 1000 + i * 250
            x = i / 24.0
            if "Episode" in tag:
                value = min(900.0, 420 + i * 2.7 + 18 * math.sin(x / 2.0))
            elif "Policy" in tag:
                value = max(0.08, 0.72 * math.exp(-i / 140.0) + 0.03 * math.sin(x))
            elif "Loss" in tag:
                value = max(0.01, 2.4 * math.exp(-i / 90.0) + 0.12 * math.sin(x * 1.7))
            else:
                value = 3200 + 1050 * (1 - math.exp(-i / 55.0)) + 75 * math.sin(x)
            points.append((step, round(value, 4)))
        return points

    async def tensorboard_scalar_catalog(self) -> TensorboardScalarCatalog:
        tags = [
            "Reward / Total reward (mean)",
            "Episode / Total timesteps (mean)",
            "Policy / Standard deviation",
            "Loss / Policy loss",
        ]
        infos = []
        for tag in tags:
            pts = self._fake_scalar_points(tag)
            infos.append(
                TensorboardTagInfo(
                    tag=tag,
                    group=_scalar_group(tag),
                    display_name=_scalar_display_name(tag),
                    points=len(pts),
                    first_step=pts[0][0],
                    last_step=pts[-1][0],
                    last_value=pts[-1][1],
                    source="fake TensorBoard stream",
                    formula=_scalar_formula(tag, "fake"),
                    description=_scalar_description(tag),
                )
            )
        return TensorboardScalarCatalog(
            available=True,
            source="fake",
            run="fake_2026-06-22_taili_amp",
            event_file="fake/events.out.tfevents",
            reward_tag="Reward / Total reward (mean)",
            tags=infos,
            refreshed_at=time.time(),
        )

    async def tensorboard_scalar_series(self, tags: list[str], max_points: int = 1200) -> TensorboardSeriesResponse:
        clean_tags = [tag for tag in tags if tag] or ["Reward / Total reward (mean)"]
        series = []
        for tag in clean_tags[:12]:
            pts = self._fake_scalar_points(tag)
            if len(pts) > max_points:
                stride = max(1, math.ceil(len(pts) / max_points))
                pts = pts[::stride]
            series.append(
                TensorboardScalarSeries(
                    tag=tag,
                    group=_scalar_group(tag),
                    display_name=_scalar_display_name(tag),
                    source="fake TensorBoard stream",
                    formula=_scalar_formula(tag, "fake"),
                    points=[TensorboardScalarPoint(step=step, value=value) for step, value in pts],
                )
            )
        return TensorboardSeriesResponse(
            available=True,
            source="fake",
            run="fake_2026-06-22_taili_amp",
            event_file="fake/events.out.tfevents",
            series=series,
        )

    async def training_telemetry(self) -> TrainingTelemetry:
        elapsed = time.time() - self._t0
        points = []
        for i in range(120):
            step = self._iter - (119 - i) * 50
            x = i / 12.0
            terrain = min(6.0, 2.0 + i / 30.0)
            phase = "phi2" if terrain >= 4 else "phi1"
            points.append(
                {
                    "type": "train_tick",
                    "step": max(0, step),
                    "total_steps": 15000,
                    "fps": 18.0 + 2.0 * math.sin(x),
                    "elapsed": f"00:{int(elapsed // 60):02d}:{int(elapsed % 60):02d}",
                    "eta": "00:10:00",
                    "reward": {
                        "total": 1.0 + 0.2 * math.sin(x),
                        "track": 0.18 + 0.03 * math.sin(x / 2.0),
                        "lin_err": 0.28 - 0.02 * math.sin(x / 2.0),
                        "speed": 0.20 + 0.02 * math.sin(x),
                        "gait": 0.96 + 0.02 * math.sin(x / 3.0),
                        "stand": 0.0,
                        "clearance": 0.05 + 0.01 * math.sin(x / 2.0),
                        "slip": 0.10 + 0.02 * math.sin(x / 3.0),
                        "torque": -0.08,
                    },
                    "curriculum": {
                        "phase": phase,
                        "terrain_mean": round(terrain, 2),
                        "terrain_max": 6,
                        "dr_level": 1 if phase == "phi2" else 0,
                        "progress_gate": 0.60 + 0.05 * math.sin(x / 3.0),
                        "gait_gate": 0.88 + 0.03 * math.sin(x / 4.0),
                        "blocked_by": "progress",
                        "next_gate": "terrain_level_up",
                    },
                    "health": {"fall_rate": 0.012, "reset_rate": 0.03},
                    "command": {
                        "cmd_vx": 0.5,
                        "cmd_vy": 0.0,
                        "cmd_wz": 0.0,
                        "actual_vx": 0.38 + 0.04 * math.sin(x),
                        "actual_vy": 0.01 * math.sin(x / 2.0),
                        "actual_wz": 0.02 * math.cos(x / 3.0),
                        "v_along": 0.38 + 0.04 * math.sin(x),
                        "speed_xy": 0.39 + 0.04 * math.sin(x),
                        "lin_err": 0.12 - 0.02 * math.sin(x / 2.0),
                        "progress_ratio": 0.60 + 0.05 * math.sin(x / 3.0),
                        "gait_match": 0.88 + 0.03 * math.sin(x / 4.0),
                        "diagonal_contact": 0.82 + 0.04 * math.sin(x / 5.0),
                        "duty_balance": 0.80 + 0.05 * math.cos(x / 6.0),
                        "stance_slip": 0.10 + 0.02 * math.sin(x / 3.0),
                    },
                    "paths": {
                        "run_dir": "fake/taili_runs/fake_2026-06-22_taili_amp",
                        "telemetry_jsonl": "fake/tp_train.telemetry.jsonl",
                        "log": "fake/tp_train.log",
                        "console_log": "fake/console.log",
                        "checkpoint_dir": "fake/checkpoints",
                        "tensorboard_dir": "fake",
                        "effective_config": "fake/effective_config.yaml",
                    },
                }
            )
        import json

        telemetry = build_telemetry(
            source="fake",
            run_id="fake_2026-06-22_taili_amp",
            running=self._running,
            log_path="fake/tp_train.log",
            telemetry_path="fake/tp_train.telemetry.jsonl",
            console_log_path="fake/console.log",
            checkpoint_dir="fake/checkpoints",
            event_file="fake/events.out.tfevents",
            effective_config_path="fake/effective_config.yaml",
            path_evidence=["fake source contract"],
            jsonl_text="\n".join(json.dumps(item) for item in points),
        )
        from .definitions import definitions_for_telemetry

        telemetry.definitions = definitions_for_telemetry(telemetry)
        return telemetry

    async def remote_machine_status(self) -> RemoteMachineStatus:
        return RemoteMachineStatus(
            available=True,
            source="fake",
            generated_at=time.time(),
            host="fake-local",
            uptime="fake uptime 02:10",
            load_avg="0.4 0.5 0.6",
            cpu_count=16,
            memory_total_mb=32768,
            memory_used_mb=12288,
            memory_available_mb=20480,
            gpus=[
                RemoteGPUInfo(
                    index=0,
                    name="Fake RTX",
                    memory_used_mb=6200,
                    memory_total_mb=24576,
                    utilization_gpu_pct=48,
                    utilization_memory_pct=31,
                    temperature_c=62,
                    power_w=180,
                    processes=[{"pid": 1234, "name": "python", "used_memory_mb": 6200}],
                )
            ],
            disks=[RemoteDiskInfo(mount="/root/gpufree-data", size="500G", used="210G", avail="290G", use_pct="42%")],
            tmux_sessions=[RemoteTmuxSessionInfo(name="fake_train", windows=1, created="fake", attached=False)],
            training_processes=[{"pid": 1234, "command": "fake train"}],
        )


# -- Real source (lazy heavy imports; Loop-0 honest stub) ---------------------

class RealDataSource(RunDataSource):
    """Driven by a discovered BoxProfile - no hardcoded paths.

    status: newest run + latest checkpoint iter + running flag.
    stream: SFTP the newest run's tfevents -> tbparse locally -> reward curve.
    The reward tag is VERIFIED against the actual file (don't trust the discovered default);
    if absent, the available tags are surfaced instead of faking zeros.
    """

    def __init__(self, settings: LocomotionConsoleSettings):
        self.settings = settings
        self.s = settings
        self._scalar_cache = _ScalarCache()
        self._remote = None
        self._remote_failure_until = 0.0
        self._remote_failure_error = ""
        self._remote_failure_count = 0
        from .box_profile import BoxProfile
        from .framework_profile import get_framework_profile, merge_box_profile
        self.framework = get_framework_profile(settings.framework_id)
        self.profile = merge_box_profile(BoxProfile.load() or BoxProfile(), self.framework)

    def _get_remote(self):
        now = time.time()
        if self._remote_failure_until > now:
            remaining = self._remote_failure_until - now
            raise _RemoteCooldownError(
                f"remote SSH is in cooldown for {remaining:.1f}s after: {self._remote_failure_error}"
            )
        if self._remote is None:
            # Lazy: only import/connect when the real source is actually used.
            from autotuner.training.remote import RemoteSSH
            from .config_manager import effective_remote_config

            # The console's own config (config/ssh.json + saved remote profile) is the source of
            # truth. `runtime_state` is a legacy CLI/adapter side-channel that is NOT present in the
            # console deployment — importing it unconditionally made every real-mode connection fail
            # with ModuleNotFoundError, so the console reported the (reachable) box as unavailable.
            cfg = effective_remote_config(self.settings)
            try:  # optional: honor a runtime_state override only where that module exists
                import runtime_state
                if runtime_state.get_config_path():
                    cfg = runtime_state.get_config()
                else:
                    runtime_state.set_config(cfg, "locomotion_console.effective_remote")
            except Exception:
                pass
            if not str(cfg.get("ssh_host") or "").strip():
                raise RuntimeError("no ssh_host configured (set config/ssh.json or the remote profile)")
            self._remote = RemoteSSH(cfg)
            # Console status/telemetry endpoints are polled aggressively. A dead
            # SSH daemon must not make each poll spend 60+ seconds inside retry
            # backoff; action/deploy paths can still choose explicit retries.
            self._remote.default_retries = 1
            self._remote.connect_timeout = 4
            self._remote.banner_timeout = 4
            self._remote.auth_timeout = 4
        return self._remote

    def _remember_remote_success(self) -> None:
        self._remote_failure_until = 0.0
        self._remote_failure_error = ""
        self._remote_failure_count = 0

    def _remember_remote_failure(self, error: Exception) -> str:
        if isinstance(error, _RemoteCooldownError):
            return f"{type(error).__name__}: {error}"
        self._remote_failure_count += 1
        text = f"{type(error).__name__}: {error}"
        self._remote_failure_error = text
        self._remote_failure_until = time.time() + min(45.0, 5.0 * self._remote_failure_count)
        try:
            if self._remote is not None:
                self._remote.close()
        except Exception:
            pass
        self._remote = None
        self._scalar_cache = _ScalarCache(error=text)
        return text

    def _remote_unavailable_message(self, error: Exception | str) -> str:
        if isinstance(error, Exception):
            text = f"{type(error).__name__}: {error}"
        else:
            text = str(error)
        return f"Remote SSH unavailable. Local backend/UI remain usable. {text}"

    def _remote_unavailable_telemetry(self, error: Exception | str) -> TrainingTelemetry:
        log_path = self._train_log_path("")
        message = self._remote_unavailable_message(error)
        telemetry = build_telemetry(
            source="real",
            run_id="remote-unavailable",
            running=False,
            log_path=log_path,
            telemetry_path=self._telemetry_jsonl_path(log_path),
            source_stats={
                "source_kind": "error",
                "remote_ok": False,
                "cooldown_until": int(self._remote_failure_until or 0),
            },
            error=message,
        )
        telemetry.mode = "error"
        telemetry.limitations.append("SSH 不可用：训练监控显示为离线快照；配置页、本地历史和 LLM 仍可用。")
        if telemetry.snapshot is not None:
            telemetry.snapshot.mode = "error"
            telemetry.snapshot.status = "error"
            telemetry.snapshot.conclusion = message
            telemetry.snapshot.notes.append("Remote-dependent reads are paused until SSH reconnects.")
        return telemetry

    def _remote_unavailable_machine_status(self, error: Exception | str) -> RemoteMachineStatus:
        return RemoteMachineStatus(
            available=False,
            source="real",
            generated_at=time.time(),
            host=self.profile.hostname,
            error=self._remote_unavailable_message(error),
        )

    @staticmethod
    def _float_field(value: object) -> float | None:
        try:
            text = str(value).strip()
            if not text or text.lower() in {"n/a", "nan", "none"}:
                return None
            return float(text)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _int_field(value: object) -> int | None:
        try:
            text = str(value).strip()
            if not text:
                return None
            return int(float(text))
        except (TypeError, ValueError):
            return None

    def _fetch_remote_machine_status(self, remote) -> RemoteMachineStatus:
        import csv
        import io

        host = (remote.exec_out("hostname 2>/dev/null || echo unknown", timeout=5) or self.profile.hostname).strip()
        uptime = (remote.exec_out("uptime -p 2>/dev/null || true", timeout=5) or "").strip()
        load_avg = (remote.exec_out("cat /proc/loadavg 2>/dev/null | awk '{print $1\" \"$2\" \"$3}' || true", timeout=5) or "").strip()
        cpu_text = (remote.exec_out("getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || true", timeout=5) or "").strip()
        mem_raw = remote.exec_out(
            "/opt/conda/bin/python - <<'PY' 2>/dev/null || python3 - <<'PY'\n"
            "from pathlib import Path\n"
            "fields = {}\n"
            "for line in Path('/proc/meminfo').read_text().splitlines():\n"
            "    key, _, value = line.partition(':')\n"
            "    parts = value.strip().split()\n"
            "    if parts:\n"
            "        fields[key] = float(parts[0]) / 1024.0\n"
            "mem_total = fields.get('MemTotal', 0.0)\n"
            "mem_avail = fields.get('MemAvailable', 0.0)\n"
            "swap_total = fields.get('SwapTotal', 0.0)\n"
            "swap_free = fields.get('SwapFree', 0.0)\n"
            "print(f'mem_total={mem_total:.0f} mem_used={max(mem_total - mem_avail, 0.0):.0f} mem_avail={mem_avail:.0f} swap_total={swap_total:.0f} swap_used={max(swap_total - swap_free, 0.0):.0f}')\n"
            "PY",
            timeout=5,
        ) or ""
        mem: dict[str, float] = {}
        for token in mem_raw.split():
            key, sep, value = token.partition("=")
            if sep:
                parsed = self._float_field(value)
                if parsed is not None:
                    mem[key] = parsed
        disks: list[RemoteDiskInfo] = []
        disk_raw = remote.exec_out(
            "df -h /root/gpufree-data /root /tmp 2>/dev/null | awk 'NR>1 {print $6\"|\"$2\"|\"$3\"|\"$4\"|\"$5}'",
            timeout=8,
        ) or ""
        seen_mounts: set[str] = set()
        for line in disk_raw.splitlines():
            mount, _, rest = line.partition("|")
            if not mount or mount in seen_mounts:
                continue
            seen_mounts.add(mount)
            size, used, avail, use_pct = (rest.split("|") + ["", "", "", ""])[:4]
            disks.append(RemoteDiskInfo(mount=mount, size=size, used=used, avail=avail, use_pct=use_pct))

        tmux_sessions: list[RemoteTmuxSessionInfo] = []
        tmux_raw = remote.exec_out(
            "tmux list-sessions -F '#S|#{session_windows}|#{session_created_string}|#{session_attached}' 2>/dev/null || true",
            timeout=5,
        ) or ""
        for line in tmux_raw.splitlines():
            parts = line.split("|")
            if not parts or not parts[0]:
                continue
            tmux_sessions.append(
                RemoteTmuxSessionInfo(
                    name=parts[0],
                    windows=self._int_field(parts[1] if len(parts) > 1 else 0) or 0,
                    created=parts[2] if len(parts) > 2 else "",
                    attached=(parts[3] if len(parts) > 3 else "0") == "1",
                )
            )

        training_processes: list[dict[str, object]] = []
        proc_raw = remote.exec_out(
            "ps -eo pid,etime,pcpu,pmem,cmd --sort=-pcpu | "
            "grep -E '[t]aili_blind_runtime\\.train_taili|[t]aili_blind_runtime\\.launch_taili_train' | "
            "grep -v grep | head -12",
            timeout=8,
        ) or ""
        for line in proc_raw.splitlines():
            parts = line.strip().split(None, 4)
            if len(parts) < 5:
                continue
            training_processes.append(
                {
                    "pid": self._int_field(parts[0]) or 0,
                    "etime": parts[1],
                    "cpu_pct": self._float_field(parts[2]),
                    "mem_pct": self._float_field(parts[3]),
                    "command": parts[4][-240:],
                }
            )

        gpus: list[RemoteGPUInfo] = []
        gpu_query = (
            "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,"
            "utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null || true"
        )
        gpu_raw = remote.exec_out(gpu_query, timeout=8) or ""
        proc_query = (
            "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory "
            "--format=csv,noheader,nounits 2>/dev/null || true"
        )
        proc_gpu_raw = remote.exec_out(proc_query, timeout=8) or ""
        gpu_processes: list[dict[str, object]] = []
        for row in csv.reader(io.StringIO(proc_gpu_raw)):
            if len(row) < 4:
                continue
            gpu_processes.append(
                {
                    "gpu_uuid": row[0].strip(),
                    "pid": self._int_field(row[1]),
                    "name": row[2].strip(),
                    "used_memory_mb": self._float_field(row[3]),
                }
            )
        for row in csv.reader(io.StringIO(gpu_raw)):
            if len(row) < 8:
                continue
            gpus.append(
                RemoteGPUInfo(
                    index=self._int_field(row[0]) or 0,
                    name=row[1].strip(),
                    memory_used_mb=self._float_field(row[2]),
                    memory_total_mb=self._float_field(row[3]),
                    utilization_gpu_pct=self._float_field(row[4]),
                    utilization_memory_pct=self._float_field(row[5]),
                    temperature_c=self._float_field(row[6]),
                    power_w=self._float_field(row[7]),
                    processes=gpu_processes,
                )
            )
        raw = {
            "mem": mem,
            "nvidia_smi_seen": bool(gpu_raw.strip()),
            "commands": [
                "hostname",
                "uptime -p",
                "cat /proc/loadavg",
                "free -m",
                "df -h /root/gpufree-data /root /tmp",
                "tmux list-sessions",
                "ps -eo pid,etime,pcpu,pmem,cmd",
                "nvidia-smi --query-gpu ...",
                "nvidia-smi --query-compute-apps ...",
            ],
            "training_process_count": len(training_processes),
        }
        return RemoteMachineStatus(
            available=True,
            source="real",
            generated_at=time.time(),
            host=host,
            uptime=uptime,
            load_avg=load_avg,
            cpu_count=self._int_field(cpu_text),
            memory_total_mb=mem.get("mem_total"),
            memory_used_mb=mem.get("mem_used"),
            memory_available_mb=mem.get("mem_avail"),
            swap_used_mb=mem.get("swap_used"),
            swap_total_mb=mem.get("swap_total"),
            gpus=gpus,
            disks=disks,
            tmux_sessions=tmux_sessions,
            training_processes=training_processes,
            raw=raw,
        )

    # -- remote resolution (structure from profile, instance resolved live) --

    def _run_globs(self) -> tuple[str, ...]:
        globs = tuple(getattr(self.framework, "run_globs", ()) or ())
        if globs:
            return globs
        return (self.profile.runs_glob,) if self.profile.runs_glob else ()

    def _run_glob_shell(self) -> str:
        return " ".join(self._run_globs())

    def _newest_run(self, remote) -> str:
        globs = self._run_glob_shell()
        if not globs:
            return ""
        cmd = f"ls -dt {globs} 2>/dev/null"
        if self.s.run_filter:
            cmd += f" | grep -- {self.s.run_filter!r}"
        cmd += " | head -1"
        out = remote.exec_out(cmd)
        return (out or "").strip().rstrip("/")

    def get_acceptance(self) -> dict:
        """Score the newest run's persisted physeval logs against the spec (§2 verdict).

        Read-only and best-effort: it reads `physeval_*.log` files that `acceptance_run` (or a manual
        physeval) left in the run directory and aggregates them with the unit-tested scorer. Any failure
        returns ``{available: False, reason: ...}`` — this is a SEPARATE code path from get_status, so it
        can never degrade the live training view. It does NOT launch physeval (that contends with the GPU).
        """
        import shlex

        from autotuner.blind_locomotion import acceptance_aggregate as AGG
        try:
            remote = self._get_remote()
            run = self._newest_run(remote)
            if not run:
                return {"available": False, "reason": "no run found"}
            listing = remote.exec_out(f"ls -1 {shlex.quote(run)}/physeval_*.log 2>/dev/null", timeout=8)
            paths = [p.strip() for p in (listing or "").splitlines() if p.strip()][:8]
            if not paths:
                return {"available": False, "reason": "no physeval logs yet — run acceptance_run to measure",
                        "run": run}
            texts = [remote.exec_out(f"cat {shlex.quote(p)}", timeout=15) or "" for p in paths]
            verdict = AGG.aggregate(texts)
            # per-gate detail (A1[fwd05] -> {ok, detail}) so the copilot can explain WHY a family
            # fails ("median 0.14 > 0.10"), not just that it failed. families[] gives per-family status.
            merged = AGG.merge_runs(AGG.parse_scorecard(t) for t in texts)
            verdict["gates"] = {k: {"ok": bool(v.get("ok")), "detail": str(v.get("detail", ""))}
                                for k, v in merged.items()}
            verdict["available"] = True
            verdict["run"] = run
            verdict["scorecards_read"] = len(paths)
            return verdict
        except Exception as exc:  # never propagate into the request path
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    def _latest_ckpt_iter(self, remote, run: str) -> int:
        import shlex

        if not run:
            return 0
        sub = self.profile.checkpoints_subdir
        checkpoint_dir = f"{run.rstrip('/')}/{sub.strip('/')}"
        # Do not grep digits from the full path: run ids contain dates such as
        # 20260703, which previously won over agent_5000.pt and broke status /
        # resume selection.  Only basename agent_<step>.pt is authoritative.
        out = remote.exec_out(
            "bash -lc "
            + shlex.quote(
                f"dir={shlex.quote(checkpoint_dir)}; "
                "[ -d \"$dir\" ] || exit 0; "
                "find \"$dir\" -maxdepth 1 -type f -name 'agent_*.pt' -printf '%f\\n' 2>/dev/null | "
                "sed -nE 's/^agent_([0-9]+)\\.pt$/\\1/p' | sort -n | tail -1"
            ),
            timeout=10,
        )
        try:
            return int((out or "0").strip() or 0)
        except ValueError:
            return 0

    def _is_running(self, remote) -> bool:
        out = self._training_process_probe(remote)
        return bool((out or "").strip())

    def _training_process_probe(self, remote) -> str:
        import shlex

        # Training status must not match diagnostics just because both commands
        # mention the IsaacLab task name. Only payload-owned training entry
        # points count as training processes here.
        pattern = r"[t]aili_blind_runtime\.train_taili|[t]aili_blind_runtime\.launch_taili_train"
        return remote.exec_out(
            "bash -lc "
            + shlex.quote(f"pgrep -fa {shlex.quote(pattern)} | grep -v pgrep || true"),
            timeout=8,
        ) or ""

    def _latest_payload_root(self, remote) -> str:
        import shlex

        roots = [
            "/root/gpufree-data/training_payloads/taili_blind_runtime_*",
        ]
        # The entries in ``roots`` are trusted framework-owned glob patterns.
        # Do not shell-quote the ``*`` itself: quoting it prevents expansion and
        # makes the console report "payload not found" even when deployment
        # succeeded.  Append "/" so ls only considers extracted directories and
        # never picks the uploaded .tar.gz archive.
        patterns = " ".join(f"{item}/" for item in roots)
        out = remote.exec_out(
            "bash -lc "
            + shlex.quote(
                "ls -td " + patterns + " 2>/dev/null | head -1"
            ),
            timeout=10,
        )
        return (out or "").strip().rstrip("/").splitlines()[0] if (out or "").strip() else ""

    @staticmethod
    def _remote_timestamp(remote) -> str:
        out = remote.exec_out("date +%Y%m%d_%H%M%S", timeout=5)
        return (out or "").strip() or str(int(time.time()))

    def _run_boot_id(self, remote) -> str:
        return (remote.exec_out("cat /proc/sys/kernel/random/boot_id 2>/dev/null || true", timeout=5) or "").strip()

    def _recorded_run_boot_id(self, remote, run: str) -> str:
        import shlex

        if not run:
            return ""
        run_clean = run.rstrip("/")
        script = (
            f"run={shlex.quote(run_clean)}; "
            "if [ -s \"$run/remote_boot_id.txt\" ]; then "
            "  head -1 \"$run/remote_boot_id.txt\"; exit 0; "
            "fi; "
            "if [ -s \"$run/console_start.json\" ]; then "
            "  sed -nE 's/.*\"remote_boot_id\"[[:space:]]*:[[:space:]]*\"([^\"]+)\".*/\\1/p' "
            "    \"$run/console_start.json\" | head -1; exit 0; "
            "fi"
        )
        return (remote.exec_out("bash -lc " + shlex.quote(script), timeout=5) or "").strip()

    def _run_interruption_reason(self, remote, run: str, running: bool) -> str:
        if running or not run:
            return ""
        try:
            run_boot = self._recorded_run_boot_id(remote, run)
            current_boot = self._run_boot_id(remote)
        except Exception:
            return ""
        if run_boot and current_boot and run_boot != current_boot:
            return (
                "remote boot id changed after this run started "
                f"({run_boot[:8]} -> {current_boot[:8]}), so the training tmux/process was interrupted"
            )
        return ""

    def _apply_runtime_state(
        self,
        remote,
        run: str,
        running: bool,
        telemetry: TrainingTelemetry,
    ) -> tuple[str, str]:
        reason = self._run_interruption_reason(remote, run, running)
        if not reason:
            return telemetry.runtime_state, ""
        telemetry.runtime_state = "interrupted"
        telemetry.stale = True
        telemetry.source_stats["runtime_state"] = "interrupted"
        telemetry.source_stats["interruption_reason"] = reason
        telemetry.limitations.insert(0, "Remote reboot detected: current values are the last telemetry sample before interruption.")
        if telemetry.snapshot is not None:
            telemetry.snapshot.runtime_state = "interrupted"
            telemetry.snapshot.stale = True
            telemetry.snapshot.notes.insert(0, reason)
            prefix = "远程机/容器重启后训练已中断；下面只是中断前最后一次 telemetry。"
            if telemetry.snapshot.conclusion:
                telemetry.snapshot.conclusion = prefix + " " + telemetry.snapshot.conclusion
            else:
                telemetry.snapshot.conclusion = prefix
        return "interrupted", reason

    def _tfevents_name(self) -> str:
        import os
        # basename of the discovered glob (handles absolute or relative form)
        return os.path.basename(self.profile.tfevents_glob) or "events.out.tfevents.*"

    def _train_log_path(self, run: str = "") -> str:
        configured = (self.settings.remote_log_path or "").strip()
        profile_log = (self.profile.train_log or "").strip()
        if configured:
            return configured
        if run:
            return run.rstrip("/") + "/train.log"
        if profile_log and profile_log not in {"/root/robot_lab/train.log", "/tmp/rl_train.log"}:
            return profile_log
        return "/root/gpufree-data/logs/taili_train.log"

    def _telemetry_jsonl_path(self, log_path: str) -> str:
        if log_path.endswith(".log"):
            return log_path[:-4] + ".telemetry.jsonl"
        return log_path + ".telemetry.jsonl"

    def _find_first_file(self, remote, candidates: list[str]) -> str:
        import shlex

        safe = [item for item in candidates if item]
        if not safe:
            return ""
        script = "\n".join(
            f"if [ -s {shlex.quote(path)} ]; then echo {shlex.quote(path)}; exit 0; fi"
            for path in safe
        )
        if not script:
            return ""
        out = (remote.exec_out("bash -lc " + shlex.quote(script), timeout=10) or "").strip()
        lines = out.splitlines()
        return lines[0].strip() if lines else ""

    def _find_newest_matching_file(self, remote, run: str, pattern: str, *, maxdepth: int = 2) -> str:
        import shlex

        if not run:
            return ""
        script = (
            f"find {shlex.quote(run)} -maxdepth {int(maxdepth)} -type f -name {shlex.quote(pattern)} "
            "-printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-"
        )
        return (remote.exec_out("bash -lc " + shlex.quote(script), timeout=10) or "").strip()

    def _resolve_telemetry_paths(self, remote, run: str) -> _TelemetryPaths:
        """Resolve the concrete telemetry/log artifacts for a run.

        The priority is:
        1. explicit LOCOMOTION_CONSOLE_REMOTE_LOG override;
        2. current run directory contract from taili_blind_runtime/launch_taili_train.py;
        3. newest matching telemetry/log file inside the run;
        4. legacy profile/fallback log path.

        This keeps discovery deterministic and allowlisted. The LLM receives the
        result as evidence; it does not choose arbitrary remote paths.
        """
        configured = (self.settings.remote_log_path or "").strip()
        evidence: list[str] = []
        if configured:
            log_path = configured
            telemetry_path = self._telemetry_jsonl_path(log_path)
            console_log_path = log_path
            checkpoint_dir = f"{run.rstrip('/')}/{self.profile.checkpoints_subdir}" if run else ""
            effective_config_path = ""
            evidence.append("configured LOCOMOTION_CONSOLE_REMOTE_LOG")
        elif run:
            run_clean = run.rstrip("/")
            telemetry_candidates = [
                f"{run_clean}/train.telemetry.jsonl",
                f"{run_clean}/telemetry.jsonl",
                f"{run_clean}/train.jsonl",
            ]
            telemetry_path = self._find_first_file(remote, telemetry_candidates)
            if telemetry_path:
                evidence.append("run contract telemetry file")
            else:
                telemetry_path = self._find_newest_matching_file(remote, run_clean, "*.telemetry.jsonl", maxdepth=2)
                if telemetry_path:
                    evidence.append("newest *.telemetry.jsonl under run")
            log_candidates = [
                f"{run_clean}/train.log",
                f"{run_clean}/console.log",
            ]
            log_path = self._find_first_file(remote, log_candidates)
            if log_path:
                evidence.append("run contract log file")
            else:
                log_path = self._find_newest_matching_file(remote, run_clean, "*.log", maxdepth=2)
                if log_path:
                    evidence.append("newest *.log under run")
            console_log_path = f"{run_clean}/console.log"
            checkpoint_dir = f"{run_clean}/{self.profile.checkpoints_subdir}"
        else:
            log_path = self._train_log_path("")
            telemetry_path = self._telemetry_jsonl_path(log_path)
            console_log_path = log_path
            checkpoint_dir = ""
            effective_config_path = ""
            evidence.append("fallback log path because no run was resolved")

        if not run:
            checkpoint_dir = ""
            effective_config_path = ""
        else:
            run_clean = run.rstrip("/")
            checkpoint_dir = f"{run_clean}/{self.profile.checkpoints_subdir}"
            effective_config_path = self._find_first_file(
                remote,
                [
                    f"{run_clean}/effective_config.yaml",
                    f"{run_clean}/config/effective_config.yaml",
                ],
            )
            if effective_config_path:
                evidence.append("run contract effective_config.yaml")
        if not telemetry_path and log_path:
            telemetry_path = self._telemetry_jsonl_path(log_path)
            evidence.append("derived telemetry path from log path")
        return _TelemetryPaths(
            run=run,
            log_path=log_path,
            telemetry_path=telemetry_path,
            console_log_path=console_log_path,
            checkpoint_dir=checkpoint_dir,
            effective_config_path=effective_config_path,
            evidence=tuple(evidence),
        )

    def _newest_tfevents(self, remote, run: str) -> tuple[str, int, int]:
        import shlex

        name = self._tfevents_name()
        raw = remote.exec_out(
            "bash -lc "
            + shlex.quote(
                f"ev=$(ls -t {shlex.quote(run.rstrip('/') + '/')}{name} 2>/dev/null | head -1); "
                "if [ -n \"$ev\" ]; then stat -c '%n|%s|%Y' \"$ev\"; fi"
            )
        )
        line = (raw or "").strip().splitlines()[:1]
        if not line:
            return "", 0, 0
        path, _, rest = line[0].partition("|")
        size_text, _, mtime_text = rest.partition("|")
        try:
            size = int(size_text or 0)
            mtime = int(mtime_text or 0)
        except ValueError:
            size, mtime = 0, 0
        return path.strip(), size, mtime

    def _load_scalar_dataframe(self, remote, run: str, *, max_age_s: float = 5.0):
        """Fetch and parse the newest TensorBoard event file for a run.

        This is the single authoritative reader for live training scalar values. The
        websocket reward stream, scalar catalog, and scalar series endpoints all use
        this path so UI and LLM explanations agree on source and formula.
        """
        import hashlib
        import os
        import tempfile

        from tbparse import SummaryReader

        now = time.time()
        remote_ev, size, mtime = self._newest_tfevents(remote, run)
        if not remote_ev:
            self._scalar_cache = _ScalarCache(run=run, error="No TensorBoard event file found for the current run.")
            return None, "", "No TensorBoard event file found for the current run."
        cache = self._scalar_cache
        unchanged = (
            cache.run == run
            and cache.remote_event == remote_ev
            and cache.remote_size == size
            and cache.remote_mtime == mtime
            and cache.dataframe is not None
            and now - cache.fetched_at < max_age_s
        )
        if unchanged:
            return cache.dataframe, remote_ev, ""
        # Stable, non-salted name (builtin hash() is per-process randomized → a new temp file every
        # restart, forever accumulating). A hashlib digest of (run, remote_ev) reuses/overwrites the
        # same path across restarts; and we delete the previous cache's file when it changes.
        digest = hashlib.sha1(f"{run}\0{remote_ev}".encode("utf-8")).hexdigest()[:16]
        local = os.path.join(tempfile.gettempdir(), f"locomotion_console_{digest}.tfevents")
        prev_local = getattr(cache, "local_event", "") if cache else ""
        if prev_local and prev_local != local and os.path.exists(prev_local):
            try:
                os.unlink(prev_local)
            except OSError:
                pass
        remote.get(remote_ev, local)
        df = SummaryReader(local).scalars
        self._scalar_cache = _ScalarCache(
            run=run,
            remote_event=remote_ev,
            local_event=local,
            remote_size=size,
            remote_mtime=mtime,
            fetched_at=now,
            dataframe=df,
            error="",
        )
        return df, remote_ev, ""

    def _iter_scalar_rows(self, df, tag: str):
        if df is None or len(df) == 0:
            return []
        if "tag" in df.columns:
            sub = df[df["tag"] == tag]
            return [(int(r.step), float(r.value)) for r in sub.itertuples()]
        if tag not in df.columns:
            return []
        return [
            (int(step_v), float(val_v))
            for step_v, val_v in df[["step", tag]].itertuples(index=False, name=None)
        ]

    def _scalar_tags_from_dataframe(self, df) -> list[str]:
        if df is None or len(df) == 0:
            return []
        if "tag" in df.columns:
            return sorted(str(item) for item in df["tag"].unique().tolist())
        return sorted(str(c) for c in df.columns if c != "step")

    def _fetch_scalar_catalog(self, remote, run: str) -> TensorboardScalarCatalog:
        df, event_file, error = self._load_scalar_dataframe(remote, run)
        if error:
            return TensorboardScalarCatalog(
                available=False,
                source="real",
                run=run,
                event_file=event_file,
                reward_tag=self.profile.reward_tag,
                error=error,
                refreshed_at=time.time(),
            )
        tags: list[TensorboardTagInfo] = []
        for tag in self._scalar_tags_from_dataframe(df):
            pts = self._iter_scalar_rows(df, tag)
            if not pts:
                continue
            tags.append(
                TensorboardTagInfo(
                    tag=tag,
                    group=_scalar_group(tag),
                    display_name=_scalar_display_name(tag),
                    points=len(pts),
                    first_step=pts[0][0],
                    last_step=pts[-1][0],
                    last_value=pts[-1][1],
                    source=event_file,
                    formula=_scalar_formula(tag, event_file),
                    description=_scalar_description(tag),
                )
            )
        return TensorboardScalarCatalog(
            available=bool(tags),
            source="real",
            run=run,
            event_file=event_file,
            reward_tag=self.profile.reward_tag,
            tags=tags,
            refreshed_at=time.time(),
        )

    async def tensorboard_scalar_catalog(self) -> TensorboardScalarCatalog:
        try:
            remote = self._get_remote()
            run = await asyncio.to_thread(self._newest_run, remote)
            if not run:
                return TensorboardScalarCatalog(
                    available=False,
                    source="real",
                    reward_tag=self.profile.reward_tag,
                    error="No training run found.",
                    refreshed_at=time.time(),
                )
            result = await asyncio.to_thread(self._fetch_scalar_catalog, remote, run)
            self._remember_remote_success()
            return result
        except Exception as exc:  # noqa: BLE001
            message = self._remember_remote_failure(exc)
            return TensorboardScalarCatalog(
                available=False,
                source="real",
                reward_tag=self.profile.reward_tag,
                error=self._remote_unavailable_message(message),
                refreshed_at=time.time(),
            )

    def _fetch_scalar_series(self, remote, run: str, tags: list[str], max_points: int) -> TensorboardSeriesResponse:
        df, event_file, error = self._load_scalar_dataframe(remote, run)
        if error:
            return TensorboardSeriesResponse(
                available=False,
                source="real",
                run=run,
                event_file=event_file,
                error=error,
            )
        series: list[TensorboardScalarSeries] = []
        for tag in tags[:12]:
            pts = self._iter_scalar_rows(df, tag)
            if max_points > 0 and len(pts) > max_points:
                stride = max(1, math.ceil(len(pts) / max_points))
                pts = pts[::stride]
            series.append(
                TensorboardScalarSeries(
                    tag=tag,
                    group=_scalar_group(tag),
                    display_name=_scalar_display_name(tag),
                    source=event_file,
                    formula=_scalar_formula(tag, event_file),
                    points=[TensorboardScalarPoint(step=step, value=value) for step, value in pts],
                )
            )
        return TensorboardSeriesResponse(
            available=bool(series),
            source="real",
            run=run,
            event_file=event_file,
            series=series,
        )

    async def tensorboard_scalar_series(self, tags: list[str], max_points: int = 1200) -> TensorboardSeriesResponse:
        try:
            remote = self._get_remote()
            run = await asyncio.to_thread(self._newest_run, remote)
            if not run:
                return TensorboardSeriesResponse(available=False, source="real", error="No training run found.")
            clean_tags = [tag for tag in (tag.strip() for tag in tags) if tag]
            if not clean_tags:
                clean_tags = [self.profile.reward_tag]
            max_points = max(50, min(int(max_points or 1200), 5000))
            result = await asyncio.to_thread(self._fetch_scalar_series, remote, run, clean_tags, max_points)
            self._remember_remote_success()
            return result
        except Exception as exc:  # noqa: BLE001
            message = self._remember_remote_failure(exc)
            return TensorboardSeriesResponse(
                available=False,
                source="real",
                error=self._remote_unavailable_message(message),
            )

    def _fetch_training_telemetry(self, remote, run: str, running: bool) -> TrainingTelemetry:
        import shlex

        paths = self._resolve_telemetry_paths(remote, run)
        log_path = paths.log_path
        telemetry_path = paths.telemetry_path
        event_file, _event_size, _event_mtime = self._newest_tfevents(remote, run) if run else ("", 0, 0)
        # Hot poll: bound the read with an explicit short timeout so a slow/hung remote does not
        # stall the telemetry loop for the 30s default.
        jsonl_text = remote.exec_out(f"tail -n 240 {shlex.quote(telemetry_path)} 2>/dev/null", timeout=10) or ""
        source_stats: dict[str, object] = {}
        if telemetry_path:
            stat_text = remote.exec_out(
                "bash -lc "
                + shlex.quote(
                    f"if [ -e {shlex.quote(telemetry_path)} ]; then "
                    f"stat -c 'size=%s mtime=%Y' {shlex.quote(telemetry_path)}; "
                    f"wc -l < {shlex.quote(telemetry_path)} | awk '{{print \"lines=\"$1}}'; "
                    "date +remote_now=%s; "
                    "fi"
                ),
                timeout=10,
            ) or ""
            for token in stat_text.split():
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                try:
                    parsed: object = int(value)
                except ValueError:
                    parsed = value
                source_stats[f"{key}_s" if key in {"mtime", "remote_now"} else key] = parsed
        effective_config_text = ""
        if paths.effective_config_path:
            effective_config_text = remote.exec_out(
                f"sed -n '1,260p' {shlex.quote(paths.effective_config_path)} 2>/dev/null",
                timeout=10,
            ) or ""
        log_text = ""
        side_log_text = ""
        side_log_path = paths.console_log_path or log_path
        if side_log_path:
            side_log_text = remote.exec_out(f"tail -n 240 {shlex.quote(side_log_path)} 2>/dev/null", timeout=10) or ""
            if side_log_text.strip():
                source_stats["side_log_tail_lines"] = len(side_log_text.splitlines())
                source_stats["side_log_path"] = side_log_path
        if not jsonl_text.strip():
            log_tail_paths = [path for path in [log_path, paths.console_log_path] if path]
            for candidate in dict.fromkeys(log_tail_paths):
                log_text = remote.exec_out(f"tail -n 500 {shlex.quote(candidate)} 2>/dev/null") or ""
                if log_text.strip():
                    log_path = candidate
                    break
        telemetry = build_telemetry(
            source="real",
            run_id=run.rsplit("/", 1)[-1] if run else "",
            running=running,
            log_path=log_path,
            telemetry_path=telemetry_path,
            console_log_path=paths.console_log_path,
            checkpoint_dir=paths.checkpoint_dir,
            event_file=event_file,
            effective_config_path=paths.effective_config_path,
            effective_config_text=effective_config_text,
            path_evidence=list(paths.evidence),
            source_stats=source_stats,
            jsonl_text=jsonl_text,
            log_text=log_text,
            side_log_text=side_log_text,
        )
        from .definitions import definitions_for_telemetry

        telemetry.definitions = definitions_for_telemetry(telemetry)
        self._apply_runtime_state(remote, run, running, telemetry)
        if not telemetry.available and paths.evidence:
            telemetry.limitations.append("telemetry path discovery: " + "; ".join(paths.evidence))
        return telemetry

    async def training_telemetry(self) -> TrainingTelemetry:
        try:
            remote = self._get_remote()
            run = await asyncio.to_thread(self._newest_run, remote)
            running = await asyncio.to_thread(self._is_running, remote)
            telemetry = await asyncio.to_thread(self._fetch_training_telemetry, remote, run, running)
            self._remember_remote_success()
            return telemetry
        except Exception as exc:  # noqa: BLE001
            message = self._remember_remote_failure(exc)
            return self._remote_unavailable_telemetry(message)

    async def remote_machine_status(self) -> RemoteMachineStatus:
        try:
            remote = self._get_remote()
            status = await asyncio.to_thread(self._fetch_remote_machine_status, remote)
            self._remember_remote_success()
            return status
        except Exception as exc:  # noqa: BLE001
            message = self._remember_remote_failure(exc)
            return self._remote_unavailable_machine_status(message)

    def _degraded_status(self, error: Exception) -> RunStatus:
        return RunStatus(
            run_id="remote-unavailable",
            task=self.profile.task_id,
            robot_id="taili_dog_39kg",
            latest_iter=0,
            total_iter=0,
            running=False,
            runtime_state="remote_unavailable",
            source="real",
            note=(
                f"{self.profile.hostname}: remote is unavailable; "
                f"active framework {self.framework.id} ({self.framework.status})"
            ),
            remote_ok=False,
            degraded=True,
            error=f"{type(error).__name__}: {error}",
        )

    async def get_status(self) -> RunStatus:
        try:
            remote = self._get_remote()
            run = await asyncio.to_thread(self._newest_run, remote)
            it = await asyncio.to_thread(self._latest_ckpt_iter, remote, run) if run else 0
            running = await asyncio.to_thread(self._is_running, remote)
            telemetry = await asyncio.to_thread(self._fetch_training_telemetry, remote, run, running) if run else None
        except Exception as exc:  # noqa: BLE001 - remote outages are a displayable state
            self._remember_remote_failure(exc)
            return self._degraded_status(exc)
        self._remember_remote_success()
        runtime_state = telemetry.runtime_state if telemetry is not None else ("live" if running else "stopped")
        interruption_reason = ""
        if telemetry is None and run and not running:
            interruption_reason = await asyncio.to_thread(self._run_interruption_reason, remote, run, running)
            if interruption_reason:
                runtime_state = "interrupted"
        telemetry_step = telemetry.latest.step if telemetry is not None and telemetry.latest is not None else 0
        latest_iter = telemetry_step or it
        total_iter = (
            telemetry.latest.total_steps
            if telemetry is not None and telemetry.latest is not None and telemetry.latest.total_steps
            else max(latest_iter, it, 15000)
        )
        note = (
            f"{self.profile.hostname}: active framework {self.framework.id} "
            f"({self.framework.status}), newest run under {self.profile.experiment}"
        )
        if interruption_reason:
            note += f"; {interruption_reason}"
        return RunStatus(
            run_id=run.split("/")[-1] if run else "(no run found)",
            task=self.profile.task_id,
            robot_id="taili_dog_39kg",
            latest_iter=latest_iter,
            total_iter=total_iter,
            running=running,
            runtime_state=runtime_state,
            stale=(telemetry.stale if telemetry is not None else False),
            telemetry_age_s=(telemetry.telemetry_age_s if telemetry is not None else None),
            tmux_session="rl_train",
            source="real",
            note=note,
            remote_ok=True,
            degraded=False,
        )

    def _fetch_reward_curve(self, remote, run: str):
        """SFTP newest tfevents of `run` to local, parse reward curve with tbparse.

        Returns (points, available_tags). points = [(step, reward), ...].
        """
        df, _event_file, _error = self._load_scalar_dataframe(remote, run)
        if df is None or len(df) == 0:
            return [], []
        rtag = self.profile.reward_tag
        tags = self._scalar_tags_from_dataframe(df)
        pts = self._iter_scalar_rows(df, rtag)
        return pts, tags

    async def stream(self, stop: asyncio.Event) -> AsyncIterator[MetricPoint]:
        try:
            remote = self._get_remote()
            run = await asyncio.to_thread(self._newest_run, remote)
        except Exception as exc:  # noqa: BLE001
            self._remember_remote_failure(exc)
            yield MetricPoint(
                iter=0,
                ts=time.time(),
                reward=0.0,
                ep_len=0.0,
                extra={"remote_error": 1, "message": self._remote_unavailable_message(exc)},
            )
            return
        if not run:
            yield MetricPoint(iter=0, ts=time.time(), reward=0.0, ep_len=0.0,
                              extra={"error": "no run found"})
            return
        last_step = -1
        warned = False
        while not stop.is_set():
            try:
                pts, tags = await asyncio.to_thread(self._fetch_reward_curve, remote, run)
            except Exception:  # noqa: BLE001
                pts, tags = [], []
            if not pts and tags and not warned:
                # verify-don't-trust: reward tag absent -> surface what's actually there
                warned = True
                yield MetricPoint(iter=last_step + 1, ts=time.time(), reward=0.0,
                                  ep_len=0.0, extra={"reward_tag_missing": 1,
                                                     "n_tags": len(tags)})
            for step, reward in pts:
                if step <= last_step:
                    continue
                last_step = step
                yield MetricPoint(iter=step, ts=time.time(), reward=reward,
                                  ep_len=0.0, terrain=0.0, phase=0, extra={})
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.s.poll_interval_s)
            except asyncio.TimeoutError:
                pass

    # -- config safety net (backup -> edit -> rollback) --

    def _config_sha(self, remote, path: str) -> str:
        out = remote.exec_out(f"sha256sum {path} 2>/dev/null | awk '{{print $1}}'")
        return (out or "").strip()

    def _backup_config(self, remote) -> str:
        """Snapshot the env_cfg to a timestamped backup + a 'latest' slot. Returns backup path."""
        p = self.profile.env_cfg_path
        if not p:
            return ""
        ts = remote.exec_out("date +%Y%m%d_%H%M%S").strip()
        bak = f"{p}.locomotion_console_bak_{ts}"
        remote.exec_out(f"cp -f {p} {bak} && cp -f {p} {p}.locomotion_console_latest_bak")
        return bak

    def _rollback_config(self, remote) -> bool:
        """Restore the env_cfg from the latest console backup."""
        p = self.profile.env_cfg_path
        if not p:
            return False
        out = remote.exec_out(
            f"test -f {p}.locomotion_console_latest_bak && cp -f {p}.locomotion_console_latest_bak {p} && echo ok")
        return "ok" in (out or "")

    def _edit_config_value(self, remote, key: str, value: str) -> dict:
        """Backup-then-edit a single dataclass field's value in the env_cfg.

        ALWAYS backs up first (rollback available). Reads the line back so the caller can verify
        the change took the intended effect - never trust the write blind.
        """
        import re
        import shlex
        p = self.profile.env_cfg_path
        if not p:
            return {"ok": False, "detail": "env_cfg_path not discovered"}
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key or ""):
            return {"ok": False, "detail": f"unsafe key: {key!r}"}
        # `value` is spliced into the sed PROGRAM (not just the shell). shlex.quote protects the
        # outer shell but NOT the sed script, where the raw value could carry the `|` delimiter,
        # `\`, `&`, or GNU sed's e/w commands to write/execute arbitrary remote files. Restrict it
        # to the legitimate env_cfg scalar/list grammar (numbers incl. -3e-4, booleans, simple
        # strings, lists) — this excludes |, \, &, #, and all control chars.
        if not re.fullmatch(r"[A-Za-z0-9_.+\-\[\], \"']*", value or ""):
            return {"ok": False, "detail": f"unsafe value: {value!r}"}
        pq = shlex.quote(p)
        old = (remote.exec_out(f"grep -nE '^\\s*{key}\\s*[:=]' {pq} 2>/dev/null | head -1") or "").strip()
        if not old:
            return {"ok": False, "detail": f"key not found in env_cfg: {key}"}
        self._backup_config(remote)
        # preserve indent + optional type annotation + trailing comment; replace only the value
        sed = (rf"s|^(\s*{key}\s*(:\s*[A-Za-z0-9_\[\], ]*)?=\s*)[^#\n]*"
               rf"|\1{value} |")
        remote.exec_out(f"sed -i -E {shlex.quote(sed)} {pq}")
        new = (remote.exec_out(f"grep -nE '^\\s*{key}\\s*[:=]' {pq} 2>/dev/null | head -1") or "").strip()
        # self-check: the new value must actually appear; otherwise auto-rollback (no silent damage)
        if value not in new:
            self._rollback_config(remote)
            return {"ok": False, "key": key, "old_line": old, "new_line": new,
                    "detail": f"edit readback mismatch for {key} -> auto-rolled back"}
        return {"ok": True, "key": key, "old_line": old, "new_line": new,
                "backup": "taken (rollback available)"}

    # -- action helpers --

    def _resolve_checkpoint(self, remote, run: str) -> str:
        """Full path to the latest checkpoint of `run` (agent_<maxiter>.pt, else best_agent.pt)."""
        if not run:
            return ""
        sub = self.profile.checkpoints_subdir
        it = self._latest_ckpt_iter(remote, run)
        if it > 0:
            return f"{run}/{sub}/agent_{it}.pt"
        return (remote.exec_out(f'ls "{run}/{sub}/"best_agent.pt 2>/dev/null | head -1') or "").strip()

    def _launch_tmux(self, remote, session: str, cmd: str) -> None:
        """Launch `cmd` detached in a fresh tmux session on the box."""
        import shlex
        remote.exec_out(f"tmux kill-session -t {session} 2>/dev/null; "
                        f"tmux new-session -d -s {session}")
        remote.exec_out(f"tmux send-keys -t {session} {shlex.quote(cmd)} Enter")

    def _deploy_payload(self, remote) -> tuple[str, str, int]:
        import shlex

        from autotuner.training_payloads.taili_blind_runtime.build_payload import build_payload

        result = build_payload()
        remote_root = "/root/gpufree-data/training_payloads"
        remote_archive = f"{remote_root}/{result.archive.name}"
        remote_payload = f"{remote_root}/{result.root_name}"
        remote.exec_out(f"mkdir -p {shlex.quote(remote_root)}", timeout=10)
        remote.put(str(result.archive), remote_archive)
        script = (
            "set -euo pipefail; "
            f"archive={shlex.quote(remote_archive)}; "
            f"payload={shlex.quote(remote_payload)}; "
            "rm -rf \"$payload\"; "
            "mkdir -p \"$payload\"; "
            "tar -xzf \"$archive\" -C \"$payload\"; "
            "test -f \"$payload/sitecustomize.py\"; "
            "test -f \"$payload/taili_blind_runtime/train_taili.py\"; "
            "test -f \"$payload/taili_blind_runtime/launch_taili_train.py\"; "
            "test -f \"$payload/taili_blind_runtime/blind_tp_env.py\"; "
            "test -f \"$payload/taili_blind_runtime/diagnose_taili_cases.py\"; "
            "test -f \"$payload/taili_blind_runtime/taili_blind_config.yaml\"; "
            "test -f \"$payload/taili_blind_runtime/assets/robots/taili-dog/robot.urdf\"; "
            "test -d \"$payload/taili_blind_runtime/assets/robots/taili-dog/meshes\"; "
            "PYTHONPATH=\"$payload${PYTHONPATH:+:$PYTHONPATH}\" "
            "/opt/conda/envs/isaaclab/bin/python - <<'PY'\n"
            "import importlib.util\n"
            "mods = ['taili_blind_runtime', 'taili_blind_runtime.launch_taili_train', 'taili_blind_runtime.train_taili', 'taili_blind_runtime.diagnose_taili_cases']\n"
            "missing = [m for m in mods if importlib.util.find_spec(m) is None]\n"
            "if missing:\n"
            "    raise SystemExit('missing modules: ' + ','.join(missing))\n"
            "print('payload_import_ok')\n"
            "PY\n"
            "printf '%s\\n' \"$payload\""
        )
        out = remote.exec_out("bash -lc " + shlex.quote(script), timeout=120)
        payload_line = ""
        for line in (out or "").splitlines():
            if line.startswith(remote_root + "/"):
                payload_line = line.strip()
        return payload_line or remote_payload, remote_archive, result.file_count

    def _start_payload_training(self, remote, *, resume: bool = False) -> tuple[str, str]:
        import shlex

        if self._is_running(remote):
            raise RuntimeError("training is already running")
        payload = self._latest_payload_root(remote)
        if not payload:
            raise RuntimeError("taili_blind_runtime payload not found under /root/gpufree-data/training_payloads")
        ts = self._remote_timestamp(remote)
        run_id = f"taili_train_{ts}" + ("_resume" if resume else "_console")
        run_dir = f"/root/gpufree-data/taili_runs/{run_id}"
        checkpoint_arg = ""
        if resume:
            previous_run = self._newest_run(remote)
            checkpoint = self._resolve_checkpoint(remote, previous_run)
            if not checkpoint:
                raise RuntimeError("resume requested, but no checkpoint was found")
            checkpoint_arg = " --checkpoint " + shlex.quote(checkpoint)
        boot_id = self._run_boot_id(remote)
        remote.exec_out(f"mkdir -p {shlex.quote(run_dir)}", timeout=10)
        # SAFE REGIME (audit 0706): this operator-facing launch used --num_envs 4096 (OOMs — 2048 already
        # walls ~16k), checkpoint every 5000 (loses more on a stall restart), and no curriculum-phase restore.
        # Match the campaign's proven regime: 1024 envs, 2000-step checkpoints, and TAILI_INIT_PHASE=3 on a
        # resume (the checkpoint does not store the phase, so without it a trained resume drops to flat phi0).
        init_phase_env = "export TAILI_INIT_PHASE=3; " if checkpoint_arg else ""
        inner = (
            "set -e; "
            f"cd {shlex.quote(payload)}; "
            "export PYTHONUNBUFFERED=1; "
            "export TAILI_CHECKPOINT_INTERVAL=2000; "
            "export TAILI_WRITE_INTERVAL=auto; "
            f"{init_phase_env}"
            f"printf '%s\\n' {shlex.quote(boot_id)} > {shlex.quote(run_dir + '/remote_boot_id.txt')}; "
            "/opt/conda/envs/isaaclab/bin/python -m taili_blind_runtime.launch_taili_train "
            "--python /opt/conda/envs/isaaclab/bin/python "
            "--data-root /root/gpufree-data "
            f"--run-id {shlex.quote(run_id)} "
            "--total-steps 1500000 "
            "--telemetry-interval 10 "
            f"{checkpoint_arg} "
            "--headless -- --num_envs 1024"
        )
        command = (
            f"bash -lc {shlex.quote(inner)}; "
            "rc=$?; "
            "printf '\\n[locomotion-console] training command exited with rc=%s.\\n' \"$rc\"; "
            "printf '[locomotion-console] Full stdout/stderr is in the run console.log; this tmux pane is kept for inspection.\\n'; "
            "printf '[locomotion-console] Start a new run from the console UI or exit this shell manually.\\n'; "
            "exec bash -l"
        )
        self._launch_tmux(remote, "rl_train", command)
        marker = (
            "bash -lc "
            + shlex.quote(
                f"cat > {shlex.quote(run_dir + '/console_start.json')} <<'JSON'\n"
                + "{\n"
                + f"  \"run_id\": \"{run_id}\",\n"
                + f"  \"run_dir\": \"{run_dir}\",\n"
                + f"  \"payload\": \"{payload}\",\n"
                + "  \"tmux_session\": \"rl_train\",\n"
                + f"  \"remote_boot_id\": \"{boot_id}\",\n"
                + f"  \"resume\": {str(bool(resume)).lower()}\n"
                + "}\nJSON"
            )
        )
        remote.exec_out(marker, timeout=10)
        return run_id, run_dir

    async def action_kill(self) -> ActionResult:
        try:
            remote = self._get_remote()
            import shlex

            pattern = r"[t]aili_blind_runtime\.train_taili|[t]aili_blind_runtime\.launch_taili_train"
            command = (
                "bash -lc "
                + shlex.quote(
                    "tmux kill-session -t rl_train 2>/dev/null || true; "
                    f"pkill -f {shlex.quote(pattern)} 2>/dev/null || true; "
                    "echo ok"
                )
            )
            await asyncio.to_thread(remote.exec_out, command)
            self._remember_remote_success()
            return ActionResult(action="kill", ok=True,
                                message="stopped rl_train tmux session and Taili training launcher/processes")
        except Exception as e:  # noqa: BLE001
            message = self._remember_remote_failure(e)
            return ActionResult(action="kill", ok=False, message=f"kill failed: {self._remote_unavailable_message(message)}")

    async def action_deploy_payload(self) -> ActionResult:
        try:
            remote = self._get_remote()
            if await asyncio.to_thread(self._is_running, remote):
                self._remember_remote_success()
                return ActionResult(action="deploy_payload", ok=False, message="refusing to deploy while training is running")
            payload, archive, file_count = await asyncio.to_thread(self._deploy_payload, remote)
            self._remember_remote_success()
            return ActionResult(
                action="deploy_payload",
                ok=True,
                message=f"deployed {payload}; archive={archive}; files={file_count}",
            )
        except Exception as e:  # noqa: BLE001
            message = self._remember_remote_failure(e)
            return ActionResult(action="deploy_payload", ok=False, message=f"deploy failed: {self._remote_unavailable_message(message)}")

    async def action_start(self) -> ActionResult:
        try:
            remote = self._get_remote()
            run_id, run_dir = await asyncio.to_thread(self._start_payload_training, remote, resume=False)
            self._remember_remote_success()
            return ActionResult(
                action="start",
                ok=True,
                message=f"started {run_id} in tmux 'rl_train'; run_dir={run_dir}",
            )
        except Exception as e:  # noqa: BLE001
            message = self._remember_remote_failure(e)
            return ActionResult(action="start", ok=False, message=f"start failed: {self._remote_unavailable_message(message)}")

    async def action_resume(self) -> ActionResult:
        try:
            remote = self._get_remote()
            if await asyncio.to_thread(self._is_running, remote):
                self._remember_remote_success()
                return ActionResult(action="resume", ok=False, message="training is already running")
            run_id, run_dir = await asyncio.to_thread(self._start_payload_training, remote, resume=True)
            self._remember_remote_success()
            return ActionResult(action="resume", ok=True,
                                message=f"resumed as {run_id} in tmux 'rl_train'; run_dir={run_dir}")
        except Exception as e:  # noqa: BLE001
            message = self._remember_remote_failure(e)
            return ActionResult(action="resume", ok=False, message=f"resume failed: {self._remote_unavailable_message(message)}")

    def _resolve_run(self, remote, run: str) -> str:
        if not run:
            return self._newest_run(remote)
        import shlex
        # run reaches a remote shell; repr() is NOT shell-safe (a single quote flips it to a
        # double-quoted string where $()/backticks still expand). Allowlist + shlex.quote.
        if not re.fullmatch(r"[A-Za-z0-9._/\-]+", run):
            return ""
        out = remote.exec_out(
            f"ls -dt {self._run_glob_shell()} 2>/dev/null | grep -- {shlex.quote(run)} | head -1")
        return (out or "").strip().rstrip("/")

    async def action_physeval(self, run: str = "") -> PhysevalResult:
        if not self.profile.physeval_cmd:
            return PhysevalResult(ok=False, summary="physeval command has not been discovered in the profile")
        try:
            remote = self._get_remote()
            training_running = await asyncio.to_thread(self._is_running, remote)
            run_path = await asyncio.to_thread(self._resolve_run, remote, run)
            ckpt = await asyncio.to_thread(self._resolve_checkpoint, remote, run_path)
            if not ckpt:
                self._remember_remote_success()
                return PhysevalResult(ok=False, summary="checkpoint not found")
            log = self.profile.physeval_log
            cmd = self.profile.physeval_cmd.replace("{checkpoint}", ckpt) + f" 2>&1 | tee {log}"
            # clear the prior eval log so a fresh read isn't confused by stale output
            await asyncio.to_thread(remote.exec_out, f"rm -f {log}")
            await asyncio.to_thread(self._launch_tmux, remote, "locomotion_console_eval", cmd)
            self._remember_remote_success()
            warning = (
                " Training is currently running; this evaluation may contend for GPU/IsaacLab resources."
                if training_running
                else ""
            )
            return PhysevalResult(ok=True, checkpoint=ckpt,
                                  summary=f"physeval started in tmux 'locomotion_console_eval' for "
                                          f"{ckpt.split('/')[-1]}; log: {log}. Use get_eval_result after it finishes."
                                          f"{warning}")
        except Exception as e:  # noqa: BLE001
            message = self._remember_remote_failure(e)
            return PhysevalResult(ok=False, summary=f"physeval failed: {self._remote_unavailable_message(message)}")

    async def action_run_acceptance(self, run: str = "", terrains: str = "flat",
                                    checkpoint: str = "best_agent.pt") -> ActionResult:
        """Launch a spec-acceptance MEASUREMENT (physeval → taili_spec §2 score). Drives the box via
        the product `acceptance_run` CLI in a detached LOCAL subprocess (the console is local; the CLI
        SSHes to the box). It self-refuses if training is active (acceptance_run's own GPU guard), so
        this never contends with a converging run. The scored verdict then surfaces via get_acceptance."""
        import os
        import re
        import subprocess
        import sys
        from pathlib import Path

        try:
            terr = [t for t in re.split(r"[,\s]+", terrains or "flat") if re.fullmatch(r"[a-z_]+", t)] or ["flat"]
            if checkpoint and not re.fullmatch(r"[A-Za-z0-9._\-]+", checkpoint):
                return ActionResult(action="run_acceptance", ok=False, message=f"invalid checkpoint name: {checkpoint!r}")
            run_id = run if (run and re.fullmatch(r"[A-Za-z0-9._\-]+", run)) else "newest"
            repo_root = str(Path(__file__).resolve().parents[2])
            args = [sys.executable, "-m", "autotuner.training.acceptance_run", run_id,
                    "--terrains", *terr, "--checkpoint", checkpoint or "best_agent.pt", "--num-envs", "64"]
            # detached: the measurement outlives the request; result is read later via get_acceptance
            subprocess.Popen(args, cwd=repo_root, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True,
                             env={**os.environ, "PYTHONPATH": repo_root})
            return ActionResult(
                action="run_acceptance",
                ok=True,
                message=(f"acceptance measurement launched ({run_id}/{checkpoint} on {terr}); it self-refuses "
                         f"if training is active. The §2 verdict appears via get_acceptance when physeval "
                         f"finishes (~5 min)."),
            )
        except Exception as e:  # noqa: BLE001
            return ActionResult(action="run_acceptance", ok=False, message=f"could not launch acceptance measurement: {e}")

    async def action_run_campaign(self, run: str = "", checkpoint: str = "best_agent.pt",
                                  max_iters: int = 4) -> ActionResult:
        """Launch an AUTONOMOUS tuning campaign (measure→analyze→tune→train→re-measure, unattended,
        keeping improvements + rolling back regressions) as a detached local subprocess that drives
        the box. This is the system completing a tuning task on its own. Long-running (hours); progress
        is in the campaign log + get_acceptance. Refuses obviously-bad inputs; the campaign manages
        training itself (stall-recovery built in)."""
        import os
        import re
        import subprocess
        import sys
        from pathlib import Path

        try:
            if checkpoint and not re.fullmatch(r"[A-Za-z0-9._\-]+", checkpoint):
                return ActionResult(action="run_campaign", ok=False, message=f"invalid checkpoint: {checkpoint!r}")
            if not (run and re.fullmatch(r"[A-Za-z0-9._\-]+", run)):
                return ActionResult(action="run_campaign", ok=False, message="a valid run id is required")
            iters = max(1, min(int(max_iters), 12))
            repo_root = str(Path(__file__).resolve().parents[2])
            args = [sys.executable, "-m", "autotuner.training.tune_orchestrator", run, checkpoint,
                    "--max-iters", str(iters), "--out", f"/tmp/campaign_{run}.json"]
            subprocess.Popen(args, cwd=repo_root, stdout=open(f"/tmp/campaign_{run}.log", "w"),
                             stderr=subprocess.STDOUT, start_new_session=True,
                             env={**os.environ, "PYTHONPATH": repo_root})
            return ActionResult(
                action="run_campaign", ok=True,
                message=(f"autonomous tuning campaign launched ({run}/{checkpoint}, up to {iters} iters). "
                         f"It measures, proposes+applies tuning, trains with stall-recovery, re-measures, "
                         f"and keeps only improvements. Watch /tmp/campaign_{run}.log; verdicts via "
                         f"get_acceptance."),
            )
        except Exception as e:  # noqa: BLE001
            return ActionResult(action="run_campaign", ok=False, message=f"could not launch campaign: {e}")

    async def action_produce_policy(self, max_iters: int = 8) -> ActionResult:
        """THE PRODUCT ACTION: end-to-end produce the best benchmark policy — bootstrap from the
        best-known checkpoint, then autonomously loop full-battery measure -> analyze -> tune ->
        train -> re-measure until benchmark passes / levers exhausted / budget ends, and emit the
        deliverable report. Detached; hours-long; progress in /tmp/produce_policy.log."""
        import os
        import subprocess
        import sys
        from pathlib import Path

        try:
            iters = max(1, min(int(max_iters), 16))
            repo_root = str(Path(__file__).resolve().parents[2])
            args = [sys.executable, "-m", "autotuner.training.tune_orchestrator", "auto", "auto",
                    "--produce", "--max-iters", str(iters), "--steps-per-iter", "18000",
                    "--num-envs", "1024", "--out", "/tmp/policy_report.json"]
            subprocess.Popen(args, cwd=repo_root, stdout=open("/tmp/produce_policy.log", "w"),
                             stderr=subprocess.STDOUT, start_new_session=True,
                             env={**os.environ, "PYTHONPATH": repo_root})
            return ActionResult(
                action="produce_policy", ok=True,
                message=(f"end-to-end policy production launched (up to {iters} iterations, full "
                         f"battery flat+4 terrains+push). It bootstraps from the best-known "
                         f"checkpoint and stops when the benchmark passes or levers are exhausted. "
                         f"Progress: /tmp/produce_policy.log; deliverable: /tmp/policy_report.json."))
        except Exception as e:  # noqa: BLE001
            return ActionResult(action="produce_policy", ok=False, message=f"could not launch: {e}")

    async def action_edit_config(self, key: str = "", value: str = "") -> ActionResult:
        try:
            remote = self._get_remote()
            res = await asyncio.to_thread(self._edit_config_value, remote, key, value)
            self._remember_remote_success()
            if not res.get("ok"):
                return ActionResult(action="edit_config", ok=False, message=res.get("detail", "failed"))
            return ActionResult(action="edit_config", ok=True,
                                message=f"updated with backup available for rollback: {res['old_line']}  ->  {res['new_line']}")
        except Exception as e:  # noqa: BLE001
            message = self._remember_remote_failure(e)
            return ActionResult(action="edit_config", ok=False, message=f"edit_config failed: {self._remote_unavailable_message(message)}")

    async def action_rollback_config(self) -> ActionResult:
        try:
            remote = self._get_remote()
            ok = await asyncio.to_thread(self._rollback_config, remote)
            self._remember_remote_success()
            return ActionResult(action="rollback_config", ok=ok,
                                message="rolled back env_cfg from the latest backup" if ok else "no rollback backup is available")
        except Exception as e:  # noqa: BLE001
            message = self._remember_remote_failure(e)
            return ActionResult(action="rollback_config", ok=False, message=f"rollback failed: {self._remote_unavailable_message(message)}")


def make_source(settings: Optional[LocomotionConsoleSettings] = None) -> RunDataSource:
    from .config import get_settings
    s = settings or get_settings()
    return RealDataSource(s) if s.source == "real" else FakeDataSource(s)
