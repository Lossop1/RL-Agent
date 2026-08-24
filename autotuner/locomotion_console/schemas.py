"""Pydantic API models for the locomotion console.

These are the wire contract between the FastAPI backend and the React frontend. The live
metric point intentionally mirrors the fields of `autotuner.tuning.events.SmokeProgressEvent`
so the real source can map a parse_log snapshot straight onto it.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RunStatus(BaseModel):
    """Current run as the locomotion console understands it."""
    run_id: str
    task: str
    robot_id: str
    latest_iter: int
    total_iter: int
    running: bool
    runtime_state: Literal["live", "stale", "stopped", "interrupted", "remote_unavailable", "unknown"] = "unknown"
    source: Literal["fake", "real"]
    note: str = ""
    stale: bool = False
    telemetry_age_s: Optional[float] = None
    tmux_session: str = ""
    remote_ok: bool = True
    degraded: bool = False
    error: str = ""


class RemoteGPUInfo(BaseModel):
    index: int = 0
    name: str = ""
    memory_used_mb: Optional[float] = None
    memory_total_mb: Optional[float] = None
    utilization_gpu_pct: Optional[float] = None
    utilization_memory_pct: Optional[float] = None
    temperature_c: Optional[float] = None
    power_w: Optional[float] = None
    processes: List[dict[str, Any]] = Field(default_factory=list)


class RemoteDiskInfo(BaseModel):
    mount: str = ""
    size: str = ""
    used: str = ""
    avail: str = ""
    use_pct: str = ""


class RemoteTmuxSessionInfo(BaseModel):
    name: str
    windows: int = 0
    created: str = ""
    attached: bool = False


class RemoteMachineStatus(BaseModel):
    available: bool = False
    source: Literal["fake", "real"]
    generated_at: float = 0.0
    host: str = ""
    uptime: str = ""
    load_avg: str = ""
    cpu_count: Optional[int] = None
    memory_total_mb: Optional[float] = None
    memory_used_mb: Optional[float] = None
    memory_available_mb: Optional[float] = None
    swap_used_mb: Optional[float] = None
    swap_total_mb: Optional[float] = None
    gpus: List[RemoteGPUInfo] = Field(default_factory=list)
    disks: List[RemoteDiskInfo] = Field(default_factory=list)
    tmux_sessions: List[RemoteTmuxSessionInfo] = Field(default_factory=list)
    training_processes: List[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class MetricPoint(BaseModel):
    """One streamed sample. ts is local wall-clock seconds (frontend x-axis fallback)."""
    iter: int
    ts: float
    reward: float
    ep_len: float
    terrain: float = 0.0
    phase: int = 0
    # Catch-all for extra parsed channels without a schema change.
    extra: dict = Field(default_factory=dict)


class TensorboardTagInfo(BaseModel):
    tag: str
    group: str = "Other"
    display_name: str = ""
    points: int = 0
    first_step: Optional[int] = None
    last_step: Optional[int] = None
    last_value: Optional[float] = None
    source: str = ""
    formula: str = ""
    description: str = ""


class TensorboardScalarCatalog(BaseModel):
    available: bool = False
    source: Literal["fake", "real"]
    run: str = ""
    event_file: str = ""
    reader: str = "tbparse.SummaryReader(...).scalars"
    reward_tag: str = ""
    tags: List[TensorboardTagInfo] = Field(default_factory=list)
    refreshed_at: float = 0.0
    error: str = ""


class TensorboardScalarPoint(BaseModel):
    step: int
    value: float


class TensorboardScalarSeries(BaseModel):
    tag: str
    group: str = "Other"
    display_name: str = ""
    source: str = ""
    formula: str = ""
    points: List[TensorboardScalarPoint] = Field(default_factory=list)


class TensorboardSeriesResponse(BaseModel):
    available: bool = False
    source: Literal["fake", "real"]
    run: str = ""
    event_file: str = ""
    series: List[TensorboardScalarSeries] = Field(default_factory=list)
    error: str = ""


class TrainingTelemetryPoint(BaseModel):
    step: int = 0
    total_steps: Optional[int] = None
    ts: float = 0.0
    elapsed: str = ""
    eta: str = ""
    fps: Optional[float] = None
    reward: dict[str, float] = Field(default_factory=dict)
    curriculum: dict[str, Any] = Field(default_factory=dict)
    health: dict[str, float] = Field(default_factory=dict)
    command: dict[str, float] = Field(default_factory=dict)
    counters: dict[str, float] = Field(default_factory=dict)
    paths: dict[str, str] = Field(default_factory=dict)
    checkpoint: str = ""
    raw: str = ""


class DataProvenanceItem(BaseModel):
    key: str
    label: str
    path: str = ""
    role: str = ""
    available: bool = False
    evidence: List[str] = Field(default_factory=list)


class MetricSnapshotItem(BaseModel):
    key: str
    label: str
    value: Any = None
    numeric_value: Optional[float] = None
    unit: str = ""
    precision: int = 3
    tone: Literal["good", "warn", "bad", "neutral"] = "neutral"
    direction: Literal["higher", "lower", "target", "info"] = "info"
    target: Optional[float] = None
    target_label: str = ""
    section: str = ""
    source: str = ""
    formula: str = ""
    meaning: str = ""
    reading: str = ""


class MetricSnapshotGroup(BaseModel):
    id: str
    title: str
    summary: str = ""
    items: List[MetricSnapshotItem] = Field(default_factory=list)


class SnapshotBlocker(BaseModel):
    key: str
    label: str = ""
    value: Optional[float] = None
    target: Optional[float] = None
    direction: Literal["higher", "lower", "target", "info"] = "info"
    gap: Optional[float] = None
    severity: Literal["info", "warn", "bad"] = "info"
    source: str = ""
    detail: str = ""


class RunSnapshot(BaseModel):
    run_id: str = ""
    source: Literal["fake", "real"] = "real"
    mode: Literal["jsonl", "log_fallback", "fake", "missing", "error"] = "missing"
    running: bool = False
    runtime_state: Literal["live", "stale", "stopped", "interrupted", "remote_unavailable", "unknown"] = "unknown"
    stale: bool = False
    telemetry_age_s: Optional[float] = None
    status: Literal["ok", "watch", "blocked", "missing", "error"] = "missing"
    conclusion: str = ""
    generated_at: float = 0.0
    step: int = 0
    total_steps: Optional[int] = None
    progress_pct: Optional[float] = None
    elapsed: str = ""
    eta: str = ""
    fps: Optional[float] = None
    phase: str = ""
    command_mode: str = ""
    active_dirs: str = ""
    blocked_by: str = ""
    next_gate: str = ""
    latest_checkpoint: str = ""
    provenance: List[DataProvenanceItem] = Field(default_factory=list)
    metric_groups: List[MetricSnapshotGroup] = Field(default_factory=list)
    blockers: List[SnapshotBlocker] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class TrainingTelemetry(BaseModel):
    available: bool = False
    source: Literal["fake", "real"]
    run_id: str = ""
    running: bool = False
    runtime_state: Literal["live", "stale", "stopped", "interrupted", "remote_unavailable", "unknown"] = "unknown"
    stale: bool = False
    telemetry_age_s: Optional[float] = None
    log_path: str = ""
    telemetry_path: str = ""
    # 本次 run 的 effective_config 原文（当前奖励权重/门控阈值的来源）。exclude=True:
    # 服务端工具可取,但不进 API 响应/model_dump,避免把整段 YAML 塞进前端负载。
    effective_config_text: str = Field(default="", exclude=True)
    mode: Literal["jsonl", "log_fallback", "fake", "missing", "error"] = "missing"
    source_stats: dict[str, Any] = Field(default_factory=dict)
    latest: Optional[TrainingTelemetryPoint] = None
    history: List[TrainingTelemetryPoint] = Field(default_factory=list)
    snapshot: Optional[RunSnapshot] = None
    scoreboard: Optional["Scoreboard"] = None
    definitions: List["DefinitionInfo"] = Field(default_factory=list)
    summary: str = ""
    limitations: List[str] = Field(default_factory=list)
    error: str = ""


class DefinitionInfo(BaseModel):
    key: str
    label: str
    category: Literal["reward", "curriculum", "terrain", "diagnostic", "system"] = "system"
    formula: str = ""
    meaning: str = ""
    current_reading: str = ""
    source: str = ""
    related: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)


class DefinitionQueryResult(BaseModel):
    query: str = ""
    items: List[DefinitionInfo] = Field(default_factory=list)


class ActionResult(BaseModel):
    action: str
    ok: bool
    message: str


class SpecCoverageItem(BaseModel):
    id: str
    title: str
    group: str
    training_mechanism: str = ""
    evaluation: str = ""
    status: Literal["complete", "partial", "missing"] = "missing"
    gaps: List[str] = Field(default_factory=list)
    next_actions: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)


class SpecCoverageReport(BaseModel):
    spec_path: str = ""
    generated_at: float = 0.0
    summary: str = ""
    legend: dict[str, str] = Field(default_factory=dict)
    items: List[SpecCoverageItem] = Field(default_factory=list)


class CommandEval(BaseModel):
    name: str
    vx: float
    vy: float
    wz: float
    tracked: bool
    foot_clearance_cm: float
    duty: float
    base_h_m: float


class PhysevalResult(BaseModel):
    ok: bool
    summary: str
    commands: List[CommandEval] = Field(default_factory=list)
    checkpoint: Optional[str] = None


class ContextRequestInfo(BaseModel):
    page: str = ""
    intent: str = ""
    focus: List[str] = Field(default_factory=list)
    visible: dict[str, Any] = Field(default_factory=dict)
    max_chars: int = 3500
    include_remote_status: bool = True


class ContextEnvelopeInfo(BaseModel):
    page: str = ""
    intent: str = ""
    budget_chars: int = 3500
    injected_chars: int = 0
    selected_sources: List[str] = Field(default_factory=list)
    required_sources: List[str] = Field(default_factory=list)
    route: dict[str, Any] = Field(default_factory=dict)
    remote_summary: dict[str, Any] = Field(default_factory=dict)
    prompt: str = ""


class ChatRequest(BaseModel):
    message: str
    ui_mode: str = ""
    context: str = ""
    context_request: Optional[ContextRequestInfo] = None


class ChatResponse(BaseModel):
    reply: str
    steps: int = 0
    mode: str = ""
    elapsed_s: float = 0.0
    suggestions: List[str] = Field(default_factory=list)
    # tool calls the soul made to ground its answer (shown as evidence)
    transcript: List[dict] = Field(default_factory=list)
    # set when the soul proposes an action needing operator confirmation
    proposed_action: Optional[dict] = None
    # grounding self-check: {checked, grounded, unsupported[]}
    grounding: Optional[dict] = None
    proposal_id: Optional[str] = None
    context_envelope: Optional[ContextEnvelopeInfo] = None


class ChatProposalInfo(BaseModel):
    id: str
    created_at: str
    updated_at: str
    status: Literal["pending", "executed", "failed", "cancelled"]
    name: str
    args: dict = Field(default_factory=dict)
    reply: str = ""
    result: str = ""
    kind: str = "action"
    risk: Literal["low", "medium", "high"] = "medium"
    evidence: List[str] = Field(default_factory=list)
    requires: List[str] = Field(default_factory=list)
    expected_result: str = ""
    rollback: str = ""


class ChatProposalHistory(BaseModel):
    items: List[ChatProposalInfo] = Field(default_factory=list)


class ExecuteRequest(BaseModel):
    name: str
    args: dict = Field(default_factory=dict)


class CancelProposalRequest(BaseModel):
    name: str


class ExecuteResponse(BaseModel):
    ok: bool
    detail: str


class LLMReadinessInfo(BaseModel):
    configured: bool
    api_key_resolved: bool
    unsafe_literal_key: bool
    model: str
    base_url: str
    audit_count: int = 0
    tools: List[str] = Field(default_factory=list)
    workflow_count: int = 0
    ok: bool = False
    issues: List[str] = Field(default_factory=list)


class AgentWorkbenchAttemptInfo(BaseModel):
    run_id: str = ""
    running: bool = False
    runtime_state: Literal["live", "stale", "stopped", "interrupted", "remote_unavailable", "unknown"] = "unknown"
    source: Literal["fake", "real"] = "real"
    step: int = 0
    total_steps: Optional[int] = None
    progress_pct: Optional[float] = None
    phase: str = ""
    command_mode: str = ""
    blocked_by: str = ""
    latest_checkpoint: str = ""
    telemetry_age_s: Optional[float] = None
    remote_ok: bool = True
    summary: str = ""


class AgentWorkbenchEvidenceInfo(BaseModel):
    id: str
    label: str
    status: Literal["ok", "warning", "error", "missing", "unknown"] = "unknown"
    detail: str = ""
    source: str = ""
    values: dict[str, Any] = Field(default_factory=dict)
    provenance: List[DataProvenanceItem] = Field(default_factory=list)


class AgentWorkbenchJudgementInfo(BaseModel):
    status: Literal["ready", "running", "blocked", "needs_evidence", "remote_unavailable", "error"] = "needs_evidence"
    title: str = ""
    summary: str = ""
    confidence: Literal["high", "medium", "low"] = "low"
    evidence_ids: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)


class AgentWorkbenchActionInfo(BaseModel):
    id: str
    label: str
    kind: Literal["ask_agent", "diagnostic", "training", "deployment", "safety"] = "ask_agent"
    risk: Literal["low", "medium", "high"] = "low"
    enabled: bool = True
    requires_confirmation: bool = True
    endpoint: str = ""
    method: str = "POST"
    reason: str = ""
    rollback: str = ""


class AgentWorkbenchInfo(BaseModel):
    generated_at: float
    objective: str
    service_loop: List[str] = Field(default_factory=list)
    attempt: AgentWorkbenchAttemptInfo
    llm: LLMReadinessInfo
    judgement: AgentWorkbenchJudgementInfo
    evidence: List[AgentWorkbenchEvidenceInfo] = Field(default_factory=list)
    proposals: List[ChatProposalInfo] = Field(default_factory=list)
    actions: List[AgentWorkbenchActionInfo] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class LLMEvidenceItem(BaseModel):
    id: str
    label: str
    status: Literal["ok", "warning", "error", "unknown"] = "unknown"
    detail: str = ""
    data: dict = Field(default_factory=dict)


class LLMWorkflowRequest(BaseModel):
    job_id: Optional[str] = None
    checkpoint: Optional[str] = None
    include_llm: bool = True


class LLMWorkflowResult(BaseModel):
    workflow: str
    ok: bool
    generated_by: Literal["deterministic", "llm", "deterministic+llm"]
    summary: str
    evidence: List[LLMEvidenceItem] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list)
    proposals: List[ChatProposalInfo] = Field(default_factory=list)
    transcript: List[dict] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)


class DiagnosticPreset(BaseModel):
    id: str
    label: str
    description: str
    category: Literal["quick", "direction", "environment", "robustness"]
    estimated_minutes: int


class DiagnosticCheckpoint(BaseModel):
    path: str
    name: str
    run_name: str
    iteration: Optional[int] = None
    kind: Literal["latest", "iteration", "best", "unknown"] = "unknown"
    is_default: bool = False
    note: str = ""
    framework_id: str = ""
    framework_label: str = ""
    framework_status: str = ""
    diagnostic_task: str = ""


class DiagnosticCatalog(BaseModel):
    presets: List[DiagnosticPreset]
    plan_templates: dict[str, dict[str, Any]] = Field(default_factory=dict)
    checkpoints: List[DiagnosticCheckpoint] = Field(default_factory=list)
    checkpoint: Optional[str] = None
    checkpoint_name: Optional[str] = None
    training_running: bool = False
    source: Literal["fake", "real"]
    framework_id: str = ""
    framework_label: str = ""
    framework_status: str = ""
    framework_note: str = ""
    available: bool = True
    message: str = ""


class DiagnosticCommandSpec(BaseModel):
    id: str = ""
    label: str = ""
    mode: str = "stand"
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    duration_s: float = 3.0
    settle_s: float = 0.0
    ramp_s: float = 0.0
    repeats: int = 1


class DiagnosticTerrainSpec(BaseModel):
    type: str = "flat"
    level: int = 0
    params: dict[str, Any] = Field(default_factory=dict)


class DiagnosticDRCaseSpec(BaseModel):
    level: int = 0
    population: str = "forced"
    stress_profile: str = ""
    stress_factor: str = ""
    friction: Optional[float] = None
    mass_scale: Optional[float] = None
    stiffness_scale: Optional[float] = None
    damping_scale: Optional[float] = None
    latency_steps: Optional[int] = None
    label: str = ""


class DiagnosticPushSpec(BaseModel):
    enabled: bool = False
    segment: int = 0
    time_s: float = 1.0
    vector: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class DiagnosticRecordingSpec(BaseModel):
    sample_hz: float = 50.0
    max_rows: int = 5000
    save_playback: bool = True


class DiagnosticCriteriaSpec(BaseModel):
    tracking_error_max: Optional[float] = 0.20
    min_height_m: Optional[float] = 0.47
    stable_fraction_min: Optional[float] = 0.90
    slip_max: Optional[float] = 0.20
    hard_impact_max: Optional[int] = 0


class DiagnosticPlan(BaseModel):
    name: str = ""
    template: str = ""
    init_phase: int = 0
    num_envs: int = 1
    reset_policy: str = "per_case"
    reset_initialization: str = "command_start"
    commands: List[DiagnosticCommandSpec] = Field(default_factory=list)
    terrains: List[DiagnosticTerrainSpec] = Field(default_factory=list)
    dr_cases: List[DiagnosticDRCaseSpec] = Field(default_factory=list)
    pushes: List[DiagnosticPushSpec] = Field(default_factory=list)
    recording: DiagnosticRecordingSpec = Field(default_factory=DiagnosticRecordingSpec)
    criteria: DiagnosticCriteriaSpec = Field(default_factory=DiagnosticCriteriaSpec)
    notes: List[str] = Field(default_factory=list)


class DiagnosticRunRequest(BaseModel):
    preset: str
    checkpoint: Optional[str] = None
    plan: Optional[DiagnosticPlan] = None


class FrameworkProfileInfo(BaseModel):
    id: str
    label: str
    status: str
    experiment: str
    task_id: str
    diagnostic_task: str
    note: str = ""
    active: bool = False
    desired: bool = False


class FrameworkSelectionRequest(BaseModel):
    framework_id: str


class FrameworkSelectionResult(BaseModel):
    active_framework_id: str
    desired_framework_id: str
    requires_restart: bool
    saved_path: str
    message: str


class RemoteProfileInfo(BaseModel):
    host: str
    port: int
    user: str
    work_dir: str
    conda_env: str
    diagnostic_tool_root: str
    diagnostic_output_root: str
    diagnostic_robot_root: str
    diagnostic_python: str
    log_format: str
    password_configured: bool
    source: str


class RemoteProfileUpdate(BaseModel):
    host: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    password: Optional[str] = None
    work_dir: Optional[str] = None
    conda_env: Optional[str] = None
    diagnostic_tool_root: Optional[str] = None
    diagnostic_output_root: Optional[str] = None
    diagnostic_robot_root: Optional[str] = None
    diagnostic_python: Optional[str] = None
    log_format: Optional[str] = None


class RemoteProfileUpdateResult(BaseModel):
    profile: RemoteProfileInfo
    saved_path: str
    requires_restart: bool
    message: str


class RemoteConnectionTestInfo(BaseModel):
    ok: bool
    host: str
    port: int
    user: str
    work_dir: str
    latency_ms: int
    detail: str
    pwd: str = ""


class RobotProfileInfo(BaseModel):
    id: str
    label: str
    status: str
    dof: int
    base_link: str
    joint_order: List[str] = Field(default_factory=list)
    leg_order: List[str] = Field(default_factory=list)
    foot_links: List[str] = Field(default_factory=list)
    diagnostic_spec: str
    capabilities: List[str] = Field(default_factory=list)
    note: str = ""
    valid: bool = True
    issues: List[str] = Field(default_factory=list)


class LLMProfileInfo(BaseModel):
    configured: bool
    model: str
    base_url: str
    api_key_env_var: str
    api_key_resolved: bool
    unsafe_literal_key: bool
    advisory_only: bool = True
    actions_require_confirmation: bool = True


class LLMProfileUpdate(BaseModel):
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env_var: Optional[str] = None


class LLMProfileUpdateResult(BaseModel):
    profile: LLMProfileInfo
    saved_path: str
    message: str


class LLMAuditItemInfo(BaseModel):
    file: str
    schema_name: str
    model: str
    elapsed_s: float
    attempt: int
    ok: bool
    error: str = ""
    created_at: str


class LLMAuditHistory(BaseModel):
    items: List[LLMAuditItemInfo] = Field(default_factory=list)


class ProfileSummaryInfo(BaseModel):
    id: str
    label: str
    status: str
    detail: str = ""


class ConfigSetInfo(BaseModel):
    id: str
    label: str
    status: str
    task_goal: str
    remote: ProfileSummaryInfo
    robot: ProfileSummaryInfo
    framework: ProfileSummaryInfo
    llm: ProfileSummaryInfo
    notes: List[str] = Field(default_factory=list)
    contract: Optional[ProfileSummaryInfo] = None


class DiagnosticStageStatus(BaseModel):
    id: str
    label: str
    state: Literal["pending", "running", "complete", "error", "cancelled"]
    progress: float = 0.0
    rows_written: int = 0


class DiagnosticJobStatus(BaseModel):
    state: Literal["idle", "starting", "running", "complete", "error", "cancelled"]
    job_id: Optional[str] = None
    preset: Optional[str] = None
    preset_label: Optional[str] = None
    checkpoint: Optional[str] = None
    framework_id: Optional[str] = None
    output_dir: Optional[str] = None
    progress: float = 0.0
    elapsed_s: float = 0.0
    message: str = ""
    stages: List[DiagnosticStageStatus] = Field(default_factory=list)
    log_tail: List[str] = Field(default_factory=list)
    plan_summary: dict[str, Any] = Field(default_factory=dict)


class ArtifactRefInfo(BaseModel):
    kind: str
    label: str
    uri: str = ""
    available: bool = False
    note: str = ""


class DiagnosticLogInfo(BaseModel):
    path: str = ""
    available: bool = False
    complete: bool = False
    truncated: bool = False
    bytes: int = 0
    lines: int = 0
    shown_lines: int = 0
    tail: List[str] = Field(default_factory=list)
    error_excerpt: List[str] = Field(default_factory=list)
    note: str = ""


class DiagnosticValueInfo(BaseModel):
    key: str
    label: str
    group: str = "summary"
    value: Any = None
    unit: str = ""
    source: str = ""
    fields: List[str] = Field(default_factory=list)
    formula: str = ""
    interpretation: str = ""
    confidence: Literal["high", "medium", "low"] = "high"


class DiagnosticHistoryItem(BaseModel):
    job_id: str
    state: Literal["idle", "starting", "running", "complete", "error", "cancelled"]
    preset: str = ""
    preset_label: str = ""
    checkpoint: str = ""
    checkpoint_name: str = ""
    framework_id: str = ""
    diagnostic_task: str = ""
    config_set_id: str = ""
    source: Literal["fake", "real"] = "fake"
    output_dir: str = ""
    progress: float = 0.0
    message: str = ""
    created_at: str = ""
    updated_at: str = ""
    elapsed_s: float = 0.0
    artifacts: List[ArtifactRefInfo] = Field(default_factory=list)
    note: str = ""
    archived: bool = False


class DiagnosticHistory(BaseModel):
    items: List[DiagnosticHistoryItem] = Field(default_factory=list)


class DiagnosticHistoryUpdate(BaseModel):
    note: Optional[str] = None
    archived: Optional[bool] = None


class DiagnosticHistoryDeleteResult(BaseModel):
    ok: bool
    job_id: str
    message: str


class DiagnosticReport(BaseModel):
    job_id: str
    preset: str
    preset_label: str
    checkpoint: str
    output_dir: str
    artifacts: List[ArtifactRefInfo] = Field(default_factory=list)
    log: DiagnosticLogInfo = Field(default_factory=DiagnosticLogInfo)
    value_explanations: List[DiagnosticValueInfo] = Field(default_factory=list)
    coverage: dict[str, Any] = Field(default_factory=dict)
    commands: List[dict[str, Any]] = Field(default_factory=list)
    posture: dict[str, Any] = Field(default_factory=dict)
    events: List[dict[str, Any]] = Field(default_factory=list)
    notes: List[dict[str, Any]] = Field(default_factory=list)
    terrain: dict[str, Any] = Field(default_factory=dict)
    robustness: dict[str, Any] = Field(default_factory=dict)
    plan: dict[str, Any] = Field(default_factory=dict)


class DiagnosticPlaybackFoot(BaseModel):
    position: list[float]
    contact: bool = False
    force_norm: Optional[float] = None
    force_w_x: Optional[float] = None
    force_w_y: Optional[float] = None
    force_w_z: Optional[float] = None
    normal_force: Optional[float] = None
    tangent_force: Optional[float] = None
    clearance: Optional[float] = None


class DiagnosticPlaybackFrame(BaseModel):
    t: float
    stage: str
    case_id: int
    env_id: int = 0
    segment_id: int
    command_mode: str
    base_position: list[float]
    base_quaternion_wxyz: list[float]
    joints: list[float]
    feet: dict[str, DiagnosticPlaybackFoot] = Field(default_factory=dict)
    reset_observed: bool = False
    done: bool = False
    terrain_height: Optional[float] = None
    terrain: str = ""                       # terrain type for this frame's case (record.csv terrain_type)
    terrain_level: Optional[int] = None


class DiagnosticPlaybackPrimitive(BaseModel):
    """Simulator-independent terrain primitive used by browser playback."""

    id: str = ""
    type: str = "box"
    center: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    size: list[float] = Field(default_factory=lambda: [1.0, 1.0, 0.1])
    color: str = ""


class DiagnosticPlaybackRobot(BaseModel):
    """Robot asset and naming contract consumed by the generic viewer."""

    id: str = ""
    label: str = ""
    urdf_url: str = ""
    base_link: str = ""
    joint_order: list[str] = Field(default_factory=list)
    leg_order: list[str] = Field(default_factory=list)


class DiagnosticPlaybackScene(BaseModel):
    """Optional scene geometry; the viewer does not know product or simulator names."""

    id: str = ""
    label: str = ""
    terrain_primitives: list[DiagnosticPlaybackPrimitive] = Field(default_factory=list)


class DiagnosticPlayback(BaseModel):
    available: bool
    message: str = ""
    source: Literal["fake", "real", "local"]
    output_dir: Optional[str] = None
    manifest_path: Optional[str] = None
    result_path: Optional[str] = None
    fps: float = 50.0
    joint_order: list[str] = Field(default_factory=list)
    leg_order: list[str] = Field(default_factory=list)
    robot: DiagnosticPlaybackRobot = Field(default_factory=DiagnosticPlaybackRobot)
    scene: DiagnosticPlaybackScene = Field(default_factory=DiagnosticPlaybackScene)
    frames: list[DiagnosticPlaybackFrame] = Field(default_factory=list)
    source_rows: int = 0
    stride: int = 1
    selected_env_id: Optional[int] = None
    available_env_ids: List[int] = Field(default_factory=list)


class TraditionalControlControllerInfo(BaseModel):
    id: str
    label: str
    description: str = ""
    provider_id: str = ""
    provider_ids: list[str] = Field(default_factory=list)


class TraditionalControlSceneInfo(BaseModel):
    id: str
    label: str
    description: str = ""
    terrain_type: str = ""
    command: list[float] = Field(default_factory=list)
    duration_s: float = 3.0
    provider_id: str = ""
    provider_ids: list[str] = Field(default_factory=list)
    controller_ids: list[str] = Field(default_factory=list)


class TraditionalControlProductInfo(BaseModel):
    id: str
    label: str
    robot: DiagnosticPlaybackRobot = Field(default_factory=DiagnosticPlaybackRobot)
    controllers: list[TraditionalControlControllerInfo] = Field(default_factory=list)
    scenes: list[TraditionalControlSceneInfo] = Field(default_factory=list)
    data_sources: list[TraditionalControlDataSourceInfo] = Field(default_factory=list)


class TraditionalControlDataSourceInfo(BaseModel):
    id: str
    label: str
    # Provider-defined capability.  The service reserves ``replay`` for its
    # local artifact loader; other kinds are delegated to the provider.
    kind: str = Field(min_length=1, max_length=80)
    available: bool = True
    description: str = ""
    replay_id: Optional[str] = None
    provider_id: str = ""


class TraditionalControlCatalog(BaseModel):
    available: bool = True
    message: str = ""
    products: list[TraditionalControlProductInfo] = Field(default_factory=list)
    data_sources: list[TraditionalControlDataSourceInfo] = Field(default_factory=list)
    recent_runs: list[dict[str, Any]] = Field(default_factory=list)


class TraditionalControlRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1, max_length=120)
    controller_id: str = Field(min_length=1, max_length=120)
    scene_id: str = Field(min_length=1, max_length=120)
    duration_s: float = Field(gt=0.0, le=600.0)
    data_source_id: str = Field(min_length=1, max_length=120)
    replay_id: Optional[str] = Field(default=None, max_length=240)


class TraditionalControlJobStatus(BaseModel):
    state: Literal["idle", "starting", "running", "complete", "error", "cancelled"]
    run_id: Optional[str] = None
    product_id: Optional[str] = None
    controller_id: Optional[str] = None
    scene_id: Optional[str] = None
    data_source_id: Optional[str] = None
    output_dir: Optional[str] = None
    manifest_path: Optional[str] = None
    result_path: Optional[str] = None
    playback_available: bool = False
    progress: float = 0.0
    elapsed_s: float = 0.0
    message: str = ""
    error: str = ""


class TraditionalControlHistory(BaseModel):
    items: list[TraditionalControlJobStatus] = Field(default_factory=list)


class TraditionalControlMetricInfo(BaseModel):
    id: str
    label: str
    value: Any = None
    unit: str = ""
    precision: int = 3


class TraditionalControlResult(BaseModel):
    available: bool = True
    verdict: Literal["passed", "failed", "unknown"] = "unknown"
    summary: str = ""
    metrics: list[TraditionalControlMetricInfo] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


# ── 目标记分牌（Objective Scoreboard）────────────────────────────────────────
# 一块屏幕，按五类目标（速度跟踪 / 步态 / 效率 / 地形 / 鲁棒性）排开。
# 每个指标三件事：现在是多少 / 底线是多少 / 这条底线哪来的。
# 只讲事实：状态只有 达标 / 未达底线 / 未定底线 / 无数据，绝不替用户下“好/坏”判断。
# 底线的出处（provenance）必须显式：来自规范、来自本次 run 配置、还是出处不明，
# 一律标清楚——绝不把“默认值”或“出处不明的写死值”冒充成本次 run 的真实验收标准。

# 底线可信度：spec=规范文档；config=本次 run 有效配置；derived=由规范量推导；
# default=配置缺失时的兜底默认；unknown=代码里写死但没标出处。
FloorConfidence = Literal["spec", "config", "derived", "default", "unknown"]
# 指标状态（纯事实，不含价值判断）。
MetricStatus = Literal["meets", "below", "no_floor", "no_data"]


class ScoreboardMetric(BaseModel):
    """记分牌上的一行：一个可测量指标的当前值、底线、底线出处与互相拆台关系。"""
    key: str
    label: str
    family: str                                  # 所属目标族 id
    value: Any = None                            # 当前值（原始）
    numeric_value: Optional[float] = None        # 数值化后的当前值（无法数值化则 None）
    unit: str = ""
    precision: int = 3
    reading: str = ""                            # 展示串，如 "0.084 m/s"
    floor: Optional[float] = None                # 底线数值（无底线则 None）
    floor_op: Literal["<=", ">=", ""] = ""       # 底线方向：<= 上界、>= 下界
    floor_label: str = ""                        # 人读底线，如 "≤0.10 m/s"
    floor_source: str = ""                       # 出处文字（带 file:line 或规范章节）
    floor_confidence: FloorConfidence = "unknown"
    status: MetricStatus = "no_data"
    # 当前值本身的可信度：地形扫描派生量（粗糙度/局部障碍高）标 low。
    value_confidence: Literal["high", "medium", "low"] = "high"
    tensions: List[str] = Field(default_factory=list)   # 与之直接拆台的其它指标 key
    meaning: str = ""                            # 一句话说明这个指标看的是什么
    note: str = ""                               # 附注：口径差异、出处存疑等


class ScoreboardFamily(BaseModel):
    """一个目标族（五类之一）及其下的指标。"""
    id: str
    title: str
    summary: str = ""
    metrics: List[ScoreboardMetric] = Field(default_factory=list)


class ScoreboardContext(BaseModel):
    """课程状态带。没有它，今天的数字和昨天的没法比：
    penalty_gate 会重标所有“晚期惩罚”的强度，地形/阶段决定了当前在考哪一关。"""
    phase: str = ""
    command_mode: str = ""
    active_dirs: str = ""
    terrain_mean: Optional[float] = None
    terrain_max: Optional[float] = None
    penalty_gate: Optional[float] = None
    dr_level: Optional[float] = None
    note: str = ""


class ScoreboardTension(BaseModel):
    """两个指标互相拆台：改一个更好，另一个通常会更差。"""
    a: str
    b: str
    reason: str = ""


class ScoreboardCoverage(BaseModel):
    """覆盖自检：如实报告记分牌的盲区，而不是假装数全了。

    unmapped_reward_terms 是“遥测里正在被优化、却还没上板”的奖励项——它让板子
    自己暴露遗漏。这是对“要求很多很杂、总有遗漏”的正面回答：把遗漏显示出来。"""
    mapped_reward_terms: List[str] = Field(default_factory=list)
    unmapped_reward_terms: List[str] = Field(default_factory=list)
    note: str = ""


class Scoreboard(BaseModel):
    """目标记分牌整体：若干目标族 + 课程状态带 + 全局拆台关系表 + 覆盖自检。

    目标族不锁死为五类：这五类不是全集，会按代码与验收规范持续补充；缺的部分由
    coverage 自检显式列出，绝不假装完整。"""
    available: bool = False
    run_id: str = ""
    generated_at: float = 0.0
    step: int = 0
    stale: bool = False
    status: Literal["ok", "watch", "blocked", "missing", "error"] = "missing"
    context: ScoreboardContext = Field(default_factory=ScoreboardContext)
    families: List[ScoreboardFamily] = Field(default_factory=list)
    tensions: List[ScoreboardTension] = Field(default_factory=list)
    coverage: ScoreboardCoverage = Field(default_factory=ScoreboardCoverage)
    notes: List[str] = Field(default_factory=list)
