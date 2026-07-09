"""Pydantic API models for the locomotion console.

These are the wire contract between the FastAPI backend and the React frontend. The live
metric point intentionally mirrors the fields of `autotuner.tuning.events.SmokeProgressEvent`
so the real source can map a parse_log snapshot straight onto it.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


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
    mode: Literal["jsonl", "log_fallback", "fake", "missing", "error"] = "missing"
    source_stats: dict[str, Any] = Field(default_factory=dict)
    latest: Optional[TrainingTelemetryPoint] = None
    history: List[TrainingTelemetryPoint] = Field(default_factory=list)
    snapshot: Optional[RunSnapshot] = None
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


class DiagnosticPlayback(BaseModel):
    available: bool
    message: str = ""
    source: Literal["fake", "real"]
    output_dir: Optional[str] = None
    fps: float = 50.0
    joint_order: list[str] = Field(default_factory=list)
    leg_order: list[str] = Field(default_factory=list)
    frames: list[DiagnosticPlaybackFrame] = Field(default_factory=list)
    source_rows: int = 0
    stride: int = 1
    selected_env_id: Optional[int] = None
    available_env_ids: List[int] = Field(default_factory=list)
