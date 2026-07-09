// REST + WS client for the locomotion console backend.
// Default to a same-origin relative base so the app works behind any host/scheme
// (localhost, reverse proxy, HTTPS). Override with VITE_API_BASE for a split origin.

export const API_BASE =
  (import.meta as any).env?.VITE_API_BASE ?? "";

// Derive the WebSocket base. When API_BASE is relative (same origin), build it
// from the current page so HTTPS pages get wss:// (avoiding mixed-content blocks).
export const WS_BASE = (() => {
  if (API_BASE) return API_BASE.replace(/^http/, "ws");
  if (typeof window !== "undefined" && window.location) {
    const { protocol, host } = window.location;
    return `${protocol === "https:" ? "wss" : "ws"}://${host}`;
  }
  return "";
})();

// Runtime console token: a browser-set localStorage value wins, else the build-time
// env fallback. Its mere presence on a mutating request defeats CSRF; when the server
// has LOCOMOTION_CONSOLE_TOKEN set, the value must match.
function consoleToken(): string {
  try {
    return (typeof localStorage !== "undefined" && localStorage.getItem("LOCOMOTION_CONSOLE_TOKEN")) || (import.meta as any).env?.VITE_CONSOLE_TOKEN || "";
  } catch {
    return (import.meta as any).env?.VITE_CONSOLE_TOKEN || "";
  }
}

export function authHeaders(): Record<string, string> {
  return { "X-Console-Token": consoleToken() };
}

export interface RunStatus {
  run_id: string;
  task: string;
  robot_id: string;
  latest_iter: number;
  total_iter: number;
  running: boolean;
  runtime_state: "live" | "stale" | "stopped" | "interrupted" | "remote_unavailable" | "unknown";
  source: "fake" | "real";
  note: string;
  stale: boolean;
  telemetry_age_s: number | null;
  tmux_session: string;
  remote_ok: boolean;
  degraded: boolean;
  error: string;
}

export interface RemoteGPUInfo {
  index: number;
  name: string;
  memory_used_mb: number | null;
  memory_total_mb: number | null;
  utilization_gpu_pct: number | null;
  utilization_memory_pct: number | null;
  temperature_c: number | null;
  power_w: number | null;
  processes: Record<string, any>[];
}

export interface RemoteDiskInfo {
  mount: string;
  size: string;
  used: string;
  avail: string;
  use_pct: string;
}

export interface RemoteTmuxSessionInfo {
  name: string;
  windows: number;
  created: string;
  attached: boolean;
}

export interface RemoteMachineStatus {
  available: boolean;
  source: "fake" | "real";
  generated_at: number;
  host: string;
  uptime: string;
  load_avg: string;
  cpu_count: number | null;
  memory_total_mb: number | null;
  memory_used_mb: number | null;
  memory_available_mb: number | null;
  swap_used_mb: number | null;
  swap_total_mb: number | null;
  gpus: RemoteGPUInfo[];
  disks: RemoteDiskInfo[];
  tmux_sessions: RemoteTmuxSessionInfo[];
  training_processes: Record<string, any>[];
  raw: Record<string, any>;
  error: string;
}

export interface ContextRequestInfo {
  page: string;
  intent: string;
  focus: string[];
  visible: Record<string, any>;
  max_chars: number;
  include_remote_status: boolean;
}

export interface ContextEnvelopeInfo {
  page: string;
  intent: string;
  budget_chars: number;
  injected_chars: number;
  selected_sources: string[];
  required_sources: string[];
  route: Record<string, any>;
  remote_summary: Record<string, any>;
  prompt: string;
}

export interface ProfileSummary {
  id: string;
  label: string;
  status: string;
  detail: string;
}

export interface ConfigSetInfo {
  id: string;
  label: string;
  status: string;
  task_goal: string;
  remote: ProfileSummary;
  robot: ProfileSummary;
  framework: ProfileSummary;
  llm: ProfileSummary;
  notes: string[];
}

export interface FrameworkProfileInfo {
  id: string;
  label: string;
  status: string;
  experiment: string;
  task_id: string;
  diagnostic_task: string;
  note: string;
  active: boolean;
  desired: boolean;
}

export interface FrameworkSelectionResult {
  active_framework_id: string;
  desired_framework_id: string;
  requires_restart: boolean;
  saved_path: string;
  message: string;
}

// Framework Library ② — mechanism components (M1-M9 + blind-TP) and named compositions.
export interface FrameworkComponentInfo {
  id: string;
  label: string;
  role: string;
  adapt_kind: string;
  applies_to: string[];
  module: string;
  version: string;
  requires: string[];
  note: string;
}

export interface FrameworkCompositionInfo {
  id: string;
  label: string;
  status: string;
  component_ids: string[];
  adapt_plan: Record<string, string[]>;
  note: string;
}

export interface FrameworkCatalogInfo {
  components: FrameworkComponentInfo[];
  compositions: FrameworkCompositionInfo[];
}

// Adapter chain (③④⑦) dry-run preview for a robot: derived values + consistency + readiness.
export interface DeployItemInfo {
  remote: string;
  role: string;
  size: number;
}

export interface AdaptationPreviewInfo {
  robot: string;
  available: boolean;
  message: string;
  composition_id: string;
  composition_valid: boolean;
  effort: Record<string, number>;
  kp: Record<string, number>;
  n_joints: number;
  mass_kg: number;
  base_h_target: number | null;
  roundtrip_ok: boolean;
  roundtrip_bad: string[];
  consistency_verdict: string;
  n_agree: number;
  n_corrected: number;
  n_flagged: number;
  sim2real_hazards_fixed: string[];
  ready: boolean;
  blockers: string[];
  plan_items: DeployItemInfo[];
}

// Robot import (§3 C front door): is a URDF adaptable by the framework?
export interface KnownUrdfsInfo {
  urdfs: Record<string, string>;
}

export interface RobotImportInfo {
  urdf: string;
  available: boolean;
  message: string;
  adaptable: boolean;
  n_joints: number;
  mass_kg: number;
  leg_length_m: number;
  roles: string[];
  issues: string[];
  derived_preview: Record<string, unknown>;
}

// ⑥ NL-edit guard: validate a proposed framework-content edit against the robot's derived safe band.
export interface EditableKnobsInfo {
  knobs: Record<string, string>;
}

export interface EditValidationInfo {
  field: string;
  available: boolean;
  message: string;
  target_field: string;
  old: number;
  new: number;
  safe_range: [number, number];
  accepted: boolean;
  risk: string;
  rationale: string;
}

export interface LLMProfileInfo {
  configured: boolean;
  model: string;
  base_url: string;
  api_key_env_var: string;
  api_key_resolved: boolean;
  unsafe_literal_key: boolean;
  advisory_only: boolean;
  actions_require_confirmation: boolean;
}

export interface RemoteProfileInfo {
  host: string;
  port: number;
  user: string;
  work_dir: string;
  conda_env: string;
  diagnostic_tool_root: string;
  diagnostic_output_root: string;
  diagnostic_robot_root: string;
  diagnostic_python: string;
  log_format: string;
  password_configured: boolean;
  source: string;
}

export interface RemoteProfileUpdateResult {
  profile: RemoteProfileInfo;
  saved_path: string;
  requires_restart: boolean;
  message: string;
}

export interface RemoteConnectionTestInfo {
  ok: boolean;
  host: string;
  port: number;
  user: string;
  work_dir: string;
  latency_ms: number;
  detail: string;
  pwd: string;
}

export interface RobotProfileInfo {
  id: string;
  label: string;
  status: string;
  dof: number;
  base_link: string;
  joint_order: string[];
  leg_order: string[];
  foot_links: string[];
  diagnostic_spec: string;
  capabilities: string[];
  note: string;
  valid: boolean;
  issues: string[];
}

export interface LLMProfileUpdate {
  model?: string;
  base_url?: string;
  api_key_env_var?: string;
}

export interface LLMProfileUpdateResult {
  profile: LLMProfileInfo;
  saved_path: string;
  message: string;
}

export interface LLMAuditItemInfo {
  file: string;
  schema_name: string;
  model: string;
  elapsed_s: number;
  attempt: number;
  ok: boolean;
  error: string;
  created_at: string;
}

export interface LLMAuditHistory {
  items: LLMAuditItemInfo[];
}

export interface MetricPoint {
  iter: number;
  ts: number;
  reward: number;
  ep_len: number;
  terrain: number;
  phase: number;
  extra: Record<string, unknown>;
}

export interface TensorboardTagInfo {
  tag: string;
  group: string;
  display_name: string;
  points: number;
  first_step: number | null;
  last_step: number | null;
  last_value: number | null;
  source: string;
  formula: string;
  description: string;
}

export interface TensorboardScalarCatalog {
  available: boolean;
  source: "fake" | "real";
  run: string;
  event_file: string;
  reader: string;
  reward_tag: string;
  tags: TensorboardTagInfo[];
  refreshed_at: number;
  error: string;
}

export interface TensorboardScalarPoint {
  step: number;
  value: number;
}

export interface TensorboardScalarSeries {
  tag: string;
  group: string;
  display_name: string;
  source: string;
  formula: string;
  points: TensorboardScalarPoint[];
}

export interface TensorboardSeriesResponse {
  available: boolean;
  source: "fake" | "real";
  run: string;
  event_file: string;
  series: TensorboardScalarSeries[];
  error: string;
}

export interface DefinitionInfo {
  key: string;
  label: string;
  category: "reward" | "curriculum" | "terrain" | "diagnostic" | "system";
  formula: string;
  meaning: string;
  current_reading: string;
  source: string;
  related: string[];
  actions: string[];
}

export interface DefinitionQueryResult {
  query: string;
  items: DefinitionInfo[];
}

export interface TrainingTelemetryPoint {
  step: number;
  total_steps: number | null;
  ts: number;
  elapsed: string;
  eta: string;
  fps: number | null;
  reward: Record<string, number>;
  curriculum: Record<string, unknown>;
  health: Record<string, number>;
  command: Record<string, number>;
  counters: Record<string, number>;
  paths: Record<string, string>;
  checkpoint: string;
  raw: string;
}

export interface DataProvenanceItem {
  key: string;
  label: string;
  path: string;
  role: string;
  available: boolean;
  evidence: string[];
}

export interface MetricSnapshotItem {
  key: string;
  label: string;
  value: unknown;
  numeric_value: number | null;
  unit: string;
  precision: number;
  tone: "good" | "warn" | "bad" | "neutral";
  direction: "higher" | "lower" | "target" | "info";
  target: number | null;
  target_label: string;
  section: string;
  source: string;
  formula: string;
  meaning: string;
  reading: string;
}

export interface MetricSnapshotGroup {
  id: string;
  title: string;
  summary: string;
  items: MetricSnapshotItem[];
}

export interface SnapshotBlocker {
  key: string;
  label: string;
  value: number | null;
  target: number | null;
  direction: "higher" | "lower" | "target" | "info";
  gap: number | null;
  severity: "info" | "warn" | "bad";
  source: string;
  detail: string;
}

export interface RunSnapshot {
  run_id: string;
  source: "fake" | "real";
  mode: "jsonl" | "log_fallback" | "fake" | "missing" | "error";
  running: boolean;
  runtime_state: "live" | "stale" | "stopped" | "interrupted" | "remote_unavailable" | "unknown";
  stale: boolean;
  telemetry_age_s: number | null;
  status: "ok" | "watch" | "blocked" | "missing" | "error";
  conclusion: string;
  generated_at: number;
  step: number;
  total_steps: number | null;
  progress_pct: number | null;
  elapsed: string;
  eta: string;
  fps: number | null;
  phase: string;
  command_mode: string;
  active_dirs: string;
  blocked_by: string;
  next_gate: string;
  latest_checkpoint: string;
  provenance: DataProvenanceItem[];
  metric_groups: MetricSnapshotGroup[];
  blockers: SnapshotBlocker[];
  notes: string[];
}

export interface TrainingTelemetry {
  available: boolean;
  source: "fake" | "real";
  run_id: string;
  running: boolean;
  runtime_state: "live" | "stale" | "stopped" | "interrupted" | "remote_unavailable" | "unknown";
  stale: boolean;
  telemetry_age_s: number | null;
  log_path: string;
  telemetry_path: string;
  mode: "jsonl" | "log_fallback" | "fake" | "missing" | "error";
  source_stats: Record<string, unknown>;
  latest: TrainingTelemetryPoint | null;
  history: TrainingTelemetryPoint[];
  snapshot: RunSnapshot | null;
  definitions: DefinitionInfo[];
  summary: string;
  limitations: string[];
  error: string;
}

export interface ActionResult {
  action: string;
  ok: boolean;
  message: string;
}

export interface SpecCoverageItem {
  id: string;
  title: string;
  group: string;
  training_mechanism: string;
  evaluation: string;
  status: "complete" | "partial" | "missing";
  gaps: string[];
  next_actions: string[];
  evidence: string[];
}

export interface SpecCoverageReport {
  spec_path: string;
  generated_at: number;
  summary: string;
  legend: Record<string, string>;
  items: SpecCoverageItem[];
}

export interface CommandEval {
  name: string;
  vx: number;
  vy: number;
  wz: number;
  tracked: boolean;
  foot_clearance_cm: number;
  duty: number;
  base_h_m: number;
}

export interface PhysevalResult {
  ok: boolean;
  summary: string;
  commands: CommandEval[];
  checkpoint: string | null;
}

export async function getStatus(): Promise<RunStatus> {
  const r = await fetchWithTimeout(`${API_BASE}/run/current`, { headers: authHeaders() }, 15000);
  if (!r.ok) throw new Error(`status ${r.status}`);
  return r.json();
}

export function getTrainingTelemetry(): Promise<TrainingTelemetry> {
  return jsonRequest("/run/current/telemetry");
}

export function getRemoteMachineStatus(): Promise<RemoteMachineStatus> {
  return jsonRequest("/remote/status");
}

export function getDefinitions(query = ""): Promise<DefinitionQueryResult> {
  const suffix = query ? `?query=${encodeURIComponent(query)}` : "";
  return jsonRequest(`/definitions${suffix}`);
}

export function getSpecCoverage(): Promise<SpecCoverageReport> {
  return jsonRequest("/spec/coverage");
}

export async function getActiveConfigSet(): Promise<ConfigSetInfo> {
  const r = await fetchWithTimeout(`${API_BASE}/config/active`, { headers: authHeaders() });
  if (!r.ok) throw new Error(`config ${r.status}`);
  return r.json();
}

export async function getFrameworkProfiles(): Promise<FrameworkProfileInfo[]> {
  const r = await fetchWithTimeout(`${API_BASE}/config/frameworks`, { headers: authHeaders() });
  if (!r.ok) throw new Error(`frameworks ${r.status}`);
  return r.json();
}

export function getFrameworkCatalog(): Promise<FrameworkCatalogInfo> {
  return jsonRequest("/config/framework/catalog");
}

export function getAdaptationPreview(robot = "taili"): Promise<AdaptationPreviewInfo> {
  return jsonRequest(`/config/adaptation/preview?robot=${encodeURIComponent(robot)}`);
}

export function getKnownUrdfs(): Promise<KnownUrdfsInfo> {
  return jsonRequest("/config/robot/import/known");
}

export function validateRobotUrdf(urdf: string): Promise<RobotImportInfo> {
  return jsonRequest(`/config/robot/import?urdf=${encodeURIComponent(urdf)}`);
}

export function getEditableKnobs(): Promise<EditableKnobsInfo> {
  return jsonRequest("/config/framework/edit/knobs");
}

export function validateFrameworkEdit(
  field: string,
  value: number,
  robot = "taili"
): Promise<EditValidationInfo> {
  return jsonRequest(
    `/config/framework/edit/validate?field=${encodeURIComponent(field)}&value=${value}&robot=${encodeURIComponent(robot)}`
  );
}

export function getLLMProfile(): Promise<LLMProfileInfo> {
  return jsonRequest("/config/llm");
}

export function updateLLMProfile(body: LLMProfileUpdate): Promise<LLMProfileUpdateResult> {
  return jsonRequest("/config/llm", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function selectFrameworkProfile(frameworkId: string): Promise<FrameworkSelectionResult> {
  return jsonRequest("/config/framework/select", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ framework_id: frameworkId }),
  });
}

export function getLLMAudit(limit = 12): Promise<LLMAuditHistory> {
  return jsonRequest(`/llm/audit?limit=${limit}`);
}

export function getRemoteProfile(): Promise<RemoteProfileInfo> {
  return jsonRequest("/config/remote");
}

export function updateRemoteProfile(body: Partial<RemoteProfileInfo> & { password?: string }): Promise<RemoteProfileUpdateResult> {
  return jsonRequest("/config/remote", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function testRemoteProfile(): Promise<RemoteConnectionTestInfo> {
  return jsonRequest("/config/remote/test", { method: "POST" });
}

export function getRobotProfile(): Promise<RobotProfileInfo> {
  return jsonRequest("/config/robot");
}

export async function postAction(
  action: "start" | "resume" | "kill" | "deploy-payload"
): Promise<ActionResult> {
  const r = await fetchWithTimeout(`${API_BASE}/action/${action}`, {
    method: "POST",
    headers: authHeaders(),
  });
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    throw new Error(data.detail || `action ${action} failed (${r.status})`);
  }
  return r.json();
}

export async function runPhyseval(): Promise<PhysevalResult> {
  const r = await fetchWithTimeout(`${API_BASE}/action/physeval`, {
    method: "POST",
    headers: authHeaders(),
  });
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    throw new Error(data.detail || `physeval failed (${r.status})`);
  }
  return r.json();
}

export interface DiagnosticPreset {
  id: string;
  label: string;
  description: string;
  category: "quick" | "direction" | "environment" | "robustness";
  estimated_minutes: number;
}

export interface DiagnosticCatalog {
  presets: DiagnosticPreset[];
  plan_templates?: Record<string, Partial<DiagnosticPlan> & Record<string, unknown>>;
  checkpoints: {
    path: string;
    name: string;
    run_name: string;
    iteration: number | null;
    kind?: "latest" | "iteration" | "best" | "unknown";
    is_default?: boolean;
    note?: string;
    framework_id: string;
    framework_label: string;
    framework_status: string;
    diagnostic_task: string;
  }[];
  checkpoint: string | null;
  checkpoint_name: string | null;
  training_running: boolean;
  source: "fake" | "real";
  framework_id: string;
  framework_label: string;
  framework_status: string;
  framework_note: string;
  available: boolean;
  message: string;
}

export interface DiagnosticStageStatus {
  id: string;
  label: string;
  state: "pending" | "running" | "complete" | "error" | "cancelled";
  progress: number;
  rows_written: number;
}

export interface DiagnosticCommandSpec {
  id: string;
  label: string;
  mode: string;
  vx: number;
  vy: number;
  wz: number;
  duration_s: number;
  settle_s: number;
  ramp_s: number;
  repeats: number;
}

export interface DiagnosticTerrainSpec {
  type: string;
  level: number;
  params: Record<string, unknown>;
}

export interface DiagnosticDRCaseSpec {
  level: number;
  friction?: number | null;
  mass_scale?: number | null;
  stiffness_scale?: number | null;
  damping_scale?: number | null;
  latency_steps?: number | null;
  label: string;
}

export interface DiagnosticPushSpec {
  enabled: boolean;
  segment: number;
  time_s: number;
  vector: [number, number, number];
}

export interface DiagnosticPlan {
  name: string;
  template: string;
  num_envs: number;
  reset_policy: string;
  reset_initialization: string;
  commands: DiagnosticCommandSpec[];
  terrains: DiagnosticTerrainSpec[];
  dr_cases: DiagnosticDRCaseSpec[];
  pushes: DiagnosticPushSpec[];
  recording: {
    sample_hz: number;
    max_rows: number;
    save_playback: boolean;
  };
  criteria: {
    tracking_error_max?: number | null;
    min_height_m?: number | null;
    stable_fraction_min?: number | null;
    slip_max?: number | null;
    hard_impact_max?: number | null;
  };
  notes: string[];
  estimated_rollout_s?: number;
}

export interface DiagnosticJobStatus {
  state: "idle" | "starting" | "running" | "complete" | "error" | "cancelled";
  job_id: string | null;
  preset: string | null;
  preset_label: string | null;
  checkpoint: string | null;
  framework_id: string | null;
  output_dir: string | null;
  progress: number;
  elapsed_s: number;
  message: string;
  stages: DiagnosticStageStatus[];
  log_tail: string[];
  plan_summary: Record<string, unknown>;
}

export interface ArtifactRefInfo {
  kind: string;
  label: string;
  uri: string;
  available: boolean;
  note: string;
}

export interface DiagnosticLogInfo {
  path: string;
  available: boolean;
  complete: boolean;
  truncated: boolean;
  bytes: number;
  lines: number;
  shown_lines: number;
  tail: string[];
  error_excerpt: string[];
  note: string;
}

export interface DiagnosticValueInfo {
  key: string;
  label: string;
  group: string;
  value: unknown;
  unit: string;
  source: string;
  fields: string[];
  formula: string;
  interpretation: string;
  confidence: "high" | "medium" | "low";
}

export interface DiagnosticHistoryItem {
  job_id: string;
  state: DiagnosticJobStatus["state"];
  preset: string;
  preset_label: string;
  checkpoint: string;
  checkpoint_name: string;
  framework_id: string;
  diagnostic_task: string;
  config_set_id: string;
  source: "fake" | "real";
  output_dir: string;
  progress: number;
  message: string;
  created_at: string;
  updated_at: string;
  elapsed_s: number;
  artifacts: ArtifactRefInfo[];
  note: string;
  archived: boolean;
}

export interface DiagnosticHistory {
  items: DiagnosticHistoryItem[];
}

export interface DiagnosticCommand {
  mode: string;
  attempts: number;
  major_event_fraction: number;
  done_fraction: number;
  reset_fraction: number;
  stable_fraction: number;
  raw_progress_m: number | null;
  controlled_progress_m: number | null;
  first_major_event_s: number | null;
  tracking_error: number | null;
}

export interface DiagnosticReport {
  job_id: string;
  preset: string;
  preset_label: string;
  checkpoint: string;
  output_dir: string;
  artifacts: ArtifactRefInfo[];
  log: DiagnosticLogInfo;
  value_explanations: DiagnosticValueInfo[];
  coverage: {
    rows?: number;
    cases?: number;
    attempts?: number;
    terrain_types?: string[];
    terrain_types_requested?: string[];
    terrain_height_sources?: string[];
    dr_levels?: string[];
    dr_levels_requested?: string[];
    push_frames?: number;
  };
  commands: DiagnosticCommand[];
  posture: {
    max_roll_deg?: number | null;
    max_pitch_deg?: number | null;
    min_height_m?: number | null;
    stable_fraction?: number | null;
  };
  events: { type: string; count: number }[];
  notes: { priority: "high" | "medium" | "low"; type: string; message: string }[];
  terrain: {
    requested_types?: string[];
    types?: string[];
    height_sources?: string[];
    readiness?: string;
    gaps?: string[];
    terrain_command_rows?: { label: string; value: string; tone: "good" | "warn" | "bad" | "neutral" }[];
  };
  robustness: { dr_levels_requested?: string[]; dr_levels?: string[]; dr_columns?: string[]; push_frames?: number };
  plan: Partial<DiagnosticPlan> & Record<string, unknown>;
}

export interface DiagnosticPlaybackFoot {
  position: [number, number, number];
  contact: boolean;
  force_norm?: number | null;
  clearance?: number | null;
}

export interface DiagnosticPlaybackFrame {
  t: number;
  stage: string;
  case_id: number;
  env_id: number;
  segment_id: number;
  command_mode: string;
  base_position: [number, number, number];
  base_quaternion_wxyz: [number, number, number, number];
  joints: number[];
  feet: Record<string, DiagnosticPlaybackFoot>;
  reset_observed: boolean;
  done: boolean;
  terrain_height?: number | null;
  terrain?: string;
  terrain_level?: number | null;
}

export interface DiagnosticPlayback {
  available: boolean;
  message: string;
  source: "fake" | "real";
  output_dir?: string | null;
  fps: number;
  joint_order: string[];
  leg_order: string[];
  frames: DiagnosticPlaybackFrame[];
  source_rows: number;
  stride: number;
  selected_env_id?: number | null;
  available_env_ids: number[];
}

async function jsonRequest<T>(url: string, init: RequestInit = {}): Promise<T> {
  const response = await fetchWithTimeout(`${API_BASE}${url}`, {
    ...init,
    headers: { ...authHeaders(), ...(init.headers || {}) },
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detailText = String(data.detail || `请求失败 (${response.status})`);
    if (response.status === 403 && /token/i.test(detailText)) {
      throw new Error("需要操作令牌：请到「配置中心 → 操作令牌」粘贴后端令牌并保存，然后重试本操作。");
    }
    throw new Error(detailText);
  }
  return data as T;
}

export interface AcceptanceVerdict {
  available: boolean;
  reason?: string;
  passed?: boolean | null;
  n_present?: number | null;
  n_hard?: number | null;
  missing?: string[];
  run?: string;
  gates?: Record<string, { ok: boolean; detail: string }>;
}

export function getAcceptance(): Promise<AcceptanceVerdict> {
  return jsonRequest("/run/current/acceptance");
}

export function getDiagnosticCatalog(): Promise<DiagnosticCatalog> {
  return jsonRequest("/diagnostics/catalog");
}

export function getDiagnosticStatus(): Promise<DiagnosticJobStatus> {
  return jsonRequest("/diagnostics/status");
}

export function getDiagnosticHistory(limit = 30): Promise<DiagnosticHistory> {
  return jsonRequest(`/diagnostics/history?limit=${limit}`);
}

export function startDiagnostic(
  preset: string,
  checkpoint?: string | null,
  plan?: Partial<DiagnosticPlan> | null
): Promise<DiagnosticJobStatus> {
  return jsonRequest("/diagnostics/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ preset, checkpoint, plan }),
  });
}

export function cancelDiagnostic(): Promise<DiagnosticJobStatus> {
  return jsonRequest("/diagnostics/cancel", { method: "POST" });
}

export function getDiagnosticReport(jobId?: string | null): Promise<DiagnosticReport> {
  const suffix = jobId ? `?job_id=${encodeURIComponent(jobId)}` : "";
  return jsonRequest(`/diagnostics/report${suffix}`);
}

export function getDiagnosticPlayback(maxFrames = 900, jobId?: string | null): Promise<DiagnosticPlayback> {
  const params = new URLSearchParams({ max_frames: String(maxFrames) });
  if (jobId) params.set("job_id", jobId);
  return jsonRequest(`/diagnostics/playback?${params.toString()}`);
}

export function getTensorboardScalarCatalog(): Promise<TensorboardScalarCatalog> {
  return jsonRequest("/run/current/tensorboard/scalars");
}

export function getTensorboardSeries(tags: string[], maxPoints = 1200): Promise<TensorboardSeriesResponse> {
  const params = new URLSearchParams({ max_points: String(maxPoints) });
  tags.forEach((tag) => params.append("tags", tag));
  return jsonRequest(`/run/current/tensorboard/series?${params.toString()}`);
}

export function updateDiagnosticHistoryItem(
  jobId: string,
  body: { note?: string; archived?: boolean }
): Promise<DiagnosticHistoryItem> {
  return jsonRequest(`/diagnostics/history/${encodeURIComponent(jobId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function deleteDiagnosticHistoryItem(jobId: string): Promise<{ ok: boolean; job_id: string; message: string }> {
  return jsonRequest(`/diagnostics/history/${encodeURIComponent(jobId)}`, { method: "DELETE" });
}

export interface ProposedAction {
  name: string;
  args: Record<string, any>;
}

export interface Grounding {
  checked: boolean;
  grounded: boolean | null;
  unsupported: string[];
}

export interface ChatResponse {
  reply: string;
  steps: number;
  mode: string;
  elapsed_s: number;
  suggestions: string[];
  transcript: { tool?: string; args?: any; result?: any }[];
  proposed_action?: ProposedAction | null;
  grounding?: Grounding | null;
  proposal_id?: string | null;
  context_envelope?: ContextEnvelopeInfo | null;
}

export interface ChatProposalInfo {
  id: string;
  created_at: string;
  updated_at: string;
  status: "pending" | "executed" | "failed" | "cancelled";
  name: string;
  args: Record<string, any>;
  reply: string;
  result: string;
  kind: string;
  risk: "low" | "medium" | "high";
  evidence: string[];
  requires: string[];
  expected_result: string;
  rollback: string;
}

export interface ChatProposalHistory {
  items: ChatProposalInfo[];
}

export interface LLMReadinessInfo {
  configured: boolean;
  api_key_resolved: boolean;
  unsafe_literal_key: boolean;
  model: string;
  base_url: string;
  audit_count: number;
  tools: string[];
  workflow_count: number;
  ok: boolean;
  issues: string[];
}

export interface LLMEvidenceItem {
  id: string;
  label: string;
  status: "ok" | "warning" | "error" | "unknown";
  detail: string;
  data: Record<string, any>;
}

export interface LLMWorkflowResult {
  workflow: string;
  ok: boolean;
  generated_by: "deterministic" | "llm" | "deterministic+llm";
  summary: string;
  evidence: LLMEvidenceItem[];
  findings: string[];
  proposals: ChatProposalInfo[];
  transcript: Record<string, any>[];
  issues: string[];
}

export interface AgentWorkbenchAttemptInfo {
  run_id: string;
  running: boolean;
  runtime_state: RunStatus["runtime_state"];
  source: "fake" | "real";
  step: number;
  total_steps: number | null;
  progress_pct: number | null;
  phase: string;
  command_mode: string;
  blocked_by: string;
  latest_checkpoint: string;
  telemetry_age_s: number | null;
  remote_ok: boolean;
  summary: string;
}

export interface AgentWorkbenchEvidenceInfo {
  id: string;
  label: string;
  status: "ok" | "warning" | "error" | "missing" | "unknown";
  detail: string;
  source: string;
  values: Record<string, any>;
  provenance: DataProvenanceItem[];
}

export interface AgentWorkbenchJudgementInfo {
  status: "ready" | "running" | "blocked" | "needs_evidence" | "remote_unavailable" | "error";
  title: string;
  summary: string;
  confidence: "high" | "medium" | "low";
  evidence_ids: string[];
  gaps: string[];
}

export interface AgentWorkbenchActionInfo {
  id: string;
  label: string;
  kind: "ask_agent" | "diagnostic" | "training" | "deployment" | "safety";
  risk: "low" | "medium" | "high";
  enabled: boolean;
  requires_confirmation: boolean;
  endpoint: string;
  method: string;
  reason: string;
  rollback: string;
}

export interface AgentWorkbenchInfo {
  generated_at: number;
  objective: string;
  service_loop: string[];
  attempt: AgentWorkbenchAttemptInfo;
  llm: LLMReadinessInfo;
  judgement: AgentWorkbenchJudgementInfo;
  evidence: AgentWorkbenchEvidenceInfo[];
  proposals: ChatProposalInfo[];
  actions: AgentWorkbenchActionInfo[];
  notes: string[];
}

function fetchWithTimeout(url: string, init: RequestInit = {}, timeoutMs = 90000): Promise<Response> {
  const controller = new AbortController();
  const externalSignal = init.signal;
  let timedOut = false;
  if (externalSignal) {
    if (externalSignal.aborted) controller.abort();
    else externalSignal.addEventListener("abort", () => controller.abort(), { once: true });
  }
  const timer = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  return fetch(url, { ...init, signal: controller.signal }).catch((reason) => {
    if (reason?.name === "AbortError") {
      throw new Error(timedOut ? `request timeout after ${Math.round(timeoutMs / 1000)}s` : "request canceled");
    }
    throw reason;
  }).finally(() => window.clearTimeout(timer));
}

export async function chat(
  message: string,
  ui_mode = "",
  context = "",
  signal?: AbortSignal,
  context_request?: Partial<ContextRequestInfo>
): Promise<ChatResponse> {
  const r = await fetchWithTimeout(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ message, ui_mode, context, context_request }),
    signal,
  }, 75000);
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    throw new Error(data.detail || `chat ${r.status}`);
  }
  return r.json();
}

export function routeChatContext(
  message: string,
  ui_mode = "",
  context = "",
  context_request?: Partial<ContextRequestInfo>
): Promise<ContextEnvelopeInfo> {
  return jsonRequest("/chat/context/route", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, ui_mode, context, context_request }),
  });
}

export async function resetChat(): Promise<{ ok: boolean }> {
  const r = await fetch(`${API_BASE}/chat/reset`, { method: "POST", headers: authHeaders() });
  if (!r.ok) throw new Error(`chat ${r.status}`);
  return r.json();
}

export interface ExecuteResponse {
  ok: boolean;
  detail: string;
}

export async function executeAction(
  name: string,
  args: Record<string, any>
): Promise<ExecuteResponse> {
  const r = await fetch(`${API_BASE}/chat/execute`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ name, args }),
  });
  if (!r.ok) throw new Error(`execute ${r.status}`);
  return r.json();
}

export function getChatProposals(limit = 12): Promise<ChatProposalHistory> {
  return jsonRequest(`/chat/proposals?limit=${limit}`);
}

export function cancelChatProposal(name: string): Promise<ChatProposalInfo> {
  return jsonRequest("/chat/proposals/cancel", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export function getLLMReadiness(): Promise<LLMReadinessInfo> {
  return jsonRequest("/llm/readiness");
}

export function getAgentWorkbench(): Promise<AgentWorkbenchInfo> {
  return jsonRequest<AgentWorkbenchInfo>("/agent/workbench").then((data) => {
    if (!data || typeof data !== "object" || !data.attempt || !data.judgement || !Array.isArray(data.evidence)) {
      throw new Error("工作台接口返回结构无效，请检查开发服务器代理 /agent -> 8000。");
    }
    return data;
  });
}

export function runLLMSystemAudit(includeLlm = true): Promise<LLMWorkflowResult> {
  return jsonRequest("/llm/workflows/system-audit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ include_llm: includeLlm }),
  });
}

export function runLLMDiagnosticBrief(jobId?: string | null, includeLlm = true): Promise<LLMWorkflowResult> {
  return jsonRequest("/llm/workflows/diagnostic-brief", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_id: jobId ?? null, include_llm: includeLlm }),
  });
}

// Open the metric stream. Returns the socket so the caller can close it.
export function openStream(
  onPoint: (p: MetricPoint) => void,
  onError?: (e: string) => void
): WebSocket {
  const token = consoleToken();
  const q = token ? `?token=${encodeURIComponent(token)}` : "";
  const ws = new WebSocket(`${WS_BASE}/run/current/stream${q}`);
  ws.onmessage = (ev) => {
    let data: any;
    try {
      data = JSON.parse(ev.data);
    } catch {
      // Drop a malformed frame instead of throwing inside the handler.
      return;
    }
    if (data.error) onError?.(`${data.error}: ${data.message}`);
    else onPoint(data as MetricPoint);
  };
  ws.onerror = () => {
    console.error("websocket error on /run/current/stream");
    onError?.("websocket error");
  };
  return ws;
}
