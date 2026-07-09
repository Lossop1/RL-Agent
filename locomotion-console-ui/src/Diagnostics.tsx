import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import {
  cancelDiagnostic,
  getDiagnosticCatalog,
  getDiagnosticHistory,
  getDiagnosticPlayback,
  getDiagnosticReport,
  getDiagnosticStatus,
  getTrainingTelemetry,
  startDiagnostic,
  type DiagnosticCatalog,
  type DiagnosticHistoryItem,
  type DiagnosticJobStatus,
  type DiagnosticPlayback,
  type DiagnosticPlaybackFrame,
  type DiagnosticPlan,
  type DiagnosticReport,
  type TrainingTelemetry,
  type TrainingTelemetryPoint,
} from "./api";
import { formatError } from "./i18n/format";
import LineChart from "./LineChart";

const RobotViewer = lazy(() => import("./RobotViewer"));

const DEFAULT_DIAG_FIELDS = ["base_z", "terrain_height", "contact_count"];
const DEFAULT_LOG_FIELDS = [
  "reward.total",
  "command.actual_vx",
  "command.lin_err",
  "command.stance_slip_high_fraction",
  "curriculum.terrain_mean",
  "curriculum.progress_gate",
  "health.fall_rate",
];
const DIAGNOSTIC_POLL_MS = 3000;
const ACTIVE_REFRESH_MS = 15000;

type TelemetryGroupKey = "reward" | "curriculum" | "health" | "command" | "counters";
type ScaleMode = "normalized" | "raw";
type CommandDirection = "stand" | "forward" | "backward" | "left" | "right" | "yaw_left" | "yaw_right" | "custom";

interface ChartOption {
  key: string;
  label: string;
  values: Array<number | null>;
  group: string;
  unit?: string;
  description?: string;
}

export default function Diagnostics({ active = true }: { active?: boolean }) {
  const [catalog, setCatalog] = useState<DiagnosticCatalog | null>(null);
  const [job, setJob] = useState<DiagnosticJobStatus | null>(null);
  const [history, setHistory] = useState<DiagnosticHistoryItem[]>([]);
  const [report, setReport] = useState<DiagnosticReport | null>(null);
  const [playback, setPlayback] = useState<DiagnosticPlayback | null>(null);
  const [telemetry, setTelemetry] = useState<TrainingTelemetry | null>(null);
  const [preset, setPreset] = useState("terrain");
  const [checkpoint, setCheckpoint] = useState("best");
  const [planDraft, setPlanDraft] = useState<Partial<DiagnosticPlan> | null>(null);
  const [useJsonOverride, setUseJsonOverride] = useState(false);
  const [planText, setPlanText] = useState("");
  const [diagFields, setDiagFields] = useState<string[]>(DEFAULT_DIAG_FIELDS);
  const [logFields, setLogFields] = useState<string[]>(DEFAULT_LOG_FIELDS);
  const [logScaleMode, setLogScaleMode] = useState<ScaleMode>("normalized");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const initializedRef = useRef(false);
  const draftKeyRef = useRef("");
  const refreshInFlightRef = useRef(false);

  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    if (!active) return;
    void refresh({ silent: true });
  }, [active]);

  useEffect(() => {
    if (!active) return;
    const intervalMs = job?.state === "running" || job?.state === "starting"
      ? DIAGNOSTIC_POLL_MS
      : ACTIVE_REFRESH_MS;
    const timer = window.setInterval(() => void refresh({ silent: true }), intervalMs);
    return () => window.clearInterval(timer);
  }, [active, job?.state, job?.job_id]);

  const selectedPreset = useMemo(
    () => catalog?.presets.find((item) => item.id === preset) ?? null,
    [catalog, preset],
  );
  const selectedTemplate = useMemo(
    () => catalog?.plan_templates?.[preset] ?? null,
    [catalog, preset],
  );

  useEffect(() => {
    const key = `${preset}:${stablePlanKey(selectedTemplate)}`;
    if (key === draftKeyRef.current) return;
    draftKeyRef.current = key;
    setPlanDraft(normalizeDraft(selectedTemplate));
    setPlanText("");
    setUseJsonOverride(false);
  }, [preset, selectedTemplate]);

  const effectivePlan = planDraft ?? selectedTemplate;
  const diagOptions = useMemo(() => buildDiagnosticOptions(report, playback), [report, playback]);
  const logOptions = useMemo(() => buildLogOptions(telemetry), [telemetry]);
  const selectedDiagSeries = diagOptions
    .filter((item) => diagFields.includes(item.key))
    .map((item, index) => toSeries(item, index, false));
  const selectedLogSeries = logOptions
    .filter((item) => logFields.includes(item.key))
    .map((item, index) => toSeries(logScaleMode === "normalized" ? { ...item, values: normalizeSeries(item.values) } : item, index, false));

  async function refresh(options: { silent?: boolean } = {}) {
    if (refreshInFlightRef.current) return;
    refreshInFlightRef.current = true;
    if (!options.silent) setBusy("refresh");
    try {
      const [nextCatalog, nextJob, nextHistory, nextTelemetry] = await Promise.all([
        getDiagnosticCatalog(),
        getDiagnosticStatus(),
        getDiagnosticHistory(12),
        getTrainingTelemetry().catch(() => null),
      ]);
      setCatalog(nextCatalog);
      setJob(nextJob);
      setHistory(nextHistory.items);
      setTelemetry(nextTelemetry);
      if (!initializedRef.current) {
        setPreset(nextJob.preset || nextCatalog.presets.find((item) => item.id === "terrain")?.id || nextCatalog.presets[0]?.id || "quick");
        setCheckpoint(nextCatalog.checkpoint || "best");
        initializedRef.current = true;
      }
      await loadReport(nextJob.job_id || nextHistory.items[0]?.job_id || null);
      setError("");
    } catch (reason) {
      setError(formatError(reason));
    } finally {
      refreshInFlightRef.current = false;
      if (!options.silent) setBusy("");
    }
  }

  async function loadReport(jobId: string | null | undefined) {
    if (!jobId) {
      setReport(null);
      setPlayback(null);
      return;
    }
    const nextReport = await getDiagnosticReport(jobId).catch(() => null);
    setReport(nextReport);
    setPlayback(await getDiagnosticPlayback(1200, jobId).catch(() => null));
  }

  async function run() {
    setBusy("run");
    try {
      const plan = useJsonOverride ? parsePlanText(planText) : effectivePlan;
      const next = await startDiagnostic(preset, checkpoint || "best", plan);
      setJob(next);
      setError("");
      await refresh();
    } catch (reason) {
      setError(formatError(reason));
    } finally {
      setBusy("");
    }
  }

  async function cancel() {
    setBusy("cancel");
    try {
      setJob(await cancelDiagnostic());
      await refresh();
    } catch (reason) {
      setError(formatError(reason));
    } finally {
      setBusy("");
    }
  }

  return (
    <section className="primary-panel tool-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">物理诊断</p>
          <h1>诊断运行、调参和回放</h1>
        </div>
        <button className="secondary-button" disabled={Boolean(busy)} onClick={() => void refresh()}>
          刷新
        </button>
      </div>

      {error && <div className="inline-alert">{error}</div>}

      <div className="diagnostic-layout">
        <section className="tool-section diagnostic-control">
          <h2>启动诊断</h2>
          <label>
            <span>预设</span>
            <select value={preset} onChange={(event) => setPreset(event.target.value)}>
              {(catalog?.presets ?? []).map((item) => (
                <option key={item.id} value={item.id}>{item.label}</option>
              ))}
            </select>
          </label>
          <label>
            <span>检查点</span>
            <select value={checkpoint} onChange={(event) => setCheckpoint(event.target.value)}>
              <option value="best">后端默认：{catalog?.checkpoint_name || catalog?.checkpoint || "最新可用检查点"}</option>
              {(catalog?.checkpoints ?? []).map((item) => (
                <option key={item.path} value={item.path}>
                  {item.name}{item.is_default ? "（默认）" : ""} · {item.framework_label || item.framework_id}
                </option>
              ))}
            </select>
          </label>
          <p>{selectedPreset?.description || "诊断会在远端运行，并生成报告、曲线和机器人回放。"}</p>
          <DetailGrid items={summarizePlan(effectivePlan)} />
          <PlanControls plan={effectivePlan} onChange={setPlanDraft} />
          <details className="plan-editor" open={useJsonOverride}>
            <summary>
              <label className="inline-check">
                <input
                  type="checkbox"
                  checked={useJsonOverride}
                  onChange={(event) => {
                    const checked = event.target.checked;
                    setUseJsonOverride(checked);
                    if (checked && !planText) setPlanText(prettyJson(effectivePlan ?? {}));
                  }}
                />
                <span>高级 JSON 覆盖</span>
              </label>
            </summary>
            <div className="plan-editor-head">
              <span>只在表单覆盖不了测试意图时使用。</span>
              <button className="secondary-button" type="button" onClick={() => setPlanText(prettyJson(effectivePlan ?? {}))}>
                载入当前表单
              </button>
            </div>
            <textarea
              spellCheck={false}
              value={useJsonOverride ? planText : prettyJson(effectivePlan ?? {})}
              readOnly={!useJsonOverride}
              onChange={(event) => setPlanText(event.target.value)}
            />
          </details>
          <div className="button-row">
            <button className="primary-button" disabled={Boolean(busy) || catalog?.available === false} onClick={() => void run()}>
              启动诊断
            </button>
          </div>
        </section>

        <section className="tool-section diagnostic-status">
          <div className="diagnostic-status-head">
            <h2>当前任务</h2>
            <button className="secondary-button danger-button" disabled={!job || job.state !== "running" || Boolean(busy)} onClick={() => void cancel()}>
              中断诊断
            </button>
          </div>
          <DetailGrid items={[
            ["状态", job?.state || "idle"],
            ["任务", job?.job_id || "无"],
            ["预设", job?.preset_label || job?.preset || "无"],
            ["进度", `${Math.round((job?.progress ?? 0) * 100)}%`],
            ["输出", job?.output_dir || "无"],
          ]} />
          {job?.stages?.length ? (
            <div className="stage-list">
              {job.stages.map((stage) => (
                <div key={stage.id}>
                  <strong>{stage.label}</strong>
                  <span>{stage.state} · {Math.round(stage.progress * 100)}% · {stage.rows_written} 行</span>
                </div>
              ))}
            </div>
          ) : null}
          {job?.message && <p>{job.message}</p>}
          {job?.log_tail?.length ? (
            <details className="run-log" open>
              <summary>诊断日志</summary>
              <pre>{job.log_tail.slice(-80).join("\n")}</pre>
            </details>
          ) : null}
        </section>
      </div>

      <section className="tool-section chart-section">
        <ChartBlock
          title="诊断曲线"
          count={diagFields.length}
          options={diagOptions}
          selected={diagFields}
          onToggle={(key) => toggleList(diagFields, key, setDiagFields, 8)}
          series={selectedDiagSeries}
          emptyLabel="先运行诊断或选择历史报告"
        />
      </section>

      <section className="tool-section chart-section">
        <div className="panel-heading compact-heading">
          <div>
            <p className="eyebrow">训练日志</p>
            <h2>日志曲线</h2>
          </div>
          <div className="segmented-control">
            <button className={logScaleMode === "normalized" ? "active" : ""} onClick={() => setLogScaleMode("normalized")}>归一化</button>
            <button className={logScaleMode === "raw" ? "active" : ""} onClick={() => setLogScaleMode("raw")}>原始值</button>
          </div>
        </div>
        <ChartBlock
          title=""
          count={logFields.length}
          options={logOptions}
          selected={logFields}
          onToggle={(key) => toggleList(logFields, key, setLogFields, 8)}
          series={selectedLogSeries}
          emptyLabel="暂无训练日志字段"
        />
      </section>

      <section className="tool-section render-section">
        <div className="panel-heading compact-heading">
          <div>
            <p className="eyebrow">物理回放</p>
            <h2>机器人渲染证据</h2>
          </div>
          <span className="render-source">{report?.job_id || "暂无报告"}</span>
        </div>
        <Suspense fallback={<div className="viewer-fallback">正在加载渲染器</div>}>
          <RobotViewer report={report} jobId={report?.job_id ?? job?.job_id ?? null} active={active} />
        </Suspense>
      </section>

      <section className="tool-section">
        <h2>最新报告</h2>
        {report ? (
          <div className="report-grid">
            <DetailGrid items={[
              ["任务", report.job_id],
              ["检查点", report.checkpoint || "无"],
              ["行数", String(report.coverage.rows ?? "无")],
              ["稳定比例", formatNumber(report.posture.stable_fraction)],
              ["最低高度", formatNumber(report.posture.min_height_m)],
              ["事件数", String(report.events.reduce((sum, item) => sum + item.count, 0))],
            ]} />
            <div className="note-list">
              {report.notes.slice(0, 8).map((note) => <span key={`${note.type}-${note.message}`}>{note.message}</span>)}
            </div>
          </div>
        ) : (
          <p className="muted">暂无可读报告。</p>
        )}
      </section>

      <section className="tool-section">
        <h2>历史</h2>
        <div className="history-list">
          {history.map((item) => (
            <button key={item.job_id} onClick={async () => await loadReport(item.job_id)}>
              <strong>{item.preset_label || item.preset}</strong>
              <span>{item.state} · {item.checkpoint_name || item.checkpoint || "检查点未知"}</span>
            </button>
          ))}
          {history.length === 0 && <p className="muted">没有诊断历史。</p>}
        </div>
      </section>
    </section>
  );
}

function PlanControls({
  plan,
  onChange,
}: {
  plan: Partial<DiagnosticPlan> | null;
  onChange: (next: Partial<DiagnosticPlan>) => void;
}) {
  if (!plan) return null;
  const commands = plan.commands ?? [];
  const terrains = plan.terrains ?? [];
  const recording = plan.recording ?? { sample_hz: 50, max_rows: 5000, save_playback: true };
  const updatePlan = (patch: Partial<DiagnosticPlan>) => onChange({ ...plan, ...patch });
  const updateRecording = (patch: Partial<DiagnosticPlan["recording"]>) => {
    updatePlan({ recording: { ...recording, ...patch } });
  };
  const updateCommand = (index: number, patch: Partial<NonNullable<DiagnosticPlan["commands"]>[number]>) => {
    updatePlan({ commands: commands.map((command, i) => (i === index ? { ...command, ...patch } : command)) });
  };
  const addCommand = () => updatePlan({
    commands: [...commands, newCommand("forward", commands.length)],
  });
  const removeCommand = (index: number) => {
    if (commands.length <= 1) return;
    updatePlan({ commands: commands.filter((_, i) => i !== index) });
  };
  const updateTerrain = (index: number, patch: Partial<NonNullable<DiagnosticPlan["terrains"]>[number]>) => {
    updatePlan({ terrains: terrains.map((terrain, i) => (i === index ? { ...terrain, ...patch } : terrain)) });
  };
  const updateTerrainType = (index: number, type: string) => {
    const terrain = terrains[index];
    const params = terrain?.params ?? {};
    if (type === "stairs_up") {
      updateTerrain(index, { type, params: { ...params, direction: "up" } });
    } else if (type === "stairs_down") {
      updateTerrain(index, { type, params: { ...params, direction: "down" } });
    } else {
      updateTerrain(index, { type });
    }
  };
  const updateTerrainParam = (index: number, key: string, value: unknown) => {
    const terrain = terrains[index];
    if (!terrain) return;
    updateTerrain(index, { params: { ...(terrain.params ?? {}), [key]: value } });
  };
  const addTerrain = () => updatePlan({
    terrains: [...terrains, { type: "stairs_up", level: 5, params: { step_height: 0.18, direction: "up" } }],
  });
  const removeTerrain = (index: number) => {
    if (terrains.length <= 1) return;
    updatePlan({ terrains: terrains.filter((_, i) => i !== index) });
  };

  return (
    <div className="plan-controls">
      <div className="plan-control-grid">
        <label>
          <span>环境数量</span>
          <input type="number" min={1} max={64} step={1} value={plan.num_envs ?? 1}
                 onChange={(event) => updatePlan({ num_envs: clampIntInput(event.currentTarget.value, 1, 64, 1) })} />
        </label>
        <label>
          <span>记录频率 Hz</span>
          <input type="number" min={10} max={100} step={5} value={recording.sample_hz ?? 50}
                 onChange={(event) => updateRecording({ sample_hz: clampNumberInput(event.currentTarget.value, 10, 100, 50) })} />
        </label>
        <label>
          <span>最大记录行</span>
          <input type="number" min={100} max={200000} step={100} value={recording.max_rows ?? 5000}
                 onChange={(event) => updateRecording({ max_rows: clampIntInput(event.currentTarget.value, 100, 200000, 5000) })} />
        </label>
      </div>
      <div className="terrain-editor-head">
        <h3>命令</h3>
        <button className="secondary-button" type="button" onClick={addCommand}>添加命令</button>
      </div>
      <div className="command-editor">
        {commands.map((command, index) => {
          const direction = directionForCommand(command);
          const isCustom = direction === "custom";
          return (
            <div className="command-row" key={`${command.id || command.label || "command"}-${index}`}>
              <label>
                <span>方向</span>
                <select
                  value={direction}
                  onChange={(event) => updateCommand(index, commandForDirection(command, event.currentTarget.value as CommandDirection))}
                >
                  <option value="stand">站立</option>
                  <option value="forward">前进</option>
                  <option value="backward">后退</option>
                  <option value="left">左移</option>
                  <option value="right">右移</option>
                  <option value="yaw_left">左转</option>
                  <option value="yaw_right">右转</option>
                  <option value="custom">自定义</option>
                </select>
              </label>
              {isCustom ? (
                <>
                  <label>
                    <span>vx m/s</span>
                    <input type="number" min={-1.5} max={1.5} step={0.05} value={numberParam(command.vx, 0)}
                           onChange={(event) => updateCommand(index, { vx: clampNumberInput(event.currentTarget.value, -1.5, 1.5, 0), mode: "custom" })} />
                  </label>
                  <label>
                    <span>vy m/s</span>
                    <input type="number" min={-0.7} max={0.7} step={0.05} value={numberParam(command.vy, 0)}
                           onChange={(event) => updateCommand(index, { vy: clampNumberInput(event.currentTarget.value, -0.7, 0.7, 0), mode: "custom" })} />
                  </label>
                  <label>
                    <span>wz rad/s</span>
                    <input type="number" min={-1.5} max={1.5} step={0.05} value={numberParam(command.wz, 0)}
                           onChange={(event) => updateCommand(index, { wz: clampNumberInput(event.currentTarget.value, -1.5, 1.5, 0), mode: "custom" })} />
                  </label>
                </>
              ) : (
                <label>
                  <span>{direction === "yaw_left" || direction === "yaw_right" ? "角速度 rad/s" : "速度 m/s"}</span>
                  <input type="number" min={0} max={direction === "yaw_left" || direction === "yaw_right" ? 1.5 : 1.5} step={0.05}
                         value={speedForCommand(command, direction)}
                         disabled={direction === "stand"}
                         onChange={(event) => updateCommand(index, commandForDirection(command, direction, clampNumberInput(event.currentTarget.value, 0, 1.5, defaultSpeedForDirection(direction))))} />
                </label>
              )}
              <label>
                <span>持续 s</span>
                <input type="number" min={0.2} max={30} step={0.2} value={numberParam(command.duration_s, 3)}
                       onChange={(event) => updateCommand(index, { duration_s: clampNumberInput(event.currentTarget.value, 0.2, 30, 3) })} />
              </label>
              <label>
                <span>准备 s</span>
                <input type="number" min={0} max={10} step={0.2} value={numberParam(command.settle_s, 0)}
                       onChange={(event) => updateCommand(index, { settle_s: clampNumberInput(event.currentTarget.value, 0, 10, 0) })} />
              </label>
              <label>
                <span>重复</span>
                <input type="number" min={1} max={10} step={1} value={numberParam(command.repeats, 1)}
                       onChange={(event) => updateCommand(index, { repeats: clampIntInput(event.currentTarget.value, 1, 10, 1) })} />
              </label>
              <button className="icon-button" type="button" title="删除命令" disabled={commands.length <= 1} onClick={() => removeCommand(index)}>
                ×
              </button>
            </div>
          );
        })}
      </div>
      <div className="terrain-editor-head">
        <h3>地形</h3>
        <button className="secondary-button" type="button" onClick={addTerrain}>添加地形</button>
      </div>
      <div className="terrain-editor">
        {terrains.map((terrain, index) => {
          const params = terrain.params ?? {};
          const isStairs = terrain.type === "stairs" || terrain.type === "stairs_up" || terrain.type === "stairs_down";
          return (
            <div className="terrain-row" key={`${terrain.type}-${index}`}>
              <label>
                <span>类型</span>
                <select value={terrain.type} onChange={(event) => updateTerrainType(index, event.currentTarget.value)}>
                  <option value="flat">平地</option>
                  <option value="slope">上坡</option>
                  <option value="slope_inv">下坡</option>
                  <option value="rough">粗糙地形</option>
                  <option value="stairs_up">上楼梯</option>
                  <option value="stairs_down">下楼梯</option>
                  <option value="boxes">箱体</option>
                </select>
              </label>
              <label>
                <span>等级</span>
                <input type="number" min={0} max={9} step={1} value={terrain.level ?? 0}
                       onChange={(event) => updateTerrain(index, { level: clampIntInput(event.currentTarget.value, 0, 9, 0) })} />
              </label>
              {isStairs && (
                <>
                  <label>
                    <span>台阶高 m</span>
                    <input type="number" min={0.02} max={0.4} step={0.01} value={numberParam(params.step_height, 0.18)}
                           onChange={(event) => updateTerrainParam(index, "step_height", clampNumberInput(event.currentTarget.value, 0.02, 0.4, 0.18))} />
                  </label>
                  <label>
                    <span>方向</span>
                    <select value={stringParam(params.direction, terrain.type === "stairs_down" ? "down" : "up")}
                            onChange={(event) => updateTerrainType(index, event.currentTarget.value === "up" ? "stairs_up" : "stairs_down")}>
                      <option value="up">上楼梯</option>
                      <option value="down">下楼梯</option>
                    </select>
                  </label>
                </>
              )}
              <button className="icon-button" type="button" title="删除地形" disabled={terrains.length <= 1} onClick={() => removeTerrain(index)}>
                ×
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ChartBlock({
  title,
  count,
  options,
  selected,
  onToggle,
  series,
  emptyLabel,
}: {
  title: string;
  count: number;
  options: ChartOption[];
  selected: string[];
  onToggle: (key: string) => void;
  series: Array<{ values: Array<number | null>; color: string; label: string; width: number }>;
  emptyLabel: string;
}) {
  return (
    <>
      {title && (
        <div className="panel-heading compact-heading">
          <div>
            <p className="eyebrow">曲线</p>
            <h2>{title}</h2>
          </div>
          <span className="render-source">{count} 条</span>
        </div>
      )}
      <div className="chart-field-picker">
        {options.map((option) => (
          <label key={option.key} title={option.description || option.key}>
            <input type="checkbox" checked={selected.includes(option.key)} onChange={() => onToggle(option.key)} />
            <span>{option.label}</span>
            {option.unit ? <small>{option.unit}</small> : null}
          </label>
        ))}
        {options.length === 0 && <p className="muted">{emptyLabel}</p>}
      </div>
      <div className="diagnostic-chart">
        <LineChart series={series} height={220} label={title || "日志曲线"} emptyLabel="请选择字段" />
      </div>
    </>
  );
}

function DetailGrid({ items }: { items: Array<[string, string]> }) {
  return (
    <dl className="detail-grid">
      {items.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function normalizeDraft(plan: Partial<DiagnosticPlan> | null | undefined): Partial<DiagnosticPlan> | null {
  if (!plan) return null;
  const cloned = JSON.parse(JSON.stringify(plan)) as Partial<DiagnosticPlan>;
  if (Array.isArray(cloned.terrains)) {
    cloned.terrains = cloned.terrains.map((terrain) => {
      if (terrain.type !== "stairs") return terrain;
      const direction = stringParam(terrain.params?.direction, "up");
      return {
        ...terrain,
        type: direction === "down" ? "stairs_down" : "stairs_up",
        params: { ...(terrain.params ?? {}), direction: direction === "down" ? "down" : "up" },
      };
    });
  }
  return { ...cloned, num_envs: cloned.num_envs ?? 1 };
}

function parsePlanText(text: string): Partial<DiagnosticPlan> {
  try {
    const parsed = JSON.parse(text);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("plan 必须是 JSON 对象");
    }
    return parsed as Partial<DiagnosticPlan>;
  } catch (reason) {
    if (reason instanceof Error) throw new Error(`诊断计划 JSON 无效：${reason.message}`);
    throw new Error("诊断计划 JSON 无效");
  }
}

function summarizePlan(plan: Partial<DiagnosticPlan> | null): Array<[string, string]> {
  if (!plan) return [["来源", "等待后端模板"], ["环境数量", "1"]];
  return [
    ["模板", String(plan.template || "无")],
    ["环境数量", String(plan.num_envs ?? 1)],
    ["命令数", String(plan.commands?.length ?? 0)],
    ["地形", (plan.terrains ?? []).map((item) => `${item.type}:${item.level}`).join(", ") || "无"],
    ["DR", (plan.dr_cases ?? []).map((item) => item.label || `DR${item.level}`).join(", ") || "无"],
  ];
}

function newCommand(direction: CommandDirection, index: number): NonNullable<DiagnosticPlan["commands"]>[number] {
  return commandForDirection({
    id: `command_${index + 1}`,
    label: "Forward",
    mode: "forward",
    vx: 0.5,
    vy: 0,
    wz: 0,
    duration_s: 2,
    settle_s: 0,
    ramp_s: 0,
    repeats: 1,
  }, direction);
}

function commandForDirection(
  command: Partial<NonNullable<DiagnosticPlan["commands"]>[number]>,
  direction: CommandDirection,
  speed = defaultSpeedForDirection(direction),
): NonNullable<DiagnosticPlan["commands"]>[number] {
  const base = {
    id: command.id || direction,
    label: command.label || labelForCommandDirection(direction),
    mode: command.mode || direction,
    vx: numberParam(command.vx, 0),
    vy: numberParam(command.vy, 0),
    wz: numberParam(command.wz, 0),
    duration_s: numberParam(command.duration_s, 3),
    settle_s: numberParam(command.settle_s, 0),
    ramp_s: numberParam(command.ramp_s, 0),
    repeats: Math.round(numberParam(command.repeats, 1)),
  };
  const absSpeed = Math.abs(speed);
  if (direction === "custom") return { ...base, mode: "custom" };
  if (direction === "stand") return { ...base, label: "Stand", mode: "stand", vx: 0, vy: 0, wz: 0 };
  if (direction === "forward") return { ...base, label: "Forward", mode: "forward", vx: absSpeed, vy: 0, wz: 0 };
  if (direction === "backward") return { ...base, label: "Backward", mode: "backward", vx: -absSpeed, vy: 0, wz: 0 };
  if (direction === "left") return { ...base, label: "Left", mode: "lateral", vx: 0, vy: absSpeed, wz: 0 };
  if (direction === "right") return { ...base, label: "Right", mode: "lateral", vx: 0, vy: -absSpeed, wz: 0 };
  if (direction === "yaw_left") return { ...base, label: "Yaw left", mode: "yaw", vx: 0, vy: 0, wz: absSpeed };
  return { ...base, label: "Yaw right", mode: "yaw", vx: 0, vy: 0, wz: -absSpeed };
}

function directionForCommand(command: Partial<NonNullable<DiagnosticPlan["commands"]>[number]>): CommandDirection {
  const vx = numberParam(command.vx, 0);
  const vy = numberParam(command.vy, 0);
  const wz = numberParam(command.wz, 0);
  const mode = stringParam(command.mode, "").toLowerCase();
  if (Math.abs(vx) <= 0.1 && Math.abs(vy) <= 0.1 && Math.abs(wz) <= 0.1) return "stand";
  const activeAxes = [Math.abs(vx) > 0.1, Math.abs(vy) > 0.1, Math.abs(wz) > 0.1].filter(Boolean).length;
  if (activeAxes > 1 || mode === "custom" || mode === "mixed") return "custom";
  if (Math.abs(wz) > 0.1) return wz >= 0 ? "yaw_left" : "yaw_right";
  if (Math.abs(vy) > 0.1) return vy >= 0 ? "left" : "right";
  if (Math.abs(vx) > 0.1) return vx >= 0 ? "forward" : "backward";
  return "custom";
}

function speedForCommand(command: Partial<NonNullable<DiagnosticPlan["commands"]>[number]>, direction: CommandDirection) {
  if (direction === "stand") return 0;
  if (direction === "left" || direction === "right") return Math.abs(numberParam(command.vy, defaultSpeedForDirection(direction)));
  if (direction === "yaw_left" || direction === "yaw_right") return Math.abs(numberParam(command.wz, defaultSpeedForDirection(direction)));
  return Math.abs(numberParam(command.vx, defaultSpeedForDirection(direction)));
}

function defaultSpeedForDirection(direction: CommandDirection) {
  if (direction === "backward") return 0.4;
  if (direction === "left" || direction === "right") return 0.3;
  if (direction === "yaw_left" || direction === "yaw_right") return 0.6;
  if (direction === "stand") return 0;
  return 0.5;
}

function labelForCommandDirection(direction: CommandDirection) {
  const labels: Record<CommandDirection, string> = {
    stand: "Stand",
    forward: "Forward",
    backward: "Backward",
    left: "Left",
    right: "Right",
    yaw_left: "Yaw left",
    yaw_right: "Yaw right",
    custom: "Custom",
  };
  return labels[direction];
}

function buildDiagnosticOptions(report: DiagnosticReport | null, playback: DiagnosticPlayback | null): ChartOption[] {
  const options: ChartOption[] = [];
  if (report?.commands.length) {
    options.push(
      reportSeries("tracking_error", "跟踪误差", report.commands.map((item) => item.tracking_error), "报告"),
      reportSeries("stable_fraction", "稳定比例", report.commands.map((item) => item.stable_fraction), "报告", "0-1"),
      reportSeries("raw_progress", "原始位移", report.commands.map((item) => item.raw_progress_m), "报告", "m"),
      reportSeries("done_fraction", "终止比例", report.commands.map((item) => item.done_fraction), "报告", "0-1"),
    );
  }
  const frames = playback?.frames ?? [];
  if (frames.length) {
    options.push(
      frameSeries("base_x", "机身 X", frames, (frame) => frame.base_position[0], "m"),
      frameSeries("base_y", "机身 Y", frames, (frame) => frame.base_position[1], "m"),
      frameSeries("base_z", "机身高度", frames, (frame) => frame.base_position[2], "m"),
      frameSeries("terrain_height", "地形高度", frames, (frame) => frame.terrain_height, "m"),
      frameSeries("contact_count", "接触脚数", frames, (frame) => Object.values(frame.feet).filter((foot) => foot.contact).length, "feet"),
      frameSeries("reset_observed", "重置标记", frames, (frame) => frame.reset_observed ? 1 : 0),
      frameSeries("done", "结束标记", frames, (frame) => frame.done ? 1 : 0),
    );
    for (const leg of ["FL", "FR", "RL", "RR"]) {
      options.push(
        frameSeries(`${leg}_clearance`, `${leg} 离地`, frames, (frame) => frame.feet[leg]?.clearance, "m"),
        frameSeries(`${leg}_force`, `${leg} 足力`, frames, (frame) => frame.feet[leg]?.force_norm, "N"),
      );
    }
  }
  return options.filter((item) => countNumbers(item.values) > 0);
}

function buildLogOptions(telemetry: TrainingTelemetry | null): ChartOption[] {
  const history = telemetry?.history ?? [];
  if (!history.length) return [];
  const groups: Array<[TelemetryGroupKey, string]> = [
    ["reward", "奖励"],
    ["command", "命令/步态"],
    ["curriculum", "课程"],
    ["health", "健康"],
    ["counters", "计数"],
  ];
  const options: ChartOption[] = [];
  for (const [group, groupLabel] of groups) {
    for (const key of numericKeys(history, group)) {
      const optionKey = `${group}.${key}`;
      if (isRedundantLogField(optionKey)) continue;
      options.push({
        key: optionKey,
        label: labelForField(optionKey, groupLabel),
        values: history.map((point) => finiteOrNull(recordFor(point, group)[key])),
        group: groupLabel,
        unit: unitForField(optionKey),
        description: optionKey,
      });
    }
  }
  return options
    .filter((item) => countNumbers(item.values) > 1 && hasSignal(item.values))
    .sort((a, b) => rankField(a.key) - rankField(b.key) || a.label.localeCompare(b.label));
}

function frameSeries(
  key: string,
  label: string,
  frames: DiagnosticPlaybackFrame[],
  pick: (frame: DiagnosticPlaybackFrame) => number | null | undefined,
  unit = "",
): ChartOption {
  return { key, label, values: frames.map((frame) => finiteOrNull(pick(frame))), group: "回放", unit, description: key };
}

function reportSeries(key: string, label: string, values: Array<number | null>, group: string, unit = ""): ChartOption {
  return { key, label, values: values.map(finiteOrNull), group, unit, description: key };
}

function numericKeys(history: TrainingTelemetryPoint[], group: TelemetryGroupKey) {
  const keys = new Set<string>();
  for (const point of history) {
    for (const [key, value] of Object.entries(recordFor(point, group))) {
      if (typeof value === "number" && Number.isFinite(value)) keys.add(key);
    }
  }
  return [...keys].sort();
}

function recordFor(point: TrainingTelemetryPoint, group: TelemetryGroupKey): Record<string, unknown> {
  return point[group] as Record<string, unknown>;
}

function labelForField(key: string, fallbackGroup: string) {
  const labels: Record<string, string> = {
    "reward.total": "总奖励",
    "command.cmd_vx": "目标前向速度",
    "command.actual_vx": "实际前向速度",
    "command.v_along": "沿命令速度",
    "command.lin_err": "线速度误差",
    "command.gait_match": "步态匹配",
    "command.stance_slip_high_fraction": "高滑移比例",
    "curriculum.progress_gate": "课程推进 gate",
    "curriculum.terrain_mean": "平均地形等级",
    "curriculum.terrain_max": "最高地形等级",
    "health.fall_rate": "摔倒率",
    "health.base_h": "机身高度",
    "health.tilt_deg": "倾角",
  };
  return labels[key] ?? `${fallbackGroup}.${key.split(".").slice(1).join(".")}`;
}

function unitForField(key: string) {
  if (/(vx|vy|speed|v_along|lin_err)/.test(key)) return "m/s";
  if (/(wz|yaw_err)/.test(key)) return "rad/s";
  if (/tilt_deg/.test(key)) return "deg";
  if (/(base_h|height)/.test(key)) return "m";
  if (/(terrain|level|dr_level)/.test(key)) return "level";
  if (/(rate|gate|ratio|fraction|match|slip|progress|score)/.test(key)) return "0-1";
  return "";
}

function rankField(key: string) {
  const order = DEFAULT_LOG_FIELDS.indexOf(key);
  if (order >= 0) return order;
  if (key.startsWith("reward.")) return 100;
  if (key.startsWith("command.")) return 200;
  if (key.startsWith("curriculum.")) return 300;
  if (key.startsWith("health.")) return 400;
  return 500;
}

function isRedundantLogField(key: string) {
  return /^reward\..*_gate$/.test(key) || /^command\.duty_(fl|fr|rl|rr)$/i.test(key);
}

function toSeries(option: ChartOption, index: number, dashed: boolean) {
  return {
    values: option.values,
    color: chartColor(index),
    label: `${option.label}${option.unit ? ` ${option.unit}` : ""}`,
    width: index === 0 ? 2.2 : 1.7,
    dashed,
  };
}

function toggleList(current: string[], key: string, onChange: (next: string[]) => void, maxItems: number) {
  onChange(current.includes(key) ? current.filter((item) => item !== key) : [...current, key].slice(-maxItems));
}

function normalizeSeries(values: Array<number | null>) {
  const numbers = values.filter(isNumber);
  if (numbers.length < 2) return values;
  const min = Math.min(...numbers);
  const max = Math.max(...numbers);
  const span = max - min;
  if (!Number.isFinite(span) || span < 1e-12) return values.map((value) => value == null ? null : 0.5);
  return values.map((value) => value == null ? null : (value - min) / span);
}

function hasSignal(values: Array<number | null>) {
  const numbers = values.filter(isNumber);
  if (numbers.length < 2) return false;
  const first = numbers[0];
  return numbers.some((value) => Math.abs(value - first) > 1e-9);
}

function countNumbers(values: Array<number | null>) {
  return values.filter(isNumber).length;
}

function finiteOrNull(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function isNumber(value: number | null): value is number {
  return typeof value === "number";
}

function clampNumberInput(value: string, min: number, max: number, fallback: number) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

function clampIntInput(value: string, min: number, max: number, fallback: number) {
  return Math.round(clampNumberInput(value, min, max, fallback));
}

function numberParam(value: unknown, fallback: number) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function stringParam(value: unknown, fallback: string) {
  return typeof value === "string" && value ? value : fallback;
}

function prettyJson(value: unknown) {
  return JSON.stringify(value ?? {}, null, 2);
}

function stablePlanKey(value: unknown) {
  return JSON.stringify(value ?? null);
}

function formatNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "无";
}

function chartColor(index: number) {
  return ["#0066cc", "#0a7f41", "#b42318", "#936100", "#7c3aed", "#0f766e", "#c2410c", "#475569"][index % 8];
}
