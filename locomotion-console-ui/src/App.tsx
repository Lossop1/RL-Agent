import { Component, useEffect, useMemo, useState, type ErrorInfo, type ReactNode } from "react";
import {
  cancelChatProposal,
  executeAction,
  getAgentWorkbench,
  getTrainingTelemetry,
  postAction,
  type AgentWorkbenchActionInfo,
  type AgentWorkbenchEvidenceInfo,
  type AgentWorkbenchInfo,
  type ChatProposalInfo,
  type TrainingTelemetry,
  type TrainingTelemetryPoint,
} from "./api";
import Chat, { INITIAL_CHAT_MESSAGES, type ChatMessage } from "./Chat";
import ConfigWorkspace from "./ConfigWorkspace";
import Diagnostics from "./Diagnostics";
import Scoreboard from "./Scoreboard";
import TraditionalControlDemo from "./TraditionalControlDemo";
import { formatError } from "./i18n/format";
import LineChart from "./LineChart";

type ToolView = "agent" | "scoreboard" | "diagnostics" | "traditional-control" | "config";
type DirectAction = "deploy-payload" | "start" | "resume" | "kill";

const WORKBENCH_POLL_MS = 5000;

class DiagnosticsErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Diagnostics page rendering failed", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <section className="primary-panel tool-panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">物理诊断</p>
            <h1>诊断页面暂时无法渲染</h1>
          </div>
        </div>
        <div className="inline-alert">诊断数据仍保存在远端，当前只是页面渲染失败。</div>
        <button className="secondary-button" onClick={() => this.setState({ error: null })}>
          重新加载诊断页面
        </button>
      </section>
    );
  }
}

class AgentWorkbenchErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Agent workbench rendering failed", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <main className="agent-layout loading-layout">
        <section className="primary-panel">
          <p className="eyebrow">智能体工作台</p>
          <h1>工作台暂时无法渲染</h1>
          <p>服务上下文已建立，但部分遥测或工作台数据不完整，页面已保留在当前工作区。</p>
          <div className="inline-alert" role="alert">
            {this.state.error.message || "未知页面错误"}
          </div>
          <button className="secondary-button" onClick={() => window.location.reload()}>
            重新加载页面
          </button>
        </section>
      </main>
    );
  }
}

const AGENT_TREND_PRESETS = [
  {
    id: "phase",
    label: "阶段",
    description: "阶段推进是否被真实能力支撑。",
    keys: [
      "curriculum.progress_gate",
      "curriculum.gait_gate",
      "curriculum.diagonal_gate",
      "curriculum.duty_balance_gate",
      "curriculum.slip_gate",
      "curriculum.progress_yaw",
    ],
  },
  {
    id: "terrain",
    label: "地形",
    description: "真实地形、离散地形和地形事件有没有推进。",
    keys: [
      "curriculum.terrain_real_mean",
      "curriculum.terrain_discrete_mean",
      "curriculum.terrain_discrete_max",
      "reward.terrain_probe_event_scope_mean",
      "reward.terrain_contact_quality",
      "reward.terrain_event_collapse",
    ],
  },
  {
    id: "gait",
    label: "步态",
    description: "支撑、对称、滑移和落脚是否干净。",
    keys: [
      "command.gait_match",
      "command.diagonal_contact",
      "command.duty_balance",
      "command.stance_slip_high_fraction",
      "reward.landing_impact",
      "reward.touchdown_slip",
    ],
  },
  {
    id: "stability",
    label: "稳定",
    description: "机身摇晃、倾斜、低高度和终止风险。",
    keys: [
      "reward.base_wxy",
      "health.tilt_deg",
      "health.support_instability",
      "health.base_h_min",
      "health.fall_rate",
      "health.terminal_rate",
    ],
  },
] as const;

type TrendScaleMode = "normalized" | "raw";
type AgentTrendPresetId = typeof AGENT_TREND_PRESETS[number]["id"];
type TelemetryGroupKey = "reward" | "curriculum" | "health" | "command" | "counters";

interface AgentTrendOption {
  key: string;
  label: string;
  values: Array<number | null>;
  group: string;
  unit?: string;
}

interface GateRow {
  label: string;
  condition: string;
  value: unknown;
  target: unknown;
  targetSource?: string;
  direction: "higher" | "lower";
  gap: number | null;
  pass: boolean | null;
}

export default function App() {
  const [view, setView] = useState<ToolView>("agent");
  const [visitedViews, setVisitedViews] = useState<Record<ToolView, boolean>>({
    agent: true,
    scoreboard: false,
    diagnostics: false,
    "traditional-control": false,
    config: false,
  });
  const [workbench, setWorkbench] = useState<AgentWorkbenchInfo | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [context, setContext] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(INITIAL_CHAT_MESSAGES);
  const [trainingTelemetry, setTrainingTelemetry] = useState<TrainingTelemetry | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;
    async function poll() {
      try {
        const next = await getAgentWorkbench();
        if (!cancelled) {
          setWorkbench(next);
          setError("");
        }
      } catch (reason) {
        if (!cancelled) setError(formatError(reason));
      } finally {
        if (!cancelled) timer = window.setTimeout(poll, WORKBENCH_POLL_MS);
      }
    }
    void poll();
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer: number | null = null;
    async function pollTelemetry() {
      try {
        const next = await getTrainingTelemetry();
        if (!cancelled) setTrainingTelemetry(next);
      } catch {
        if (!cancelled) setTrainingTelemetry(null);
      } finally {
        if (!cancelled) timer = window.setTimeout(pollTelemetry, WORKBENCH_POLL_MS);
      }
    }
    void pollTelemetry();
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    if (!workbench) return;
    const attempt = workbench.attempt;
    const evidence = arrayValue<AgentWorkbenchEvidenceInfo>(workbench.evidence);
    const values = evidence.find((item) => item.id === "telemetry")?.values ?? {};
    const curriculum = recordValue(values.curriculum);
    const command = recordValue(values.command);
    const gait = recordValue(values.gait);
    const health = recordValue(values.health);
    const reward = recordValue(values.reward);
    setContext([
      "page=agent_workbench",
      `run=${attempt.run_id || "unknown"}`,
      `state=${attempt.runtime_state}`,
      `step=${attempt.step}`,
      `phase=${attempt.phase || "unknown"}`,
      `blocked_by=${attempt.blocked_by || "none"}`,
      `judgement=${workbench.judgement.title}`,
      `cmd_vx=${values.cmd_vx ?? "unknown"}`,
      `actual_vx=${values.actual_vx ?? "unknown"}`,
      `progress=${values.progress_gate ?? values.progress_ratio ?? "unknown"}`,
      `progress_fwd=${curriculum.progress_fwd ?? "unknown"}`,
      `progress_back=${curriculum.progress_back ?? "unknown"}`,
      `progress_lat=${curriculum.progress_lat ?? "unknown"}`,
      `progress_yaw=${curriculum.progress_yaw ?? "unknown"}`,
      `terrain_real=${curriculum.terrain_real_mean ?? values.terrain_mean ?? "unknown"}`,
      `terrain_discrete=${curriculum.terrain_discrete_mean ?? "unknown"}`,
      `dr_level=${values.dr_level ?? "unknown"}`,
      `gait=${gait.gait_match ?? "unknown"}`,
      `diag=${gait.diagonal_contact ?? "unknown"}`,
      `duty=${gait.duty_balance ?? "unknown"}`,
      `slip=${gait.stance_slip ?? command.stance_slip ?? "unknown"}`,
      `high_slip=${gait.stance_slip_high_fraction ?? command.stance_slip_high_fraction ?? "unknown"}`,
      `fall=${health.fall_rate ?? "unknown"}`,
      `base_wxy=${reward.base_wxy ?? "unknown"}`,
      `landing_impact=${reward.landing_impact ?? "unknown"}`,
      `touchdown_slip=${reward.touchdown_slip ?? "unknown"}`,
    ].join(" | "));
  }, [workbench]);

  async function refreshWorkbench() {
    setBusy("refresh");
    try {
      const [workbenchResult, telemetryResult] = await Promise.allSettled([
        getAgentWorkbench(),
        getTrainingTelemetry(),
      ]);
      if (workbenchResult.status === "fulfilled") {
        setWorkbench(workbenchResult.value);
        setError("");
      } else {
        setError(formatError(workbenchResult.reason));
      }
      if (telemetryResult.status === "fulfilled") {
        setTrainingTelemetry(telemetryResult.value);
      }
    } finally {
      setBusy(null);
    }
  }

  async function runDirectAction(action: DirectAction) {
    const confirmed = window.confirm(confirmTextForAction(action));
    if (!confirmed) return;
    setBusy(action);
    try {
      const result = await postAction(action);
      if (!result.ok) setError(result.message || `${action} 执行失败`);
      await refreshWorkbench();
    } catch (reason) {
      setError(formatError(reason));
    } finally {
      setBusy(null);
    }
  }

  async function confirmProposal(proposal: ChatProposalInfo) {
    setBusy(`proposal:${proposal.name}`);
    try {
      const response = await executeAction(proposal.name, proposal.args);
      if (!response.ok) setError(response.detail || "proposal 执行失败");
      await refreshWorkbench();
    } catch (reason) {
      setError(formatError(reason));
    } finally {
      setBusy(null);
    }
  }

  async function cancelProposal(proposal: ChatProposalInfo) {
    setBusy(`cancel:${proposal.name}`);
    try {
      await cancelChatProposal(proposal.name);
      await refreshWorkbench();
    } catch (reason) {
      setError(formatError(reason));
    } finally {
      setBusy(null);
    }
  }

  const pendingProposals = useMemo(
    () => arrayValue<ChatProposalInfo>(workbench?.proposals).filter((item) => item.status === "pending"),
    [workbench],
  );

  function openView(nextView: ToolView) {
    setView(nextView);
    setVisitedViews((current) => ({ ...current, [nextView]: true }));
  }

  return (
    <div className="product-shell">
      <header className="product-topbar">
        <div className="product-brand">
          <span className="brand-glyph">策</span>
          <div>
            <strong>运动策略智能体</strong>
            <span>训练迭代工作台</span>
          </div>
        </div>
        <nav className="product-nav" aria-label="工作区">
          <button className={view === "agent" ? "active" : ""} onClick={() => openView("agent")}>智能体</button>
          <button className={view === "scoreboard" ? "active" : ""} onClick={() => openView("scoreboard")}>目标记分牌</button>
          <button className={view === "diagnostics" ? "active" : ""} onClick={() => openView("diagnostics")}>诊断工具</button>
          <button className={view === "traditional-control" ? "active" : ""} onClick={() => openView("traditional-control")}>传统控制</button>
          <button className={view === "config" ? "active" : ""} onClick={() => openView("config")}>配置工具</button>
        </nav>
        <button className="secondary-button" disabled={busy === "refresh"} onClick={() => void refreshWorkbench()}>
          刷新状态
        </button>
      </header>

      {error && (
        <section className="system-alert" role="alert">
          <strong>服务状态读取失败</strong>
          <span>{error}</span>
        </section>
      )}

      <div className="workspace-keepalive">
        <div className="workspace-pane" hidden={view !== "agent"}>
          <AgentWorkbenchErrorBoundary>
            <AgentWorkbench
              workbench={workbench}
              busy={busy}
              pendingProposals={pendingProposals}
              onDirectAction={runDirectAction}
              onConfirmProposal={confirmProposal}
              onCancelProposal={cancelProposal}
              onOpenDiagnostics={() => openView("diagnostics")}
              trainingTelemetry={trainingTelemetry}
              chatContext={context}
              chatMessages={chatMessages}
              setChatMessages={setChatMessages}
            />
          </AgentWorkbenchErrorBoundary>
        </div>
        {visitedViews.scoreboard && (
          <main className="tool-host workspace-pane" hidden={view !== "scoreboard"}>
            <Scoreboard active={view === "scoreboard"} />
          </main>
        )}
        {visitedViews.diagnostics && (
          <main className="tool-host workspace-pane" hidden={view !== "diagnostics"}>
            <DiagnosticsErrorBoundary>
              <Diagnostics active={view === "diagnostics"} />
            </DiagnosticsErrorBoundary>
          </main>
        )}
        {visitedViews["traditional-control"] && (
          <main className="tool-host workspace-pane" hidden={view !== "traditional-control"}>
            <TraditionalControlDemo active={view === "traditional-control"} />
          </main>
        )}
        {visitedViews.config && (
          <main className="tool-host workspace-pane" hidden={view !== "config"}>
            <ConfigWorkspace />
          </main>
        )}
      </div>
    </div>
  );
}

function AgentWorkbench({
  workbench,
  busy,
  pendingProposals,
  onDirectAction,
  onConfirmProposal,
  onCancelProposal,
  onOpenDiagnostics,
  trainingTelemetry,
  chatContext,
  chatMessages,
  setChatMessages,
}: {
  workbench: AgentWorkbenchInfo | null;
  busy: string | null;
  pendingProposals: ChatProposalInfo[];
  onDirectAction: (action: DirectAction) => Promise<void>;
  onConfirmProposal: (proposal: ChatProposalInfo) => Promise<void>;
  onCancelProposal: (proposal: ChatProposalInfo) => Promise<void>;
  onOpenDiagnostics: () => void;
  trainingTelemetry: TrainingTelemetry | null;
  chatContext: string;
  chatMessages: ChatMessage[];
  setChatMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
}) {
  if (!workbench) {
    return (
      <main className="agent-layout loading-layout">
        <section className="primary-panel">
          <p className="eyebrow">正在建立服务上下文</p>
          <h1>读取智能体工作台状态</h1>
          <p>后端会聚合模型、训练、遥测、配置和提案。前端不会自行推断训练结论。</p>
        </section>
      </main>
    );
  }

  return (
    <main className="agent-layout">
      <section className="agent-main">
        <ObjectivePanel workbench={workbench} />
        <RunControlPanel
          workbench={workbench}
          busy={busy}
          onDirectAction={onDirectAction}
        />
        <TelemetryPanel workbench={workbench} trainingTelemetry={trainingTelemetry} />
        <JudgementPanel workbench={workbench} />
        <EvidencePanel evidence={workbench.evidence} />
      </section>
      <aside className="agent-side">
        <Chat
          mode="training"
          context={chatContext}
          messages={chatMessages}
          setMessages={setChatMessages}
        />
        <ActionPanel
          actions={workbench.actions}
          busy={busy}
          onOpenDiagnostics={onOpenDiagnostics}
        />
        <ProposalPanel
          proposals={pendingProposals}
          busy={busy}
          onConfirm={onConfirmProposal}
          onCancel={onCancelProposal}
        />
        <ProcessLog workbench={workbench} busy={busy} />
      </aside>
    </main>
  );
}

function ObjectivePanel({ workbench }: { workbench: AgentWorkbenchInfo }) {
  const attempt = workbench.attempt;
  return (
    <section className="primary-panel hero-panel flat-panel">
      <div className="workspace-title">
        <p className="eyebrow">目标</p>
        <h1>{workbench.objective}</h1>
      </div>
      <dl className="attempt-bar">
        <div><dt>状态</dt><dd>{attemptStateLabel(attempt.running, attempt.runtime_state)}</dd></div>
        <div><dt>步数</dt><dd>{attempt.step}{attempt.total_steps ? ` / ${attempt.total_steps}` : ""}</dd></div>
        <div><dt>阶段</dt><dd>{attempt.phase || "未知"}</dd></div>
        <div><dt>阻塞</dt><dd>{attempt.blocked_by || "无"}</dd></div>
        <div><dt>检查点</dt><dd title={attempt.latest_checkpoint}>{shortPath(attempt.latest_checkpoint)}</dd></div>
      </dl>
    </section>
  );
}

function JudgementPanel({ workbench }: { workbench: AgentWorkbenchInfo }) {
  const judgement = workbench.judgement;
  const gaps = arrayValue<string>(judgement.gaps);
  const notes = arrayValue<string>(workbench.notes);
  return (
    <section className={`primary-panel judgement-panel flat-panel ${judgement.status}`}>
      <div className="panel-heading">
        <div>
          <p className="eyebrow">智能体判断</p>
          <h2>{judgement.title}</h2>
        </div>
        <span className={`confidence ${judgement.confidence}`}>可信度 {confidenceLabel(judgement.confidence)}</span>
      </div>
      <p className="judgement-summary">{judgement.summary}</p>
      {gaps.length > 0 && (
        <div className="gap-list">
          <strong>证据缺口</strong>
          {gaps.map((gap) => <span key={gap}>{gap}</span>)}
        </div>
      )}
      <div className="note-list">
        {notes.map((note) => <span key={note}>{note}</span>)}
      </div>
    </section>
  );
}

function RunControlPanel({
  workbench,
  busy,
  onDirectAction,
}: {
  workbench: AgentWorkbenchInfo;
  busy: string | null;
  onDirectAction: (action: DirectAction) => Promise<void>;
}) {
  const attempt = workbench.attempt;
  const actions = arrayValue<AgentWorkbenchActionInfo>(workbench.actions);
  const evidence = arrayValue<AgentWorkbenchEvidenceInfo>(workbench.evidence);
  const actionsById = new Map(actions.map((action) => [action.id, action]));
  const remote = evidence.find((item) => item.id === "remote");
  const telemetry = evidence.find((item) => item.id === "telemetry");
  const stateReliable = attempt.remote_ok
    && !["stale", "remote_unavailable", "unknown"].includes(attempt.runtime_state);
  const controls: Array<{ id: string; action: DirectAction; label: string; tone: "primary" | "secondary" | "danger" }> = [
    { id: "deploy-payload", action: "deploy-payload", label: "部署当前包", tone: "secondary" },
    { id: "start-training", action: "start", label: "启动全新训练", tone: "primary" },
    { id: "resume-training", action: "resume", label: "从检查点继续", tone: "secondary" },
    { id: "kill-training", action: "kill", label: "停止训练", tone: "danger" },
  ];
  return (
    <section className="primary-panel run-control-panel flat-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">运行控制</p>
          <h2>{attempt.running ? "训练正在运行" : "训练未运行"}</h2>
        </div>
        <span className={`run-state ${attempt.running ? "running" : "idle"}`}>
          {attempt.runtime_state || "unknown"}
        </span>
      </div>
      <div className="run-control-grid">
        {controls.map((control) => {
          const info = actionsById.get(control.id);
          const className = control.tone === "primary"
            ? "primary-button"
            : control.tone === "danger"
              ? "danger-button"
              : "secondary-button";
          return (
            <button
              key={control.id}
              className={className}
              disabled={Boolean(busy) || info?.enabled === false || (!stateReliable && control.action !== "kill")}
              title={info?.reason || control.label}
              onClick={() => void onDirectAction(control.action)}
            >
              {busy === control.action ? "执行中" : control.label}
            </button>
          );
        })}
      </div>
      <dl className="run-control-facts">
        <div><dt>当前 run</dt><dd>{attempt.run_id || "无"}</dd></div>
        <div><dt>远端</dt><dd>{remote?.detail || evidenceStatusLabel(remote?.status || "unknown")}</dd></div>
        <div><dt>遥测</dt><dd>{telemetry?.detail || evidenceStatusLabel(telemetry?.status || "unknown")}</dd></div>
        <div><dt>检查点</dt><dd title={attempt.latest_checkpoint}>{shortPath(attempt.latest_checkpoint)}</dd></div>
      </dl>
      {!stateReliable && (
        <p className="control-safety-note" role="status">
          当前运行状态不可信，已暂停启动、继续和部署。恢复远端连接并取得新遥测后再操作。
        </p>
      )}
      <p className="compact-copy">“启动全新训练”会创建新 run，不带 checkpoint；“从检查点继续”会自动解析最新可用 checkpoint。</p>
    </section>
  );
}

function ProposalPanel({
  proposals,
  busy,
  onConfirm,
  onCancel,
}: {
  proposals: ChatProposalInfo[];
  busy: string | null;
  onConfirm: (proposal: ChatProposalInfo) => Promise<void>;
  onCancel: (proposal: ChatProposalInfo) => Promise<void>;
}) {
  return (
    <section className="side-panel compact-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">待授权提案</p>
          <h2>{proposals.length ? `${proposals.length} 个待确认` : "无待确认动作"}</h2>
        </div>
      </div>
      {proposals.length === 0 ? (
        <p className="muted compact-copy">智能体生成需要授权的动作后，会在这里确认或取消。</p>
      ) : (
        <div className="proposal-table">
          {proposals.map((proposal) => (
            <div className="proposal-row" key={proposal.id}>
              <div>
                <strong>{proposal.name}</strong>
                <span>{proposal.expected_result || proposal.reply || "等待确认"}</span>
              </div>
              <span className={`risk-pill ${proposal.risk}`}>风险 {riskLabel(proposal.risk)}</span>
              <div className="row-actions">
                <button
                  className="primary-button"
                  disabled={Boolean(busy)}
                  onClick={() => void onConfirm(proposal)}
                >
                  确认执行
                </button>
                <button
                  className="secondary-button"
                  disabled={Boolean(busy)}
                  onClick={() => void onCancel(proposal)}
                >
                  取消
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function EvidencePanel({ evidence }: { evidence: AgentWorkbenchEvidenceInfo[] }) {
  const visibleEvidence = arrayValue<AgentWorkbenchEvidenceInfo>(evidence).filter((item) => item.id !== "telemetry");
  return (
    <section className="primary-panel flat-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">证据</p>
          <h2>数据来源状态</h2>
        </div>
      </div>
      <div className="evidence-table">
        {visibleEvidence.map((item) => (
          <div className={`evidence-row ${item.status}`} key={item.id}>
            <strong>{item.label}</strong>
            <span>{evidenceStatusLabel(item.status)}</span>
            <p>{item.detail || "无详情"}</p>
            <code>{item.source}</code>
          </div>
        ))}
      </div>
    </section>
  );
}

function TelemetryPanel({
  workbench,
  trainingTelemetry,
}: {
  workbench: AgentWorkbenchInfo;
  trainingTelemetry: TrainingTelemetry | null;
}) {
  const evidence = arrayValue<AgentWorkbenchEvidenceInfo>(workbench.evidence);
  const telemetry = evidence.find((item) => item.id === "telemetry");
  const values = telemetry?.values ?? {};
  const latestPoint = trainingTelemetry?.latest ?? null;
  const [trendPresetId, setTrendPresetId] = useState<AgentTrendPresetId>("phase");
  const [trendScale, setTrendScale] = useState<TrendScaleMode>("normalized");
  const trendOptions = useMemo(() => buildAgentTrendOptions(trainingTelemetry), [trainingTelemetry]);
  const trendPreset = AGENT_TREND_PRESETS.find((item) => item.id === trendPresetId) ?? AGENT_TREND_PRESETS[0];
  const selectedTrendOptions = trendPreset.keys
    .map((key) => trendOptions.find((option) => option.key === key))
    .filter((option): option is AgentTrendOption => Boolean(option));
  const trendSeries = selectedTrendOptions.map((option, index) => ({
    values: trendScale === "normalized" ? normalizeTrend(option.values) : option.values,
    color: trendColor(index),
    label: `${option.label}${trendScale === "raw" && option.unit ? ` ${option.unit}` : ""}`,
    width: index === 0 ? 2.2 : 1.7,
  }));
  const timeline = {
    ...recordValue(values.timeline),
    step: latestPoint?.step ?? recordValue(values.timeline).step,
    total_steps: latestPoint?.total_steps ?? recordValue(values.timeline).total_steps,
    elapsed: latestPoint?.elapsed ?? recordValue(values.timeline).elapsed,
    eta: latestPoint?.eta ?? recordValue(values.timeline).eta,
    fps: latestPoint?.fps ?? recordValue(values.timeline).fps,
  };
  const curriculum = { ...recordValue(values.curriculum), ...recordValue(latestPoint?.curriculum) };
  const command = { ...recordValue(values.command), ...recordValue(latestPoint?.command) };
  const gait = {
    ...recordValue(values.gait),
    gait_match: latestPoint?.command?.gait_match ?? recordValue(values.gait).gait_match,
    diagonal_contact: latestPoint?.command?.diagonal_contact ?? recordValue(values.gait).diagonal_contact,
    duty_balance: latestPoint?.command?.duty_balance ?? recordValue(values.gait).duty_balance,
    duty_spread_window: latestPoint?.command?.duty_spread_window ?? recordValue(values.gait).duty_spread_window,
    stance_slip: latestPoint?.command?.stance_slip ?? recordValue(values.gait).stance_slip,
    stance_slip_high_fraction: latestPoint?.command?.stance_slip_high_fraction ?? recordValue(values.gait).stance_slip_high_fraction,
  };
  const health = { ...recordValue(values.health), ...recordValue(latestPoint?.health) };
  const reward = { ...recordValue(values.reward), ...recordValue(latestPoint?.reward) };
  const paths = { ...recordValue(values.paths), ...recordValue(latestPoint?.paths) };
  const step = numberOrNull(timeline.step ?? values.step ?? workbench.attempt.step);
  const total = numberOrNull(timeline.total_steps ?? values.total_steps ?? workbench.attempt.total_steps);
  const progressPct = total && step !== null ? Math.max(0, Math.min(100, (step / total) * 100)) : null;
  const phaseGate = recordValue(curriculum.phase_gate);
  const runtimeBlockers = Array.isArray(phaseGate.runtime_blockers)
    ? phaseGate.runtime_blockers.map((item: unknown) => textValue(item)).filter(Boolean).join(",")
    : "";
  const blockedBy = textValue(
    curriculum.phase_gate_blockers
      ?? curriculum.blocked_by
      ?? values.blocked_by
      ?? workbench.attempt.blocked_by,
  ) || runtimeBlockers;
  const telemetryAge = trainingTelemetry?.telemetry_age_s ?? values.telemetry_age_s ?? workbench.attempt.telemetry_age_s;
  const telemetryStale = trainingTelemetry?.stale ?? values.stale;
  const status = telemetryStale
    ? "warning"
    : trainingTelemetry?.available
      ? "ok"
      : (telemetry?.status ?? "unknown");
  const freshnessNotice = telemetryStale
    ? `遥测已过期（${formatAge(telemetryAge)}）；以下数值是最后已知快照，不代表当前训练状态。`
    : status !== "ok"
      ? "当前没有可用遥测，页面不会用零值代替缺失数据。"
      : "";

  return (
    <section className={`primary-panel telemetry-panel flat-panel ${status}`}>
      <div className="panel-heading telemetry-heading">
        <div>
          <p className="eyebrow">当前遥测</p>
          <h2>{telemetryStale ? "最后已知训练快照" : status === "ok" ? "训练状态快照" : "遥测不可用"}</h2>
        </div>
        <div className="telemetry-state">
          <span className={`run-state ${workbench.attempt.running ? "running" : "idle"}`}>
            {attemptStateLabel(workbench.attempt.running, workbench.attempt.runtime_state)}
          </span>
          <span className={`evidence-pill ${status}`}>{evidenceStatusLabel(status)}</span>
        </div>
      </div>

      {freshnessNotice && <p className={`telemetry-notice ${status}`} role="status">{freshnessNotice}</p>}

      <div className="telemetry-hero">
        <MetricCell label="步数" value={formatStep(step, total)} />
        <MetricCell label="阶段" value={textValue(curriculum.phase ?? workbench.attempt.phase) || "未知"} />
        <MetricCell label="阻塞" value={blockedBy || "无"} tone={blockedBy ? "warn" : "good"} />
        <MetricCell label="时效" value={formatAge(telemetryAge)} tone={telemetryStale ? "warn" : "neutral"} />
      </div>

      {progressPct !== null && (
        <div className="telemetry-progress" aria-label="训练进度">
          <span style={{ width: `${progressPct}%` }} />
        </div>
      )}

      <div className="telemetry-dashboard">
        <PhaseGateDetails curriculum={curriculum} command={command} gait={gait} health={health} />
        <DirectionProgressDetails curriculum={curriculum} />
        <TerrainDetails curriculum={curriculum} reward={reward} />
        <QualityDetails command={command} gait={gait} />
        <StabilityDetails health={health} reward={reward} />
      </div>

      <details className="telemetry-trends">
        <summary>趋势</summary>
        <TrendDetails
          preset={trendPreset}
          series={trendSeries}
          selectedCount={selectedTrendOptions.length}
          hasHistory={Boolean(trainingTelemetry?.history?.length)}
          scale={trendScale}
          onScaleChange={setTrendScale}
        />
        <div className="trend-preset-tabs" aria-label="选择趋势">
          {AGENT_TREND_PRESETS.map((preset) => (
            <button
              key={preset.id}
              className={preset.id === trendPresetId ? "active" : ""}
              onClick={() => setTrendPresetId(preset.id)}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </details>

      <details className="telemetry-meta">
        <summary>路径与采样</summary>
        <div className="telemetry-footer">
          <div className="telemetry-paths">
            <PathLine label="JSONL" value={textValue(paths.telemetry_jsonl ?? values.telemetry_path)} />
            <PathLine label="日志" value={textValue(paths.train_log ?? values.log_path)} />
            <PathLine label="配置" value={textValue(paths.effective_config)} />
          </div>
          <div className="telemetry-mini">
            <span>mode {textValue(values.mode) || "unknown"}</span>
            <span>history {formatNumberLike(values.history_points)}</span>
            <span>fps {formatNumberLike(timeline.fps ?? values.fps)}</span>
            <span>eta {textValue(timeline.eta ?? values.eta) || "无"}</span>
          </div>
        </div>
      </details>
    </section>
  );
}

function PhaseGateDetails({
  curriculum,
  command,
  gait,
  health,
}: {
  curriculum: Record<string, any>;
  command: Record<string, any>;
  gait: Record<string, any>;
  health: Record<string, any>;
}) {
  const gate = recordValue(curriculum.phase_gate);
  const conditions = recordValue(gate.conditions);
  const conditionsSource = recordValue(gate.conditions_source);
  const phaseIndex = numberOrNull(gate.phase_index);
  const terrainStartPhase = numberOrNull(conditions.terrain_start_phase);
  const terrainPhase = phaseIndex !== null && terrainStartPhase !== null && phaseIndex >= terrainStartPhase;
  const commandMode = textValue(curriculum.command_mode);
  const mixedPhase = commandMode === "mixed";
  const activeDirs = textValue(curriculum.active_dirs).split(",").map((item) => item.trim()).filter(Boolean);
  const progressFormula = progressGateFormula(curriculum, activeDirs);
  const rows = [
    gateRow("方向推进", progressFormula, curriculum.phase_gate_progress_value ?? curriculum.progress_gate ?? command.progress_ratio, conditions.progress_min, "higher", conditionsSource.progress_min),
    gateRow("执行覆盖", "当前阶段所有参与方向的最小有效执行覆盖", curriculum.phase_gate_execution_value ?? curriculum.execution_gate, conditions.execution_min, "higher", conditionsSource.execution_min),
    gateRow(
      "支撑滑移",
      terrainPhase ? "地形阶段仍要求整体支撑滑移不过高" : "平地阶段要求支撑期滑移不过高",
      curriculum.phase_gate_slip_value ?? gait.stance_slip ?? command.stance_slip ?? curriculum.slip_gate,
      terrainPhase ? conditions.terrain_slip_max : conditions.slip_max,
      "lower",
      terrainPhase ? conditionsSource.terrain_slip_max : conditionsSource.slip_max,
    ),
  ];
  if (terrainPhase) {
    rows.push(
      gateRow("平地对角支撑", "只在平地样本上统计，地形样本不会被强套直线模板", curriculum.phase_gate_flat_diagonal_value ?? command.flat_diagonal_contact, conditions.diagonal_min, "higher", conditionsSource.diagonal_min),
      gateRow("平地 duty 区间", "四腿 duty 落在允许区间内", curriculum.phase_gate_flat_duty_target_value ?? command.flat_duty_target, conditions.duty_target_min, "higher", conditionsSource.duty_target_min),
      gateRow("平地 duty 对称", "左右、前后和交叉支撑时间保持对称", curriculum.phase_gate_flat_duty_symmetry_value ?? command.flat_duty_symmetry, conditions.duty_symmetry_min, "higher", conditionsSource.duty_symmetry_min),
      gateRow("平地 duty 覆盖", "各线性方向都完成足够多的真实接触周期", curriculum.phase_gate_flat_duty_valid_value ?? command.flat_duty_valid, conditions.duty_valid_min, "higher", conditionsSource.duty_valid_min),
      gateRow("平地实际周期", "各线性方向的真实接触周期与目标周期一致", curriculum.phase_gate_flat_period_value ?? command.flat_period, conditions.period_min, "higher", conditionsSource.period_min),
      gateRow("Yaw 支撑", "Yaw 使用旋转专属支撑结构，不套用直线对角模板", curriculum.phase_gate_flat_yaw_gait_value ?? command.yaw_gait_gate, conditions.yaw_gait_min, "higher", conditionsSource.yaw_gait_min),
      gateRow("平地倾斜 p95", "稳态移动时机身倾斜尾部", curriculum.phase_gate_flat_tilt_p95_value ?? command.flat_tilt_p95, conditions.flat_tilt_p95_max, "lower", conditionsSource.flat_tilt_p95_max),
      gateRow("平地角速度", "稳态移动时 roll/pitch 角速度均值", curriculum.phase_gate_flat_wxy_value ?? command.flat_wxy_mean, conditions.flat_wxy_max, "lower", conditionsSource.flat_wxy_max),
      gateRow("平地角加速度 p95", "roll/pitch 以及稳态 yaw 的角加速度尾部", curriculum.phase_gate_flat_ang_accel_p95_value ?? command.flat_ang_accel_p95, conditions.flat_ang_accel_p95_max, "lower", conditionsSource.flat_ang_accel_p95_max),
      gateRow("平地高度误差 p95", "移动机身高度相对目标的尾部误差", curriculum.phase_gate_flat_height_error_p95_value ?? command.flat_height_error_p95, conditions.flat_height_error_p95_max, "lower", conditionsSource.flat_height_error_p95_max),
      gateRow("平地触地速度 p95", "真实承重触地前一帧的向下足速尾部", curriculum.phase_gate_flat_touchdown_vz_p95_value ?? command.flat_touchdown_vz_p95, conditions.flat_touchdown_vz_p95_max, "lower", conditionsSource.flat_touchdown_vz_p95_max),
      gateRow("平地滑移尾部", "平地支撑滑移高值比例", curriculum.phase_gate_flat_slip_high_value ?? command.flat_slip_high, conditions.flat_slip_high_max, "lower", conditionsSource.flat_slip_high_max),
      gateRow("平地轨迹 p95", "四腿中最差足端轨迹误差尾部", curriculum.phase_gate_flat_trajectory_p95_value ?? command.flat_trajectory_worst_p95, conditions.flat_trajectory_p95_max, "lower", conditionsSource.flat_trajectory_p95_max),
      gateRow("平地误触发", "平地样本错误触发地形响应的比例", curriculum.phase_gate_flat_false_terrain_response_value ?? command.flat_false_terrain_response, conditions.flat_false_terrain_response_max, "lower", conditionsSource.flat_false_terrain_response_max),
    );
  } else {
    rows.push(
      gateRow("对角支撑", "平地阶段要求对角支撑一致性达标", gait.diagonal_contact ?? command.diagonal_contact ?? curriculum.diagonal_gate, conditions.diagonal_min, "higher", conditionsSource.diagonal_min),
      gateRow("duty 区间", "四腿支撑占空比处于允许区间", command.duty_target_score, conditions.duty_target_min, "higher", conditionsSource.duty_target_min),
      gateRow("duty 对称", "左右、前后和交叉支撑时间保持对称", command.duty_symmetry_score, conditions.duty_symmetry_min, "higher", conditionsSource.duty_symmetry_min),
      gateRow("duty 有效覆盖", "统计窗口已覆盖足够多的有效步态周期", command.duty_cycle_valid_frac, conditions.duty_valid_min, "higher", conditionsSource.duty_valid_min),
      gateRow("实际周期", "实际接触周期与目标周期相符", command.gait_period_score, conditions.period_min, "higher", conditionsSource.period_min),
      gateRow("yaw 支撑", "当前阶段要求 yaw 时使用旋转专属支撑结构", command.yaw_gait_gate, conditions.yaw_gait_min, "higher", conditionsSource.yaw_gait_min),
      gateRow("腾空", "平地阶段要求足端腾空指标达标", command.feet_air_time ?? curriculum.air_gate, conditions.air_min, "higher", conditionsSource.air_min),
    );
  }
  if (terrainPhase && mixedPhase) {
    rows.push(
      gateRow("真实地形", "混合地形阶段要求真实地形均值达到门槛", curriculum.terrain_real_mean ?? curriculum.terrain_mean, conditions.terrain_min, "higher", conditionsSource.terrain_min),
      gateRow("离散地形", "混合地形阶段要求 boxes/stairs 等离散地形达到门槛", curriculum.terrain_discrete_mean, conditions.discrete_terrain_min, "higher", conditionsSource.discrete_terrain_min),
      gateRow("boxes 等级", "boxes 不能被其他地形均值代偿", curriculum.terrain_boxes_mean, conditions.boxes_min, "higher", conditionsSource.boxes_min),
      gateRow("下楼等级", "下楼梯不能被其他地形均值代偿", curriculum.terrain_stairs_mean, conditions.stairs_down_min, "higher", conditionsSource.stairs_down_min),
      gateRow("上楼等级", "上楼梯不能被其他地形均值代偿", curriculum.terrain_stairs_up_mean, conditions.stairs_up_min, "higher", conditionsSource.stairs_up_min),
      gateRow("boxes 成功", "稳定穿越并升级的滚动成功率", curriculum.terrain_boxes_success_rate, conditions.boxes_success_min, "higher", conditionsSource.boxes_success_min),
      gateRow("下楼成功", "真实承重换层、方向位移和稳定终点同时满足", curriculum.terrain_stairs_down_success_rate, conditions.stairs_down_success_min, "higher", conditionsSource.stairs_down_success_min),
      gateRow("上楼成功", "真实承重换层、方向位移和稳定终点同时满足", curriculum.terrain_stairs_up_success_rate, conditions.stairs_up_success_min, "higher", conditionsSource.stairs_up_success_min),
      gateRow("boxes 崩溃", "boxes 终止、低高度或失稳的滚动比例", curriculum.terrain_boxes_collapse_rate, conditions.boxes_collapse_max, "lower", conditionsSource.boxes_collapse_max),
      gateRow("下楼崩溃", "下楼终止、低高度或失稳的滚动比例", curriculum.terrain_stairs_down_collapse_rate, conditions.stairs_down_collapse_max, "lower", conditionsSource.stairs_down_collapse_max),
      gateRow("上楼崩溃", "上楼终止、低高度或失稳的滚动比例", curriculum.terrain_stairs_up_collapse_rate, conditions.stairs_up_collapse_max, "lower", conditionsSource.stairs_up_collapse_max),
      gateRow("摔倒率", "混合地形阶段要求摔倒率低于门槛", health.fall_rate ?? curriculum.fall_gate, conditions.fall_max, "lower", conditionsSource.fall_max),
    );
  }
  const ruleSummary = [
    gate.condition_set_complete ? `运行时最终 gate ${gate.runtime_gate_ok ? "通过" : "未通过"}` : "当前包未直接输出最终 gate，按有效配置重建",
    `连续 ${formatNumberLike(conditions.phase_intervals)} 次`,
    terrainPhase ? "地形训练已启用" : "地形训练未启用",
    mixedPhase ? "当前检查分类型地形能力" : "当前不以地形等级阻塞 count",
  ].join(" · ");
  return (
    <DetailPanel
      title="阶段门控"
      meta={[
        `phase ${textValue(curriculum.phase) || "未知"}`,
        `count ${formatNumberLike(curriculum.phase_count)} / ${formatNumberLike(conditions.phase_intervals)}`,
        `gate@${formatNumberLike(curriculum.phase_gate_eval_step)}`,
        `mode ${commandMode || "未知"}`,
      ]}
    >
      <div className="rule-summary">{ruleSummary}</div>
      <GateTable rows={rows} />
      <p className="detail-note">来源：{gate.available ? textValue(gate.source) : "当前 telemetry 未提供 effective_config 门控解析"}</p>
      <details className="rule-details">
        <summary>规则详情</summary>
        <dl>
          <div><dt>阈值出处</dt><dd>门槛列标「默认」表示配置里没读到、用了兜底值；标「定值」表示定义性常量、不随策略调整；无标记表示来自本次训练的实际配置。</dd></div>
          <div><dt>推进公式</dt><dd>{progressFormula}</dd></div>
          <div><dt>偏航</dt><dd>{activeDirs.includes("yaw") ? "偏航参与当前阶段门控。" : "偏航当前只记录，不参与阶段门控。"}</dd></div>
          <div><dt>质量公式</dt><dd>{terrainPhase ? "使用整体滑移，加上仅由平地样本计算的核心质量与接触步态；地形样本本身不强套平地模板。" : "要求滑移、对角支撑、duty、实际周期、yaw 支撑和腾空同时满足。"}</dd></div>
          <div><dt>连续计数</dt><dd>满足一次门控 phase_count +1，不满足则回退；达到 phase_intervals 后进入下一阶段。</dd></div>
          <div><dt>超时机制</dt><dd>{"只有 phase_timeout_enable=true 才会强制放行；当前关闭时等待不会替代能力。"}</dd></div>
          <div><dt>地形阶段</dt><dd>{"phase >= terrain_start_phase 会启用地形训练；只有 mixed 命令阶段才把分类型地形等级、成功率、崩溃率和摔倒率加入 count 门控。"}</dd></div>
          <div><dt>当前配置</dt><dd>terrain_start_phase={formatNumberLike(conditions.terrain_start_phase)}，phase_max_steps={formatNumberLike(conditions.phase_max_steps)}。</dd></div>
        </dl>
      </details>
    </DetailPanel>
  );
}

function DirectionProgressDetails({ curriculum }: { curriculum: Record<string, any> }) {
  const active = new Set(textValue(curriculum.active_dirs).split(",").map((item) => item.trim()).filter(Boolean));
  const rows = [
    { label: "前进", key: "fwd", value: curriculum.progress_fwd },
    { label: "后退", key: "back", value: curriculum.progress_back },
    { label: "横移", key: "lat", value: curriculum.progress_lat },
    { label: "Yaw", key: "yaw", value: curriculum.progress_yaw },
  ];
  return (
    <DetailPanel
      title="方向 Progress"
      meta={[
        `gate ${formatNumberLike(curriculum.progress_gate)}`,
        `active ${textValue(curriculum.active_dirs) || "无"}`,
        `lag ${textValue(curriculum.progress_lagging_dir) || "无"}`,
      ]}
    >
      <div className="direction-grid">
        {rows.map((row) => (
          <div key={row.key}>
            <span>{row.label}</span>
            <strong>{formatNumberLike(row.value)}</strong>
            <small>{active.has(row.key) ? "参与门控" : "记录"}</small>
          </div>
        ))}
      </div>
    </DetailPanel>
  );
}

function TerrainDetails({
  curriculum,
  reward,
}: {
  curriculum: Record<string, any>;
  reward: Record<string, any>;
}) {
  const rows = [
    ["real", curriculum.terrain_real_mean, curriculum.terrain_real_max],
    ["discrete", curriculum.terrain_discrete_mean, curriculum.terrain_discrete_max],
    ["rough", curriculum.terrain_rough_mean, curriculum.terrain_rough_max],
    ["boxes", curriculum.terrain_boxes_mean, curriculum.terrain_boxes_max],
    ["stairs", curriculum.terrain_stairs_mean, curriculum.terrain_stairs_max],
    ["stairs_up", curriculum.terrain_stairs_up_mean, curriculum.terrain_stairs_up_max],
    ["slope", curriculum.terrain_slope_mean, curriculum.terrain_slope_max],
    ["slope_inv", curriculum.terrain_slope_inv_mean, curriculum.terrain_slope_inv_max],
  ];
  return (
    <DetailPanel
      title="地形课程"
      meta={[
        `up ${formatNumberLike(curriculum.terrain_discrete_move_up_rate ?? curriculum.terrain_move_up_rate)}`,
        `down ${formatNumberLike(curriculum.terrain_discrete_move_down_rate ?? curriculum.terrain_move_down_rate)}`,
        `fail ${formatNumberLike(curriculum.terrain_discrete_failure_down_rate ?? curriculum.terrain_failure_down_rate)}`,
      ]}
    >
      <table className="compact-metric-table terrain-table">
        <thead><tr><th>类型</th><th>均值</th><th>最大值</th></tr></thead>
        <tbody>
          {rows.map(([label, mean, max]) => (
            <tr key={String(label)}>
              <td>{label}</td>
              <td>{formatNumberLike(mean)}</td>
              <td>{formatNumberLike(max)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="detail-inline">
        <span>event {formatNumberLike(reward.terrain_probe_event_scope_mean)}</span>
        <span>contact {formatNumberLike(reward.terrain_contact_quality)}</span>
        <span>collapse {formatNumberLike(reward.terrain_event_collapse)}</span>
        <span>eligible {formatNumberLike(curriculum.terrain_eligible_frac)}</span>
      </div>
    </DetailPanel>
  );
}

function QualityDetails({
  command,
  gait,
}: {
  command: Record<string, any>;
  gait: Record<string, any>;
}) {
  const rows = [
    ["gait", gait.gait_match ?? command.gait_match],
    ["diag", gait.diagonal_contact ?? command.diagonal_contact],
    ["duty", gait.duty_balance ?? command.duty_balance],
    ["duty spread", gait.duty_spread_window ?? command.duty_spread_window],
    ["high slip", gait.stance_slip_high_fraction ?? command.stance_slip_high_fraction],
  ];
  return (
    <DetailPanel title="质量细节" meta={[`lin_err ${formatNumberLike(command.lin_err, "m/s")}`, `yaw_err ${formatNumberLike(command.yaw_err, "rad/s")}`]}>
      <div className="quality-grid">
        {rows.map(([label, value]) => (
          <div key={String(label)}>
            <span>{label}</span>
            <strong>{formatNumberLike(value)}</strong>
          </div>
        ))}
      </div>
    </DetailPanel>
  );
}

function StabilityDetails({
  health,
  reward,
}: {
  health: Record<string, any>;
  reward: Record<string, any>;
}) {
  const rows = [
    ["base_wxy", reward.base_wxy],
    ["landing", reward.landing_impact],
    ["touch slip", reward.touchdown_slip],
    ["tilt", health.tilt_deg],
    ["fall", health.fall_rate],
  ];
  return (
    <DetailPanel title="稳定细节" meta={[`base_h ${formatNumberLike(health.base_h, "m")}`, `terminal ${formatNumberLike(health.terminal_rate)}`]}>
      <div className="quality-grid">
        {rows.map(([label, value]) => (
          <div key={String(label)}>
            <span>{label}</span>
            <strong>{formatNumberLike(value)}</strong>
          </div>
        ))}
      </div>
    </DetailPanel>
  );
}

function TrendDetails({
  preset,
  series,
  selectedCount,
  hasHistory,
  scale,
  onScaleChange,
}: {
  preset: typeof AGENT_TREND_PRESETS[number];
  series: Array<{ values: Array<number | null>; color: string; label: string; width: number }>;
  selectedCount: number;
  hasHistory: boolean;
  scale: TrendScaleMode;
  onScaleChange: (mode: TrendScaleMode) => void;
}) {
  return (
    <DetailPanel title={`${preset.label}趋势`} meta={[preset.description]}>
      <div className="trend-toolbar compact">
        <span>{selectedCount} 条曲线</span>
        <div className="segmented-control">
          <button className={scale === "normalized" ? "active" : ""} onClick={() => onScaleChange("normalized")}>归一化</button>
          <button className={scale === "raw" ? "active" : ""} onClick={() => onScaleChange("raw")}>原始值</button>
        </div>
      </div>
      <div className="agent-trend-chart">
        <LineChart
          series={series}
          height={190}
          label={selectedCount ? `${preset.label}趋势` : "关键趋势"}
          emptyLabel={hasHistory ? "当前预设暂无字段" : "暂无训练历史"}
        />
      </div>
    </DetailPanel>
  );
}

function DetailPanel({ title, meta, children }: { title: string; meta?: string[]; children: React.ReactNode }) {
  return (
    <div className="detail-panel">
      <div className="detail-panel-head">
        <h3>{title}</h3>
        {meta?.length ? <div>{meta.map((item) => <span key={item}>{item}</span>)}</div> : null}
      </div>
      {children}
    </div>
  );
}

function GateTable({ rows }: { rows: GateRow[] }) {
  return (
    <table className="compact-metric-table gate-table">
      <thead>
        <tr><th>项</th><th>当前</th><th>门槛</th><th>差距</th><th>条件</th></tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.label} className={row.pass === false ? "miss" : ""}>
            <td>{row.label}</td>
            <td>{formatNumberLike(row.value)}</td>
            <td>{formatNumberLike(row.target)}<GateSource source={row.targetSource} /></td>
            <td className={row.pass === false ? "bad-delta" : "good-delta"}>{formatSignedNumber(row.gap)}</td>
            <td>{row.condition}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function GateSource({ source }: { source?: string }) {
  if (source === "default") return <span className="gate-src default" title="配置里没读到这个阈值，用的是兜底默认值">默认</span>;
  if (source === "constant") return <span className="gate-src constant" title="定义性常量，不是按策略调整的阈值">定值</span>;
  return null;
}

function MetricCell({ label, value, tone = "neutral" }: { label: string; value: string; tone?: "good" | "warn" | "bad" | "neutral" }) {
  return (
    <div className={`telemetry-cell ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function PathLine({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <code title={value}>{shortPath(value)}</code>
    </div>
  );
}

function ActionPanel({
  actions,
  busy,
  onOpenDiagnostics,
}: {
  actions: AgentWorkbenchActionInfo[];
  busy: string | null;
  onOpenDiagnostics: () => void;
}) {
  const safeActions = arrayValue<AgentWorkbenchActionInfo>(actions);
  const askAction = safeActions.find((action) => action.kind === "ask_agent");
  const diagnosticAction = safeActions.find((action) => action.kind === "diagnostic");
  const askAgent = () => window.dispatchEvent(new CustomEvent("locomotion-console-send-command", {
    detail: "/ask 读取当前证据，解释现在的训练状态；如果需要动作，只生成待授权提案，不直接改变训练状态。",
  }));
  return (
    <section className="side-panel compact-panel command-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">操作</p>
          <h2>必要入口</h2>
        </div>
      </div>
      <div className="command-strip">
        <button className="primary-button" disabled={Boolean(busy) || askAction?.enabled === false} onClick={askAgent}>
          问智能体
        </button>
        <button className="secondary-button" disabled={Boolean(busy) || diagnosticAction?.enabled === false} onClick={onOpenDiagnostics}>
          诊断
        </button>
      </div>
      <p className="compact-copy">训练启动、继续、部署和停止在主工作台“运行控制”中执行。</p>
    </section>
  );
}

function ProcessLog({ workbench, busy }: { workbench: AgentWorkbenchInfo; busy: string | null }) {
  const rows = processRows(workbench, busy);
  return (
    <section className="side-panel compact-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">过程日志</p>
          <h2>最近系统事件</h2>
        </div>
      </div>
      <ol className="process-log">
        {rows.map((row) => (
          <li key={row.key}>
            <time>{row.time}</time>
            <span>{row.text}</span>
          </li>
        ))}
      </ol>
    </section>
  );
}

function processRows(workbench: AgentWorkbenchInfo, busy: string | null) {
  const evidence = arrayValue<AgentWorkbenchEvidenceInfo>(workbench.evidence);
  const proposals = arrayValue<ChatProposalInfo>(workbench.proposals);
  const notes = arrayValue<string>(workbench.notes);
  const time = new Date(workbench.generated_at * 1000).toLocaleTimeString("zh-CN", { hour12: false });
  const rows = [
    { key: "status", time, text: `状态：${workbench.attempt.summary || stateLabel(workbench.attempt.runtime_state)}` },
    { key: "judgement", time, text: `判断：${workbench.judgement.title}` },
    { key: "evidence", time, text: `证据：${evidence.filter((item) => item.status === "ok").length}/${evidence.length} 可用` },
  ];
  const remote = evidence.find((item) => item.id === "remote");
  if (remote) {
    rows.push({ key: "remote", time, text: `远端：${remote.detail || evidenceStatusLabel(remote.status)}` });
  }
  const pending = proposals.filter((item) => item.status === "pending").length;
  if (pending > 0) rows.push({ key: "proposal", time, text: `待确认：${pending} 个提案` });
  if (busy) rows.unshift({ key: "busy", time: "现在", text: `正在执行：${busy}` });
  for (const note of notes.slice(0, 2)) {
    rows.push({ key: `note:${note}`, time, text: note });
  }
  return rows;
}

function stateLabel(state: string) {
  const labels: Record<string, string> = {
    live: "在线",
    stale: "遥测过期",
    stopped: "已停止",
    interrupted: "中断",
    remote_unavailable: "远端不可用",
    unknown: "未知",
  };
  return labels[state] ?? state;
}

function attemptStateLabel(running: boolean, state: string) {
  if (state === "live" && running) return "训练中";
  if (state === "stale" && running) return "训练状态未知（遥测过期）";
  return stateLabel(state);
}

function confidenceLabel(value: string) {
  if (value === "high") return "高";
  if (value === "medium") return "中";
  return "低";
}

function riskLabel(value: string) {
  if (value === "high") return "高";
  if (value === "medium") return "中";
  return "低";
}

function evidenceStatusLabel(value: string) {
  const labels: Record<string, string> = {
    ok: "可用",
    warning: "警告",
    error: "错误",
    missing: "缺失",
    unknown: "未知",
  };
  return labels[value] ?? value;
}

function shortPath(path: string) {
  if (!path) return "无";
  const normalized = path.replace(/\\/g, "/");
  const parts = normalized.split("/").filter(Boolean);
  return parts.length <= 3 ? path : `.../${parts.slice(-3).join("/")}`;
}

function recordValue(value: unknown): Record<string, any> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, any> : {};
}

function arrayValue<T>(value: unknown): T[] {
  return Array.isArray(value) ? value as T[] : [];
}

function numberOrNull(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function textValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "";
  return String(value);
}

function formatNumberLike(value: unknown, unit = "") {
  const num = numberOrNull(value);
  if (num === null) return textValue(value) || "无";
  const abs = Math.abs(num);
  const digits = abs >= 100 ? 0 : abs >= 10 ? 1 : 3;
  return `${num.toFixed(digits)}${unit ? ` ${unit}` : ""}`;
}

function formatSignedNumber(value: unknown) {
  const num = numberOrNull(value);
  if (num === null) return "无";
  const abs = Math.abs(num);
  const digits = abs >= 100 ? 0 : abs >= 10 ? 1 : 3;
  return `${num >= 0 ? "+" : ""}${num.toFixed(digits)}`;
}

function formatStep(step: number | null, total: number | null) {
  if (step === null && total === null) return "无";
  if (total === null) return String(step ?? "无");
  return `${step ?? 0} / ${total}`;
}

function formatAge(value: unknown) {
  const age = numberOrNull(value);
  if (age === null) return "未知";
  if (age < 60) return `${age.toFixed(0)}s`;
  if (age < 3600) return `${(age / 60).toFixed(1)}min`;
  return `${(age / 3600).toFixed(1)}h`;
}

function progressGateFormula(curriculum: Record<string, any>, activeDirs: string[]) {
  const labels: Record<string, string> = {
    fwd: "前进",
    back: "后退",
    lat: "横移",
    yaw: "Yaw",
  };
  const values: Record<string, unknown> = {
    fwd: curriculum.progress_fwd,
    back: curriculum.progress_back,
    lat: curriculum.progress_lat,
    yaw: curriculum.progress_yaw,
  };
  const active = activeDirs.length ? activeDirs : ["fwd", "back", "lat", "yaw"];
  const terms = active.map((key) => `${labels[key] ?? key} ${formatNumberLike(values[key])}`);
  const yawNote = active.includes("yaw") ? "" : `；Yaw ${formatNumberLike(values.yaw)} 不参与门控`;
  return `min(${terms.join("，")})${yawNote}`;
}

function gateRow(
  label: string,
  condition: string,
  value: unknown,
  target: unknown,
  direction: "higher" | "lower",
  targetSource?: string,
): GateRow {
  const numericValue = numberOrNull(value);
  const numericTarget = numberOrNull(target);
  let pass: boolean | null = null;
  let gap: number | null = null;
  if (numericValue !== null && numericTarget !== null) {
    pass = direction === "higher" ? numericValue >= numericTarget : numericValue <= numericTarget;
    gap = direction === "higher" ? numericValue - numericTarget : numericTarget - numericValue;
  }
  return { label, condition, value, target, targetSource, direction, gap, pass };
}

function buildAgentTrendOptions(telemetry: TrainingTelemetry | null): AgentTrendOption[] {
  const history = telemetry?.history ?? [];
  if (!history.length) return [];
  const groups: Array<[TelemetryGroupKey, string]> = [
    ["curriculum", "课程"],
    ["command", "命令/步态"],
    ["health", "健康"],
    ["reward", "奖励"],
    ["counters", "计数"],
  ];
  const options: AgentTrendOption[] = [];
  for (const [group, groupLabel] of groups) {
    for (const key of numericTrendKeys(history, group)) {
      const optionKey = `${group}.${key}`;
      options.push({
        key: optionKey,
        label: labelForTrendField(optionKey, groupLabel),
        values: history.map((point) => finiteTrendOrNull(recordForTrend(point, group)[key])),
        group: groupLabel,
        unit: unitForTrendField(optionKey),
      });
    }
  }
  return options
    .filter((item) => countTrendNumbers(item.values) > 1)
    .sort((a, b) => rankTrendField(a.key) - rankTrendField(b.key) || a.label.localeCompare(b.label));
}

function numericTrendKeys(history: TrainingTelemetryPoint[], group: TelemetryGroupKey) {
  const keys = new Set<string>();
  for (const point of history) {
    for (const [key, value] of Object.entries(recordForTrend(point, group))) {
      if (typeof value === "number" && Number.isFinite(value)) keys.add(key);
    }
  }
  return [...keys].sort();
}

function recordForTrend(point: TrainingTelemetryPoint, group: TelemetryGroupKey): Record<string, unknown> {
  return recordValue(point[group]);
}

function labelForTrendField(key: string, fallbackGroup: string) {
  const labels: Record<string, string> = {
    "reward.total": "总奖励",
    "reward.base_wxy": "机身摇晃",
    "reward.landing_impact": "落地冲击",
    "reward.touchdown_slip": "触地滑移",
    "reward.terrain_support_transfer": "地形支撑转移",
    "reward.terrain_contact_quality": "地形接触质量",
    "reward.terrain_event_collapse": "触地塌陷惩罚",
    "reward.terrain_probe_contact_event_mean": "地形接触事件",
    "reward.terrain_probe_event_scope_mean": "地形事件范围",
    "command.cmd_vx": "目标前向速度",
    "command.actual_vx": "实际前向速度",
    "command.v_along": "沿命令速度",
    "command.lin_err": "线速度误差",
    "command.yaw_err": "Yaw 误差",
    "command.gait_match": "步态匹配",
    "command.diagonal_contact": "对角支撑",
    "command.duty_balance": "占空平衡",
    "command.stance_slip": "支撑滑移",
    "command.stance_slip_high_fraction": "高滑移比例",
    "command.transition_active_frac": "过渡占比",
    "command.transition_strength": "过渡强度",
    "curriculum.progress_gate": "推进 gate",
    "curriculum.progress_fwd": "前进推进",
    "curriculum.progress_back": "后退推进",
    "curriculum.progress_lat": "横移推进",
    "curriculum.progress_yaw": "Yaw 推进",
    "curriculum.gait_gate": "步态 gate",
    "curriculum.diagonal_gate": "对角 gate",
    "curriculum.duty_balance_gate": "占空 gate",
    "curriculum.slip_gate": "滑移 gate",
    "curriculum.fall_gate": "摔倒 gate",
    "curriculum.terrain_real_mean": "真实地形均值",
    "curriculum.terrain_real_max": "真实地形最高",
    "curriculum.terrain_discrete_mean": "离散地形均值",
    "curriculum.terrain_discrete_max": "离散地形最高",
    "curriculum.terrain_discrete_move_up_rate": "地形升级率",
    "curriculum.terrain_discrete_failure_down_rate": "地形失败降级率",
    "health.fall_rate": "摔倒率",
    "health.terminal_rate": "终止率",
    "health.base_h": "机身高度",
    "health.base_h_min": "最低机身高度",
    "health.tilt_deg": "倾角",
    "health.support_instability": "支撑不稳定",
    "health.torque_util": "力矩使用",
  };
  return labels[key] ?? `${fallbackGroup}.${key.split(".").slice(1).join(".")}`;
}

function unitForTrendField(key: string) {
  if (/(vx|vy|speed|v_along|lin_err)/.test(key)) return "m/s";
  if (/(wz|yaw_err)/.test(key)) return "rad/s";
  if (/tilt_deg/.test(key)) return "deg";
  if (/(base_h|height)/.test(key)) return "m";
  if (/(terrain|level|dr_level)/.test(key)) return "level";
  if (/(rate|gate|ratio|fraction|match|slip|progress|score|balance)/.test(key)) return "0-1";
  return "";
}

function rankTrendField(key: string) {
  const defaultRank = (AGENT_TREND_PRESETS.flatMap((preset) => preset.keys) as readonly string[]).indexOf(key);
  if (defaultRank >= 0) return defaultRank;
  if (key.startsWith("curriculum.progress_")) return 20;
  if (key.startsWith("curriculum.terrain_")) return 40;
  if (key.startsWith("command.")) return 80;
  if (key.startsWith("health.")) return 120;
  if (key.startsWith("reward.terrain_")) return 160;
  if (key.startsWith("reward.")) return 200;
  return 300;
}

function normalizeTrend(values: Array<number | null>) {
  const numbers = values.filter(isTrendNumber);
  if (numbers.length < 2) return values;
  const min = Math.min(...numbers);
  const max = Math.max(...numbers);
  const span = max - min;
  if (!Number.isFinite(span) || span < 1e-12) return values.map((value) => value == null ? null : 0.5);
  return values.map((value) => value == null ? null : (value - min) / span);
}

function finiteTrendOrNull(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function isTrendNumber(value: number | null): value is number {
  return typeof value === "number";
}

function countTrendNumbers(values: Array<number | null>) {
  return values.filter(isTrendNumber).length;
}

function trendColor(index: number) {
  return ["#0066cc", "#0a7f41", "#b42318", "#936100", "#7c3aed", "#0f766e", "#c2410c", "#475569"][index % 8];
}

function confirmTextForAction(action: DirectAction) {
  const labels: Record<DirectAction, string> = {
    "deploy-payload": "部署当前上传包到远端。不会启动训练，但会更新远端运行代码和配置。",
    start: "启动全新训练。会创建新的 run，不会从 checkpoint 继续。",
    resume: "从最新可用 checkpoint 继续训练。会创建新的 resume run。",
    kill: "停止当前远端训练进程和 rl_train tmux 会话。",
  };
  return `${labels[action]}\n\n确认执行？`;
}
