import { useEffect, useMemo, useState } from "react";
import {
  cancelChatProposal,
  executeAction,
  getAgentWorkbench,
  postAction,
  type AgentWorkbenchActionInfo,
  type AgentWorkbenchEvidenceInfo,
  type AgentWorkbenchInfo,
  type ChatProposalInfo,
} from "./api";
import Chat, { INITIAL_CHAT_MESSAGES, type ChatMessage } from "./Chat";
import ConfigWorkspace from "./ConfigWorkspace";
import Diagnostics from "./Diagnostics";
import { formatError } from "./i18n/format";

type ToolView = "agent" | "diagnostics" | "config";
type DirectAction = "deploy-payload" | "start" | "resume" | "kill";

const WORKBENCH_POLL_MS = 5000;

export default function App() {
  const [view, setView] = useState<ToolView>("agent");
  const [visitedViews, setVisitedViews] = useState<Record<ToolView, boolean>>({
    agent: true,
    diagnostics: false,
    config: false,
  });
  const [workbench, setWorkbench] = useState<AgentWorkbenchInfo | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [context, setContext] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(INITIAL_CHAT_MESSAGES);

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
    if (!workbench) return;
    const attempt = workbench.attempt;
    const values = workbench.evidence.find((item) => item.id === "telemetry")?.values ?? {};
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
      `terrain_mean=${values.terrain_mean ?? "unknown"}`,
      `dr_level=${values.dr_level ?? "unknown"}`,
    ].join(" | "));
  }, [workbench]);

  async function refreshWorkbench() {
    setBusy("refresh");
    try {
      setWorkbench(await getAgentWorkbench());
      setError("");
    } catch (reason) {
      setError(formatError(reason));
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
    () => workbench?.proposals.filter((item) => item.status === "pending") ?? [],
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
          <button className={view === "diagnostics" ? "active" : ""} onClick={() => openView("diagnostics")}>诊断工具</button>
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
          <AgentWorkbench
            workbench={workbench}
            busy={busy}
            pendingProposals={pendingProposals}
            onDirectAction={runDirectAction}
            onConfirmProposal={confirmProposal}
            onCancelProposal={cancelProposal}
            onOpenDiagnostics={() => openView("diagnostics")}
            chatContext={context}
            chatMessages={chatMessages}
            setChatMessages={setChatMessages}
          />
        </div>
        {visitedViews.diagnostics && (
          <main className="tool-host workspace-pane" hidden={view !== "diagnostics"}>
            <Diagnostics active={view === "diagnostics"} />
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
        <TelemetryPanel workbench={workbench} />
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
        <div><dt>状态</dt><dd>{attempt.running ? "训练中" : stateLabel(attempt.runtime_state)}</dd></div>
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
      {judgement.gaps.length > 0 && (
        <div className="gap-list">
          <strong>证据缺口</strong>
          {judgement.gaps.map((gap) => <span key={gap}>{gap}</span>)}
        </div>
      )}
      <div className="note-list">
        {workbench.notes.map((note) => <span key={note}>{note}</span>)}
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
  const actionsById = new Map(workbench.actions.map((action) => [action.id, action]));
  const remote = workbench.evidence.find((item) => item.id === "remote");
  const telemetry = workbench.evidence.find((item) => item.id === "telemetry");
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
              disabled={Boolean(busy) || info?.enabled === false}
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
  const visibleEvidence = evidence.filter((item) => item.id !== "telemetry");
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

function TelemetryPanel({ workbench }: { workbench: AgentWorkbenchInfo }) {
  const telemetry = workbench.evidence.find((item) => item.id === "telemetry");
  const values = telemetry?.values ?? {};
  const status = telemetry?.status ?? "unknown";
  const snapshot = recordValue(values.snapshot);
  const timeline = recordValue(values.timeline);
  const curriculum = recordValue(values.curriculum);
  const command = recordValue(values.command);
  const gait = recordValue(values.gait);
  const health = recordValue(values.health);
  const paths = recordValue(values.paths);
  const blockers = Array.isArray(snapshot.blockers) ? snapshot.blockers.slice(0, 3) : [];
  const step = numberOrNull(timeline.step ?? values.step ?? workbench.attempt.step);
  const total = numberOrNull(timeline.total_steps ?? values.total_steps ?? workbench.attempt.total_steps);
  const progressPct = total && step !== null ? Math.max(0, Math.min(100, (step / total) * 100)) : null;
  const blockedBy = textValue(curriculum.blocked_by ?? values.blocked_by ?? workbench.attempt.blocked_by);
  const conclusion = textValue(snapshot.conclusion ?? values.summary ?? telemetry?.detail);

  return (
    <section className={`primary-panel telemetry-panel flat-panel ${status}`}>
      <div className="panel-heading telemetry-heading">
        <div>
          <p className="eyebrow">当前遥测</p>
          <h2>{status === "ok" ? "训练状态快照" : "遥测不可用"}</h2>
        </div>
        <div className="telemetry-state">
          <span className={`run-state ${workbench.attempt.running ? "running" : "idle"}`}>
            {workbench.attempt.running ? "训练中" : stateLabel(workbench.attempt.runtime_state)}
          </span>
          <span className={`evidence-pill ${status}`}>{evidenceStatusLabel(status)}</span>
        </div>
      </div>

      {conclusion && <p className="telemetry-conclusion">{conclusion}</p>}

      <div className="telemetry-hero">
        <MetricCell label="步数" value={formatStep(step, total)} />
        <MetricCell label="阶段" value={textValue(curriculum.phase ?? workbench.attempt.phase) || "未知"} />
        <MetricCell label="阻塞" value={blockedBy || "无"} tone={blockedBy ? "warn" : "good"} />
        <MetricCell label="时效" value={formatAge(values.telemetry_age_s ?? workbench.attempt.telemetry_age_s)} tone={values.stale ? "warn" : "neutral"} />
      </div>

      {progressPct !== null && (
        <div className="telemetry-progress" aria-label="训练进度">
          <span style={{ width: `${progressPct}%` }} />
        </div>
      )}

      <div className="telemetry-sections">
        <TelemetryGroup
          title="课程"
          items={[
            ["progress", formatNumberLike(curriculum.progress_gate ?? values.progress_gate)],
            ["fwd/back/lat/yaw", formatDirectionProgress(curriculum)],
            ["最弱方向", textValue(curriculum.progress_lagging_dir) || "无"],
            ["terrain", formatNumberLike(curriculum.terrain_mean ?? values.terrain_mean)],
            ["DR", formatNumberLike(curriculum.dr_level ?? values.dr_level)],
          ]}
        />
        <TelemetryGroup
          title="命令"
          items={[
            ["cmd vx", formatNumberLike(command.cmd_vx ?? values.cmd_vx, "m/s")],
            ["actual vx", formatNumberLike(command.actual_vx ?? values.actual_vx, "m/s")],
            ["v along", formatNumberLike(command.v_along, "m/s")],
            ["lin err", formatNumberLike(command.lin_err ?? values.lin_err, "m/s")],
            ["yaw err", formatNumberLike(command.yaw_err, "rad/s")],
          ]}
        />
        <TelemetryGroup
          title="步态"
          items={[
            ["gait", formatNumberLike(gait.gait_match)],
            ["diag", formatNumberLike(gait.diagonal_contact)],
            ["duty", formatNumberLike(gait.duty_balance)],
            ["slip", formatNumberLike(gait.stance_slip)],
            ["high slip", formatNumberLike(gait.stance_slip_high_fraction)],
          ]}
        />
        <TelemetryGroup
          title="健康"
          items={[
            ["fall", formatNumberLike(health.fall_rate ?? values.fall_rate)],
            ["base h", formatNumberLike(health.base_h ?? values.base_h, "m")],
            ["min h", formatNumberLike(health.base_h_min, "m")],
            ["tilt", formatNumberLike(health.tilt_deg, "deg")],
            ["torque", formatNumberLike(health.torque_util)],
          ]}
        />
      </div>

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

      {blockers.length > 0 && (
        <div className="telemetry-blockers">
          {blockers.map((item, index) => {
            const blocker = recordValue(item);
            return (
              <span key={`${textValue(blocker.key) || "blocker"}-${index}`}>
                {textValue(blocker.label) || textValue(blocker.key) || "阻塞项"}：{formatNumberLike(blocker.value)}
                {blocker.target !== undefined && ` / ${formatNumberLike(blocker.target)}`}
              </span>
            );
          })}
        </div>
      )}
    </section>
  );
}

function TelemetryGroup({ title, items }: { title: string; items: Array<[string, string]> }) {
  return (
    <div className="telemetry-group">
      <h3>{title}</h3>
      <dl>
        {items.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
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
  const askAction = actions.find((action) => action.kind === "ask_agent");
  const diagnosticAction = actions.find((action) => action.kind === "diagnostic");
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
  const time = new Date(workbench.generated_at * 1000).toLocaleTimeString("zh-CN", { hour12: false });
  const rows = [
    { key: "status", time, text: `状态：${workbench.attempt.summary || stateLabel(workbench.attempt.runtime_state)}` },
    { key: "judgement", time, text: `判断：${workbench.judgement.title}` },
    { key: "evidence", time, text: `证据：${workbench.evidence.filter((item) => item.status === "ok").length}/${workbench.evidence.length} 可用` },
  ];
  const remote = workbench.evidence.find((item) => item.id === "remote");
  if (remote) {
    rows.push({ key: "remote", time, text: `远端：${remote.detail || evidenceStatusLabel(remote.status)}` });
  }
  const pending = workbench.proposals.filter((item) => item.status === "pending").length;
  if (pending > 0) rows.push({ key: "proposal", time, text: `待确认：${pending} 个提案` });
  if (busy) rows.unshift({ key: "busy", time: "现在", text: `正在执行：${busy}` });
  for (const note of workbench.notes.slice(0, 2)) {
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

function formatDirectionProgress(curriculum: Record<string, any>) {
  const parts = [
    ["F", curriculum.progress_fwd],
    ["B", curriculum.progress_back],
    ["L", curriculum.progress_lat],
    ["Y", curriculum.progress_yaw],
  ].map(([label, value]) => `${label}:${formatNumberLike(value)}`);
  return parts.join("  ");
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
