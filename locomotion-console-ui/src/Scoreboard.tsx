import { useEffect, useRef, useState } from "react";
import {
  getScoreboard,
  type Scoreboard as ScoreboardData,
  type ScoreboardContext,
  type ScoreboardCoverage,
  type ScoreboardFamily,
  type ScoreboardMetric,
  type ScoreboardTension,
} from "./api";
import { formatError } from "./i18n/format";

// 记分牌是训练目标的「事实看板」：只报现在读数、底线和出处，不做好坏判断。
// 定期刷新，切到本页时立刻拉一次。
const SCOREBOARD_POLL_MS = 10000;

export default function Scoreboard({ active = true }: { active?: boolean }) {
  const [board, setBoard] = useState<ScoreboardData | null>(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const refreshInFlightRef = useRef(false);

  async function refresh(options: { silent?: boolean } = {}) {
    if (refreshInFlightRef.current) return;
    refreshInFlightRef.current = true;
    if (!options.silent) setBusy("refresh");
    try {
      const next = await getScoreboard();
      setBoard(next);
      setError("");
    } catch (reason) {
      setError(formatError(reason));
    } finally {
      refreshInFlightRef.current = false;
      if (!options.silent) setBusy("");
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    if (!active) return;
    void refresh({ silent: true });
    const timer = window.setInterval(() => void refresh({ silent: true }), SCOREBOARD_POLL_MS);
    return () => window.clearInterval(timer);
  }, [active]);

  return (
    <section className="primary-panel tool-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">目标记分牌</p>
          <h1>训练目标达成记分牌</h1>
        </div>
        <button className="secondary-button" disabled={Boolean(busy)} onClick={() => void refresh()}>
          刷新
        </button>
      </div>

      <p className="chart-help">
        每个指标只报三件事：现在读数、底线、底线出处。状态是中性事实标签，不代表好坏；
        “未达底线”是提醒，不是失败判定。跨时间比较前，请先看顶部课程状态带。
      </p>

      {error && <div className="inline-alert">{error}</div>}

      {!board && !error && <p className="muted">正在读取记分牌…</p>}

      {board && (
        <>
          <CurriculumBand board={board} context={board.context} />

          {!board.available && (
            <div className="sb-unavailable">
              <strong>记分牌暂不可用</strong>
              <p>可能是当前没有运行、遥测缺失或读取出错。下方“说明与注意”列出了原因。</p>
            </div>
          )}

          {board.families.map((family) => (
            <FamilySection key={family.id} family={family} />
          ))}

          {board.available && board.families.length === 0 && (
            <p className="muted">当前没有可展示的指标族。</p>
          )}

          {board.tensions.length > 0 && <TensionsLegend tensions={board.tensions} />}

          <CoverageSection coverage={board.coverage} />

          {board.notes.length > 0 && <NotesSection notes={board.notes} />}
        </>
      )}
    </section>
  );
}

// 课程状态带：读数的可比前提。值离开这个上下文就没法跨时间比较，所以放在最显眼处。
function CurriculumBand({ board, context }: { board: ScoreboardData; context: ScoreboardContext }) {
  return (
    <section className="scoreboard-band">
      <div className="scoreboard-band-head">
        <div>
          <p className="eyebrow">课程状态带</p>
          <h2>读数的可比前提</h2>
        </div>
        <div className="sb-band-tags">
          <span className={`evidence-pill ${boardStatusPillClass(board.status)}`}>
            {boardStatusLabel(board.status)}
          </span>
          {board.stale && <span className="sb-stale">数据可能不是实时</span>}
        </div>
      </div>
      <div className="scoreboard-band-cells">
        <BandCell label="阶段" value={context.phase || "未知"} />
        <BandCell label="命令模式" value={context.command_mode || "未知"} />
        <BandCell label="激活方向" value={context.active_dirs || "未知"} />
        <BandCell label="地形均值" value={formatMaybeNumber(context.terrain_mean)} />
        <BandCell label="地形最高" value={formatMaybeNumber(context.terrain_max)} />
        <BandCell label="惩罚门" value={formatMaybeNumber(context.penalty_gate)} />
        <BandCell label="域随机等级" value={formatMaybeNumber(context.dr_level)} />
      </div>
      {context.note && <p className="sb-band-note">{context.note}</p>}
      <div className="sb-run-facts">
        <span>运行：{board.run_id || "未知"}</span>
        <span>步数：{formatInteger(board.step)}</span>
        <span>生成：{formatTime(board.generated_at)}</span>
      </div>
    </section>
  );
}

function BandCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="sb-band-cell">
      <span>{label}</span>
      <strong title={value}>{value}</strong>
    </div>
  );
}

function FamilySection({ family }: { family: ScoreboardFamily }) {
  return (
    <section className="tool-section sb-family">
      <div className="sb-family-head">
        <h2>{family.title}</h2>
        <span className="render-source">{family.metrics.length} 项</span>
      </div>
      {family.summary && <p className="sb-family-summary">{family.summary}</p>}
      <div className="scoreboard-metrics">
        {family.metrics.map((metric) => (
          <MetricRow key={metric.key} metric={metric} />
        ))}
        {family.metrics.length === 0 && <p className="muted">这个族当前没有可读指标。</p>}
      </div>
    </section>
  );
}

function MetricRow({ metric }: { metric: ScoreboardMetric }) {
  return (
    <div className="sb-metric">
      <div className="sb-metric-head">
        <div className="sb-metric-title">
          <strong>{metric.label}</strong>
          <span className="sb-key">{metric.key}</span>
        </div>
        <span className={`sb-status ${metric.status}`}>{metricStatusLabel(metric.status)}</span>
      </div>
      <div className="sb-metric-facts">
        <div className="sb-fact">
          <span>现在</span>
          <strong title={metric.reading}>{metric.reading || "—"}</strong>
          {metric.value_confidence !== "high" && (
            <em className="sb-vconf" title="读数可信度">
              {valueConfidenceLabel(metric.value_confidence)}
            </em>
          )}
        </div>
        <div className="sb-fact">
          <span>底线</span>
          <strong title={metric.floor_label}>{metric.floor_label || "未定"}</strong>
        </div>
        <div className="sb-fact">
          <span>出处</span>
          <strong title={metric.floor_source}>{metric.floor_source || "—"}</strong>
          <ProvenanceTag confidence={metric.floor_confidence} />
        </div>
      </div>
      {metric.meaning && <p className="sb-meaning">{metric.meaning}</p>}
      {metric.tensions.length > 0 && (
        <p className="sb-tensions" title="改善这个指标，可能会拖累下面这些指标">
          拆台：{metric.tensions.join("、")}
        </p>
      )}
      {metric.note && <p className="sb-note">{metric.note}</p>}
    </div>
  );
}

// 出处可信度标签：spec/config/derived 是可信来源；default 是配置缺失时的兜底；
// unknown 必须显眼报警，以免把凭空来的底线当成真的验收目标。
function ProvenanceTag({ confidence }: { confidence: ScoreboardMetric["floor_confidence"] }) {
  const info = PROVENANCE_LABELS[confidence] ?? PROVENANCE_LABELS.unknown;
  return (
    <span className={`sb-prov ${confidence}`} title={info.hint}>
      {info.text}
    </span>
  );
}

const PROVENANCE_LABELS: Record<string, { text: string; hint: string }> = {
  spec: { text: "定义规格", hint: "来自规格定义，可信" },
  config: { text: "本次配置", hint: "来自本次训练的实际配置，可信" },
  derived: { text: "推导得出", hint: "由本次配置推导得出，可信" },
  default: { text: "兜底默认", hint: "配置里没读到，用了兜底默认值" },
  unknown: { text: "⚠出处不明", hint: "底线来源不明，不要当成真实验收目标" },
};

function TensionsLegend({ tensions }: { tensions: ScoreboardTension[] }) {
  return (
    <section className="tool-section">
      <h2>相互拆台的目标</h2>
      <p className="sb-family-summary">
        这些指标彼此牵制：改善一个，另一个可能变差。看板不替你取舍，只把牵制关系摆出来。
      </p>
      <div className="sb-tensions-legend">
        {tensions.map((tension, index) => (
          <div className="sb-tension" key={`${tension.a}-${tension.b}-${index}`}>
            <span className="sb-tension-pair">
              {tension.a} ↔ {tension.b}
            </span>
            <span className="sb-tension-reason">{tension.reason}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

// 覆盖自检：看板主动报告自己还没覆盖到的奖励项。这是刻意做的功能，用来暴露盲区。
function CoverageSection({ coverage }: { coverage: ScoreboardCoverage }) {
  const mapped = coverage.mapped_reward_terms ?? [];
  const unmapped = coverage.unmapped_reward_terms ?? [];
  const total = mapped.length + unmapped.length;
  return (
    <section className="tool-section">
      <div className="sb-family-head">
        <h2>覆盖自检</h2>
        <span className="render-source">{mapped.length} / {total} 已覆盖</span>
      </div>
      {coverage.note && <p className="sb-family-summary">{coverage.note}</p>}
      {unmapped.length > 0 && (
        <div className="sb-coverage-block">
          <p className="sb-coverage-title">板子还没覆盖的目标</p>
          <div className="gap-list">
            {unmapped.map((term) => (
              <span key={term}>{term}</span>
            ))}
          </div>
        </div>
      )}
      {mapped.length > 0 && (
        <div className="sb-coverage-block">
          <p className="sb-coverage-title">板子已经覆盖的奖励项</p>
          <div className="note-list">
            {mapped.map((term) => (
              <span key={term}>{term}</span>
            ))}
          </div>
        </div>
      )}
      {total === 0 && <p className="muted">当前没有可核对的奖励项映射。</p>}
    </section>
  );
}

function NotesSection({ notes }: { notes: string[] }) {
  return (
    <section className="tool-section">
      <h2>说明与注意</h2>
      <div className="note-list">
        {notes.map((note, index) => (
          <span key={`${index}-${note}`}>{note}</span>
        ))}
      </div>
    </section>
  );
}

function metricStatusLabel(status: ScoreboardMetric["status"]): string {
  const labels: Record<string, string> = {
    meets: "达标",
    below: "未达底线",
    no_floor: "未定底线",
    no_data: "无数据",
  };
  return labels[status] ?? status;
}

function boardStatusLabel(status: ScoreboardData["status"]): string {
  const labels: Record<string, string> = {
    ok: "正常",
    watch: "关注",
    blocked: "受阻",
    missing: "缺数据",
    error: "错误",
  };
  return labels[status] ?? status;
}

function boardStatusPillClass(status: ScoreboardData["status"]): string {
  const classes: Record<string, string> = {
    ok: "ok",
    watch: "warning",
    blocked: "warning",
    missing: "missing",
    error: "error",
  };
  return classes[status] ?? "missing";
}

function valueConfidenceLabel(confidence: ScoreboardMetric["value_confidence"]): string {
  const labels: Record<string, string> = {
    high: "读数高",
    medium: "读数中",
    low: "读数低",
  };
  return labels[confidence] ?? confidence;
}

function formatMaybeNumber(value: number | null): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  const digits = abs >= 100 ? 0 : abs >= 10 ? 1 : 2;
  return value.toFixed(digits);
}

function formatInteger(value: number): string {
  return Number.isFinite(value) ? Math.round(value).toString() : "—";
}

function formatTime(ts: number): string {
  if (!ts || !Number.isFinite(ts)) return "未知";
  return new Date(ts * 1000).toLocaleString("zh-CN", { hour12: false });
}
