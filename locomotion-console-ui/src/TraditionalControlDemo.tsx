import { useEffect, useMemo, useState } from "react";
import RobotViewer from "./RobotViewer";
import {
  cancelTraditionalControlRun,
  getTraditionalControlCatalog,
  getTraditionalControlHistory,
  getTraditionalControlPlayback,
  getTraditionalControlManifest,
  getTraditionalControlResult,
  getTraditionalControlStatus,
  startTraditionalControlRun,
  type DiagnosticPlayback,
  type TraditionalControlCatalog,
  type TraditionalControlJobStatus,
  type TraditionalControlResult,
  DEFAULT_TRADITIONAL_CONTROL_PLAYBACK_FRAMES,
} from "./api";
import { formatError } from "./i18n/format";

const POLL_MS = 700;

export default function TraditionalControlDemo({ active = true }: { active?: boolean }) {
  const [catalog, setCatalog] = useState<TraditionalControlCatalog | null>(null);
  const [job, setJob] = useState<TraditionalControlJobStatus | null>(null);
  const [playback, setPlayback] = useState<DiagnosticPlayback | null>(null);
  const [result, setResult] = useState<TraditionalControlResult | null>(null);
  const [manifest, setManifest] = useState<Record<string, unknown> | null>(null);
  const [historyRuns, setHistoryRuns] = useState<TraditionalControlJobStatus[]>([]);
  const [productId, setProductId] = useState("");
  const [controllerId, setControllerId] = useState("");
  const [sceneId, setSceneId] = useState("");
  const [duration, setDuration] = useState(3);
  const [dataSourceId, setDataSourceId] = useState("");
  const [replayId, setReplayId] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const products = catalog?.products ?? [];
  const product = useMemo(
    () => products.find((item) => item.id === productId) ?? products[0] ?? null,
    [products, productId],
  );
  const controllers = product?.controllers ?? [];
  const selectedController = useMemo(
    () => controllers.find((item) => item.id === controllerId) ?? controllers[0] ?? null,
    [controllers, controllerId],
  );
  const productScenes = product?.scenes ?? [];
  const scenes = useMemo(() => {
    if (!product || !selectedController) return [];
    return productScenes.filter((item) =>
      (!item.controller_ids.length || item.controller_ids.includes(selectedController.id))
      && (!item.provider_ids.length || item.provider_ids.some((id) => selectedController.provider_ids.includes(id)))
    );
  }, [product, productScenes, selectedController]);
  const selectedScene = useMemo(
    () => scenes.find((item) => item.id === sceneId) ?? scenes[0] ?? null,
    [scenes, sceneId],
  );
  const dataSources = useMemo(() => {
    if (!product || !selectedController || !selectedScene) return [];
    return (product.data_sources ?? []).filter((item) =>
      selectedController.provider_ids.includes(item.provider_id)
      && selectedScene.provider_ids.includes(item.provider_id)
    );
  }, [product, selectedController, selectedScene]);
  const replayRuns = historyRuns.filter(
    (item) => item.state === "complete"
      && item.product_id === product?.id
      && item.controller_id === selectedController?.id
      && item.scene_id === selectedScene?.id,
  ) ?? [];
  const selectedSource = dataSources.find((item) => item.id === dataSourceId)
    ?? dataSources.find((item) => item.available)
    ?? null;

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [next, current] = await Promise.all([
          getTraditionalControlCatalog(),
          getTraditionalControlStatus(),
        ]);
        if (cancelled) return;
        setCatalog(next);
        setHistoryRuns(await loadHistoryOrFallback(next.recent_runs));
        setError(next.message || "");
        if (current.state !== "idle") {
          setJob(current);
          if (current.product_id) setProductId(current.product_id);
          if (current.controller_id) setControllerId(current.controller_id);
          if (current.scene_id) setSceneId(current.scene_id);
          if (current.data_source_id) setDataSourceId(current.data_source_id);
          if (current.run_id && current.state === "complete") {
            const [frames, values, record] = await Promise.all([
              getTraditionalControlPlayback(DEFAULT_TRADITIONAL_CONTROL_PLAYBACK_FRAMES, current.run_id),
              getTraditionalControlResult(current.run_id),
              getTraditionalControlManifest(current.run_id),
            ]);
            if (!cancelled) {
              setPlayback(frames);
              setResult(values);
              setManifest(record);
            }
          }
        } else if (!productId && next.products?.[0]) {
          setProductId(next.products[0].id);
        }
      } catch (reason) {
        if (!cancelled) setError(formatError(reason));
      }
    }
    void load();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!product) return;
    const source = dataSources.find((item) => item.id === dataSourceId)
      ?? dataSources.find((item) => item.available);
    if (source && source.id !== dataSourceId) setDataSourceId(source.id);
    if (!controllers.some((item) => item.id === controllerId)) {
      setControllerId(controllers[0]?.id ?? "");
    }
    if (!scenes.some((item) => item.id === sceneId)) {
      setSceneId(scenes[0]?.id ?? "");
    }
  }, [product, controllerId, sceneId, dataSourceId, dataSources, scenes]);

  useEffect(() => {
    if (!selectedScene) return;
    setDuration(selectedScene.duration_s);
  }, [selectedScene]);

  useEffect(() => {
    if (selectedSource?.kind !== "replay") {
      if (replayId) setReplayId("");
      return;
    }
    if (replayId && !replayRuns.some((item) => item.run_id === replayId)) {
      setReplayId("");
    }
  }, [selectedSource, replayId, replayRuns]);

  useEffect(() => {
    if (!active || !job?.run_id || !["starting", "running"].includes(job.state)) return;
    let cancelled = false;
    const timer = window.setInterval(async () => {
      try {
        const next = await getTraditionalControlStatus(job.run_id);
        if (cancelled) return;
        setJob(next);
        if (next.state === "complete") {
          const [frames, values, record, refreshedCatalog] = await Promise.all([
            getTraditionalControlPlayback(DEFAULT_TRADITIONAL_CONTROL_PLAYBACK_FRAMES, next.run_id),
            getTraditionalControlResult(next.run_id),
            getTraditionalControlManifest(next.run_id),
            getTraditionalControlCatalog(),
          ]);
          const history = await loadHistoryOrFallback(refreshedCatalog.recent_runs);
          if (!cancelled) {
            setPlayback(frames);
            setResult(values);
            setManifest(record);
            setCatalog(refreshedCatalog);
            setHistoryRuns(history);
          }
        }
      } catch (reason) {
        if (!cancelled) setError(formatError(reason));
      }
    }, POLL_MS);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [active, job?.run_id, job?.state]);

  async function start() {
    if (!product || !selectedController || !selectedScene || !selectedSource) return;
    setBusy("run");
    setError("");
    setPlayback(null);
    setResult(null);
    setManifest(null);
    try {
      const next = await startTraditionalControlRun({
        product_id: product.id,
        controller_id: selectedController.id,
        scene_id: selectedScene.id,
        duration_s: Math.max(0.01, Math.min(600, duration)),
        data_source_id: selectedSource.id,
        replay_id: selectedSource.kind === "replay" ? replayId || null : null,
      });
      setJob(next);
      if (next.state === "complete" && next.run_id) {
        const [frames, values, record, refreshedCatalog] = await Promise.all([
          getTraditionalControlPlayback(DEFAULT_TRADITIONAL_CONTROL_PLAYBACK_FRAMES, next.run_id),
          getTraditionalControlResult(next.run_id),
          getTraditionalControlManifest(next.run_id),
          getTraditionalControlCatalog(),
        ]);
        const history = await loadHistoryOrFallback(refreshedCatalog.recent_runs);
        setPlayback(frames);
        setResult(values);
        setManifest(record);
        setCatalog(refreshedCatalog);
        setHistoryRuns(history);
      }
    } catch (reason) {
      setError(formatError(reason));
    } finally {
      setBusy("");
    }
  }

  async function cancel() {
    setBusy("cancel");
    try {
      setJob(await cancelTraditionalControlRun());
    } catch (reason) {
      setError(formatError(reason));
    } finally {
      setBusy("");
    }
  }

  async function openHistory(item: TraditionalControlJobStatus) {
    if (!item.run_id || !item.playback_available) return;
    setBusy(`history:${item.run_id}`);
    setError("");
    try {
      const [status, frames] = await Promise.all([
        getTraditionalControlStatus(item.run_id),
        getTraditionalControlPlayback(DEFAULT_TRADITIONAL_CONTROL_PLAYBACK_FRAMES, item.run_id),
      ]);
      const [values, record] = await Promise.allSettled([
        getTraditionalControlResult(item.run_id),
        getTraditionalControlManifest(item.run_id),
      ]);
      setJob(status);
      setPlayback(frames);
      setResult(values.status === "fulfilled" ? values.value : null);
      setManifest(record.status === "fulfilled" ? record.value : null);
      if (status.product_id) setProductId(status.product_id);
      if (status.controller_id) setControllerId(status.controller_id);
      if (status.scene_id) setSceneId(status.scene_id);
      if (status.data_source_id) setDataSourceId(status.data_source_id);
    } catch (reason) {
      setError(formatError(reason));
    } finally {
      setBusy("");
    }
  }

  return (
    <section className="traditional-control-workspace">
      <div className="primary-panel traditional-control-header">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">传统控制</p>
            <h1>传统控制测试与回放</h1>
            <p className="panel-copy">选择控制器和场景执行测试，并查看机器人、足端接触与地形的完整回放。</p>
          </div>
          {job && <span className={`status-chip ${job.state}`}>{job.state}</span>}
        </div>
        {error && <div className="inline-alert">{error}</div>}
        <div className="traditional-control-form">
          <label>
            产品
            <select value={product?.id ?? ""} onChange={(event) => setProductId(event.target.value)} disabled={!catalog}>
              {products.map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}
            </select>
          </label>
          <label>
            控制器
            <select value={selectedController?.id ?? ""} onChange={(event) => setControllerId(event.target.value)} disabled={!product}>
              {controllers.map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}
            </select>
          </label>
          <label>
            场景
            <select value={selectedScene?.id ?? ""} onChange={(event) => setSceneId(event.target.value)} disabled={!product}>
              {scenes.map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}
            </select>
          </label>
          <label>
            时长 (s)
            <input type="number" min="0.01" max="600" step="0.01" value={duration} onChange={(event) => setDuration(Number(event.target.value))} />
          </label>
          <label>
            数据源
            <select value={dataSourceId} onChange={(event) => setDataSourceId(event.target.value)} disabled={!catalog}>
              {dataSources.map((item) => <option value={item.id} key={item.id} disabled={!item.available}>{item.label}</option>)}
            </select>
          </label>
          {selectedSource?.kind === "replay" && (
            <label>
              回放记录
              <select value={replayId} onChange={(event) => setReplayId(event.target.value)}>
                <option value="">最近一次</option>
                {replayRuns.map((item) => <option value={item.run_id ?? ""} key={item.run_id}>{item.run_id}</option>)}
              </select>
            </label>
          )}
          <button className="primary-button traditional-control-run" onClick={() => void start()} disabled={Boolean(busy) || !selectedController || !selectedScene || !selectedSource?.available}>
            {busy ? "准备中..." : selectedSource?.kind === "replay" ? "加载回放" : "启动演示"}
          </button>
          {job && ["starting", "running"].includes(job.state) && (
            <button className="secondary-button traditional-control-run" onClick={() => void cancel()} disabled={Boolean(busy)}>
              取消
            </button>
          )}
        </div>
        {job && (
          <div className="traditional-control-status">
            <span>{job.message}</span>
            <progress value={job.progress} max={1} />
            <span>{Math.round(job.progress * 100)}%</span>
            {job.error && <code>{job.error}</code>}
          </div>
        )}
        {result && (
          <div className="traditional-control-result">
            <strong>{result.verdict === "passed" ? "运行通过" : result.verdict === "failed" ? "运行未通过" : "结果已生成"}</strong>
            <span>{result.summary}</span>
            {result.metrics.map((metric) => (
              <span key={metric.id}>{metric.label} {formatMetric(metric.value, metric.precision, metric.unit)}</span>
            ))}
            <span>失败原因 {result.failure_reasons.length ? result.failure_reasons.join(", ") : "无"}</span>
            <span>请求时长 {formatSeconds(job?.requested_duration_s)}</span>
            <span>实际仿真 {formatSeconds(job?.simulated_duration_s)}</span>
            <span>完成步数 {formatSteps(job?.completed_steps, job?.requested_steps)}</span>
            {isEarlyTermination(job) && <strong className="traditional-control-terminated">提前终止</strong>}
            {job?.termination_reason && <span>终止原因 {job.termination_reason}</span>}
            <span>manifest {job?.manifest_path || "--"}</span>
            <code>{job?.result_path || ""}</code>
            <small>provider {String(manifest?.provider_id ?? "--")} · trace schema {String(manifest?.provenance && (manifest.provenance as Record<string, unknown>).trace_schema || "--")}</small>
          </div>
        )}
        {manifest && (
          <details className="traditional-control-manifest">
            <summary>运行记录</summary>
            <pre>{JSON.stringify(manifest, null, 2)}</pre>
          </details>
        )}
        <section className="traditional-control-history" aria-labelledby="traditional-control-history-title">
          <div className="traditional-control-history-heading">
            <div>
              <p className="eyebrow">历史回放</p>
              <h2 id="traditional-control-history-title">最近测试记录</h2>
            </div>
            <span>{historyRuns.length} 条</span>
          </div>
          {historyRuns.length ? (
            <div className="traditional-control-history-list">
              {historyRuns.map((item) => (
                <button
                  type="button"
                  className={item.run_id === job?.run_id ? "active" : ""}
                  onClick={() => void openHistory(item)}
                  disabled={!item.playback_available || Boolean(busy) || Boolean(job && ["starting", "running"].includes(job.state))}
                  key={item.run_id}
                  title={item.playback_available ? "打开并播放这次测试" : "这次测试没有可用回放"}
                >
                  <span className={`traditional-control-history-verdict ${item.verdict}`}>{historyVerdict(item)}</span>
                  <span className="traditional-control-history-main">
                    <strong>{historyTitle(item, products)}</strong>
                    <small>{formatRunTime(item.created_at)} · {item.run_id}</small>
                  </span>
                  <span className="traditional-control-history-measure">
                    {formatSeconds(item.simulated_duration_s)} / {formatSeconds(item.requested_duration_s)}
                    <small>实际 / 请求 · 步数 {formatSteps(item.completed_steps, item.requested_steps)}</small>
                  </span>
                  <span className="traditional-control-history-action">
                    {busy === `history:${item.run_id}` ? "加载中..." : item.playback_available ? "查看" : "无回放"}
                  </span>
                </button>
              ))}
            </div>
          ) : (
            <p className="traditional-control-history-empty">还没有测试记录。</p>
          )}
        </section>
      </div>
      <RobotViewer playback={playback} active={active} mode="traditional-control" />
    </section>
  );
}

function isEarlyTermination(job: TraditionalControlJobStatus | null) {
  if (!job) return false;
  if (job.completed_steps != null && job.requested_steps != null) {
    return job.completed_steps < job.requested_steps;
  }
  if (job.simulated_duration_s != null && job.requested_duration_s != null) {
    return job.simulated_duration_s + 0.001 < job.requested_duration_s;
  }
  return false;
}

function formatSeconds(value: number | null | undefined) {
  return value == null || !Number.isFinite(value) ? "--" : `${value.toFixed(2)} s`;
}

function formatSteps(completed: number | null | undefined, requested: number | null | undefined) {
  if (completed == null && requested == null) return "--";
  return `${completed ?? "--"} / ${requested ?? "--"}`;
}

function formatRunTime(timestamp: number | null | undefined) {
  if (timestamp == null || !Number.isFinite(timestamp)) return "时间未知";
  return new Date(timestamp * 1000).toLocaleString("zh-CN", { hour12: false });
}

function historyVerdict(item: TraditionalControlJobStatus) {
  if (item.state === "error") return "执行错误";
  if (item.state === "cancelled") return "已取消";
  if (item.verdict === "passed") return "通过";
  if (item.verdict === "failed") return "未通过";
  return item.state === "complete" ? "已完成" : "未完成";
}

function historyTitle(
  item: TraditionalControlJobStatus,
  products: TraditionalControlCatalog["products"],
) {
  const product = products.find((candidate) => candidate.id === item.product_id);
  const controller = product?.controllers.find((candidate) => candidate.id === item.controller_id);
  const scene = product?.scenes.find((candidate) => candidate.id === item.scene_id);
  return `${controller?.label ?? item.controller_id ?? "未知控制器"} · ${scene?.label ?? item.scene_id ?? "未知场景"}`;
}

async function loadHistoryOrFallback(fallback: TraditionalControlJobStatus[]) {
  try {
    return (await getTraditionalControlHistory()).items;
  } catch {
    return Array.isArray(fallback) ? fallback : [];
  }
}

function formatMetric(value: unknown, precision: number, unit: string) {
  const numeric = typeof value === "number" && Number.isFinite(value);
  const text = numeric ? value.toFixed(Math.max(0, Math.min(6, precision))) : String(value ?? "--");
  return `${text}${unit ? ` ${unit}` : ""}`;
}
