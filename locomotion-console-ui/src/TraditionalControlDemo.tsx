import { useEffect, useMemo, useState } from "react";
import RobotViewer from "./RobotViewer";
import {
  cancelTraditionalControlRun,
  getTraditionalControlCatalog,
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
  const [productId, setProductId] = useState("");
  const [controllerId, setControllerId] = useState("");
  const [sceneId, setSceneId] = useState("");
  const [duration, setDuration] = useState(3);
  const [dataSourceId, setDataSourceId] = useState("");
  const [replayId, setReplayId] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const product = useMemo(
    () => catalog?.products.find((item) => item.id === productId) ?? catalog?.products[0] ?? null,
    [catalog, productId],
  );
  const selectedController = useMemo(
    () => product?.controllers.find((item) => item.id === controllerId) ?? product?.controllers[0] ?? null,
    [product, controllerId],
  );
  const scenes = useMemo(() => {
    if (!product || !selectedController) return [];
    return product.scenes.filter((item) =>
      (!item.controller_ids.length || item.controller_ids.includes(selectedController.id))
      && (!item.provider_ids.length || item.provider_ids.some((id) => selectedController.provider_ids.includes(id)))
    );
  }, [product, selectedController]);
  const selectedScene = useMemo(
    () => scenes.find((item) => item.id === sceneId) ?? scenes[0] ?? null,
    [scenes, sceneId],
  );
  const dataSources = useMemo(() => {
    if (!product || !selectedController || !selectedScene) return [];
    return product.data_sources.filter((item) =>
      selectedController.provider_ids.includes(item.provider_id)
      && selectedScene.provider_ids.includes(item.provider_id)
    );
  }, [product, selectedController, selectedScene]);
  const replayRuns = catalog?.recent_runs.filter(
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
        } else if (!productId && next.products[0]) {
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
    if (!product.controllers.some((item) => item.id === controllerId)) {
      setControllerId(product.controllers[0]?.id ?? "");
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
          const frames = await getTraditionalControlPlayback(DEFAULT_TRADITIONAL_CONTROL_PLAYBACK_FRAMES, next.run_id);
          const values = await getTraditionalControlResult(next.run_id);
          const record = await getTraditionalControlManifest(next.run_id);
          if (!cancelled) {
            setPlayback(frames);
            setResult(values);
            setManifest(record);
            setCatalog(await getTraditionalControlCatalog());
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
        setPlayback(frames);
        setResult(values);
        setManifest(record);
        setCatalog(refreshedCatalog);
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

  return (
    <section className="traditional-control-workspace">
      <div className="primary-panel traditional-control-header">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">传统控制</p>
            <h1>本机无头演示</h1>
            <p className="panel-copy">通过统一回放协议查看控制器输出、足端接触和场景几何。</p>
          </div>
          {job && <span className={`status-chip ${job.state}`}>{job.state}</span>}
        </div>
        {error && <div className="inline-alert">{error}</div>}
        <div className="traditional-control-form">
          <label>
            产品
            <select value={product?.id ?? ""} onChange={(event) => setProductId(event.target.value)} disabled={!catalog}>
              {catalog?.products.map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}
            </select>
          </label>
          <label>
            控制器
            <select value={selectedController?.id ?? ""} onChange={(event) => setControllerId(event.target.value)} disabled={!product}>
              {product?.controllers.map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}
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
      </div>
      <RobotViewer playback={playback} active={active} />
    </section>
  );
}

function formatMetric(value: unknown, precision: number, unit: string) {
  const numeric = typeof value === "number" && Number.isFinite(value);
  const text = numeric ? value.toFixed(Math.max(0, Math.min(6, precision))) : String(value ?? "--");
  return `${text}${unit ? ` ${unit}` : ""}`;
}
