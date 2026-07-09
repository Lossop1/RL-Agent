import { useEffect, useState } from "react";
import {
  getActiveConfigSet,
  getLLMProfile,
  getRemoteProfile,
  getRobotProfile,
  testRemoteProfile,
  type ConfigSetInfo,
  type LLMProfileInfo,
  type RemoteConnectionTestInfo,
  type RemoteProfileInfo,
  type RobotProfileInfo,
} from "./api";
import { formatError } from "./i18n/format";

export default function ConfigWorkspace() {
  const [config, setConfig] = useState<ConfigSetInfo | null>(null);
  const [remote, setRemote] = useState<RemoteProfileInfo | null>(null);
  const [robot, setRobot] = useState<RobotProfileInfo | null>(null);
  const [llm, setLlm] = useState<LLMProfileInfo | null>(null);
  const [remoteTest, setRemoteTest] = useState<RemoteConnectionTestInfo | null>(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    void refresh();
  }, []);

  async function refresh() {
    setBusy("refresh");
    try {
      const [nextConfig, nextRemote, nextRobot, nextLlm] = await Promise.all([
        getActiveConfigSet(),
        getRemoteProfile(),
        getRobotProfile(),
        getLLMProfile(),
      ]);
      setConfig(nextConfig);
      setRemote(nextRemote);
      setRobot(nextRobot);
      setLlm(nextLlm);
      setError("");
    } catch (reason) {
      setError(formatError(reason));
    } finally {
      setBusy("");
    }
  }

  async function testRemote() {
    setBusy("remote-test");
    try {
      setRemoteTest(await testRemoteProfile());
      setError("");
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
          <p className="eyebrow">配置工具</p>
          <h1>确认智能体使用的配置事实</h1>
        </div>
        <button className="secondary-button" disabled={Boolean(busy)} onClick={() => void refresh()}>
          刷新
        </button>
      </div>

      {error && <div className="inline-alert">{error}</div>}

      <section className="tool-section">
        <h2>工作目标</h2>
        <p>{config?.task_goal || "未读取到当前 ConfigSet。"}</p>
        <div className="profile-grid">
          <ProfileCard title="远端" label={config?.remote.label} status={config?.remote.status} detail={config?.remote.detail} />
          <ProfileCard title="机器人" label={config?.robot.label} status={config?.robot.status} detail={config?.robot.detail} />
          <ProfileCard title="框架" label={config?.framework.label} status={config?.framework.status} detail={config?.framework.detail} />
          <ProfileCard title="智能体" label={config?.llm.label} status={config?.llm.status} detail={config?.llm.detail} />
        </div>
      </section>

      <div className="tool-grid">
        <section className="tool-section">
          <h2>远端执行环境</h2>
          <DetailGrid items={[
            ["host", remote ? `${remote.user}@${remote.host}:${remote.port}` : "未读取"],
            ["work_dir", remote?.work_dir || "无"],
            ["conda_env", remote?.conda_env || "无"],
            ["诊断目录", remote?.diagnostic_output_root || "无"],
            ["密码", remote?.password_configured ? "已配置" : "未配置"],
          ]} />
          <div className="button-row">
            <button className="primary-button" disabled={Boolean(busy)} onClick={() => void testRemote()}>
              测试连接
            </button>
          </div>
          {remoteTest && (
            <p className={remoteTest.ok ? "success-text" : "danger-text"}>
              {remoteTest.ok ? "连接可用" : "连接失败"}：{remoteTest.detail}
            </p>
          )}
        </section>

        <section className="tool-section">
          <h2>智能体模型</h2>
          <DetailGrid items={[
            ["model", llm?.model || "未配置"],
            ["base_url", llm?.base_url || "无"],
            ["API key", llm?.api_key_resolved ? "已解析" : "未解析"],
            ["动作模式", llm?.actions_require_confirmation ? "必须确认" : "未强制确认"],
            ["明文风险", llm?.unsafe_literal_key ? "存在" : "无"],
          ]} />
        </section>
      </div>

      <section className="tool-section">
        <h2>机器人证据</h2>
        <DetailGrid items={[
          ["id", robot?.id || "无"],
          ["名称", robot?.label || "无"],
          ["DoF", String(robot?.dof ?? "无")],
          ["base_link", robot?.base_link || "无"],
          ["诊断规格", robot?.diagnostic_spec || "无"],
          ["状态", robot?.valid ? "可用" : "存在问题"],
        ]} />
        {robot?.issues && robot.issues.length > 0 && (
          <div className="gap-list">
            {robot.issues.map((issue) => <span key={issue}>{issue}</span>)}
          </div>
        )}
      </section>
    </section>
  );
}

function ProfileCard({
  title,
  label,
  status,
  detail,
}: {
  title: string;
  label?: string;
  status?: string;
  detail?: string;
}) {
  return (
    <article className="profile-card">
      <span>{title}</span>
      <strong>{label || "未读取"}</strong>
      <small>{status || "unknown"}</small>
      {detail && <p>{detail}</p>}
    </article>
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
