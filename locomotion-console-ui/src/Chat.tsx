import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from "react";
import {
  cancelChatProposal,
  chat,
  executeAction,
  getLLMReadiness,
  resetChat,
  type Grounding,
  type ContextEnvelopeInfo,
  type LLMReadinessInfo,
  type ProposedAction,
} from "./api";
import { formatError } from "./i18n/format";

export interface ChatMessage {
  role: "user" | "assistant";
  text: string;
  tools?: string[];
  action?: ProposedAction | null;
  grounding?: Grounding | null;
  contextEnvelope?: ContextEnvelopeInfo | null;
  elapsed_s?: number;
  suggestions?: string[];
}

export const INITIAL_CHAT_MESSAGES: ChatMessage[] = [{
  role: "assistant",
  text: "我是当前训练迭代的智能体。请给目标或问题；我会先读取证据，再给判断或待授权提案。",
}];

export default function Chat({
  mode = "training",
  context = "",
  messages,
  setMessages,
}: {
  mode?: "training" | "diagnostics";
  context?: string;
  messages: ChatMessage[];
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
}) {
  const [readiness, setReadiness] = useState<LLMReadinessInfo | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const modeRef = useRef(mode);
  const contextRef = useRef(context);

  useEffect(() => {
    modeRef.current = mode;
    contextRef.current = context;
  }, [mode, context]);

  useEffect(() => {
    void refreshReadiness();
    return () => abortRef.current?.abort();
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy]);

  useEffect(() => {
    function onSendCommand(event: Event) {
      const detail = (event as CustomEvent<unknown>).detail;
      const text = typeof detail === "string" ? detail : "";
      if (text.trim()) void send(text);
    }
    window.addEventListener("locomotion-console-send-command", onSendCommand);
    return () => window.removeEventListener("locomotion-console-send-command", onSendCommand);
  }, []);

  async function refreshReadiness() {
    try {
      setReadiness(await getLLMReadiness());
    } catch {
      setReadiness(null);
    }
  }

  async function send(value = input) {
    const text = value.trim();
    if (!text || busy) return;
    setMessages((current) => [...current, { role: "user", text }]);
    setInput("");
    setBusy("chat");
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const response = await chat(text, modeRef.current, contextRef.current, controller.signal, {
        page: modeRef.current,
        intent: text,
        focus: focusTermsFor(text, modeRef.current),
        visible: parseContext(contextRef.current),
        max_chars: 4200,
        include_remote_status: true,
      });
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          text: response.reply,
          tools: response.transcript.map((item) => item.tool || "").filter(Boolean),
          action: response.proposed_action,
          grounding: response.grounding,
          contextEnvelope: response.context_envelope,
          elapsed_s: response.elapsed_s,
          suggestions: response.suggestions,
        },
      ]);
      await refreshReadiness();
    } catch (reason) {
      const aborted = reason instanceof Error && reason.name === "AbortError";
      setMessages((current) => [...current, {
        role: "assistant",
        text: aborted ? "已取消等待。后端如果已经进入模型调用，会在超时边界自然结束。" : `请求失败：${formatError(reason)}`,
      }]);
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setBusy(null);
    }
  }

  async function confirm(action: ProposedAction) {
    setBusy("execute");
    try {
      const response = await executeAction(action.name, action.args);
      setMessages((current) => [...current, {
        role: "assistant",
        text: response.ok ? `已执行：${response.detail}` : `执行失败：${response.detail}`,
      }]);
      await refreshReadiness();
    } catch (reason) {
      setMessages((current) => [...current, { role: "assistant", text: `执行失败：${formatError(reason)}` }]);
    } finally {
      setBusy(null);
    }
  }

  async function cancel(action: ProposedAction) {
    setBusy("cancel");
    try {
      await cancelChatProposal(action.name);
      setMessages((current) => [...current, { role: "assistant", text: `已取消提案：${action.name}` }]);
    } catch (reason) {
      setMessages((current) => [...current, { role: "assistant", text: `取消失败：${formatError(reason)}` }]);
    } finally {
      setBusy(null);
    }
  }

  async function resetConversation() {
    if (busy) return;
    setBusy("reset");
    try {
      await resetChat();
      setMessages([{ role: "assistant", text: "对话和待授权提案已清空。请重新给出目标或问题。" }]);
      await refreshReadiness();
    } catch (reason) {
      setMessages((current) => [...current, { role: "assistant", text: `清空失败：${formatError(reason)}` }]);
    } finally {
      setBusy(null);
    }
  }

  const quickPrompts = [
    "判断失败点",
    "生成调参提案",
    "检查配置是否生效",
  ];
  const quickCommands: Record<string, string> = {
    判断失败点: "基于当前工作台证据，一句话判断训练失败点。",
    生成调参提案: "从第一性原理给出下一轮调参提案，必须包含证据、风险、回滚。",
    检查配置是否生效: "检查当前配置链路是否真的生效：本地配置、上传包、运行目录配置、最终生效配置。",
  };

  return (
    <section className="side-panel chat-panel compact-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">智能体</p>
          <h2>证据驱动交互</h2>
        </div>
        <span className={`agent-state ${readiness?.ok ? "ok" : "warn"}`}>
          {readiness?.ok ? "LLM 就绪" : "LLM 待检查"}
        </span>
      </div>

      <div className="chat-tools">
        {quickPrompts.map((prompt) => (
          <button key={prompt} disabled={Boolean(busy)} onClick={() => void send(`/ask ${quickCommands[prompt]}`)}>
            {prompt}
          </button>
        ))}
      </div>

      <div className="chat-log" aria-live="polite">
        {messages.map((message, index) => (
          <article className={`chat-message ${message.role}`} key={`${message.role}-${index}`}>
            <div className="chat-text">{renderText(message.text)}</div>
            {message.tools && message.tools.length > 0 && (
              <details className="chat-evidence">
                <summary>读取证据：{compactTools(message.tools)}</summary>
                <code>{message.tools.join(" -> ")}</code>
              </details>
            )}
            {message.contextEnvelope && (
              <details className="chat-evidence">
                <summary>上下文路由：{message.contextEnvelope.selected_sources.join(" / ") || "无"}</summary>
                <code>
                  {`page=${message.contextEnvelope.page}; injected=${message.contextEnvelope.injected_chars}/${message.contextEnvelope.budget_chars}`}
                </code>
              </details>
            )}
            {message.grounding?.checked && (
              <span className={`grounding ${message.grounding.grounded ? "ok" : "warn"}`}>
                {message.grounding.grounded ? "有工具依据" : "存在未支撑信息"}
              </span>
            )}
            {typeof message.elapsed_s === "number" && <small className="elapsed">{message.elapsed_s.toFixed(1)}s</small>}
            {message.action && (
              <div className="proposal-actions">
                <button className="primary-button" disabled={Boolean(busy)} onClick={() => void confirm(message.action!)}>
                  确认执行 {message.action.name}
                </button>
                <button className="secondary-button" disabled={Boolean(busy)} onClick={() => void cancel(message.action!)}>
                  取消
                </button>
              </div>
            )}
            {message.suggestions && message.suggestions.length > 0 && (
              <div className="suggestions">
                {message.suggestions.slice(0, 3).map((suggestion) => (
                  <button key={suggestion} disabled={Boolean(busy)} onClick={() => void send(suggestion)}>
                    {suggestion}
                  </button>
                ))}
              </div>
            )}
          </article>
        ))}
        {busy && (
          <div className="chat-thinking">
            <span>{busy === "chat" ? "智能体正在读取证据" : "正在处理"}</span>
            {busy === "chat" && (
              <button className="secondary-button" onClick={() => abortRef.current?.abort()}>取消等待</button>
            )}
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="chat-input">
        <input
          value={input}
          disabled={Boolean(busy)}
          placeholder="输入目标、问题或 /ask ..."
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") void send();
          }}
        />
        <button className="primary-button" disabled={Boolean(busy) || !input.trim()} onClick={() => void send()}>
          发送
        </button>
        <button className="secondary-button" disabled={Boolean(busy)} onClick={() => void resetConversation()}>
          清空
        </button>
      </div>
    </section>
  );
}

function renderText(text: string) {
  const paragraphs = text.split(/\n{2,}/).map((item) => item.trim()).filter(Boolean);
  if (paragraphs.length === 0) return <p />;
  return paragraphs.map((paragraph, index) => {
    const lines = paragraph.split("\n").map((line) => line.trim()).filter(Boolean);
    const bullets = lines.filter((line) => /^[-*]\s+/.test(line));
    if (bullets.length >= 2) {
      return (
        <div key={`${index}-${paragraph.slice(0, 12)}`}>
          {lines.some((line) => !/^[-*]\s+/.test(line)) && (
            <p>{lines.find((line) => !/^[-*]\s+/.test(line))}</p>
          )}
          <ul>{bullets.map((line) => <li key={line}>{line.replace(/^[-*]\s+/, "")}</li>)}</ul>
        </div>
      );
    }
    return <p key={`${index}-${paragraph.slice(0, 12)}`}>{paragraph}</p>;
  });
}

function compactTools(tools: string[]) {
  const unique = [...new Set(tools.filter(Boolean))];
  if (unique.length <= 3) return unique.join(" -> ");
  return `${unique.slice(0, 3).join(" -> ")} +${unique.length - 3}`;
}

function focusTermsFor(text: string, mode: string) {
  const focus = new Set<string>();
  focus.add(mode);
  const lower = text.toLowerCase();
  for (const item of [
    "remote", "ssh", "gpu", "vram", "tmux", "timeout", "diagnostic", "terrain",
    "reward", "gait", "checkpoint", "payload", "config", "log", "telemetry",
    "远端", "显存", "诊断", "地形", "奖励", "步态", "检查点", "配置", "日志",
  ]) {
    if (lower.includes(item.toLowerCase())) focus.add(item);
  }
  return [...focus].slice(0, 10);
}

function parseContext(context: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const part of context.split("|")) {
    const [rawKey, ...rest] = part.trim().split("=");
    const key = rawKey?.trim();
    const value = rest.join("=").trim();
    if (key && value) out[key] = value;
  }
  return out;
}
