import { zhCN } from "./zh-CN";

const ERROR_PREFIXES: Array<[RegExp, string]> = [
  [/^Error:\s*/i, ""],
  [/^请求失败\s*\((\d+)\)$/i, "请求失败（HTTP $1）"],
  [/^status\s+(\d+)$/i, "请求失败（HTTP $1）"],
  [/^config\s+(\d+)$/i, "配置读取失败（HTTP $1）"],
  [/^frameworks\s+(\d+)$/i, "框架列表读取失败（HTTP $1）"],
  [/^chat\s+(\d+)$/i, "LLM 对话失败（HTTP $1）"],
  [/^execute\s+(\d+)$/i, "动作执行失败（HTTP $1）"],
  [/^TypeError:\s*Failed to fetch$/i, "无法连接后端，请确认服务已启动。"],
  [/^Failed to fetch$/i, "无法连接后端，请确认服务已启动。"],
];

const STATUS_LABELS: Record<string, string> = {
  draft: "草稿",
  validated: "已验证",
  reference: "参考",
  legacy: "旧版",
  missing: "缺失",
  configured: "已配置",
  ok: "正常",
  warning: "警告",
  error: "错误",
  unknown: "未知",
};

const BACKEND_TEXT_REPLACEMENTS: Array<[RegExp, string]> = [
  [/remote is unavailable/gi, "远程机不可用"],
  [/active framework/gi, "当前框架"],
  [/diagnostic catalog unavailable/gi, "诊断目录不可用"],
  [/no eval result yet/gi, "暂无评估结果"],
  [/eval still running/gi, "评估仍在运行"],
  [/done/gi, "已完成"],
  [/checkpoint not found/gi, "未找到检查点"],
  [/training is running/gi, "训练正在运行"],
  [/training stopped/gi, "训练已停止"],
  [/remote reachable/gi, "远程可达"],
  [/checkpoint available/gi, "检查点可用"],
  [/No state change/gi, "无状态变更"],
  [/inspection only/gi, "仅检查"],
  [/deploy-ready/gi, "可部署"],
  [/not ready/gi, "未就绪"],
  [/round-trip/gi, "回读校验"],
  [/sim2real hazard fixed/gi, "已修复 sim2real 隐患"],
];

export function formatError(reason: unknown): string {
  let text = reason instanceof Error ? reason.message : String(reason);
  for (const [pattern, replacement] of ERROR_PREFIXES) {
    if (pattern.test(text)) text = text.replace(pattern, replacement);
  }
  return text || "未知错误";
}

export function formatBackendText(text: string | null | undefined): string {
  if (!text) return "";
  let next = text;
  for (const [pattern, replacement] of BACKEND_TEXT_REPLACEMENTS) {
    next = next.replace(pattern, replacement);
  }
  return next;
}

export function formatBackendList(items: string[]): string {
  return formatList(items.map(formatBackendText));
}

export function formatStatus(status: string | null | undefined): string {
  if (!status) return "—";
  return STATUS_LABELS[status] ?? zhCN.diagnostics.state[status] ?? status;
}

export function formatEvidenceStatus(status: string): string {
  return STATUS_LABELS[status] ?? status;
}

export function formatList(items: string[]): string {
  return items.length ? items.join("；") : "—";
}
