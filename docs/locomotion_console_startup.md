# Locomotion Console 启动参数

本文档集中记录 Taili Locomotion Console 的本地启动命令和相关启动参数。

## 本地开发启动

从仓库根目录开始，开两个终端分别启动后端和前端。

后端，使用 fake 假数据模式：

```powershell
cd D:\RL-Dog\hand\locomotion-workspace_backup\locomotion-workspace
pip install -r requirements.txt
$env:LOCOMOTION_CONSOLE_SOURCE = "fake"
python -m autotuner.locomotion_console
```

前端：

```powershell
cd D:\RL-Dog\hand\locomotion-workspace_backup\locomotion-workspace\locomotion-console-ui
npm install
npm run dev
```

默认访问地址：

- 后端：`http://localhost:8000`
- 前端：`http://localhost:5173`

`fake` 模式使用合成遥测数据，不需要 SSH、GPU、IsaacLab 或远端训练机器。

## 真实远端模式

复制 SSH 配置模板，然后填入远端机器信息：

```powershell
Copy-Item config\ssh.json.example config\ssh.json
```

至少需要配置这些字段：

```json
{
  "ssh_host": "203.0.113.10",
  "ssh_port": 22,
  "ssh_user": "root",
  "ssh_pass": ""
}
```

如果 `config/ssh.json` 里有非空的 `ssh_host`，并且没有显式设置 `LOCOMOTION_CONSOLE_SOURCE`，后端会自动选择 `source=real`。

强制使用真实远端模式：

```powershell
$env:LOCOMOTION_CONSOLE_SOURCE = "real"
python -m autotuner.locomotion_console
```

即使存在 `config/ssh.json`，也强制使用 fake 模式：

```powershell
$env:LOCOMOTION_CONSOLE_SOURCE = "fake"
python -m autotuner.locomotion_console
```

不要提交 `config/ssh.json`，这个文件可能包含远端机器凭据。

## 后端环境变量

这些变量需要在运行 `python -m autotuner.locomotion_console` 前设置。

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `LOCOMOTION_CONSOLE_HOST` | `127.0.0.1` | 后端绑定地址。只有明确需要对外暴露时才用 `0.0.0.0`。 |
| `LOCOMOTION_CONSOLE_PORT` | `8000` | 后端 HTTP 和 WebSocket 端口。 |
| `LOCOMOTION_CONSOLE_RELOAD` | `0` | 设置为 `1`、`true` 或 `True` 时启用 uvicorn 自动重载。默认关闭，稳定性更好。 |
| `LOCOMOTION_CONSOLE_SOURCE` | 自动 | `fake` 表示本地合成数据，`real` 表示 SSH 远端数据源。自动模式下，有 `ssh_host` 就用 `real`，否则用 `fake`。 |
| `LOCOMOTION_CONSOLE_POLL_S` | `1.0` | WebSocket 指标流轮询间隔，单位秒。 |
| `LOCOMOTION_CONSOLE_RUN` | 空 | 可选的 run 目录名过滤子串。空值表示使用最新 run。 |
| `LOCOMOTION_CONSOLE_FRAMEWORK` | 项目默认值 | 当前启用的 framework profile id。也可能由 UI 保存的选择提供。 |
| `LOCOMOTION_CONSOLE_REMOTE_LOG` | 空或配置文件值 | 要 tail 的远端训练日志路径。设置后会覆盖 `config/ssh.json`。 |
| `LOCOMOTION_CONSOLE_LOG_FORMAT` | `skrl` | 日志解析格式 key，也可以在 `config/ssh.json` 中设置。 |
| `LOCOMOTION_CONSOLE_UI_ORIGIN` | `http://localhost:5173` | 前后端分离开发时允许的前端 CORS origin。 |
| `LOCOMOTION_CONSOLE_STATE_ROOT` | 系统临时目录 | 本地状态目录，用于保存 console 配置、远端 profile、LLM profile、审计日志和 campaign 状态。 |

示例：把后端改到另一个本地端口。

```powershell
$env:LOCOMOTION_CONSOLE_SOURCE = "fake"
$env:LOCOMOTION_CONSOLE_PORT = "8010"
python -m autotuner.locomotion_console
```

后端端口改变后，前端也要指向新的 API 地址：

```powershell
cd locomotion-console-ui
$env:VITE_API_BASE = "http://localhost:8010"
npm run dev
```

## 远端 SSH 配置字段

`config/ssh.json` 会被 real 模式、adapter 工具和 training 工具读取。

| 字段 | 说明 |
|---|---|
| `ssh_host` | 远端主机名或 IP。非空时 console 自动选择 real 模式。 |
| `ssh_port` | SSH 端口，通常是 `22`。 |
| `ssh_user` | SSH 用户名，当前项目惯例为 `root`。 |
| `ssh_pass` | SSH 密码。能用密钥时优先使用密钥认证。 |
| `ssh_auto_add_host_key` | 可选布尔值。`false` 表示严格校验 host key，`true` 表示首次连接时自动信任。 |
| `work_dir` | 远端 robot 或 IsaacLab 工作目录。 |
| `conda_env` | 远端 conda 环境名。 |
| `conda_sh_path` | 远端 conda shell hook 路径。 |
| `shell_init` | 可选的远端 shell 初始化片段。 |
| `remote_log_path` | 远端训练日志路径。未设置 `LOCOMOTION_CONSOLE_REMOTE_LOG` 时使用此值。 |
| `log_format` | 日志格式 key。未设置 `LOCOMOTION_CONSOLE_LOG_FORMAT` 时使用此值。 |
| `diagnostic_python` | 远端诊断脚本使用的 Python。 |
| `diagnostic_tool_root` | 远端诊断工具包根目录。 |
| `diagnostic_output_root` | 远端诊断输出目录。 |
| `diagnostic_robot_root` | 远端诊断使用的机器人项目根目录。 |

## 诊断相关环境变量

这些变量会覆盖远端配置文件中的同名含义。

| 变量 | 默认值 |
|---|---|
| `LOCOMOTION_CONSOLE_DIAG_TOOL_ROOT` | `/root/gpufree-data/tools/isaaclab_quad_diag_v051` |
| `LOCOMOTION_CONSOLE_DIAG_OUTPUT_ROOT` | `/root/gpufree-data/diag_runs` |
| `LOCOMOTION_CONSOLE_DIAG_ROBOT_ROOT` | `/root/gpufree-data/robot_lab` |
| `LOCOMOTION_CONSOLE_DIAG_PYTHON` | `/opt/conda/envs/isaaclab/bin/python` |
| `LOCOMOTION_CONSOLE_DIAG_TASK` | framework profile 默认值 |

## 鉴权和 Token 参数

当前前端在有 token 时会给 API 请求带上 `X-Console-Token`。

后端参数：

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `LOCOMOTION_CONSOLE_REQUIRE_TOKEN` | 空，关闭 | 设置为 `1`、`true` 或 `True` 时，对会修改状态的 HTTP 请求启用 token 校验。 |
| `LOCOMOTION_CONSOLE_TOKEN` | 空 | 启用 token 校验后，后端期望 `X-Console-Token` 与该值一致。 |

前端参数：

| 变量或存储项 | 说明 |
|---|---|
| `VITE_CONSOLE_TOKEN` | Vite 构建或开发时注入的备用 token。 |
| `localStorage.LOCOMOTION_CONSOLE_TOKEN` | 浏览器运行时 token。这个值优先级高于 `VITE_CONSOLE_TOKEN`。 |

PowerShell 示例：

```powershell
$env:LOCOMOTION_CONSOLE_REQUIRE_TOKEN = "1"
$env:LOCOMOTION_CONSOLE_TOKEN = "change-me"
python -m autotuner.locomotion_console
```

```powershell
cd locomotion-console-ui
$env:VITE_CONSOLE_TOKEN = "change-me"
npm run dev
```

## 前端参数

Vite 前端位于 `locomotion-console-ui/`。

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `VITE_API_BASE` | 空 | API base URL。空值表示同源请求。前后端分离开发时可设置为 `http://localhost:8000`。 |
| `VITE_CONSOLE_TOKEN` | 空 | `X-Console-Token` 的可选备用 token。 |

常用命令：

```powershell
cd locomotion-console-ui
npm install
npm run dev
npm run verify
npm run build
npm run preview
```

## LLM 和 Agent 参数

LLM 配置从 `config/llm.json` 读取，然后叠加 UI 保存到 `LOCOMOTION_CONSOLE_STATE_ROOT` 下的覆盖配置。

推荐的 `config/llm.json` 结构：

```json
{
  "api_key_env_var": "DEEPSEEK_API_KEY",
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-chat",
  "timeout_s": 35.0,
  "temperature": 0.0
}
```

`api_key_env_var` 建议填写环境变量名，不要把真实 API key 写进仓库配置。

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `LOCOMOTION_CONSOLE_LLM_TIMEOUT_S` | 配置中的 `timeout_s` 或 `35.0` | OpenAI 兼容 LLM 客户端超时时间。 |
| `LOCOMOTION_CONSOLE_LLM_AUDIT_DIR` | state-root 下的审计目录 | LLM 调用审计 JSON 文件目录。 |
| `LOCOMOTION_CONSOLE_AGENT_MAX_STEPS` | `5` | 每次聊天请求允许的最大内部 agent/tool 步数。 |
| `LOCOMOTION_CONSOLE_AGENT_TIMEOUT_S` | `70` | 聊天 agent 总超时时间，单位秒。 |
| `LOCOMOTION_CONSOLE_VERIFY_GROUNDING` | `0` | 设置为 `1`、`true` 或 `yes` 时，对回复执行 grounding 自检。 |
| `LOCOMOTION_CONSOLE_GROUNDING_TIMEOUT_S` | `12` | grounding 自检超时时间，单位秒。 |
| `LOCOMOTION_CONSOLE_LLM_AUTONOMY` | `advisory` | `advisory` 只提出建议，`assisted` 可自动执行低风险动作，`autonomous` 可自动执行低/中风险动作。破坏性动作不会自动执行。 |

## Docker 启动

容器启动：

```powershell
docker compose -f docker/docker-compose.yml up --build
```

暴露端口：

- 后端：`8000`
- 前端：`5173`

Compose 默认参数：

| 变量 | 默认值 |
|---|---|
| `LOCOMOTION_CONSOLE_SOURCE` | `fake` |
| `LOCOMOTION_CONSOLE_HOST` | `0.0.0.0` |
| `LOCOMOTION_CONSOLE_PORT` | `8000` |
| `FRONTEND_PORT` | `5173` |
| `LOCOMOTION_CONSOLE_UI_ORIGIN` | `http://localhost:5173` |

Docker 中使用 real 模式时，把 `config/ssh.json` 放在仓库的 `config/` 目录下，然后设置：

```powershell
$env:LOCOMOTION_CONSOLE_SOURCE = "real"
docker compose -f docker/docker-compose.yml up --build
```

## 快速命令

本地 fake console：

```powershell
$env:LOCOMOTION_CONSOLE_SOURCE = "fake"
python -m autotuner.locomotion_console
```

真实远端 console：

```powershell
$env:LOCOMOTION_CONSOLE_SOURCE = "real"
python -m autotuner.locomotion_console
```

代码修改后稳定重启后端：

```powershell
python -m autotuner.locomotion_console
```

后端开发时启用自动重载：

```powershell
$env:LOCOMOTION_CONSOLE_RELOAD = "1"
python -m autotuner.locomotion_console
```
