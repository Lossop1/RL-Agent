# Taili Locomotion Console

A production operations surface for Taili blind-locomotion RL training: a FastAPI backend + React
frontend that drives training, diagnostics, and acceptance evaluation on a **remote** IsaacLab GPU box
over SSH/tmux, keeping the deterministic control path outside the LLM loop.

- **Backend:** `autotuner/locomotion_console/` — `python -m autotuner.locomotion_console` (uvicorn on :8000)
- **Frontend source:** `locomotion-console-ui/`; generated output is `frontend/dist/`
- **Training strategy contract:** `autotuner/blind_locomotion/taili_blind_config.yaml` (single editable source)
- **Architecture and maintenance:** [`docs/README.md`](docs/README.md)、[`docs/repository_architecture.md`](docs/repository_architecture.md)
- **Structure gate:** `python tools/check_repository_structure.py`

## 启动项目

以下命令以 Windows PowerShell 为例。后端和前端需要分别在两个终端中运行。

### 1. 首次安装依赖

在仓库根目录执行：

```powershell
pip install -r requirements.txt
cd locomotion-console-ui
npm.cmd install
cd ..
```

### 2. 启动本地后端（无 GPU/SSH）

在第一个终端、仓库根目录执行。`fake` 模式使用本地合成数据，适合查看控制台和传统控制演示：

```powershell
$env:LOCOMOTION_CONSOLE_SOURCE = "fake"
python -m autotuner.locomotion_console
```

后端默认地址为 `http://127.0.0.1:8000`。

### 3. 启动前端

在第二个终端执行：

```powershell
cd locomotion-console-ui
npm.cmd run dev
```

浏览器打开 `http://127.0.0.1:5173`。进入页面的“传统控制”页即可启动本机无头演示并查看回放。
详细操作见 [`docs/traditional_control_demo.md`](docs/traditional_control_demo.md)。

### 4. 连接真实远程训练机（可选）

先复制并填写 SSH 配置（`config/ssh.json` 已被 Git 忽略）：

```powershell
Copy-Item config/ssh.json.example config/ssh.json
```

填写 `ssh_host` 等字段后，在后端终端使用真实数据源启动：

```powershell
$env:LOCOMOTION_CONSOLE_SOURCE = "real"
python -m autotuner.locomotion_console
```

前端启动命令和访问地址不变。SSH 配置和其他启动参数见 [`docs/locomotion_console_startup.md`](docs/locomotion_console_startup.md)。

### Docker 启动

已安装 Docker Compose 时，在仓库根目录执行：

```powershell
docker compose -f docker/docker-compose.yml up --build
```

容器会同时提供后端 `:8000` 和前端 `:5173`，默认使用 `fake` 数据源。

## Security

State-changing endpoints (`POST/PATCH/DELETE`: start/kill/resume training, deploy payloads, edit the
remote SSH target, execute LLM-proposed actions) are **authenticated**:

- Every mutating request must carry an **`X-Console-Token`** header. Its presence forces a CORS
  preflight the restricted `allow_origins` list rejects → defeats cross-origin CSRF simple-POSTs.
- Set **`LOCOMOTION_CONSOLE_TOKEN`** to require a matching secret on the header (constant-time). With
  no token set, only loopback clients may mutate — **so any non-local deployment (incl. the container,
  where the browser is non-loopback) MUST set `LOCOMOTION_CONSOLE_TOKEN`** and provide it to the UI
  (`VITE_CONSOLE_TOKEN` at build time, or `localStorage.LOCOMOTION_CONSOLE_TOKEN` at runtime).
- SSH uses host-key verification by default (`RejectPolicy` + known_hosts; opt-in TOFU via
  `ssh_auto_add_host_key`). Credentials live only in git-ignored `config/ssh.json` (0600), never in the image.
- All remote shell interpolation is `shlex.quote`'d / allowlisted; LLM-supplied args cannot reach a
  remote shell unescaped; robot-import URDF paths are confined to asset roots and parsed with
  DTD/entity rejection.

> `docker/entrypoint.sh` deletes `/.dockerenv` to make the container indistinguishable from a host.
> Nothing in the app reads it; it is left as-is but flagged — for a hardened deployment, remove that line.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest                       # backend: acceptance-scorer, reward/loss math, curriculum, API auth
cd locomotion-console-ui && npm run verify   # frontend: i18n + typecheck + build
```

The reward/curriculum/acceptance math is pure and CPU-testable (no GPU/IsaacLab); the acceptance-scorer
tests pin every spec threshold to `docs/taili_spec.md`. Training + physeval run on the remote box.
