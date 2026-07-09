# Taili Locomotion Console

A production operations surface for Taili blind-locomotion RL training: a FastAPI backend + React
frontend that drives training, diagnostics, and acceptance evaluation on a **remote** IsaacLab GPU box
over SSH/tmux, keeping the deterministic control path outside the LLM loop.

- **Backend:** `autotuner/locomotion_console/` — `python -m autotuner.locomotion_console` (uvicorn on :8000)
- **Frontend:** `locomotion-console-ui/` — Vite/React, built to `frontend/dist/` and served on :5173
- **Training strategy contract:** `autotuner/blind_locomotion/taili_blind_config.yaml` (single editable source)
- **Acceptance spec / design docs:** [`docs/`](./docs) — `taili_spec.md`, `taili_strategy_decisions.md`, `taili_tuning_followups.md`

## Run

### Local dev
```bash
# backend (synthetic source — no GPU/SSH needed)
LOCOMOTION_CONSOLE_SOURCE=fake python -m autotuner.locomotion_console
# frontend dev server (proxies to :8000)
cd locomotion-console-ui && npm install && npm run dev
```
To drive a real remote box, copy `config/ssh.json.example` → `config/ssh.json` (git-ignored) and set
`ssh_host`; the console auto-selects `source=real`.

### Container (reproducible build)
```bash
docker compose -f docker/docker-compose.yml up --build     # backend :8000, frontend :5173
```

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
