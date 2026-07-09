# Locomotion Console Progress Record

The Locomotion Console is a local FastAPI and React operations surface for RL locomotion runs. It reaches the remote IsaacLab box through the existing SSH/tmux path and keeps the deterministic control path outside the LLM loop.

## Current State

- Backend package: `autotuner/locomotion_console/`
- Frontend package name: `locomotion-console-ui`
- Backend entrypoint: `python -m autotuner.locomotion_console`
- Runtime env vars: `LOCOMOTION_CONSOLE_*`
- Diagnostic tmux sessions: `locomotion_console_diag`, `locomotion_console_train`, `locomotion_console_eval`

## Diagnostic Playback

The viewer is driven by diagnostic `record.csv` output, not static report posture values. The backend exposes `GET /diagnostics/playback`, parses continuous diagnostic rows, downsamples them, and returns IsaacLab base pose, joint positions, foot world positions, contact state, force, clearance, stage, case, segment, and command metadata.

The frontend `RobotViewer` reads that endpoint, applies the base pose and 12 URDF joint values, and advances through frames with play/pause controls.

## Run Locally

```powershell
$env:LOCOMOTION_CONSOLE_SOURCE = "real"
python -m autotuner.locomotion_console
```

```powershell
cd locomotion-console-ui
npm.cmd run dev
```

Use `LOCOMOTION_CONSOLE_SOURCE=fake` only for UI development without the remote GPU.
