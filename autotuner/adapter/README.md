# Taili Adapter Backend (`autotuner/adapter/`)

This package is currently kept as a Taili-only dry-run support surface for the console.
It derives actuator/geometry/reward-threshold values from the Taili URDF, materializes local
copies of env/asset files, and produces a deploy plan for review.

It is not the active training deployment strategy. The training path should move to the packaged
Taili blind runtime payload under `autotuner/training_payloads/taili_blind_runtime/`, so remote
machines do not depend on a mutable `robot_lab` source tree.

## Command

```bash
python -m autotuner.adapter [taili] [--composition ID] [--save ID|--load ID|--list]
                            [--work-dir DIR] [--execute --confirm [--launch]] [--force]
```

Default mode is a dry-run. It prints:

- derived Taili values;
- consistency report;
- deploy-readiness blockers;
- a deterministic file plan.

`--execute --confirm` is outward-facing: it can write remote files and optionally launch a job.
Keep it gated.

## Current scope

- Current preset: `taili`.
- Current local robot assets: `assets/robots/taili-dog/`.
- Unitree/B2 proof fixtures and tools were moved to `archive/2026-07-cleanup/b2-removed/`.

## Tests

```bash
pytest tests/autotuner/adapter tests/autotuner/framework_library -q
```
