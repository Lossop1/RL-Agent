# Local Cleanup 2026-08-04

The local workspace was using about 15.76 GiB. The largest consumers were generated
payload staging trees, old payload build products, and runtime logs rather than source
code or the active remote trainer.

Removed generated data:

- `output/target_22000_rework` (about 5.84 GiB)
- `output/target_28750_rework` (about 0.23 GiB)
- `autotuner/training_payloads/taili_blind_runtime/.build` (about 0.80 GiB)
- obsolete files from `autotuner/training_payloads/taili_blind_runtime/dist` (about 1.14 GiB)
- `.runtime_logs` (about 1.70 GiB)

Before removal, the code/config portions of the staging and build trees were archived
without checkpoints, meshes, motion clips, generated telemetry, or nested archives:

`strategy_backups/generated_payload_text_history_20260804.tar.gz`

SHA256:

`0C4C923CAC2098B5C9279B080A43E2F78238324D503024E285BF9634BF674887`

Retained material includes the current candidate staging, diagnostic comparison outputs,
the historical bootstrap archive, selected stair/terrain/quality/DR payload archives, and
the source/docs tree. The remote training run was not modified by this cleanup.
