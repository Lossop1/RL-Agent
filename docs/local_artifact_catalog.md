# Local Artifact Catalog

Updated 2026-08-14. This file describes the boundary between reproducible
project sources and local reinforcement-learning evidence. Large artifacts are
intentionally outside Git, but are retained when they can explain, reproduce,
or validate a training decision.

## Git-Owned Material

Commit source, configuration, tests, reusable tools, patches, and design or
handoff documentation under `autotuner/`, `config/`, `docs/`, `locomotion-console-ui/`,
`tests/`, and `tools/`. The root `.gitignore` excludes generated outputs,
checkpoints, telemetry, credentials, caches, and local staging trees.

`tools/ssh_askpass.cmd` is safe to version because it reads
`TAILI_SSH_PASSWORD` from the environment and contains no credential.

## Verified Local Archive

The archive created on 2026-08-14 is under:

`strategy_backups/archive_20260814/`

Important contents:

- `history7_payloads/`: compressed historical resume-chain payloads and the
  verified missing-directory variants.
- `staging/`: the 2026-08-04 candidate staging and payload comparison trees.
- `logs/`: old console logs retained for troubleshooting history.
- `manifests/SHA256SUMS.txt`: SHA-256 checksums for the archive files.

The archive packages passed a full `tar` listing check before cleanup. The
historical extracted trees were compared against their archives; exact copies
and copies covered by a later archive were removed only after that check.

## Retained Working Evidence

The following remain in place because they are useful for future inspection or
deployment:

- `output/diagnostics/`, `output/experiments/`, `output/monitoring/` and policy
  comparison results.
- `output/deployments/`, `output/deploy/` and `output/training_payloads/`.
- `output/reference_configs/`, `output/source_snapshots/`, and `output/staging/`.
- `strategy_backups/` and the generated payload history archive already stored
  there.
- Root and local checkpoint files are ignored by Git and are not part of source
  commits.

## Removed Regenerable Material

The cleanup removed root pytest caches and temporary directories, verified
duplicate historical extraction trees, generated package-verification and
pytest output, UI concept render output, and plaintext askpass scripts. These
items do not add training knowledge once the source tools and archived evidence
are retained.

When adding future artifacts, keep them only when they answer at least one of
these questions: which policy/configuration produced a result, what physical
behavior was observed, how can the result be reproduced, or what deployment
asset must be recovered. Otherwise put the artifact under an ignored output
directory and remove it after the run.
