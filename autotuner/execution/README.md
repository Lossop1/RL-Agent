# Execution Layer

This package is system-owned execution code. It is deliberately independent
of a robot product, an IsaacLab task, a reward function, and the web console.

It is the authority for the last-mile handoff, not for deciding what a robot
should learn. Product and task adapters produce a resolved contract and a
payload; this package proves, stages, activates, and records those artifacts.

## Responsibilities

- resolve and record immutable runtime identities;
- build and verify content-addressed payload metadata;
- apply exact, hash-guarded `ChangeSet` objects with rollback journals;
- decide whether a checkpoint has enough identity evidence for `resume`;
- stage immutable runtime and payload directories on a remote host;
- atomically activate a run pointer and roll it back without deleting history.

Runtime identity evidence distinguishes a declared digest from what the
process can observe. Package/version observations have status `observed`; they
are not promoted to `matched` unless the execution infrastructure supplies an
independent runtime-digest assertion. A known `mismatch` blocks preflight.

`DeploymentSpec` is the explicit handoff type. It contains only resolved
runtime identity, payload archive/manifest, and run manifest references. It
does not contain robot-specific imports or a hidden default source tree.

## Boundaries

`autotuner.execution` may use standard-library and dependency-light data
helpers. It must not import `autotuner.product`, a task environment, a robot
asset, a reward implementation, IsaacLab, or the console. Product code calls
these interfaces through resolved contracts.

The dependency direction is one-way:

```text
product manifest -> resolved contract -> payload builder -> execution spec
                                                  -> execution layer -> remote host
```

The execution layer never imports the left side. IsaacLab/Isaac Sim and
PhysX remain external runtime dependencies and are represented by the runtime
identity; they are not copied into this source package.

The product layer declares what runtime and structural identities are needed.
The payload adapter decides which product/task files are packaged. The remote
layer only verifies hashes and executes the resulting artifact; it never edits
the remote source tree in place.

Process lifecycle (tmux/systemd), SSH reconnect policy, diagnostic scheduling,
disk cleanup policy, and human authorization remain injected infrastructure
services. Keeping those concerns outside this package prevents a generic
artifact deployer from acquiring Taili-specific launch assumptions.

## Remote layout

```text
<root>/
  runtimes/<runtime_digest>/
  payloads/<payload_digest>/
  runs/<run_id>/
  active.json
```

Runtime, payload, and run directories are immutable after activation. A
rollback changes `active.json` only, so old payloads and run evidence remain
available for reproduction.

## Repository ownership

| Path | Ownership | Meaning |
| --- | --- | --- |
| `autotuner/execution/` | system source | generic hashes, changesets, compatibility, staging |
| `autotuner/product/` | system contract layer | product manifests and resolved contracts |
| `config/products/` | product input | robot/task asset declarations, not executable runtime code |
| `autotuner/blind_locomotion/` | product adapter | current Taili/IsaacLab adapter and payload recipe |
| `autotuner/training_payloads/` | generated-product boundary | reproducible payload build recipe and archives |
| `output/`, `strategy_backups/`, `docs/archive/` | run/history artifacts | never imported as source |

`autotuner/adapter/` remains a compatibility surface for the older
ConfigSet/file-plan workflow. Its `VersionedPayloadDeployExecutor` is the
bridge to this package; the legacy executor must not be used as the definition
of the new payload deployment contract.

## Failure rules

- A changed base file blocks a `ChangeSet`; it is never guessed around.
- A prepared journal can be recovered, while an unrelated post-change edit
  blocks rollback.
- A checkpoint without complete identity evidence is `blocked`, not silently
  treated as a valid resume.
- A payload archive is verified locally before upload and again by remote
  manifest checks.
