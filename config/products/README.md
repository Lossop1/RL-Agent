# Product Inputs

Files in this directory describe a robot product and its task adapter. They
are inputs to `autotuner.product`, not the implementation of the generic
execution system.

Each product declaration may reference:

- robot assets and their expected hashes;
- task/config/source roots owned by that product adapter;
- the logical task identity and, separately, the concrete runtime task IDs
  registered by the external simulator;
- runtime identity and resume-compatibility fields;
- payload and diagnostic entry points.

The resolver produces a `ResolvedProductContract`. Payload builders consume
that contract and write generated files under their own build/output roots.
Do not import files from `output/`, `strategy_backups/`, or a remote checkout
back into this directory.
