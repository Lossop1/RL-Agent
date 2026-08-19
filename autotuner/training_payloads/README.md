# Training Payloads

This directory contains product-specific payload recipes and generated
distribution artifacts. A payload is a build product, not a second copy of
the system source tree.

The build flow is:

```text
product manifest -> resolved contract -> selected payload recipe
    -> content-addressed archive + payload manifest
```

The generic execution layer only consumes the archive and manifest. It does
not discover Taili files, import IsaacLab tasks, or infer which files belong
to another robot such as Taishan. New robots add a product adapter/recipe and
contract data without changing `autotuner.execution`.
