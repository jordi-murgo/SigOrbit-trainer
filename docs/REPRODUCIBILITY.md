# Reproducibility

The project provides a best-effort deterministic protocol, not a promise of
byte-identical results across GPU models, drivers, Torch/e2cnn versions, or
dataset reconstructions.

Every run records the normalized configuration digest, dataset manifest digest,
class map, seeds, package/runtime versions, device information, deterministic
flags, aggregate metrics, and artifact hashes. Recovery checkpoints are distinct
from deployable checkpoints.

Use one of these claim labels:

- **historical**: recorded by the original spike, not independently reproduced;
- **attempted reproduction**: same declared recipe, but data/runtime equivalence
  is incomplete;
- **verified on authorized corpus**: authorized data fingerprint and protocol
  matched, with results inside predeclared tolerances.

The historical initializer was itself the result of a resumed discrete-rotation
run. The new protocol makes backbone bootstrap a first-class stage; therefore a
new result is not expected to be byte-identical to the historical checkpoint.

## Recovery semantics

Recovery checkpoints are atomically written safetensors plus strict JSON. Resume
is supported at completed epoch boundaries and requires the latest checkpoint in
the same run directory. It fails closed on configuration, dataset, class-map,
architecture, run-ID, digest, or CUDA-topology mismatch. Mid-epoch interruption
replays the incomplete epoch from its previous boundary in this alpha release.

## Evaluation protocol

Model selection uses only signer-disjoint validation identities with the
historical `(top1, median margin)` lexicographic rule. Per-angle retrieval and
rotated-embedding cosine are reported but never drive selection. The `test`
split is reserved for release candidates and requires `--allow-test-split`.
