# Validated from-scratch 257 px recipe

This document describes two validated runs: the original `c8-257-final` and the
faster `c4-257-b64` variant. The source of truth is `configs/c8-257-final.toml`
and `configs/c4-257-b64.toml`; the archived normalized configurations are in
`artifacts/`.

## Model and data contract

- SigOrbit 0.1 `CanonicalizedEncoder`;
- 257×257 grayscale input, direct bicubic square resize and normalization to
  `[-1,1]`;
- C8 or C4 widths `24,48,96,128`;
- 256-D L2-normalized embeddings;
- PK batches `P=8,K=4` (c8-257-final) or `P=16,K=4` (c4-257-b64);

The trainer imports the architecture from the pinned `sigorbit==0.1.0` package.
ArcFace is a train-only classifier and is absent from the exported encoder.

## Stage 1 — backbone

The `SteerableEncoder` and a 250-class ArcFace head start from random weights.
They train for 40 epochs with cross-entropy over ArcFace logits.

```text
AdamW lr                 1e-3
weight decay             1e-4
LR schedule              2-epoch linear warmup → cosine to zero
ArcFace scale            16
ArcFace margin target    0.35
margin schedule          zero through epoch index 2, then linear warmup
gradient clipping        global norm 5
```

Augmentation applies framing jitter; one expanded PIL rotation sampled from
the configured `discrete_rotations_deg` set; direct 257×257 bicubic resize;
translation, scale and shear jitter with nearest-neighbour interpolation; and
brightness/contrast jitter. The c8-257-final run uses
`0,15,30,45,60,75` degrees (one-sided for compatibility with the historical
recipe); the c4-257-b64 run uses `0,90` degrees. The discrete rotations are
intentionally one-sided.

Validation uses clean signer-disjoint leave-one-out retrieval. Checkpoint
selection clamps top-1 at the configured 99% floor and then maximizes median
genuine-minus-best-impostor margin. Stage 2 restores the best backbone and
ArcFace state, not the last epoch.

## Stage 2 — pose

The restored backbone and ArcFace head are frozen. A freshly
identity-initialized 32,098-parameter canonicalizer trains alone for 10 epochs at
LR `3e-3` with AdamW and weight decay `1e-4`.

For each clean image, the trainer samples a signed angle uniformly from the
current envelope, creates a fixed-canvas bicubic tensor rotation, and supervises
both clean pose `(1,0)` and rotated pose `(cos α,sin α)`. The envelope increases
linearly from ±45° to ±180° across the 10 epochs. The final pose epoch feeds the
joint stage.

## Stage 3 — joint

Canonicalizer, backbone and ArcFace head train together for 80 epochs:

```text
backbone lr              3e-4
canonicalizer lr         1e-3
ArcFace-head lr          1e-3
weight decay             1e-4
LR schedule              5-epoch linear warmup → cosine to zero
rotation envelope        ±10° → ±180° over 20 epoch indices
orientation weight      0.5
consistency weight      0.5
gradient clipping        global norm 5
```

For clean `x`, rotated `Rαx` and signer label `y`:

```text
L = L_arc + 0.5 L_orient + 0.5 L_consist

L_arc     = 0.5 [CE(ArcFace(e(x), y)) + CE(ArcFace(e(Rαx), y))]
L_orient  = circular pose loss for the clean and known rotated poses
L_consist = 1 - cosine(e(x), e(Rαx))
```

Tensor rotation and canonicalizer resampling are fixed-canvas bicubic operations
with normalized zero (gray) padding. The 80-epoch schedule uses
`joint.min_epochs = 50` and `patience = 16`: validation temporarily worsens while
the rotation envelope and ArcFace penalty increase, so ordinary short-patience
early stopping can select a checkpoint before the intended problem has been
presented. The `min_epochs` floor ensures the cosine schedule is not cut short,
while finite patience avoids wasting epochs after the best checkpoint stabilizes.
The export restores the best joint validation checkpoint, not the final state.

## Precision and reproducibility

Parameters and tensors are FP32. `runtime.precision = "tf32"` enables accelerated
CUDA matrix multiplication and convolution where supported. Both validated runs
set `run.deterministic = false` because bicubic
`grid_sampler_2d_backward_cuda` has no deterministic implementation.

The `run.seed` field controls all stochastic sources in a run: weight
initialization (`torch.manual_seed`), NumPy and Python RNG state, PK sampler
order, augmentation jitter and discrete-rotation sampling, and synthetic angle
generation for the pose and joint stages. Both validated runs use seed `31337`.
Changing the seed produces a different initialization and augmentation sequence
but the same dataset split (the split has its own `[data.split] seed`), so
results are comparable across seeds. Recovery checkpoints contain model,
ArcFace, optimizer, scheduler and RNG state as safetensors plus strict JSON
metadata. Resume is exact at a completed epoch boundary only when configuration,
dataset, class map, architecture, run ID and device topology match.

## Selection and export

Only signer-disjoint validation identities select checkpoints. The manifest
`test` split is excluded unless the operator explicitly passes
`--allow-test-split`. The final artifact contains the canonicalizer and backbone,
strict architecture/preprocessing identity and non-executable provenance; it
does not contain ArcFace or optimizer state.

Run and validate either recipe:

```bash
uv run sigorbit-train config validate configs/c8-257-final.toml
uv run sigorbit-train run configs/c8-257-final.toml
uv run sigorbit-train config validate configs/c4-257-b64.toml
uv run sigorbit-train run configs/c4-257-b64.toml
```

Observed metrics and known limitations are recorded in
[`RESULTS_c8-257-final.md`](RESULTS_c8-257-final.md) and
[`RESULTS_c4-257-b64.md`](RESULTS_c4-257-b64.md).
