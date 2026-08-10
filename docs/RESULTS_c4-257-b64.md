# c4-257-b64 — C4 steerable backbone with batch-64 training

Checkpoint `sigorbit-c4-257-b64.pt`, sha256 in `artifacts/c4-257-b64/run.json`.

Trained end to end from random init on RTX PRO 6000 Blackwell, 2026-08-10:
backbone 40 epochs -> canonicalizer pretraining 10 -> joint 80 (early-stopped
at epoch 80 after no improvement for 16 epochs past the 50-epoch floor). No
external checkpoint seeds the run. Run seed `31337`. Same deduplicated dataset
as c8-257-final, `manifest_sha256`
`65eb267c8c13a678cf5fded01f8e0b61a51a73b047e44e4128dcd637874f5928`
(5,939 / 610 / 580 across 250 / 32 / 33 signers).

## Results (validation, deduplicated, 610 queries)

| | top-1 | margin | fragile |
|---|---|---|---|
| clean | 100.0% | +0.2546 | 0.3% |
| backbone alone (best, epoch 31) | 99.7% | +0.2409 | 0.8% |

Pose pretraining reached MAE 42.2 deg at +-180.

Best joint checkpoint: epoch 70, margin +0.2546, top-1 100.0%.

## Comparison with c8-257-final

| | c8-257-final (C8, P8×K4) | c4-257-b64 (C4, P16×K4) |
|---|---|---|
| group order | 8 | 4 |
| discrete rotations | 0,15,30,45,60,75 | 0,90 |
| batch size | 32 | 64 |
| best joint margin | +0.2501 | **+0.2546** |
| best joint top-1 | 100.0% | 100.0% |
| best joint epoch | 45 | 70 |
| best backbone margin | +0.2269 | +0.2409 |
| best backbone epoch | 36 | 31 |
| seconds per epoch (backbone) | 44 s | **19 s** |
| seconds per epoch (joint) | 88 s | **38 s** |
| total wall time | ~2 h 28 min | **~55 min** |
| GPU | RTX PRO 6000 Blackwell | RTX PRO 6000 Blackwell |
| cost (on-demand) | ~$4.50 | **~$1.65** |

C4 with batch 64 matches or exceeds C8 with batch 32 on margin and top-1 while
training 2.7× faster. The SO(2) canonicalizer handles continuous rotation, so
C4 equivariance (90° symmetry) is sufficient for signatures.

## What was measured and rejected

| attempt | result | verdict |
|---|---|---|
| C8 batch 64, LR ×√2 | +0.1363 margin at backbone ep 10 | below ×1 (+0.1491) |
| C8 batch 64, LR ×2 | +0.1447 margin at backbone ep 10 | below ×1 |
| BF16 autocast (C8) | 43 s vs 44 s per epoch | no meaningful speedup |
| C8 batch 64, full run | stopped at joint ep 4 | backbone +0.2409 but joint incomplete |

## Open

- Clean margin +0.2546 against ~+0.273 for the historical reference once its
  20.6% validation duplication is discounted.
- Test split (33 signers) never touched.
- Thresholds not recalibrated for this checkpoint.
- C4 model ID (`sigorbit-c4-257-b64`) is not release-compatible with the C8 v1
  architecture contract; deployment requires a new model release.