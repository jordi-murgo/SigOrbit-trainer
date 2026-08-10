# c8-257-final — auditable from-scratch SigOrbit encoder run

Checkpoint `sigorbit-c8-257-retrained-v1.pt`, sha256 `f3f0885a79ab54f266b938ca41dd1d6c8f276637e272e4209b5d7726e5c8adbf`.

Trained end to end from random init on RTX PRO 6000 Blackwell, 2026-08-08:
backbone 40 epochs -> canonicalizer pretraining 10 -> joint 80. No external
checkpoint seeds the run. Run seed `31337`. Dataset deduplicated, `manifest_sha256`
`65eb267c8c13a678cf5fded01f8e0b61a51a73b047e44e4128dcd637874f5928`
(5,939 / 610 / 580 across 250 / 32 / 33 signers).

## Results (validation, deduplicated, 610 queries)

| | top-1 | margin | fragile |
|---|---|---|---|
| clean | 99.7% | +0.2333 | 0.66% |
| backbone alone (best, epoch 38) | 99.8% | +0.2227 | 0.2% |

Pose pretraining reached MAE 40.9 deg at +-180 (reference: 41.6).

### Rotation — clean enrolled references, rotated query

This is the deployment shape. The trainer's own `rotation_retrieval` rotates the
whole gallery at once, which an equivariant model handles almost for free; it
reports 99.8% at every angle and does not predict field behaviour.

| angle | model-canvas (pad to square, no ink lost) | raw-expand | reference 129 raw-expand |
|---|---|---|---|
| 0 | 96.6% | 99.7% | — |
| 5 | — | 99.5% | 99.0% |
| 20 | — | 98.9% | 95.1% |
| 30 | 96.6% | 97.9% | 89.3% |
| 45 | 96.2% | 97.0% | **81.7%** |
| 60 | 96.4% | 97.2% | — |
| 90 | 96.4% | 100.0% | — |
| 180 | 96.7% | 99.7% | — |

Model-canvas is flat within 1pp from 0 to 180 deg: rotation invariance is
effectively total. raw-expand beats the 129 reference by ~15pp at 45 deg.

Rotating a wide signature inside its own rectangular frame is a destructive
transform, not an orientation test: it truncates the ink and collapses top-1 to
4.8% at 90 deg while recovering to 99.7% at 180 deg, where a rectangle fits its
own bounds exactly. Pad to square first.

## What was measured and rejected

| attempt | result | verdict |
|---|---|---|
| backbone 80 epochs instead of 40 | +0.2356 vs +0.2383 | no gain for 2x cost |
| seeding stage 1 from `equivariant_rotaug.pt` | — | breaks auditability; its own provenance is a resume, not reproducible |
| `run.deterministic = true` on CUDA | crashes at the joint stage | grid_sampler_2d_backward_cuda has no deterministic kernel |
| `precision = "fp32"` (TF32 off) | 3.5x slower on Ampere+ | buys no reproducibility; TF32 is bitwise stable |
| joint `patience = 16` | stops at epoch 18, selects epoch 2 | early stopping is incompatible with the rotation curriculum |

Three independent 40-epoch backbones landed at +0.2383, +0.2356 and +0.2227.
The margin ceiling looks architectural, not a matter of schedule.

## Open

- Clean margin +0.2333 against ~+0.273 for the reference once its 20.6%
  validation duplication is discounted.
- Test split (33 signers) never touched.
- Thresholds not recalibrated for this checkpoint.
- Negative rotation angles and multiple reference selections not swept.
