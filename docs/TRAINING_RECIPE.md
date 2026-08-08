# Historical 257 px recipe

The declared selected run used C8 widths `24,48,96,128`, 256-D embeddings,
FP32, PK batches `P=8,K=4`, ArcFace scale 16/margin 0.35, 10 epochs of
angle-only pretraining, and 40 joint epochs. Joint rotation ramps from ±10° to
±180° over 20 epoch indices. Identity, orientation, and cosine-consistency loss
weights are `1.0,0.5,0.5`.

Important historical details retained as explicit compatibility settings:

- backbone discrete PIL rotations are one-sided `0,15,30,45,60,75` degrees;
- tensor rotations are fixed-canvas, bicubic, and historically use normalized
  zero (gray) padding;
- direct square resize distorts aspect ratio;
- clean source images are treated as the zero-angle pose;
- the historical selected model was chosen on clean validation top-1/margin,
  while the new protocol also reports rotation metrics.

See configuration comments; changing these creates a different recipe.

## Interpolation contract

For historical augmentation compatibility, train-time square resize is bilinear
and `RandomAffine` is nearest-neighbour; expanded discrete PIL rotation remains
bicubic with white fill. Validation uses SigOrbit's production bicubic square
preprocessing, intentionally measuring the deployed input contract rather than
the spike's bilinear evaluation transform. This is a declared protocol change.
