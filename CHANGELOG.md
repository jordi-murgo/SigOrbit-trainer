# Changelog

## 0.2.3 - 2026-08-17

- PyPI Mermaid diagrams now use a light theme (white background, light blue
  nodes, dark text) for legibility on PyPI's white page. No source code
  changes.

## 0.2.2 - 2026-08-17

- PyPI README now renders Mermaid diagrams as inline SVG images via
  mermaid.ink instead of unrenderable fenced code blocks. No source code
  changes.

## 0.2.1 - 2026-08-17

- Repository renamed to `jordi-murgo/sigorbit-trainer` (lowercase). All URLs
  in `pyproject.toml`, `CITATION.cff`, `README.md` and scripts updated to
  the new canonical repository path.
- README relative links now rewritten to absolute GitHub URLs at build time
  so they resolve correctly on PyPI. No source code changes.

## 0.2.0 - 2026-08-10

- CUDA BF16 autocast with startup capability validation and FP32 stored weights.
- Joint-stage `min_epochs` gating so finite patience does not compress or
  prematurely stop the configured cosine schedule.
- Bounded recovery storage: retain only latest and per-stage best checkpoints.
- C4 group_order support and batch-64 (P16×K4) configuration; validated run
  achieves +0.2546 margin at 2.7× speedup over the C8 batch-32 baseline.
- `scripts/prepare_dataset.sh`: standalone dataset download, deduplication and
  materialization pipeline.
- Pin `sigorbit>=0.2.0` for C4/C8 architecture support.

## 0.1.0 - 2026-08-08

- Initial offline training pipeline for the SigOrbit C_N canonicalized encoder.
- Strict TOML configuration with a generated JSON Schema and bounded values.
- Local-only dataset manifests, rights attestations, and an offline importer for
  an already-downloaded Hugging Face `DatasetDict`.
- Backbone bootstrap, angle-only pose pretraining, and joint ArcFace training.
- Pickle-free safetensors recovery checkpoints with metadata digests, device
  topology binding, atomic pointers, and epoch-boundary resume.
- SigOrbit export with a CPU-reconstructed parity check.
- Deterministic writer-level random splits with `--random-seed` and a
  `dataset split-preview` command.
- Per-angle retrieval reporting and an opt-in test-split evaluation command.
- Code-only release safeguards, CI matrix, and MIT code boundary documentation.
