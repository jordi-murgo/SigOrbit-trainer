# Changelog

## 0.1.0 - Unreleased

- Initial offline training pipeline for the SigOrbit C8 canonicalized encoder.
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
