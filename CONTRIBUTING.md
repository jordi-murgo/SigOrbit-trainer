# Contributing

Open an issue before large changes. Use synthetic non-person fixtures only; do
not submit signatures, datasets, manifests, embeddings, checkpoints, weights,
logs with private paths, or credentials. Contributions must be compatible with
MIT distribution and include tests.

```bash
uv sync --extra dev --extra hf
uv run ruff format --check src tests scripts
uv run ruff check src tests scripts
uv run mypy src/sigorbit_trainer
uv run pytest
```

By contributing, you certify that you have the right to submit the code. Dataset
and weight changes are outside the code-only release boundary.
