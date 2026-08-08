# SigOrbit Trainer

Offline, auditable training workflows for [SigOrbit](https://github.com/jordi-murgo/SigOrbit).
The project trains the 257 px SO(2)-canonicalized C8 signature encoder from random
initialization without requiring an unpublished initializer.

> **Code-only boundary:** MIT covers this repository's code only. This project
> contains no signatures, datasets, embeddings, checkpoints, or trained weights,
> and does not grant rights to any of them. A public dataset listing or Hugging
> Face card is not evidence of permission.

## Status

**Alpha / not yet published.** The historical 257 px recipe is being converted
into a resumable protocol. Historical accuracy numbers are not a reproduction
claim.

## Pipeline

1. Validate an immutable local manifest and a separate rights attestation.
2. Bootstrap the C8 backbone with PK sampling and ArcFace.
3. Pretrain the SO(2) canonicalizer using known synthetic angles.
4. Jointly optimize identity, circular orientation, and embedding consistency.
5. Evaluate on signer-disjoint validation identities and export an inference-only
   SigOrbit checkpoint.

There are no reflections: the symmetry is SO(2)+C8, not O(2)/D8.

## Install for development

```bash
git clone https://github.com/jordi-murgo/SigOrbit-trainer.git
cd SigOrbit-trainer
uv sync --extra dev
```

## Safe smoke test

The smoke source generates non-person synthetic strokes in memory:

```bash
uv run sigorbit-train config validate configs/smoke.toml
uv run sigorbit-train run configs/smoke.toml
```

Interrupted runs can resume from the directory named by
`RUN/checkpoints/latest.json`:

```bash
sigorbit-train resume CONFIG.toml --checkpoint RUN/checkpoints/stage=...-epoch=...-step=...
```

Resume is exact at a completed epoch boundary when the configuration, dataset,
class map, architecture, device topology, optimizer, scheduler, and RNG schema
match. The current alpha does not checkpoint mid-epoch.

## Training with authorized local data

Training never downloads data. If an authorized dataset is already saved as a
local Hugging Face `DatasetDict`, materialize it into the offline contract:

```bash
sigorbit-train dataset import-hf-disk \
  /secure/source-dataset \
  /secure/sigorbit-dataset \
  --dataset-id authorized-signatures \
  --revision IMMUTABLE_SOURCE_REVISION \
  --assert-genuine-only

sigorbit-train dataset attest \
  /secure/sigorbit-dataset/dataset.toml \
  /secure/sigorbit-dataset/rights.attestation.json \
  --purpose research-only \
  --authorization-reference APPROVAL_REFERENCE \
  --assert-authorized-use

sigorbit-train dataset validate \
  /secure/sigorbit-dataset/dataset.toml \
  --attestation /secure/sigorbit-dataset/rights.attestation.json

sigorbit-train run configs/c8-257-research.toml
```

The importer only reads an already-local directory; it has no Hub download
path. `attest` records the operator's assertion and does not create legal rights.
Keep both materialized data and attestation outside the repository.

The historical CEDAR/BHSig260-derived aggregate may only be used after acquiring
the sources under applicable terms and documenting authorization. Its trained
weights must not be published or used commercially without a separate legal,
privacy, and model-release review.

Read [`docs/DATA_POLICY.md`](docs/DATA_POLICY.md),
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), and
[`docs/MODEL_RELEASE_POLICY.md`](docs/MODEL_RELEASE_POLICY.md) before training.

## Randomized splits

Manifest splits can be repartitioned by writer before training, which is the
recommended way to check that a result is not an artefact of one partition:

```bash
sigorbit-train dataset split-preview CONFIG.toml --random-seed 1234
sigorbit-train run CONFIG.toml --random-seed 1234
```

Splits move whole signers, never individual samples, and the seed enters the
dataset fingerprint so a resume cannot silently change the partition. See
[`docs/DATASET_MANIFEST.md`](docs/DATASET_MANIFEST.md).

## Evaluation

Training reports clean leave-one-out retrieval plus per-angle top-1, median
margin, and fragile rate on validation identities. The manifest `test` split is
never used for selection and requires an explicit opt-in:

```bash
sigorbit-train evaluate CONFIG.toml --checkpoint RUN/model.pt --split validation
sigorbit-train evaluate CONFIG.toml --checkpoint RUN/model.pt --split test --allow-test-split
```

## Relationship with SigOrbit

`sigorbit-trainer` owns data manifests, augmentation, losses, optimization,
resume state, evaluation, and run provenance. `sigorbit` owns the encoder
architecture, inference preprocessing, and runtime checkpoint compatibility.
The trainer pins `sigorbit==0.1.0`; architecture changes require a new trainer
release and model identifier.

## Licence

MIT for this repository's code. Dataset, model-weight, privacy, and third-party
rights are separate. See `NOTICE`.
