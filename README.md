# SigOrbit Trainer

Offline, auditable training workflows for [SigOrbit](https://github.com/jordi-murgo/sigorbit).
The project trains the 257 px SO(2)-canonicalized signature encoder from random
initialization without requiring an unpublished initializer. C8 and C4
steerable backbones are supported; the C4 variant trains 2.7× faster with
comparable margin.

> **Code-only boundary:** MIT covers this repository's code only. This project
> contains no signatures, datasets, embeddings, checkpoints, or trained weights,
> and does not grant rights to any of them. A public dataset listing or Hugging
> Face card is not evidence of permission.

## Status

**Alpha.** The source is published on GitHub; there is no PyPI release yet.
The three-stage trainer, epoch-boundary resume, strict provenance records and a
completed from-scratch 257 px run are implemented. Historical checkpoint metrics
remain historical evidence, not a reproduction claim.

## Architecture and pipeline

The trainer does not define a second neural network. It pins `sigorbit==0.1.0`
and imports the runtime package's `ModelConfig`, `SteerableEncoder` and
`CanonicalizedEncoder` classes:

````mermaid
flowchart TB
    subgraph Input["Preprocessing"]
        IMG["Cropped signature image"]
        GRAY["Grayscale → 257×257 bicubic → [-1, 1]"]
        IMG --> GRAY
    end

    subgraph Canon["OrientationCanonicalizer (SO(2))"]
        direction TB
        CC1["Conv2d 1→16 k5 s2 p2<br/>+ ReLU + BatchNorm2d"]
        CC2["Conv2d 16→32 k5 s2 p2<br/>+ ReLU + BatchNorm2d"]
        CC3["Conv2d 32→64 k3 s2 p1<br/>+ ReLU + BatchNorm2d"]
        CCP["AdaptiveAvgPool2d(1) → Flatten"]
        CLIN["Linear 64→2<br/>→ (cos θ, sin θ), L2-normalized"]
        CAFF["affine_grid + grid_sample<br/>(bicubic, rotation only)"]
        CC1 --> CC2 --> CC3 --> CCP --> CLIN --> CAFF
    end

    subgraph Backbone["SteerableEncoder (C4 or C8-steerable CNN, e2cnn)"]
        direction TB
        STEM["Stem<br/>R2Conv 1→24·N regular k7 p3<br/>+ InnerBatchNorm + ReLU<br/>+ BlurPool /2 (N = group_order)"]
        L1["Layer 1 (/4)<br/>R2Conv 24→48 k5 p2 + IBN + ReLU<br/>R2Conv 48→48 k5 p2 + IBN + ReLU<br/>+ BlurPool /2"]
        L2["Layer 2 (/8)<br/>R2Conv 48→96 k5 p2 + IBN + ReLU<br/>R2Conv 96→96 k5 p2 + IBN + ReLU<br/>+ BlurPool /2"]
        L3["Layer 3 (/16)<br/>R2Conv 96→128 k5 p2 + IBN + ReLU<br/>R2Conv 128→128 k5 p2 + IBN + ReLU<br/>+ BlurPool /2"]
        GP["GroupPooling<br/>max over C4/C8 fiber<br/>→ 128 invariant channels"]
        STEM --> L1 --> L2 --> L3 --> GP
    end

    subgraph Head["Embedding head"]
        direction TB
        POOL["AdaptiveAvgPool2d(1)<br/>→ Flatten → 128"]
        FC1["Linear 128→512<br/>+ BatchNorm1d + ReLU<br/>+ Dropout 0.3"]
        FC2["Linear 512→256<br/>+ BatchNorm1d"]
        NORM["L2-normalize"]
        POOL --> FC1 --> FC2 --> NORM
    end

    GRAY --> CC1
    CAFF --> STEM
    GP --> POOL
    NORM --> OUT["256-D L2-normalized embedding"]

    style Input fill:#e8edf5,color:#1a1a1a
    style Canon fill:#dbe8f5,color:#1a1a1a
    style Backbone fill:#e0f0e0,color:#1a1a1a
    style Head fill:#f5ede0,color:#1a1a1a
    style OUT fill:#f5dede,color:#1a1a1a
````

The end-to-end workflow wraps three training stages:

1. validate an immutable local manifest and a separate rights attestation;
2. train the C4 or C8 backbone from random weights with PK sampling, discrete rotation
   augmentation and a temporary ArcFace head;
3. restore the best backbone and train only the SO(2) canonicalizer against
   known synthetic angles;
4. jointly optimize canonicalizer, backbone and ArcFace head using identity,
   circular-orientation and cosine-consistency objectives;
5. evaluate on signer-disjoint validation identities, restore the best joint
   checkpoint and export an inference-only SigOrbit artifact.

There are no reflections: the symmetry is SO(2)+C_N (N = group_order), not O(2)/D_N.
The ArcFace classifier, optimizers and schedulers exist only in recovery
checkpoints; the deployable artifact contains the canonicalizer and backbone.

## Install for development

```bash
git clone https://github.com/jordi-murgo/sigorbit-trainer.git
cd sigorbit-trainer
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

Runtime controls keep the optimization schedule separate from resource-saving
stops. `joint.epochs` remains the cosine-scheduler horizon, while
`joint.min_epochs` prevents patience from stopping the joint stage before that
floor. `runtime.precision = "bf16"` autocasts CUDA training forwards after a
startup capability check; parameters and exported weights remain FP32.

The checkpoint directory retains only targets referenced by `latest.json` and
the per-stage `best-*.json` pointers. Superseded epoch checkpoints are removed
after each atomic pointer update.

## Training with authorized local data

Training never downloads data inside the trainer process. A standalone script
downloads, deduplicates, and materializes an authorized Hugging Face imagefolder
dataset into the offline contract in one step:

```bash
HF_TOKEN=hf_... scripts/prepare_dataset.sh
```

Options: `HF_DATASET`, `HF_REVISION`, `DATASET_ID`, `DEDUPLICATE`,
`OUTPUT_DIR`, `WORK_DIR`. See `scripts/prepare_dataset.sh --help` for details.

If an authorized dataset is already saved as a local Hugging Face `DatasetDict`,
materialize it manually:

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

uv run sigorbit-train run configs/c8-257-final.toml
```

The importer only reads an already-local directory. `attest` records the
operator's assertion and does not create legal rights. Keep both materialized
data and attestation outside the repository.

The historical CEDAR/BHSig260-derived aggregate may only be used after acquiring
the sources under applicable terms and documenting authorization. Its trained
weights must not be published or used commercially without a separate legal,
privacy, and model-release review.

Read [`docs/DATA_POLICY.md`](docs/DATA_POLICY.md),
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), and
[`docs/MODEL_RELEASE_POLICY.md`](docs/MODEL_RELEASE_POLICY.md) before training.

The validated configurations and observed runs are documented in
[`docs/TRAINING_RECIPE.md`](docs/TRAINING_RECIPE.md),
[`docs/RESULTS_c8-257-final.md`](docs/RESULTS_c8-257-final.md), and
[`docs/RESULTS_c4-257-b64.md`](docs/RESULTS_c4-257-b64.md).

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
sigorbit-train evaluate CONFIG.toml \
  --checkpoint RUN/sigorbit-c8-257-retrained-v1.pt --split validation
sigorbit-train evaluate CONFIG.toml \
  --checkpoint RUN/sigorbit-c8-257-retrained-v1.pt --split test --allow-test-split
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
