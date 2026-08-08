# Local dataset manifest

Core training has no network client. A dataset root contains `dataset.toml`,
`samples.jsonl`, and image paths below that root. Paths are normalized POSIX
relative paths; absolute, escaping, symlinked, animated, multipage, unsupported,
oversized, and checksum-mismatching inputs are rejected.

```toml
schema_version = 1
dataset_id = "authorized-signatures"
revision = "immutable-source-revision"
records = "samples.jsonl"
records_sha256 = "<64 lowercase hex>"
canonical_pose = "upright"
```

Each JSONL row has exactly these fields:

```json
{"sample_id":"sha256:...","image":"images/train/ab/sample.png","image_sha256":"...","media_type":"image/png","width":640,"height":320,"signer_id":"opaque-writer-id","split":"train","kind":"genuine","source_group":"opaque-source-id","canonical_angle_degrees":0.0}
```

The validator requires signer-disjoint splits, no source-group leakage, at least
`K` genuine train samples per writer, and two samples per validation writer.
The training protocol currently rejects forgeries.

The rights attestation is strict JSON, stored outside Git, and bound to the
SHA-256 of `dataset.toml`. It records an operator assertion; it neither embeds
permission documents nor grants rights.

## Randomized writer-level splits

A manifest's own split layout is often arbitrary, and a single fixed partition
can hide overfitting to that particular set of writers. The trainer can
repartition **in memory** without ever rewriting the manifest:

```toml
[data.split]
strategy = "random_by_signer"
seed = 1234
train_fraction = 0.7
validation_fraction = 0.15
test_fraction = 0.15
```

```bash
sigorbit-train dataset split-preview CONFIG.toml --random-seed 1234
sigorbit-train run CONFIG.toml --random-seed 1234
```

Rules enforced by the implementation:

- splits are assigned **per signer**, never per sample, so a writer's images
  cannot straddle train, validation, and test;
- ordering comes from `sha256(seed, signer_id)`, so the result is reproducible
  and independent of manifest order;
- `source_group` must not span signers, otherwise the split is refused;
- fractions must sum to exactly 1.0 and use largest-remainder allocation, with
  at least two signers guaranteed for train and validation;
- every signer needs at least `max(2, K)` genuine samples;
- the seed and resulting assignment digest enter the dataset fingerprint, so a
  resume under a different split fails closed instead of silently continuing.

`--random-seed` is recorded in the resolved configuration and its digest, so the
partition stays auditable. Repeated runs under different seeds are the intended
way to report mean and spread rather than a single lucky partition.
