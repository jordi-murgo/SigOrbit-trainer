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
