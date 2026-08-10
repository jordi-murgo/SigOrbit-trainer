# Data policy

SigOrbit Trainer processes handwritten signatures, which are sensitive identity
material and may be biometric data depending on purpose and jurisdiction. The
repository ships no data and performs no network download.

## Required local contract

A non-synthetic run requires:

1. a versioned `dataset.toml` and checksum-bound `samples.jsonl`;
2. per-image SHA-256, dimensions, media type, opaque signer ID, source group,
   split, and genuine/forgery kind;
3. signer-disjoint train/validation/test splits and no source-group leakage;
4. a separate, untracked rights attestation bound to the manifest digest.

The attestation records the operator's assertion; it is not legal validation.
Do not put real names, credentials, consent documents, or absolute paths into a
manifest. The trainer logs aggregate metrics only.

## Historical sources

The selected research run used 7,560 genuine images from CEDAR and BHSig260 via
an aggregate dataset. CEDAR has no verified explicit redistribution grant in
our audit, and BHSig260 is described as available for research purposes. The
aggregate's MIT card does not repair upstream rights. Acquire every source
directly and obtain legal/privacy approval. `scripts/prepare_dataset.sh`
can download an authorized Hugging Face dataset and materialize it locally, but
the operator is responsible for verifying rights before use; the trainer itself
never downloads data at runtime.

## Operational minimum

Use encrypted storage, least-privilege access, a retention/deletion policy,
network-disabled training jobs, private output directories, and an applicable
lawful-basis/consent assessment. Never upload samples or manifests to issues,
CI, experiment trackers, or telemetry services.
