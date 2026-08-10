#!/usr/bin/env bash
# Download a Hugging Face imagefolder dataset, deduplicate decoded images, and
# materialize it into the offline sigorbit-trainer manifest contract.
#
# This script is the standalone, reusable version of the dataset pipeline that
# prime_pod_train.sh runs remotely. It does NOT require a Prime pod; it runs
# locally or on any machine with git-lfs and a sigorbit-trainer venv.
#
# Required:
#   HF_TOKEN                 HuggingFace read token (never logged)
#
# Common options (environment variables):
#   HF_DATASET               dataset repo id    (default rakshitdabral/Signature-Verification-Dataset)
#   HF_REVISION              immutable commit   (default 485e3b6f95ef93a9994b93459933770f69a2e554)
#   DATASET_ID                materialized id   (default cedar-bhsig260-research)
#   DEDUPLICATE               yes|no            (default yes)
#   OUTPUT_DIR                materialized dir  (default ./sigorbit-dataset)
#   WORK_DIR                  scratch dir       (default ./sigorbit-dataset-work)
#   DEDUPLICATE_ONLY          yes|no            skip materialize, just write dedup DatasetDict
#
# Usage:
#   HF_TOKEN=hf_... scripts/prepare_dataset.sh
#   HF_TOKEN=hf_... HF_DATASET=other/repo scripts/prepare_dataset.sh
set -euo pipefail

log()  { printf '\033[36m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

:: "${HF_TOKEN:?set HF_TOKEN to a HuggingFace read token}"
command -v git >/dev/null || die "git not found"
command -v uv >/dev/null || die "uv not found (install: curl -LsSf https://astral.sh/uv/install.sh | sh)"

HF_DATASET="${HF_DATASET:-rakshitdabral/Signature-Verification-Dataset}"
HF_REVISION="${HF_REVISION:-485e3b6f95ef93a9994b93459933770f69a2e554}"
DATASET_ID="${DATASET_ID:-cedar-bhsig260-research}"
DEDUPLICATE="${DEDUPLICATE:-yes}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/sigorbit-dataset}"
WORK_DIR="${WORK_DIR:-$(pwd)/sigorbit-dataset-work}"
DEDUPLICATE_ONLY="${DEDUPLICATE_ONLY:-no}"

TRAINER="$(cd "$(dirname "$0")/.." && pwd)"
HF_CLONE="$WORK_DIR/hf_dataset"
DS_DISK="$WORK_DIR/dataset_dedup"

log "trainer: $TRAINER"
log "dataset: $HF_DATASET @ ${HF_REVISION:0:12}"
log "output:  $OUTPUT_DIR"

# ── 1. clone + lfs ────────────────────────────────────────────────────────────
# git clone + lfs pull rather than snapshot_download: the latter repeatedly
# stalled partway through this repo (58%, 95%, 99%) and cannot resume cleanly.
log "cloning dataset via git-lfs"
rm -rf "$HF_CLONE"
GIT_LFS_SKIP_SMUDGE=1 git clone -q \
  "https://oauth2:${HF_TOKEN}@huggingface.co/datasets/${HF_DATASET}" "$HF_CLONE"
cd "$HF_CLONE"
git checkout -q "$HF_REVISION"
git lfs pull
log "images: $(find dataset -name '*.png' | wc -l)"

# ── 2. deduplicate ────────────────────────────────────────────────────────────
log "building DatasetDict (deduplicate=$DEDUPLICATE)"
cd "$TRAINER"
HF_CLONE="$HF_CLONE" DS_DISK="$DS_DISK" DEDUPLICATE="$DEDUPLICATE" \
uv run --extra hf python - << 'PY'
import hashlib, io, os
from datasets import load_dataset

ds = load_dataset("imagefolder", data_dir=os.path.join(os.environ["HF_CLONE"], "dataset"))
# The materializer refuses byte-identical images: BHSig260 shares a handful of
# scans across signer ids, which would make two identities indistinguishable.
if os.environ["DEDUPLICATE"] == "yes":
    for name in list(ds.keys()):
        split, seen, keep = ds[name], set(), []
        for i, row in enumerate(split):
            buf = io.BytesIO()
            row["image"].save(buf, format="PNG")
            digest = hashlib.sha256(buf.getvalue()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                keep.append(i)
        if len(keep) != len(split):
            print(f"{name}: {len(split)} -> {len(keep)} ({len(split)-len(keep)} duplicates)")
        ds[name] = split.select(keep)
import shutil
if os.path.exists(os.environ["DS_DISK"]):
    shutil.rmtree(os.environ["DS_DISK"])
ds.save_to_disk(os.environ["DS_DISK"])
print({name: len(split) for name, split in ds.items()})
PY

if [ "$DEDUPLICATE_ONLY" = "yes" ]; then
  log "DEDUPLICATE_ONLY=yes — dedup DatasetDict at $DS_DISK, not materializing"
  exit 0
fi

# ── 3. materialize -> attest -> validate ──────────────────────────────────────
log "materialize -> attest -> validate"
rm -rf "$OUTPUT_DIR"
uv run --extra hf sigorbit-train dataset import-hf-disk "$DS_DISK" "$OUTPUT_DIR" \
  --dataset-id "$DATASET_ID" --revision "$HF_REVISION" --assert-genuine-only
uv run --extra hf sigorbit-train dataset attest "$OUTPUT_DIR/dataset.toml" \
  "$OUTPUT_DIR/rights.attestation.json" \
  --purpose research-only \
  --authorization-reference operator-self-authorized-research \
  --assert-authorized-use
uv run --extra hf sigorbit-train dataset validate "$OUTPUT_DIR/dataset.toml" \
  --attestation "$OUTPUT_DIR/rights.attestation.json"

log "done: $OUTPUT_DIR"
log "clean up work dir: rm -rf $WORK_DIR"