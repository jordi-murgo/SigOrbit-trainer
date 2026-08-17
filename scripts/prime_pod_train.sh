#!/usr/bin/env bash
# Provision a Prime Intellect pod and run a full SigOrbit training pipeline on it.
#
# Encodes the whole flow so a run is reproducible from zero instead of a pile of
# ad-hoc SSH commands: provision, install, fetch dataset, deduplicate,
# materialize, attest, validate, smoke test, train.
#
# Required:
#   HF_TOKEN                 HuggingFace read token (never logged)
#
# Common options (environment variables):
#   GPU_ID                   `prime availability list` short id (default 6a235a = RTX 6000 Ada)
#   POD_NAME                 pod name                       (default sigorbit-train)
#   CONFIG                   config in configs/ to run      (default c8-257-research.toml)
#   RUN_NAME                 run + output directory name    (default from CONFIG basename)
#   TRAINER_REF              branch/tag/commit to check out (default main)
#   HF_DATASET               dataset repo id                (default rakshitdabral/Signature-Verification-Dataset)
#   HF_REVISION              immutable dataset commit       (default 485e3b6f95ef93a9994b93459933770f69a2e554)
#   DEDUPLICATE              yes|no                         (default yes)
#   SMOKE                    yes|no, run configs/smoke.toml (default yes)
#   KEEP_POD                 yes|no, keep pod on failure    (default no)
#   STAGE_ONLY               yes|no, prepare but do not train (default no)
#
# Usage:
#   HF_TOKEN=hf_... scripts/prime_pod_train.sh
#   HF_TOKEN=hf_... GPU_ID=ad4b66 CONFIG=c8-257-research.toml scripts/prime_pod_train.sh
#
# Training runs detached under nohup. The script prints the pod address and exits
# once training is confirmed alive; poll with scripts/prime_pod_status.sh.
set -euo pipefail

log()  { printf '\033[36m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

: "${HF_TOKEN:?set HF_TOKEN to a HuggingFace read token}"
command -v prime >/dev/null || die "prime CLI not found"

GPU_ID="${GPU_ID:-6a235a}"
POD_NAME="${POD_NAME:-sigorbit-train}"
CONFIG="${CONFIG:-c8-257-research.toml}"
RUN_NAME="${RUN_NAME:-$(basename "$CONFIG" .toml)-pod}"
TRAINER_REF="${TRAINER_REF:-main}"
TRAINER_URL="${TRAINER_URL:-https://github.com/jordi-murgo/sigorbit-trainer.git}"
HF_DATASET="${HF_DATASET:-rakshitdabral/Signature-Verification-Dataset}"
HF_REVISION="${HF_REVISION:-485e3b6f95ef93a9994b93459933770f69a2e554}"
DATASET_ID="${DATASET_ID:-cedar-bhsig260-research}"
DEDUPLICATE="${DEDUPLICATE:-yes}"
SMOKE="${SMOKE:-yes}"
KEEP_POD="${KEEP_POD:-no}"
STAGE_ONLY="${STAGE_ONLY:-no}"
DISK_GB="${DISK_GB:-200}"
VCPUS="${VCPUS:-16}"
MEMORY_GB="${MEMORY_GB:-72}"

HOME_DIR=/home/ubuntu
TRAINER=$HOME_DIR/SigOrbit-trainer
HF_CLONE=$HOME_DIR/hf_dataset
DS_DISK=$HOME_DIR/signature_dataset_disk
DATA_DIR=$HOME_DIR/sigorbit-dataset
LOG=$HOME_DIR/train.log

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=15)
POD_ID=""
POD=""

# Strip the noisy known-hosts warning without hiding real stderr.
ssh_pod() { ssh "${SSH_OPTS[@]}" "$POD" "$@" 2>&1 | grep -v "Warning: Permanently added"; }
ssh_script() { ssh "${SSH_OPTS[@]}" "$POD" 'bash -seu' 2>&1 | grep -v "Warning: Permanently added"; }

cleanup_on_failure() {
  local code=$?
  [ $code -eq 0 ] && return 0
  if [ -n "$POD_ID" ] && [ "$KEEP_POD" != "yes" ]; then
    log "failed (exit $code) — terminating pod $POD_ID; set KEEP_POD=yes to inspect"
    prime --plain pods terminate "$POD_ID" -y >/dev/null 2>&1 || true
  elif [ -n "$POD_ID" ]; then
    log "failed (exit $code) — pod $POD_ID kept: prime pods terminate $POD_ID -y"
  fi
  return $code
}
trap cleanup_on_failure EXIT

# ── 1. provision ─────────────────────────────────────────────────────────────
log "creating pod ($GPU_ID, ${DISK_GB}GB disk, $VCPUS vCPU, ${MEMORY_GB}GB RAM)"
create_out="$(prime --plain pods create --id "$GPU_ID" --name "$POD_NAME" \
  --disk-size "$DISK_GB" --vcpus "$VCPUS" --memory "$MEMORY_GB" -y 2>&1)" \
  || die "pod create failed:\n$create_out"
POD_ID="$(printf '%s' "$create_out" | grep -oE '[0-9a-f]{32}' | head -1)"
[ -n "$POD_ID" ] || die "no pod id in create output:\n$create_out"
log "pod id $POD_ID"

log "waiting for ACTIVE"
POD_IP=""
for _ in $(seq 1 60); do
  status="$(prime --plain pods status "$POD_ID" 2>&1 || true)"
  case "$status" in
    *ACTIVE*)
      POD_IP="$(printf '%s' "$status" | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' | head -1)"
      [ -n "$POD_IP" ] && break ;;
    *FAILED*|*ERROR*|*TERMINATED*) die "pod entered a terminal state:\n$status" ;;
  esac
  sleep 15
done
[ -n "$POD_IP" ] || die "pod not ACTIVE after 15 min"
POD="ubuntu@$POD_IP"
log "pod active at $POD_IP"

for _ in $(seq 1 20); do
  ssh_pod true >/dev/null 2>&1 && break
  sleep 10
done
ssh_pod 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader' \
  || die "SSH/GPU check failed"

# ── 2. environment ───────────────────────────────────────────────────────────
log "installing git-lfs + uv, cloning trainer @ $TRAINER_REF"
TRAINER_URL="$TRAINER_URL" TRAINER_REF="$TRAINER_REF" TRAINER="$TRAINER" \
ssh_script << 'EOF'
export PATH="$HOME/.local/bin:$PATH"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git-lfs
git lfs install --skip-repo
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
rm -rf "$TRAINER"
git clone -q "$TRAINER_URL" "$TRAINER"
cd "$TRAINER"
git checkout -q "$TRAINER_REF"
echo "trainer @ $(git rev-parse --short HEAD)"
uv venv --python 3.12 .venv >/dev/null
uv sync --extra hf >/dev/null
.venv/bin/python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))'
EOF

# ── 3. dataset ───────────────────────────────────────────────────────────────
# git clone + lfs pull rather than snapshot_download: the latter repeatedly
# stalled partway through this repo (58%, 95%, 99%) and cannot resume cleanly.
log "cloning dataset $HF_DATASET @ ${HF_REVISION:0:12} via git-lfs"
HF_TOKEN="$HF_TOKEN" HF_DATASET="$HF_DATASET" HF_REVISION="$HF_REVISION" HF_CLONE="$HF_CLONE" \
ssh_script << 'EOF'
rm -rf "$HF_CLONE"
GIT_LFS_SKIP_SMUDGE=1 git clone -q \
  "https://oauth2:${HF_TOKEN}@huggingface.co/datasets/${HF_DATASET}" "$HF_CLONE"
cd "$HF_CLONE"
git checkout -q "$HF_REVISION"
git lfs pull
echo "images: $(find dataset -name '*.png' | wc -l)"
EOF

log "building DatasetDict (deduplicate=$DEDUPLICATE)"
HF_CLONE="$HF_CLONE" DS_DISK="$DS_DISK" DEDUPLICATE="$DEDUPLICATE" TRAINER="$TRAINER" \
ssh_script << 'EOF'
cd "$TRAINER"
HF_CLONE="$HF_CLONE" DS_DISK="$DS_DISK" DEDUPLICATE="$DEDUPLICATE" .venv/bin/python - << 'PY'
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
ds.save_to_disk(os.environ["DS_DISK"])
print({name: len(split) for name, split in ds.items()})
PY
EOF

log "materialize -> attest -> validate"
TRAINER="$TRAINER" DS_DISK="$DS_DISK" DATA_DIR="$DATA_DIR" \
DATASET_ID="$DATASET_ID" HF_REVISION="$HF_REVISION" \
ssh_script << 'EOF'
cd "$TRAINER"
rm -rf "$DATA_DIR"
.venv/bin/sigorbit-train dataset import-hf-disk "$DS_DISK" "$DATA_DIR" \
  --dataset-id "$DATASET_ID" --revision "$HF_REVISION" --assert-genuine-only
.venv/bin/sigorbit-train dataset attest "$DATA_DIR/dataset.toml" \
  "$DATA_DIR/rights.attestation.json" \
  --purpose research-only \
  --authorization-reference operator-self-authorized-research \
  --assert-authorized-use
.venv/bin/sigorbit-train dataset validate "$DATA_DIR/dataset.toml" \
  --attestation "$DATA_DIR/rights.attestation.json"
EOF

# ── 4. config ────────────────────────────────────────────────────────────────
log "deriving pod config from configs/$CONFIG (run $RUN_NAME)"
TRAINER="$TRAINER" CONFIG="$CONFIG" RUN_NAME="$RUN_NAME" DATA_DIR="$DATA_DIR" HOME_DIR="$HOME_DIR" \
ssh_script << 'EOF'
cd "$TRAINER"
TRAINER="$TRAINER" CONFIG="$CONFIG" RUN_NAME="$RUN_NAME" DATA_DIR="$DATA_DIR" HOME_DIR="$HOME_DIR" \
python3 - << 'PY'
import os, re
cfg, run = os.environ["CONFIG"], os.environ["RUN_NAME"]
data, home = os.environ["DATA_DIR"], os.environ["HOME_DIR"]
text = open(f"configs/{cfg}").read()
subs = {
    r'(?m)^manifest = ".*"$': f'manifest = "{data}/dataset.toml"',
    r'(?m)^rights_attestation = ".*"$': f'rights_attestation = "{data}/rights.attestation.json"',
    r'(?m)^output_dir = ".*"$': f'output_dir = "{home}/sigorbit-runs/{run}"',
}
for pattern, replacement in subs.items():
    text, n = re.subn(pattern, replacement, text)
    if not n:
        raise SystemExit(f"config key not found for {pattern}")
text = re.sub(r'(?m)^name = "[^"]*"$', f'name = "{run}"', text, count=1)
open(f"configs/{run}.toml", "w").write(text)
PY
.venv/bin/sigorbit-train config validate "configs/$RUN_NAME.toml"
EOF

if [ "$SMOKE" = "yes" ]; then
  log "smoke test (synthetic, all three stages)"
  TRAINER="$TRAINER" ssh_script << 'EOF'
cd "$TRAINER"
.venv/bin/sigorbit-train run configs/smoke.toml 2>&1 | tail -5
EOF
fi

if [ "$STAGE_ONLY" = "yes" ]; then
  log "STAGE_ONLY=yes — pod prepared, not training"
  log "pod: ssh ${SSH_OPTS[*]} $POD"
  trap - EXIT
  exit 0
fi

# ── 5. train ─────────────────────────────────────────────────────────────────
log "launching training (detached)"
TRAINER="$TRAINER" RUN_NAME="$RUN_NAME" LOG="$LOG" ssh_script << 'EOF'
export PATH="$HOME/.local/bin:$PATH"
cd "$TRAINER"
rm -f "$LOG"
nohup .venv/bin/sigorbit-train run "configs/$RUN_NAME.toml" > "$LOG" 2>&1 &
echo "training pid $!"
EOF

log "waiting for the first logged step (e2cnn kernel build takes a few minutes)"
for _ in $(seq 1 30); do
  sleep 30
  if ssh_pod "grep -qE 'epoch 1/|Traceback|Error' '$LOG'" 2>/dev/null; then break; fi
done
ssh_pod "grep -vE 'UserWarning|full_mask' '$LOG' | tail -5"
ssh_pod "pgrep -f sigorbit-train >/dev/null" || die "training died on startup; see $LOG"

log "training alive on $POD_IP"
cat >&2 << SUMMARY

  pod        $POD_ID  ($POD_IP)
  config     configs/$RUN_NAME.toml
  log        $LOG
  run dir    $HOME_DIR/sigorbit-runs/$RUN_NAME

  status     POD=$POD_IP scripts/prime_pod_status.sh
  fetch      POD=$POD_IP RUN_NAME=$RUN_NAME scripts/prime_pod_fetch.sh
  terminate  prime pods terminate $POD_ID -y

SUMMARY
trap - EXIT
