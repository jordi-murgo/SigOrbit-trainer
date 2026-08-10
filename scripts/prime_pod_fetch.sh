#!/usr/bin/env bash
# Download a finished run from a Prime pod and verify integrity before the pod
# is terminated. Fetches the whole run directory, not just checkpoints: run.json
# carries the resume state and provenance, and a run without it cannot resume.
#
#   POD=216.81.245.166 RUN_NAME=c8-257-research-pod scripts/prime_pod_fetch.sh
#   POD=... RUN_NAME=... OUT_DIR=artifacts TERMINATE=<pod-id> scripts/prime_pod_fetch.sh
set -euo pipefail

log() { printf '\033[36m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { printf '\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

: "${POD:?set POD to the pod IP or user@host}"
: "${RUN_NAME:?set RUN_NAME to the run directory name}"
OUT_DIR="${OUT_DIR:-artifacts}"
REMOTE_RUNS="${REMOTE_RUNS:-/home/ubuntu/sigorbit-runs}"
LOG_FILE="${LOG_FILE:-/home/ubuntu/train.log}"
TERMINATE="${TERMINATE:-}"

case "$POD" in *@*) TARGET="$POD" ;; *) TARGET="ubuntu@$POD" ;; esac
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=15)
ssh_pod() { ssh "${SSH_OPTS[@]}" -o LogLevel=ERROR "$TARGET" "$@"; }

REMOTE_RUN="$REMOTE_RUNS/$RUN_NAME"
ssh_pod "test -d '$REMOTE_RUN'" || die "no run directory at $REMOTE_RUN"

status="$(ssh_pod "python3 -c \"import json;print(json.load(open('$REMOTE_RUN/run.json'))['status'])\"" || echo unknown)"
log "remote run status: $status"
[ "$status" = "completed" ] || log "WARNING: run is not 'completed' — artifacts may be partial"

mkdir -p "$OUT_DIR"
log "fetching $REMOTE_RUN -> $OUT_DIR/$RUN_NAME"
scp "${SSH_OPTS[@]}" -o LogLevel=ERROR -q -r "$TARGET:$REMOTE_RUN" "$OUT_DIR/"
scp "${SSH_OPTS[@]}" -o LogLevel=ERROR -q "$TARGET:$LOG_FILE" "$OUT_DIR/$RUN_NAME/train.log"

log "verifying checksums against the pod"
remote_sums="$(ssh_pod "export LC_ALL=C; cd '$REMOTE_RUN' && find . -type f \\( -name '*.pt' -o -name '*.json' \\) -print0 | sort -z | xargs -0 sha256sum")"
local_sums="$(export LC_ALL=C; cd "$OUT_DIR/$RUN_NAME" && find . -type f \( -name '*.pt' -o -name '*.json' \) -print0 | sort -z | xargs -0 sha256sum)"
if [ "$remote_sums" = "$local_sums" ]; then
  log "checksums match ($(printf '%s\n' "$remote_sums" | wc -l) files)"
else
  printf '%s\n' "$remote_sums" > /tmp/remote_sums.txt
  printf '%s\n' "$local_sums" > /tmp/local_sums.txt
  diff /tmp/remote_sums.txt /tmp/local_sums.txt >&2 || true
  die "checksum mismatch — do NOT terminate the pod"
fi

printf '\n' >&2
find "$OUT_DIR/$RUN_NAME" -name '*.pt' -exec ls -lh {} + >&2

if [ -n "$TERMINATE" ]; then
  log "terminating pod $TERMINATE"
  prime --plain pods terminate "$TERMINATE" -y
  prime --plain pods list
else
  log "artifacts verified; terminate with: prime pods terminate <pod-id> -y"
fi
