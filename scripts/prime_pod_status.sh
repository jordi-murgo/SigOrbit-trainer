#!/usr/bin/env bash
# Report training progress on a Prime pod.
#
#   POD=216.81.245.166 scripts/prime_pod_status.sh
#   POD=216.81.245.166 WATCH=yes scripts/prime_pod_status.sh   # refresh every 60 s
set -euo pipefail

: "${POD:?set POD to the pod IP or user@host}"
LOG="${LOG:-/home/ubuntu/train.log}"
WATCH="${WATCH:-no}"
INTERVAL="${INTERVAL:-60}"

case "$POD" in *@*) TARGET="$POD" ;; *) TARGET="ubuntu@$POD" ;; esac
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=15)

report() {
  ssh "${SSH_OPTS[@]}" "$TARGET" LOG="$LOG" 'bash -seu' << 'EOF' 2>&1 |
if pgrep -f sigorbit-train >/dev/null; then
  echo "state: RUNNING ($(ps -o etime= -p "$(pgrep -f sigorbit-train | head -1)" | tr -d ' ') elapsed)"
else
  echo "state: STOPPED"
fi
echo "gpu:   $(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader)"
echo "--- last epochs ---"
grep -E 'epoch [0-9]+:|stage|completed|Error|Traceback' "$LOG" 2>/dev/null | tail -8 \
  || echo "(no epoch lines yet)"
echo "--- current ---"
grep -E 'step [0-9]+/' "$LOG" 2>/dev/null | tail -1 || echo "(no step lines yet)"
EOF
  grep -v "Warning: Permanently added"
}

if [ "$WATCH" = "yes" ]; then
  while true; do
    printf '\n\033[36m[%s]\033[0m\n' "$(date +%H:%M:%S)"
    report
    sleep "$INTERVAL"
  done
else
  report
fi
