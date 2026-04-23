#!/bin/bash
# Local one-shot deploy for the n=100 → n=300 extension run:
#   1. start a100-demo
#   2. wait for SSH
#   3. scp run_n300_t512_extend.sh to ~/phase2_w2s/
#   4. launch a detached tmux session running it, tee'd to
#      ~/phase2_n300_extend.log
#
# Run from project root (the dir containing phase2_w2s/).

set -euo pipefail

INSTANCE="a100-demo"
ZONE="us-central1-c"
PROJECT="llm-applications-490420"
TMUX_SESSION="phase2_n300_extend"
LOG_FILE="\$HOME/phase2_n300_extend.log"

FILES=(
    "phase2_w2s/run_n300_t512_extend.sh"
)

for f in "${FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: expected $f in cwd=$(pwd). Run from project root."
        exit 1
    fi
done

echo "=== [1/4] Starting $INSTANCE ==="
gcloud compute instances start "$INSTANCE" \
    --zone "$ZONE" \
    --project "$PROJECT"

echo ""
echo "=== [2/4] Waiting for SSH (up to ~100s) ==="
for i in {1..20}; do
    if gcloud compute ssh "$INSTANCE" --zone "$ZONE" --project "$PROJECT" \
            --command "echo ready" &>/dev/null; then
        echo "  SSH up (attempt $i)."
        break
    fi
    printf "."
    sleep 5
done

echo ""
echo "=== [3/4] scp'ing ${#FILES[@]} file(s) to $INSTANCE:~/phase2_w2s/ ==="
gcloud compute scp "${FILES[@]}" "$INSTANCE:~/phase2_w2s/" \
    --zone "$ZONE" --project "$PROJECT"

echo ""
echo "=== [4/4] Launching detached tmux '$TMUX_SESSION' ==="
gcloud compute ssh "$INSTANCE" --zone "$ZONE" --project "$PROJECT" --command "
    set -e
    chmod +x \$HOME/phase2_w2s/run_n300_t512_extend.sh
    tmux kill-session -t $TMUX_SESSION 2>/dev/null || true
    tmux new-session -d -s $TMUX_SESSION \"bash \$HOME/phase2_w2s/run_n300_t512_extend.sh 2>&1 | tee $LOG_FILE\"
    echo ''
    echo '--- tmux sessions ---'
    tmux ls
    echo ''
    echo '--- nvidia-smi ---'
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
"

echo ""
echo "=== DONE ==="
echo ""
echo "Tail log:     gcloud compute ssh $INSTANCE --zone $ZONE --project $PROJECT --command 'tail -f \$HOME/phase2_n300_extend.log'"
echo "Attach tmux:  gcloud compute ssh $INSTANCE --zone $ZONE --project $PROJECT -- -t 'tmux a -t $TMUX_SESSION'"
echo "Stop inst:    gcloud compute instances stop $INSTANCE --zone $ZONE --project $PROJECT --discard-local-ssd=false"
