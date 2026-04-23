#!/bin/bash
# Local one-shot deploy for the 8-bit bnb sanity run:
#   1. start a100-demo (with --discard-local-ssd=false)
#   2. wait for SSH to come up
#   3. scp the 3 new files to ~/phase2_w2s/
#   4. pip-install bnb + accelerate, then launch a detached tmux session
#      running run_n50_8bit.sh with output tee'd to ~/phase2_8bit.log
#
# Run from project root (the dir containing phase2_w2s/).

set -euo pipefail

INSTANCE="a100-demo"
ZONE="us-central1-c"
PROJECT="llm-applications-490420"
TMUX_SESSION="phase2_8bit"
LOG_FILE="\$HOME/phase2_8bit.log"

FILES=(
    "phase2_w2s/generate_outputs_8bit.py"
    "phase2_w2s/generate_outputs_genarm_8bit.py"
    "phase2_w2s/run_n50_8bit.sh"
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
echo "=== [3/4] scp'ing ${#FILES[@]} files to $INSTANCE:~/phase2_w2s/ ==="
gcloud compute scp "${FILES[@]}" "$INSTANCE:~/phase2_w2s/" \
    --zone "$ZONE" --project "$PROJECT"

echo ""
echo "=== [4/4] Installing bnb + launching detached tmux '$TMUX_SESSION' ==="
gcloud compute ssh "$INSTANCE" --zone "$ZONE" --project "$PROJECT" --command "
    set -e
    # --- Local NVMe: mkfs + mount at /mnt/ssd (idempotent). Local SSD data
    # is wiped on every stop, so we re-format every time we see an unformatted
    # disk.
    if ! mountpoint -q /mnt/ssd; then
        if ! sudo blkid /dev/nvme0n1 2>/dev/null | grep -q 'TYPE=\"ext4\"'; then
            echo 'Formatting /dev/nvme0n1 as ext4...'
            sudo mkfs.ext4 -F /dev/nvme0n1
        fi
        sudo mkdir -p /mnt/ssd
        sudo mount /dev/nvme0n1 /mnt/ssd
        sudo chown -R \$USER:\$USER /mnt/ssd
        echo 'Mounted /dev/nvme0n1 at /mnt/ssd.'
    else
        echo '/mnt/ssd already mounted.'
    fi
    df -h /mnt/ssd
    # One-time cleanup of the partial HF download on the boot disk (safe to
    # re-run — no-op if already gone).
    rm -rf \$HOME/.cache/huggingface/hub/models--TheBloke--alpaca-lora-65B-HF || true
    source \$HOME/common-agency/venv/bin/activate
    if python -c 'import bitsandbytes, accelerate' 2>/dev/null; then
        echo 'bnb + accelerate already importable, skipping pip install.'
    else
        echo 'Installing bnb + accelerate (force-reinstall to clear any stale dist-info)...'
        pip install -q --force-reinstall --no-deps bitsandbytes accelerate
    fi
    chmod +x \$HOME/phase2_w2s/run_n50_8bit.sh
    tmux kill-session -t $TMUX_SESSION 2>/dev/null || true
    tmux new-session -d -s $TMUX_SESSION \"bash \$HOME/phase2_w2s/run_n50_8bit.sh 2>&1 | tee $LOG_FILE\"
    echo ''
    echo '--- tmux sessions ---'
    tmux ls
"

echo ""
echo "=== DONE ==="
echo ""
echo "Tail log:     gcloud compute ssh $INSTANCE --zone $ZONE --project $PROJECT --command 'tail -f \$HOME/phase2_8bit.log'"
echo "Attach tmux:  gcloud compute ssh $INSTANCE --zone $ZONE --project $PROJECT -- -t 'tmux a -t $TMUX_SESSION'"
echo "Stop inst:    gcloud compute instances stop $INSTANCE --zone $ZONE --project $PROJECT --discard-local-ssd=false"
