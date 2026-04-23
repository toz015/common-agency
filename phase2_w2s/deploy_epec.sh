#!/bin/bash
# Local one-shot deploy for the EPEC n=100 t=512 sweep:
#   1. start a100-demo
#   2. wait for SSH
#   3. scp the 3 new EPEC files to ~/phase2_w2s/
#   4. launch a detached tmux session running run_epec_n100_t512.sh with
#      output tee'd to ~/phase2_epec.log
#
# No bnb/accelerate reinstall — EPEC uses the existing 4-bit GPTQ base
# (auto_gptq) + fp16 ARMs; the venv from the logit-sum n=100 run already
# has peft/scipy/auto-gptq installed.
#
# Run from project root (the dir containing phase2_w2s/).

set -euo pipefail

INSTANCE="a100-demo"
ZONE="us-central1-c"
PROJECT="llm-applications-490420"
TMUX_SESSION="phase2_epec"
LOG_FILE="\$HOME/phase2_epec.log"

FILES=(
    "phase2_w2s/generate_outputs_epec_genarm_w2s.py"
    "phase2_w2s/generate_outputs_epec_parm_w2s.py"
    "phase2_w2s/run_epec_n100_t512.sh"
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
echo "=== [4/4] Launching detached tmux '$TMUX_SESSION' ==="
gcloud compute ssh "$INSTANCE" --zone "$ZONE" --project "$PROJECT" --command "
    set -e
    source \$HOME/common-agency/venv/bin/activate
    # Sanity-check the imports EPEC needs.
    python -c 'import peft, scipy.optimize, auto_gptq, transformers; print(
        \"peft\", peft.__version__,
        \"scipy ok,\",
        \"auto_gptq\", auto_gptq.__version__,
        \"transformers\", transformers.__version__)'
    chmod +x \$HOME/phase2_w2s/run_epec_n100_t512.sh
    tmux kill-session -t $TMUX_SESSION 2>/dev/null || true
    tmux new-session -d -s $TMUX_SESSION \"bash \$HOME/phase2_w2s/run_epec_n100_t512.sh 2>&1 | tee $LOG_FILE\"
    echo ''
    echo '--- tmux sessions ---'
    tmux ls
"

echo ""
echo "=== DONE ==="
echo ""
echo "Tail log:     gcloud compute ssh $INSTANCE --zone $ZONE --project $PROJECT --command 'tail -f \$HOME/phase2_epec.log'"
echo "Attach tmux:  gcloud compute ssh $INSTANCE --zone $ZONE --project $PROJECT -- -t 'tmux a -t $TMUX_SESSION'"
echo "Stop inst:    gcloud compute instances stop $INSTANCE --zone $ZONE --project $PROJECT"
