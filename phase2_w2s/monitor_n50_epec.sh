#!/bin/bash
# Monitor the W2S 65B EPEC sweep on a100-demo.
# Exits when:
#   (a) error appears in remote log,
#   (b) at least 3 prompts of first config done (stable timing),
#   (c) MAX_POLLS reached (default 3h cap; covers 65B model load + 3 prompts).
#
# Outputs:
#   /tmp/epec_monitor.status   (final summary)
#   /tmp/epec_monitor.history  (running snapshot log)

INSTANCE="a100-demo"
ZONE="us-central1-c"
PROJECT="llm-applications-490420"
REMOTE_LOG="\$HOME/phase2_n50_epec.log"
GENARM_DIR="\$HOME/common-agency/results_phase2_n50_t512_epec/genarm/EPEC_GenARM_0.0help_1.0harm_tau0.1_k50"

STATUS=/tmp/epec_monitor.status
HISTORY=/tmp/epec_monitor.history
: > "$STATUS"
: > "$HISTORY"

POLL_EVERY=90
MAX_POLLS=120     # ~3h cap (65B load ~6 min + 3 prompts at ~162s)
TARGET_N=3

ssh_check() {
    gcloud compute ssh "$INSTANCE" --zone "$ZONE" --project "$PROJECT" \
        --command "
            echo '== LOG TAIL =='
            tail -n 20 $REMOTE_LOG 2>/dev/null
            echo '== generation.json =='
            python3 -c \"
import json, os
p = os.path.expanduser('$GENARM_DIR/generation.json')
if not os.path.exists(p):
    print('  no file yet')
else:
    with open(p) as f:
        d = json.load(f)
    print(f'  n={len(d)}')
    for r in d[-3:]:
        print(f'  uid={r.get(\\\"uid\\\")} elapsed={r.get(\\\"elapsed\\\"):.1f}s fwd={r.get(\\\"fwd_time\\\"):.1f}s epec={r.get(\\\"epec_time\\\"):.1f}s ntok={r.get(\\\"n_tokens\\\")}')
\" 2>&1 || true
            echo '== nvidia-smi =='
            nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
        " 2>&1
}

i=0
while [ "$i" -lt "$MAX_POLLS" ]; do
    i=$((i + 1))
    ts=$(date +'%H:%M:%S')
    snap=$(ssh_check)
    {
        echo ""
        echo "=========================================="
        echo "[$ts] poll #$i"
        echo "=========================================="
        echo "$snap"
    } >> "$HISTORY"

    if echo "$snap" | grep -qE 'Traceback|RuntimeError|ImportError|CUDA out of memory|FileNotFoundError|NotImplementedError|size mismatch'; then
        {
            echo "EXIT REASON: ERROR DETECTED"
            echo "Poll #$i at $ts"
            echo '--- relevant log lines ---'
            echo "$snap" | grep -E 'Traceback|Error|raise|^  File|size mismatch' | head -n 30
            echo '--- full tail ---'
            echo "$snap" | tail -n 30
        } > "$STATUS"
        exit 1
    fi

    n=$(echo "$snap" | grep -oE 'n=[0-9]+' | head -n 1 | sed 's/n=//')
    if [ -n "$n" ] && [ "$n" -ge "$TARGET_N" ]; then
        {
            echo "EXIT REASON: TARGET REACHED (n=$n >= $TARGET_N for first config)"
            echo "Poll #$i at $ts"
            echo '--- final snapshot ---'
            echo "$snap"
        } > "$STATUS"
        exit 0
    fi

    sleep "$POLL_EVERY"
done

{
    echo "EXIT REASON: MAX_POLLS REACHED ($MAX_POLLS x ${POLL_EVERY}s = $((MAX_POLLS * POLL_EVERY / 60)) min)"
    echo '--- last snapshot ---'
    echo "$snap"
} > "$STATUS"
exit 2
