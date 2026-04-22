#!/bin/bash
# EPEC_PARM: remaining 7 preference settings × 50 prompts
set -e
LIMIT=50
OUTPUT_DIR=./results/HH-RLHF

PREFS=(
    "1.0 0.0 0.0"
    "0.0 1.0 0.0"
    "0.0 0.0 1.0"
    "0.5 0.3 0.2"
    "0.2 0.5 0.3"
    "0.3 0.2 0.5"
    "0.33 0.33 0.34"
    "0.8 0.1 0.1"
    "0.1 0.8 0.1"
    "0.1 0.1 0.8"
)

echo "=============================="
echo "  EPEC_PARM sweep: ${LIMIT} prompts × 10 prefs (skip existing)"
echo "=============================="

for pref in "${PREFS[@]}"; do
    read -r ah as au <<< "$pref"
    dirname="EPEC_PARM_${ah}help_${as}harm_${au}humor"
    if [ -f "$OUTPUT_DIR/$dirname/generation.json" ]; then
        echo "  SKIP (exists): $dirname"
        continue
    fi
    echo "  EPEC_PARM: help=$ah harm=$as humor=$au"
    python generate_outputs_epec_parm_hh.py \
        --alpha_helpfulness $ah --alpha_harmlessness $as --alpha_humor $au \
        --limit $LIMIT --output_dir $OUTPUT_DIR
done

# Score new outputs
echo -e "\n>>> Scoring new EPEC_PARM outputs"
for dir in $OUTPUT_DIR/EPEC_PARM_*/; do
    if [ -f "$dir/generation.json" ] && [ ! -f "$dir/mean_result.json" ]; then
        echo "  Scoring: $dir"
        python compute_reward_hh.py --path "$dir"
    fi
done

# Regenerate plots and metrics
python plot_pareto_and_metrics.py
python plot_pareto_meeting.py
