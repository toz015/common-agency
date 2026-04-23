#!/bin/bash
# Full experiment: 4 methods × 10 preferences × 200 prompts
# Skips any settings that already have 200-prompt results
# Estimated: ~4.5 days total on single L4

set -e
LIMIT=200
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

# Helper: check if result already has >= LIMIT prompts
has_enough() {
    local f="$1/generation.json"
    if [ ! -f "$f" ]; then return 1; fi
    local n=$(python3 -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null)
    [ "$n" -ge "$LIMIT" ]
}

echo "=============================="
echo "  Full: ${LIMIT} prompts × 10 prefs × 4 methods"
echo "=============================="

# --- GenARM ---
echo -e "\n>>> GenARM generation"
for pref in "${PREFS[@]}"; do
    read -r ah as au <<< "$pref"
    dirname="GenARM_${ah}help_${as}harm_${au}humor"
    if has_enough "$OUTPUT_DIR/$dirname"; then
        echo "  SKIP (>=${LIMIT}): $dirname"
        continue
    fi
    # Remove old 50-prompt result if exists
    rm -rf "$OUTPUT_DIR/$dirname"
    echo "  GenARM: help=$ah harm=$as humor=$au"
    python generate_outputs_genarm_hh.py \
        --alpha_helpfulness $ah --alpha_harmlessness $as --alpha_humor $au \
        --limit $LIMIT --output_dir $OUTPUT_DIR
done

# --- EPEC_GenARM ---
echo -e "\n>>> EPEC_GenARM generation"
for pref in "${PREFS[@]}"; do
    read -r ah as au <<< "$pref"
    dirname="EPEC_GenARM_${ah}help_${as}harm_${au}humor"
    if has_enough "$OUTPUT_DIR/$dirname"; then
        echo "  SKIP (>=${LIMIT}): $dirname"
        continue
    fi
    rm -rf "$OUTPUT_DIR/$dirname"
    echo "  EPEC_GenARM: help=$ah harm=$as humor=$au"
    python generate_outputs_epec_hh.py \
        --alpha_helpfulness $ah --alpha_harmlessness $as --alpha_humor $au \
        --limit $LIMIT --output_dir $OUTPUT_DIR
done

# --- PARM ---
echo -e "\n>>> PARM generation"
for pref in "${PREFS[@]}"; do
    read -r ah as au <<< "$pref"
    dirname="PARM_${ah}help_${as}harm_${au}humor"
    if has_enough "$OUTPUT_DIR/$dirname"; then
        echo "  SKIP (>=${LIMIT}): $dirname"
        continue
    fi
    rm -rf "$OUTPUT_DIR/$dirname"
    echo "  PARM: help=$ah harm=$as humor=$au"
    python generate_outputs_parm_hh.py \
        --alpha_helpfulness $ah --alpha_harmlessness $as --alpha_humor $au \
        --limit $LIMIT --output_dir $OUTPUT_DIR
done

# --- EPEC_PARM ---
echo -e "\n>>> EPEC_PARM generation"
for pref in "${PREFS[@]}"; do
    read -r ah as au <<< "$pref"
    dirname="EPEC_PARM_${ah}help_${as}harm_${au}humor"
    if has_enough "$OUTPUT_DIR/$dirname"; then
        echo "  SKIP (>=${LIMIT}): $dirname"
        continue
    fi
    rm -rf "$OUTPUT_DIR/$dirname"
    echo "  EPEC_PARM: help=$ah harm=$as humor=$au"
    python generate_outputs_epec_parm_hh.py \
        --alpha_helpfulness $ah --alpha_harmlessness $as --alpha_humor $au \
        --limit $LIMIT --output_dir $OUTPUT_DIR
done

# --- Score all unscored ---
echo -e "\n>>> Scoring"
for dir in $OUTPUT_DIR/*/; do
    if [ -f "$dir/generation.json" ] && [ ! -f "$dir/mean_result.json" ]; then
        echo "  Scoring: $dir"
        python compute_reward_hh.py --path "$dir"
    fi
done

# --- Plots and metrics ---
echo -e "\n>>> Generating plots and metrics"
python plot_pareto_and_metrics.py
python plot_pareto_meeting.py

echo -e "\n=============================="
echo "  Done!"
echo "=============================="
