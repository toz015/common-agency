#!/bin/bash
# Run HH-RLHF generation experiment: 3 methods × 3 preferences × 200 prompts
# Then score all outputs with 3 reward models.
#
# Methods: PARM, GenARM, EPEC
# Preferences: P1 (help-focused), P2 (safe-focused), P3 (humor-focused)

set -e
LIMIT=200
OUTPUT_DIR=./results_hh

# Preference settings
PREFS=(
    "0.5 0.3 0.2"  # P1: help-focused
    "0.2 0.5 0.3"  # P2: safe-focused
    "0.3 0.2 0.5"  # P3: humor-focused
)

echo "=============================="
echo "  HH-RLHF Experiment: ${LIMIT} prompts × 3 prefs × 3 methods"
echo "=============================="

# --- GenARM (loads base + TinyLLaMA, ~16GB) ---
echo -e "\n>>> GenARM generation"
for pref in "${PREFS[@]}"; do
    read -r ah as au <<< "$pref"
    echo "  GenARM: help=$ah harm=$as humor=$au"
    python generate_outputs_genarm_hh.py \
        --alpha_helpfulness $ah --alpha_harmlessness $as --alpha_humor $au \
        --limit $LIMIT --output_dir $OUTPUT_DIR
done

# --- EPEC (loads base + TinyLLaMA, ~16GB) ---
echo -e "\n>>> EPEC generation"
for pref in "${PREFS[@]}"; do
    read -r ah as au <<< "$pref"
    echo "  EPEC: help=$ah harm=$as humor=$au"
    python generate_outputs_epec_hh.py \
        --alpha_helpfulness $ah --alpha_harmlessness $as --alpha_humor $au \
        --limit $LIMIT --output_dir $OUTPUT_DIR
done

# --- PARM (loads base + PARM adapter via ModelArithmetic, ~16GB) ---
echo -e "\n>>> PARM generation"
for pref in "${PREFS[@]}"; do
    read -r ah as au <<< "$pref"
    echo "  PARM: help=$ah harm=$as humor=$au"
    python generate_outputs_parm_hh.py \
        --alpha_helpfulness $ah --alpha_harmlessness $as --alpha_humor $au \
        --limit $LIMIT --output_dir $OUTPUT_DIR
done

# --- Score all outputs ---
echo -e "\n>>> Scoring all outputs with reward models"
for dir in $OUTPUT_DIR/*/; do
    if [ -f "$dir/generation.json" ]; then
        echo "  Scoring: $dir"
        python compute_reward_hh.py --path "$dir"
    fi
done

echo -e "\n=============================="
echo "  Experiment complete!"
echo "  Results in: $OUTPUT_DIR"
echo "=============================="

# Print summary
echo -e "\n--- Summary ---"
for dir in $OUTPUT_DIR/*/; do
    if [ -f "$dir/mean_result.json" ]; then
        name=$(basename "$dir")
        echo "$name: $(cat $dir/mean_result.json)"
    fi
done
