#!/bin/bash
# Balanced preference vectors: 4 methods × 6 new preferences × 200 prompts
set -e
LIMIT=200
OUTPUT_DIR=./results/HH-RLHF

PREFS=(
    "0.2 0.3 0.5"
    "0.2 0.4 0.4"
    "0.3 0.3 0.4"
    "0.4 0.2 0.4"
    "0.25 0.25 0.5"
    "0.2 0.2 0.6"
)

echo "=============================="
echo "  Balanced prefs: ${LIMIT} prompts × 6 prefs × 4 methods"
echo "=============================="

# --- GenARM ---
echo -e "\n>>> GenARM generation"
for pref in "${PREFS[@]}"; do
    read -r ah as au <<< "$pref"
    dirname="GenARM_${ah}help_${as}harm_${au}humor"
    if [ -f "$OUTPUT_DIR/$dirname/generation.json" ]; then
        echo "  SKIP: $dirname"; continue
    fi
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
    if [ -f "$OUTPUT_DIR/$dirname/generation.json" ]; then
        echo "  SKIP: $dirname"; continue
    fi
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
    if [ -f "$OUTPUT_DIR/$dirname/generation.json" ]; then
        echo "  SKIP: $dirname"; continue
    fi
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
    if [ -f "$OUTPUT_DIR/$dirname/generation.json" ]; then
        echo "  SKIP: $dirname"; continue
    fi
    echo "  EPEC_PARM: help=$ah harm=$as humor=$au"
    python generate_outputs_epec_parm_hh.py \
        --alpha_helpfulness $ah --alpha_harmlessness $as --alpha_humor $au \
        --limit $LIMIT --output_dir $OUTPUT_DIR
done

# --- Score ---
echo -e "\n>>> Scoring new outputs"
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
python plot_preference_sensitivity.py

echo -e "\n=============================="
echo "  Done!"
echo "=============================="
