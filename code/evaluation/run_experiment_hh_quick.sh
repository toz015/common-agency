#!/bin/bash
# Quick test: 3 methods × 3 preferences × 50 prompts
# Estimated: 8-12 hours total

set -e
LIMIT=50
OUTPUT_DIR=./results/HH-RLHF

PREFS=(
    "0.5 0.3 0.2"  # help-focused
    "0.2 0.5 0.3"  # safe-focused
    "0.3 0.2 0.5"  # humor-focused
)

echo "=============================="
echo "  Quick test: ${LIMIT} prompts × 3 prefs × 3 methods"
echo "=============================="

# --- GenARM ---
echo -e "\n>>> GenARM generation"
for pref in "${PREFS[@]}"; do
    read -r ah as au <<< "$pref"
    echo "  GenARM: help=$ah harm=$as humor=$au"
    python generate_outputs_genarm_hh.py \
        --alpha_helpfulness $ah --alpha_harmlessness $as --alpha_humor $au \
        --limit $LIMIT --output_dir $OUTPUT_DIR
done

# --- EPEC ---
echo -e "\n>>> EPEC generation"
for pref in "${PREFS[@]}"; do
    read -r ah as au <<< "$pref"
    echo "  EPEC: help=$ah harm=$as humor=$au"
    python generate_outputs_epec_hh.py \
        --alpha_helpfulness $ah --alpha_harmlessness $as --alpha_humor $au \
        --limit $LIMIT --output_dir $OUTPUT_DIR
done

# --- PARM ---
echo -e "\n>>> PARM generation"
for pref in "${PREFS[@]}"; do
    read -r ah as au <<< "$pref"
    echo "  PARM: help=$ah harm=$as humor=$au"
    python generate_outputs_parm_hh.py \
        --alpha_helpfulness $ah --alpha_harmlessness $as --alpha_humor $au \
        --limit $LIMIT --output_dir $OUTPUT_DIR
done

# --- Score all ---
echo -e "\n>>> Scoring all outputs"
for dir in $OUTPUT_DIR/*/; do
    if [ -f "$dir/generation.json" ] && [ ! -f "$dir/mean_result.json" ]; then
        echo "  Scoring: $dir"
        python compute_reward_hh.py --path "$dir"
    fi
done

# --- Results ---
echo -e "\n=============================="
echo "  Results"
echo "=============================="
python3 -c "
import json, os
rows = []
for name in sorted(os.listdir('$OUTPUT_DIR')):
    p = f'$OUTPUT_DIR/{name}/mean_result.json'
    if os.path.isfile(p):
        d = json.load(open(p))
        rows.append((name, d['help'], d['harm'], d['humor']))
print(f\"{'Method':<50} {'Help':>8} {'Harm':>9} {'Humor':>7}\")
print('-'*77)
for method in ['EPEC', 'GenARM', 'PARM']:
    for name, h, s, u in rows:
        if name.startswith(method + '_'):
            print(f'{name:<50} {h:+8.4f} {s:+9.4f} {u:7.4f}')
    print()
"

# --- Plots and metrics ---
python plot_pareto_and_metrics.py
