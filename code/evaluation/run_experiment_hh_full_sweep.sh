#!/bin/bash
# Full sweep: 3 methods × 10 preferences × 50 prompts
# Skips any settings that already have results

set -e
LIMIT=50
OUTPUT_DIR=./results_hh

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
echo "  Full sweep: ${LIMIT} prompts × 10 prefs × 3 methods"
echo "=============================="

# --- GenARM ---
echo -e "\n>>> GenARM generation"
for pref in "${PREFS[@]}"; do
    read -r ah as au <<< "$pref"
    dirname="GenARM_${ah}help_${as}harm_${au}humor"
    if [ -f "$OUTPUT_DIR/$dirname/generation.json" ]; then
        echo "  SKIP (exists): $dirname"
        continue
    fi
    echo "  GenARM: help=$ah harm=$as humor=$au"
    python generate_outputs_genarm_hh.py \
        --alpha_helpfulness $ah --alpha_harmlessness $as --alpha_humor $au \
        --limit $LIMIT --output_dir $OUTPUT_DIR
done

# --- EPEC ---
echo -e "\n>>> EPEC generation"
for pref in "${PREFS[@]}"; do
    read -r ah as au <<< "$pref"
    dirname="EPEC_${ah}help_${as}harm_${au}humor"
    if [ -f "$OUTPUT_DIR/$dirname/generation.json" ]; then
        echo "  SKIP (exists): $dirname"
        continue
    fi
    echo "  EPEC: help=$ah harm=$as humor=$au"
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
        echo "  SKIP (exists): $dirname"
        continue
    fi
    echo "  PARM: help=$ah harm=$as humor=$au"
    python generate_outputs_parm_hh.py \
        --alpha_helpfulness $ah --alpha_harmlessness $as --alpha_humor $au \
        --limit $LIMIT --output_dir $OUTPUT_DIR
done

# --- Score unscored outputs ---
echo -e "\n>>> Scoring new outputs"
for dir in $OUTPUT_DIR/*/; do
    if [ -f "$dir/generation.json" ] && [ ! -f "$dir/mean_result.json" ]; then
        echo "  Scoring: $dir"
        python compute_reward_hh.py --path "$dir"
    fi
done

# --- Results ---
echo -e "\n=============================="
echo "  Full Results"
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
python plot_pareto_meeting.py
python plot_preference_sensitivity.py
