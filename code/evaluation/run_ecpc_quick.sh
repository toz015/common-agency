#!/bin/bash
# ECPC quick test: 3 preferences × 50 prompts
set -e
LIMIT=50
OUTPUT_DIR=./results/HH-RLHF

PREFS=(
    "0.5 0.3 0.2"
    "0.2 0.5 0.3"
    "0.3 0.2 0.5"
)

echo "=============================="
echo "  ECPC quick test: ${LIMIT} prompts × 3 prefs"
echo "=============================="

for pref in "${PREFS[@]}"; do
    read -r ah as au <<< "$pref"
    dirname="ECPC_${ah}help_${as}harm_${au}humor"
    if [ -f "$OUTPUT_DIR/$dirname/generation.json" ]; then
        echo "  SKIP (exists): $dirname"
        continue
    fi
    echo "  ECPC: help=$ah harm=$as humor=$au"
    python generate_outputs_ecpc_hh.py \
        --alpha_helpfulness $ah --alpha_harmlessness $as --alpha_humor $au \
        --limit $LIMIT --output_dir $OUTPUT_DIR
done

# Score
echo -e "\n>>> Scoring ECPC outputs"
for dir in $OUTPUT_DIR/ECPC_*/; do
    if [ -f "$dir/generation.json" ] && [ ! -f "$dir/mean_result.json" ]; then
        echo "  Scoring: $dir"
        python compute_reward_hh.py --path "$dir"
    fi
done

# Compare all methods
echo -e "\n=============================="
echo "  All Results (including ECPC)"
echo "=============================="
python3 -c "
import json, os
d = '$OUTPUT_DIR'
rows = []
for name in sorted(os.listdir(d)):
    p = f'{d}/{name}/mean_result.json'
    if os.path.isfile(p):
        r = json.load(open(p))
        rows.append((name, r['help'], r['harm'], r['humor']))
print(f\"{'Method':<50} {'Help':>8} {'Harm':>9} {'Humor':>7}\")
print('-'*77)
for method in ['ECPC', 'EPEC', 'GenARM', 'PARM']:
    method_rows = [(n,h,s,u) for n,h,s,u in rows if n.startswith(method+'_')]
    if method_rows:
        for name, h, s, u in method_rows:
            print(f'{name:<50} {h:+8.4f} {s:+9.4f} {u:7.4f}')
        print()
"

# Regenerate plots with ECPC included
python plot_pareto_and_metrics.py
python plot_pareto_meeting.py
