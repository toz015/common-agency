#!/bin/bash
# Extended sweep: 3 methods × 7 NEW preference vectors × 200 prompts
# (Settings 4-6 already done in run_experiment_hh.sh)
#
# Only score NEW results (skip dirs that already have mean_result.json)

set -e
LIMIT=200
OUTPUT_DIR=./results_hh

# 7 new preference settings (help, harm, humor)
PREFS=(
    "1.0 0.0 0.0"    # Pure help
    "0.0 1.0 0.0"    # Pure harm
    "0.0 0.0 1.0"    # Pure humor
    "0.33 0.33 0.34"  # Equal
    "0.8 0.1 0.1"    # Strong help
    "0.1 0.8 0.1"    # Strong harm
    "0.1 0.1 0.8"    # Strong humor
)

echo "=============================="
echo "  HH-RLHF Sweep: ${LIMIT} prompts × 7 prefs × 3 methods"
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

# --- Score only unscored outputs ---
echo -e "\n>>> Scoring new outputs"
for dir in $OUTPUT_DIR/*/; do
    if [ -f "$dir/generation.json" ] && [ ! -f "$dir/mean_result.json" ]; then
        echo "  Scoring: $dir"
        python compute_reward_hh.py --path "$dir"
    fi
done

# --- Full summary ---
echo -e "\n=============================="
echo "  Full Results (all 10 preferences × 3 methods)"
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
