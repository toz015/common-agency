#!/bin/bash
# Edge sweep along the (harm, humor) edge of the simplex (α_help = 0).
# 5 prefs × 4 methods × 200 prompts.
# Skips any (method, pref) that already has generation.json with >= LIMIT prompts.

set -e
LIMIT=200
OUTPUT_DIR=./results/HH-RLHF

PREFS=(
    "0.0 0.9 0.1"
    "0.0 0.7 0.3"
    "0.0 0.5 0.5"
    "0.0 0.3 0.7"
    "0.0 0.1 0.9"
)

has_enough() {
    local f="$1/generation.json"
    if [ ! -f "$f" ]; then return 1; fi
    local n=$(python3 -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null)
    [ "$n" -ge "$LIMIT" ]
}

echo "=============================="
echo "  Edge sweep: (harm, humor) edge — α_help = 0"
echo "  ${LIMIT} prompts × ${#PREFS[@]} prefs × 4 methods"
echo "=============================="

for stage in GenARM EPEC_GenARM PARM EPEC_PARM; do
    case "$stage" in
        GenARM)       script=generate_outputs_genarm_hh.py ;;
        EPEC_GenARM)  script=generate_outputs_epec_hh.py ;;
        PARM)         script=generate_outputs_parm_hh.py ;;
        EPEC_PARM)    script=generate_outputs_epec_parm_hh.py ;;
    esac
    echo -e "\n>>> ${stage} generation"
    for pref in "${PREFS[@]}"; do
        read -r ah as au <<< "$pref"
        dirname="${stage}_${ah}help_${as}harm_${au}humor"
        if has_enough "$OUTPUT_DIR/$dirname"; then
            echo "  SKIP (>=${LIMIT}): $dirname"
            continue
        fi
        rm -rf "$OUTPUT_DIR/$dirname"
        echo "  ${stage}: help=$ah harm=$as humor=$au"
        python "$script" \
            --alpha_helpfulness $ah --alpha_harmlessness $as --alpha_humor $au \
            --limit $LIMIT --output_dir $OUTPUT_DIR
    done
done

echo -e "\n>>> Scoring new generations"
for dir in $OUTPUT_DIR/*/; do
    if [ -f "$dir/generation.json" ] && [ ! -f "$dir/mean_result.json" ]; then
        echo "  Scoring: $dir"
        python compute_reward_hh.py --path "$dir"
    fi
done

echo -e "\n=============================="
echo "  Edge sweep done!  $(date)"
echo "=============================="
