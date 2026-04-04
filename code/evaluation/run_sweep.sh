#!/usr/bin/env bash
# Preference vector sweep for PARM evaluation.
# Constraint: alpha_helpfulness + alpha_harmlessness = 1
# Steps: 0.0/1.0, 0.1/0.9, ..., 1.0/0.0  (11 points)
#
# Output: results/PARM_{alpha_h}help_{alpha_harm}harm/generation.json
# Each folder name encodes the exact preference vector used.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_BASE="PKU-Alignment/alpaca-7b-reproduced"
MODEL_ARM="${SCRIPT_DIR}/../training/exp"
OUTPUT_DIR="${SCRIPT_DIR}/results"
DATASETS="${SCRIPT_DIR}/../data/test_prompt_only.json"

echo "=========================================="
echo "PARM Preference Vector Sweep"
echo "  base model : ${MODEL_BASE}"
echo "  ARM model  : ${MODEL_ARM}"
echo "  output dir : ${OUTPUT_DIR}"
echo "  dataset    : ${DATASETS}"
echo "=========================================="

# 11 steps: alpha_h in {0.0, 0.1, ..., 1.0}, alpha_harm = 1 - alpha_h
for i in $(seq 0 10); do
    # Use python for reliable floating point formatting
    alpha_h=$(python3 -c "print(f'{$i * 0.1:.1f}')")
    alpha_s=$(python3 -c "print(f'{(10 - $i) * 0.1:.1f}')")

    echo ""
    echo "------------------------------------------"
    echo "Step $((i+1))/11: alpha_helpfulness=${alpha_h}  alpha_harmlessness=${alpha_s}"
    echo "------------------------------------------"

    python "${SCRIPT_DIR}/generate_outputs.py" \
        --model_base_name_or_path "${MODEL_BASE}" \
        --model_parm_both_name_or_path "${MODEL_ARM}" \
        --alpha_helpfulness "${alpha_h}" \
        --alpha_harmlessness "${alpha_s}" \
        --output_dir "${OUTPUT_DIR}" \
        --datasets "${DATASETS}"
done

echo ""
echo "=========================================="
echo "Sweep complete. Results in: ${OUTPUT_DIR}"
echo "Folders created:"
ls "${OUTPUT_DIR}"
echo "=========================================="
