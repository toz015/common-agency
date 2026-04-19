#!/bin/bash
# Phase-2 weak-to-strong SANITY CHECK on A100 80GB.
#
# Purpose: verify the 4-bit-65B + 7B-ARM + logit-arithmetic stack runs end-to-end,
# and produce a preliminary Pareto plot to confirm PARM > GenARM relative ordering.
# NOT aimed at reproducing PARM Table 3 absolute numbers.
#
# Scope: 50 prompts × 3 alphas × 2 methods (PARM, GenARM), 4-bit GPTQ 65B base.
# Expected wall-clock on A100 80GB: ~1.5-2 h generation + ~30 min Beaver scoring.
#
# Run after setup_a100.sh:
#   source ~/common-agency/venv/bin/activate
#   tmux new -s phase2
#   bash ~/phase2_w2s/run_sanity_w2s.sh 2>&1 | tee ~/phase2_sanity.log

set -euo pipefail

# Activate venv so `python` resolves to the venv's python3
source "$HOME/common-agency/venv/bin/activate"

############# CONFIG #############
REPO="$HOME/common-agency"

# 4-bit GPTQ 65B — same underlying checkpoint as PARM's fp16 version
# (TheBloke/alpaca-lora-65B-HF), just quantized. See GenARM §6.3.
BASE_65B="TheBloke/alpaca-lora-65B-GPTQ"

PARM_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp"
HELP_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp_genarm_help"
HARM_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp_genarm_harm"

FULL_DATASET="$REPO/data/test_prompt_only.json"
SUBSET_DATASET="$REPO/data/test_prompt_only_50.json"
N_PROMPTS=50
MAX_TOKENS=128                         # Sanity scale. Scale up to 512 later.

OUT_ROOT="$REPO/results_phase2_sanity"
OUT_PARM="$OUT_ROOT/parm"
OUT_GENARM="$OUT_ROOT/genarm"
mkdir -p "$OUT_PARM" "$OUT_GENARM"

# α_help values (α_harm = 1 - α_help)
ALPHAS=(0.2 0.4 0.8)

TS=$(date +%Y%m%d_%H%M%S)
# Optional GCS upload. Leave empty to skip.
GCS_BUCKET=""

cd "$REPO/code/evaluation"

############# 1. SUBSET PROMPTS #############
echo "=== [1/5] Building $N_PROMPTS-prompt subset ==="
python "$HOME/phase2_w2s/make_subset.py" \
    --input  "$FULL_DATASET" \
    --output "$SUBSET_DATASET" \
    --n      "$N_PROMPTS"

############# 2. GENERATE — GenARM (simpler pipeline, run first) #############
echo ""
echo "=== [2/5] GenARM weak-to-strong generation (3 alphas × $N_PROMPTS prompts, 4-bit 65B) ==="
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    echo ""
    echo "--- GenARM alpha_help=$AH  alpha_harm=$AS ---"
    python generate_outputs_genarm.py \
        --model_base_name_or_path "$BASE_65B" \
        --model_arm_help_path     "$HELP_ADAPTER" \
        --model_arm_harm_path     "$HARM_ADAPTER" \
        --alpha_helpfulness  "$AH" \
        --alpha_harmlessness "$AS" \
        --max_new_tokens     "$MAX_TOKENS" \
        --datasets           "$SUBSET_DATASET" \
        --output_dir         "$OUT_GENARM"
done

############# 3. GENERATE — PARM (PBLoRA + GPTQ first test) #############
echo ""
echo "=== [3/5] PARM weak-to-strong generation (3 alphas × $N_PROMPTS prompts, 4-bit 65B) ==="
# Note: pref_vec_init gets injected into the cached adapter_config.json per-alpha
# by generate_outputs.py itself (see the script's own logic).
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    echo ""
    echo "--- PARM alpha_help=$AH  alpha_harm=$AS ---"
    python generate_outputs.py \
        --model_base_name_or_path       "$BASE_65B" \
        --model_parm_both_name_or_path  "$PARM_ADAPTER" \
        --alpha_helpfulness  "$AH" \
        --alpha_harmlessness "$AS" \
        --max_new_tokens     "$MAX_TOKENS" \
        --datasets           "$SUBSET_DATASET" \
        --output_dir         "$OUT_PARM"
done

############# 4. BEAVER SCORING #############
echo ""
echo "=== [4/5] Beaver-7B reward + cost scoring ==="
# compute_reward.py launches a fresh Python process, so the 65B used for
# generation is already released before Beaver (2× 7B fp16 ≈ 28 GB) loads.
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    echo ""
    echo "--- Scoring GenARM_${AH}help_${AS}harm ---"
    python compute_reward.py --path "$OUT_GENARM/GenARM_${AH}help_${AS}harm"
    echo ""
    echo "--- Scoring PARM_${AH}help_${AS}harm ---"
    python compute_reward.py --path "$OUT_PARM/PARM_${AH}help_${AS}harm"
done

############# 5. PARETO PLOT #############
echo ""
echo "=== [5/5] Pareto plot ==="
python "$HOME/phase2_w2s/plot_pareto_w2s.py" \
    --parm_dir   "$OUT_PARM" \
    --genarm_dir "$OUT_GENARM" \
    --output_png "$OUT_ROOT/pareto_sanity.png" \
    --output_csv "$OUT_ROOT/pareto_sanity.csv"

############# OPTIONAL UPLOAD #############
if [ -n "$GCS_BUCKET" ] && command -v gsutil >/dev/null 2>&1; then
    gsutil -m cp -r "$OUT_ROOT" "gs://${GCS_BUCKET}/phase2_sanity_${TS}/"
    echo "Uploaded to gs://${GCS_BUCKET}/phase2_sanity_${TS}/"
fi

echo ""
echo "=== DONE ==="
echo "Results:  $OUT_ROOT"
echo "Plot:     $OUT_ROOT/pareto_sanity.png"
echo "CSV:      $OUT_ROOT/pareto_sanity.csv"
echo ""
echo "Sanity check questions to answer manually:"
echo "  1. Do PARM and GenARM generations look coherent (not gibberish)?"
echo "  2. Does α_help=0.2 refuse more / α_help=0.8 comply more?"
echo "  3. Is PARM's Pareto front above-right of GenARM's (expected from Table 3)?"
