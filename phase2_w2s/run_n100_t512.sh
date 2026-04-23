#!/bin/bash
# Phase-2 weak-to-strong scaled run: 100 prompts × 3 alphas × 2 methods,
# max_new_tokens=512 (vs sanity's 50 prompts × 128 tokens).
#
# Scope: 600 generations total. Expected wall-clock ~4-5 h on A100 80GB.
# Output goes to results_phase2_n100_t512/ to keep the sanity results intact.
#
# Run after the venv + adapters are already set up (no setup_a100.sh needed
# if the instance is the same one used for the sanity check):
#   source ~/common-agency/venv/bin/activate
#   tmux new -s phase2_n100
#   bash ~/phase2_w2s/run_n100_t512.sh 2>&1 | tee ~/phase2_n100.log

set -euo pipefail
source "$HOME/common-agency/venv/bin/activate"

############# CONFIG #############
REPO="$HOME/common-agency"
BASE_65B="TheBloke/alpaca-lora-65B-GPTQ"

PARM_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp"
HELP_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp_genarm_help"
HARM_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp_genarm_harm"

FULL_DATASET="$REPO/data/test_prompt_only.json"
SUBSET_DATASET="$REPO/data/test_prompt_only_100.json"
N_PROMPTS=100
MAX_TOKENS=512
CHECKPOINT_EVERY=10

OUT_ROOT="$REPO/results_phase2_n100_t512"
OUT_PARM="$OUT_ROOT/parm"
OUT_GENARM="$OUT_ROOT/genarm"
mkdir -p "$OUT_PARM" "$OUT_GENARM"

ALPHAS=(0.2 0.4 0.8)

cd "$REPO/code/evaluation"

############# 1. SUBSET PROMPTS #############
echo "=== [1/5] Building $N_PROMPTS-prompt subset ==="
python "$HOME/phase2_w2s/make_subset.py" \
    --input  "$FULL_DATASET" \
    --output "$SUBSET_DATASET" \
    --n      "$N_PROMPTS"

############# 2. GENERATE — GenARM #############
echo ""
echo "=== [2/5] GenARM weak-to-strong generation (3 alphas × $N_PROMPTS prompts, 4-bit 65B, max_new=$MAX_TOKENS) ==="
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    echo ""
    echo "--- GenARM alpha_help=$AH  alpha_harm=$AS ---"
    python generate_outputs_genarm.py \
        --model_base_name_or_path "$BASE_65B" \
        --model_arm_help_path     "$HELP_ADAPTER" \
        --model_arm_harm_path     "$HARM_ADAPTER" \
        --alpha_helpfulness   "$AH" \
        --alpha_harmlessness  "$AS" \
        --max_new_tokens      "$MAX_TOKENS" \
        --datasets            "$SUBSET_DATASET" \
        --output_dir          "$OUT_GENARM" \
        --resume              true \
        --checkpoint_every    "$CHECKPOINT_EVERY"
done

############# 3. GENERATE — PARM #############
echo ""
echo "=== [3/5] PARM weak-to-strong generation (3 alphas × $N_PROMPTS prompts, 4-bit 65B, max_new=$MAX_TOKENS) ==="
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    echo ""
    echo "--- PARM alpha_help=$AH  alpha_harm=$AS ---"
    python generate_outputs.py \
        --model_base_name_or_path       "$BASE_65B" \
        --model_parm_both_name_or_path  "$PARM_ADAPTER" \
        --alpha_helpfulness   "$AH" \
        --alpha_harmlessness  "$AS" \
        --max_new_tokens      "$MAX_TOKENS" \
        --datasets            "$SUBSET_DATASET" \
        --output_dir          "$OUT_PARM" \
        --resume              true \
        --checkpoint_every    "$CHECKPOINT_EVERY"
done

############# 4. BEAVER SCORING #############
echo ""
echo "=== [4/5] Beaver-7B reward + cost scoring ==="
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
    --output_png "$OUT_ROOT/pareto_n100.png" \
    --output_csv "$OUT_ROOT/pareto_n100.csv"

echo ""
echo "=== DONE ==="
echo "Results:  $OUT_ROOT"
echo "Plot:     $OUT_ROOT/pareto_n100.png"
echo "CSV:      $OUT_ROOT/pareto_n100.csv"
