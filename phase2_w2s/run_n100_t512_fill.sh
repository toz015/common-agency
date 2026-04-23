#!/bin/bash
# Fill in the 4-bit logit-sum baseline (n=100, t=512) to the full 11-point
# α sweep. The existing run did {0.2, 0.4, 0.8}; this adds the remaining 8:
#   α_help ∈ {0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0}, α_harm = 1 − α_help.
#
# Writes into the SAME output dirs as run_n100_t512.sh so the final Pareto
# plot / CSV / HV computation sees all 11 points per method.
#
# Uses the canonical baseline scripts on the A100 (generate_outputs.py and
# generate_outputs_genarm.py under ~/common-agency/code/evaluation/) — the
# 65B-4bit GPTQ base + logit-sum aggregator via model_arithmetic, unchanged.
#
# Expected wall-clock: ~4-5 h per method per 3 alphas at the existing 4-bit
# throughput → ~12-15 h total for the 8 missing alphas × 2 methods.
#
# Prereqs:
#   source ~/common-agency/venv/bin/activate
#   tmux new -s phase2_n100_fill
#   bash ~/phase2_w2s/run_n100_t512_fill.sh 2>&1 | tee ~/phase2_n100_fill.log

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

# Keep the SAME output dirs as the original run so the existing α ∈ {0.2, 0.4, 0.8}
# results remain visible alongside the new ones.
OUT_ROOT="$REPO/results_phase2_n100_t512"
OUT_PARM="$OUT_ROOT/parm"
OUT_GENARM="$OUT_ROOT/genarm"
mkdir -p "$OUT_PARM" "$OUT_GENARM"

# Missing 8 alphas (the already-done {0.2, 0.4, 0.8} are intentionally
# omitted — saves ~15 min of wasted model loading per method).
ALPHAS=(0.0 0.1 0.3 0.5 0.6 0.7 0.9 1.0)

cd "$REPO/code/evaluation"

############# 1. SUBSET PROMPTS — idempotent #############
echo "=== [1/5] Building $N_PROMPTS-prompt subset (idempotent) ==="
python "$HOME/phase2_w2s/make_subset.py" \
    --input  "$FULL_DATASET" \
    --output "$SUBSET_DATASET" \
    --n      "$N_PROMPTS"

############# 2. GENERATE — GenARM (fill 8 alphas) #############
echo ""
echo "=== [2/5] GenARM fill (${#ALPHAS[@]} alphas × $N_PROMPTS prompts, 4-bit 65B, max_new=$MAX_TOKENS) ==="
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

############# 3. GENERATE — PARM (fill 8 alphas) #############
echo ""
echo "=== [3/5] PARM fill (${#ALPHAS[@]} alphas × $N_PROMPTS prompts, 4-bit 65B, max_new=$MAX_TOKENS) ==="
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

############# 4. BEAVER SCORING — only the new 8 alphas #############
echo ""
echo "=== [4/5] Beaver-7B reward + cost scoring (new alphas only) ==="
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    echo ""
    echo "--- Scoring GenARM_${AH}help_${AS}harm ---"
    python compute_reward.py --path "$OUT_GENARM/GenARM_${AH}help_${AS}harm"
    echo ""
    echo "--- Scoring PARM_${AH}help_${AS}harm ---"
    python compute_reward.py --path "$OUT_PARM/PARM_${AH}help_${AS}harm"
done

############# 5. PARETO PLOT — overwrites with full 11-point curves #############
echo ""
echo "=== [5/5] Pareto plot (all 11 alphas per method) ==="
python "$HOME/phase2_w2s/plot_pareto_w2s.py" \
    --parm_dir   "$OUT_PARM" \
    --genarm_dir "$OUT_GENARM" \
    --output_png "$OUT_ROOT/pareto_n100.png" \
    --output_csv "$OUT_ROOT/pareto_n100.csv"

echo ""
echo "=== DONE (fill) ==="
echo "Results:  $OUT_ROOT"
echo "Plot:     $OUT_ROOT/pareto_n100.png"
echo "CSV:      $OUT_ROOT/pareto_n100.csv"
