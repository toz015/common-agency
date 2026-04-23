#!/bin/bash
# Extend the 4-bit logit-sum n=100 sweep to n=300 on the SAME 11-α grid,
# accumulating on top of the existing generation.json files.
#
# Mechanism:
#   * make_subset.py takes the first N prompts (deterministic prefix), so the
#     first 100 uids of the 300-prompt subset are identical to the existing
#     n=100 subset — no overlap/duplication risk.
#   * generate_outputs.py / generate_outputs_genarm.py honor --resume true
#     and skip uids already present in generation.json → only eval100..eval299
#     get newly generated, eval0..eval99 are pass-through.
#   * compute_reward.py is re-run on ALL 22 configs after generation so
#     reward_result.json covers the full 300 prompts (old per-config JSONs
#     only had 100).
#   * plot_pareto_w2s.py writes pareto_n300.{png,csv} alongside the existing
#     pareto_n100.*.
#
# Expected wall-clock (from fill-run cadence, ~27s/prompt logit-sum avg):
#   * +200 new prompts × ~27s × 22 configs ≈ 33 h generation
#   * Beaver scoring 22 × ~8 min                 ≈ 3 h
#   * Total ≈ 36 h. Checkpoints every 10 prompts so interruption is cheap.
#
# Prereqs on the host:
#   source ~/common-agency/venv/bin/activate
#   tmux new -s phase2_n300_extend
#   bash ~/phase2_w2s/run_n300_t512_extend.sh 2>&1 | tee ~/phase2_n300_extend.log

set -euo pipefail
source "$HOME/common-agency/venv/bin/activate"

############# CONFIG #############
REPO="$HOME/common-agency"
BASE_65B="TheBloke/alpaca-lora-65B-GPTQ"

PARM_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp"
HELP_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp_genarm_help"
HARM_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp_genarm_harm"

FULL_DATASET="$REPO/data/test_prompt_only.json"
SUBSET_DATASET="$REPO/data/test_prompt_only_300.json"
N_PROMPTS=300
MAX_TOKENS=512
CHECKPOINT_EVERY=10

# Same output roots as n=100 — generation.json is appended in-place.
OUT_ROOT="$REPO/results_phase2_n100_t512"
OUT_PARM="$OUT_ROOT/parm"
OUT_GENARM="$OUT_ROOT/genarm"
mkdir -p "$OUT_PARM" "$OUT_GENARM"

# All 11 alphas — resume handles the overlap with prior n=100 runs.
ALPHAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

cd "$REPO/code/evaluation"

############# 1. SUBSET (idempotent; 300 = first-300 prefix) #############
echo "=== [1/5] Building $N_PROMPTS-prompt subset (idempotent) ==="
python "$HOME/phase2_w2s/make_subset.py" \
    --input  "$FULL_DATASET" \
    --output "$SUBSET_DATASET" \
    --n      "$N_PROMPTS"

############# 2. GENERATE — GenARM (all 11 α, resume accumulates) #############
echo ""
echo "=== [2/5] GenARM extend (${#ALPHAS[@]} α × $N_PROMPTS prompts, --resume true) ==="
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

############# 3. GENERATE — PARM (all 11 α, resume accumulates) #############
echo ""
echo "=== [3/5] PARM extend (${#ALPHAS[@]} α × $N_PROMPTS prompts, --resume true) ==="
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

############# 4. BEAVER — rescore all 22 (old reward_result only has 100) #############
echo ""
echo "=== [4/5] Beaver-7B reward + cost scoring — all 22 configs (n=$N_PROMPTS) ==="
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    echo ""
    echo "--- Scoring GenARM_${AH}help_${AS}harm ---"
    python compute_reward.py --path "$OUT_GENARM/GenARM_${AH}help_${AS}harm"
    echo ""
    echo "--- Scoring PARM_${AH}help_${AS}harm ---"
    python compute_reward.py --path "$OUT_PARM/PARM_${AH}help_${AS}harm"
done

############# 5. PARETO (n=300 alongside n=100) #############
echo ""
echo "=== [5/5] Pareto plot (n=$N_PROMPTS, all 11 α per method) ==="
python "$HOME/phase2_w2s/plot_pareto_w2s.py" \
    --parm_dir   "$OUT_PARM" \
    --genarm_dir "$OUT_GENARM" \
    --output_png "$OUT_ROOT/pareto_n${N_PROMPTS}.png" \
    --output_csv "$OUT_ROOT/pareto_n${N_PROMPTS}.csv" \
    --n_per_cell "$N_PROMPTS"

echo ""
echo "=== DONE (extend to n=$N_PROMPTS) ==="
echo "Results:  $OUT_ROOT"
echo "Plot:     $OUT_ROOT/pareto_n${N_PROMPTS}.png"
echo "CSV:      $OUT_ROOT/pareto_n${N_PROMPTS}.csv"
