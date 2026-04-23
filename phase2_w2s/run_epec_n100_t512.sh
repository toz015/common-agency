#!/bin/bash
# Phase-2 weak-to-strong EPEC sweep:
#   100 prompts × 11 alphas × 2 methods at max_new_tokens=512.
#
# Drop-in alt to run_n100_t512.sh (the logit-sum baseline): same prompts /
# tokens / 65B-4bit GPTQ base / 7B ARMs, but aggregation swapped from
# linear logit-sum (model_arithmetic) to Common-Agency EPEC equilibrium
# (per-token Nonlinear Jacobi on top-k candidates).
#
# α grid: α_help ∈ {0.0, 0.1, …, 1.0}, α_harm = 1 − α_help  (11 points per method).
#
# Output: results_epec_n100_t512/ — does NOT clobber the logit-sum baseline
# results_phase2_n100_t512/.
#
# VRAM budget on A100 80GB:
#   PARM / GenARM : 65B GPTQ (~35 GB) + 7B fp16 backbone w/ adapter(s) (~14 GB)
#                 = ~50 GB steady state; comfortable.
#
# Prereqs (same A100 instance as the logit-sum n=100 run):
#   source ~/common-agency/venv/bin/activate
#   pip list | grep -E "peft|scipy|auto-gptq"   # all should be present
#   tmux new -s phase2_epec
#   bash ~/phase2_w2s/run_epec_n100_t512.sh 2>&1 | tee ~/phase2_epec.log

set -euo pipefail
source "$HOME/common-agency/venv/bin/activate"

############# CONFIG #############
REPO="$HOME/common-agency"
BASE_65B="TheBloke/alpaca-lora-65B-GPTQ"
BACKBONE_7B="PKU-Alignment/alpaca-7b-reproduced"

PARM_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp"
HELP_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp_genarm_help"
HARM_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp_genarm_harm"

FULL_DATASET="$REPO/data/test_prompt_only.json"
SUBSET_DATASET="$REPO/data/test_prompt_only_100.json"
N_PROMPTS=100
MAX_TOKENS=512
CHECKPOINT_EVERY=10
TAU=0.1
K=50

OUT_ROOT="$REPO/results_epec_n100_t512"
OUT_PARM="$OUT_ROOT/parm"
OUT_GENARM="$OUT_ROOT/genarm"
mkdir -p "$OUT_PARM" "$OUT_GENARM"

# Full 11-point sweep: α_help ∈ {0.0, 0.1, …, 1.0}.
ALPHAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

GEN_PARM="$HOME/phase2_w2s/generate_outputs_epec_parm_w2s.py"
GEN_GENARM="$HOME/phase2_w2s/generate_outputs_epec_genarm_w2s.py"

cd "$REPO/code/evaluation"

############# 1. SUBSET PROMPTS #############
echo "=== [1/5] Building $N_PROMPTS-prompt subset ==="
python "$HOME/phase2_w2s/make_subset.py" \
    --input  "$FULL_DATASET" \
    --output "$SUBSET_DATASET" \
    --n      "$N_PROMPTS"

############# 2. GENERATE — PARM (cheaper, run first) #############
echo ""
echo "=== [2/5] EPEC-PARM (11 alphas × $N_PROMPTS prompts, 65B-4bit + 7B-PBLoRA, τ=$TAU k=$K, max_new=$MAX_TOKENS) ==="
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    echo ""
    echo "--- EPEC-PARM alpha_help=$AH  alpha_harm=$AS ---"
    python "$GEN_PARM" \
        --model_base_name_or_path       "$BASE_65B" \
        --model_backbone_name_or_path   "$BACKBONE_7B" \
        --model_parm_both_name_or_path  "$PARM_ADAPTER" \
        --alpha_helpfulness   "$AH" \
        --alpha_harmlessness  "$AS" \
        --tau                 "$TAU" \
        --k                   "$K" \
        --max_new_tokens      "$MAX_TOKENS" \
        --datasets            "$SUBSET_DATASET" \
        --output_dir          "$OUT_PARM" \
        --resume              true \
        --checkpoint_every    "$CHECKPOINT_EVERY"
done

############# 3. GENERATE — GenARM #############
echo ""
echo "=== [3/5] EPEC-GenARM (11 alphas × $N_PROMPTS prompts, 65B-4bit + 7B backbone + 2 LoRAs, τ=$TAU k=$K, max_new=$MAX_TOKENS) ==="
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    echo ""
    echo "--- EPEC-GenARM alpha_help=$AH  alpha_harm=$AS ---"
    python "$GEN_GENARM" \
        --model_base_name_or_path      "$BASE_65B" \
        --model_backbone_name_or_path  "$BACKBONE_7B" \
        --model_arm_help_path          "$HELP_ADAPTER" \
        --model_arm_harm_path          "$HARM_ADAPTER" \
        --alpha_helpfulness   "$AH" \
        --alpha_harmlessness  "$AS" \
        --tau                 "$TAU" \
        --k                   "$K" \
        --max_new_tokens      "$MAX_TOKENS" \
        --datasets            "$SUBSET_DATASET" \
        --output_dir          "$OUT_GENARM" \
        --resume              true \
        --checkpoint_every    "$CHECKPOINT_EVERY"
done

############# 4. BEAVER SCORING #############
echo ""
echo "=== [4/5] Beaver-7B reward + cost scoring ==="
# Model dir names include tau/k — the EPEC scripts produce:
#   PARM_EPEC_{ah}help_{as}harm_tau{tau}_k{k}/
#   GenARM_EPEC_{ah}help_{as}harm_tau{tau}_k{k}/
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    PARM_NAME="PARM_EPEC_${AH}help_${AS}harm_tau${TAU}_k${K}"
    GEN_NAME="GenARM_EPEC_${AH}help_${AS}harm_tau${TAU}_k${K}"
    echo ""
    echo "--- Scoring $GEN_NAME ---"
    python compute_reward.py --path "$OUT_GENARM/$GEN_NAME"
    echo ""
    echo "--- Scoring $PARM_NAME ---"
    python compute_reward.py --path "$OUT_PARM/$PARM_NAME"
done

############# 5. PARETO PLOT #############
echo ""
echo "=== [5/5] Pareto plot ==="
# NOTE: plot_pareto_w2s.py expects PARM_{...} / GenARM_{...} dir names;
# our EPEC dir names have the EPEC_ prefix inside. Pass them anyway —
# the script globs on the help/harm suffix pattern. If the script doesn't
# pick them up, run compute_hv.py / a custom aggregator offline.
python "$HOME/phase2_w2s/plot_pareto_w2s.py" \
    --parm_dir   "$OUT_PARM" \
    --genarm_dir "$OUT_GENARM" \
    --output_png "$OUT_ROOT/pareto_epec_n100.png" \
    --output_csv "$OUT_ROOT/pareto_epec_n100.csv" \
    --n_per_cell "$N_PROMPTS" || echo "[warn] plot step failed — aggregate manually via compute_hv.py"

echo ""
echo "=== DONE ==="
echo "Results:  $OUT_ROOT"
echo "Plot:     $OUT_ROOT/pareto_epec_n100.png"
echo "CSV:      $OUT_ROOT/pareto_epec_n100.csv"
