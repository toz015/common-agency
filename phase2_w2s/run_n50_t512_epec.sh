#!/bin/bash
# Phase-2 W2S — token-level EPEC sweep, n=50 prompts × 11 α × 2 methods.
#
# Pipeline:
#   1. Build 50-prompt subset (deterministic prefix → matches existing n=100/300 runs)
#   2. For each α in 0.0..1.0 (step 0.1), α_harm = 1 - α_help:
#        a. EPEC+GenARM token-level generation
#        b. EPEC+PARM   token-level generation
#      All runs use --resume so re-launching is idempotent.
#   3. Beaver-7B reward + cost scoring on every per-config dir
#
# This is a TIMING PROBE first → estimate sec/token for token-level EPEC on
# the 4-bit 65B + 7B-ARM stack. Watch the first few configs' timing.json to
# decide if we want to extend to n=300 or scale back the α grid.
#
# Prereqs on a100-demo:
#   source ~/common-agency/venv/bin/activate
#   tmux new -s phase2_n50_epec
#   bash ~/phase2_w2s/run_n50_t512_epec.sh 2>&1 | tee ~/phase2_n50_epec.log

set -euo pipefail
source "$HOME/common-agency/venv/bin/activate"

############# CONFIG #############
REPO="$HOME/common-agency"
BASE_65B="TheBloke/alpaca-lora-65B-GPTQ"

PARM_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp"
HELP_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp_genarm_help"
HARM_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp_genarm_harm"
ARM_BASE="PKU-Alignment/alpaca-7b-reproduced"

FULL_DATASET="$REPO/data/test_prompt_only.json"
SUBSET_DATASET="$REPO/data/test_prompt_only_50.json"
N_PROMPTS=50
MAX_TOKENS=512
TAU=0.1
TOPK=50

OUT_ROOT="$REPO/results_phase2_n50_t512_epec"
OUT_PARM="$OUT_ROOT/parm"
OUT_GENARM="$OUT_ROOT/genarm"
mkdir -p "$OUT_PARM" "$OUT_GENARM"

# 6-point α grid (subset of the 11-point logit-sum baseline grid). Uniform 0.2
# spacing covers both endpoints + the trade-off curve. Each removed α can be
# filled in later by re-running this script after extending ALPHAS — --resume
# at the per-prompt level keeps already-completed configs intact.
ALPHAS=(0.0 0.2 0.4 0.6 0.8 1.0)

cd "$REPO/code/evaluation"

############# 1. SUBSET #############
echo "=== [1/4] Building $N_PROMPTS-prompt subset (idempotent) ==="
python "$HOME/phase2_w2s/make_subset.py" \
    --input  "$FULL_DATASET" \
    --output "$SUBSET_DATASET" \
    --n      "$N_PROMPTS"

############# 2. EPEC+GenARM (11 α) #############
echo ""
echo "=== [2/4] EPEC+GenARM token-level (${#ALPHAS[@]} α × $N_PROMPTS prompts) ==="
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    echo ""
    echo "--- EPEC_GenARM alpha_help=$AH  alpha_harm=$AS ---"
    python generate_outputs_epec_genarm.py \
        --base                "$BASE_65B" \
        --arm_base            "$ARM_BASE" \
        --help_adapter        "$HELP_ADAPTER" \
        --harm_adapter        "$HARM_ADAPTER" \
        --alpha_helpfulness   "$AH" \
        --alpha_harmlessness  "$AS" \
        --tau                 "$TAU" \
        --k                   "$TOPK" \
        --max_new_tokens      "$MAX_TOKENS" \
        --datasets            "$SUBSET_DATASET" \
        --output_dir          "$OUT_GENARM" \
        --resume              true
done

############# 3. EPEC+PARM (11 α) #############
echo ""
echo "=== [3/4] EPEC+PARM token-level (${#ALPHAS[@]} α × $N_PROMPTS prompts) ==="
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    echo ""
    echo "--- EPEC_PARM alpha_help=$AH  alpha_harm=$AS ---"
    python generate_outputs_epec_parm.py \
        --base                "$BASE_65B" \
        --parm_base           "$ARM_BASE" \
        --parm_adapter        "$PARM_ADAPTER" \
        --alpha_helpfulness   "$AH" \
        --alpha_harmlessness  "$AS" \
        --tau                 "$TAU" \
        --k                   "$TOPK" \
        --max_new_tokens      "$MAX_TOKENS" \
        --datasets            "$SUBSET_DATASET" \
        --output_dir          "$OUT_PARM" \
        --resume              true
done

############# 4. BEAVER scoring (all 22 configs) #############
echo ""
echo "=== [4/4] Beaver-7B reward + cost scoring (all 22 EPEC configs) ==="
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    GENARM_DIR="$OUT_GENARM/EPEC_GenARM_${AH}help_${AS}harm_tau${TAU}_k${TOPK}"
    PARM_DIR="$OUT_PARM/EPEC_PARM_${AH}help_${AS}harm_tau${TAU}_k${TOPK}"
    echo ""
    echo "--- Scoring $(basename $GENARM_DIR) ---"
    python compute_reward.py --path "$GENARM_DIR"
    echo ""
    echo "--- Scoring $(basename $PARM_DIR) ---"
    python compute_reward.py --path "$PARM_DIR"
done

echo ""
echo "=== DONE — n=$N_PROMPTS, max_tokens=$MAX_TOKENS, ${#ALPHAS[@]} α × 2 methods ==="
echo "Results:   $OUT_ROOT"
echo "Timings:   look at */timing.json under each per-config dir"
