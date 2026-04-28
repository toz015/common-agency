#!/bin/bash
# Fill the 5 missing alphas (0.1, 0.3, 0.5, 0.7, 0.9) at n=300 to complete
# the 11-α grid that matches the logit-sum baseline.
#
# Existing 6 dirs (α ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}) at n=300 are unchanged.
# Each of the 5 new dirs is generated from scratch (uids 0-299) — no resume
# needed since they're new dirs.
#
# Estimated runtime: 5 α × 300 prompts × ~120s = ~50h on A100. Plus Beaver
# scoring (~15 min for 5 configs). max_tok=512.
#
# After this completes:
#   results_phase2_n50_t512_epec/genarm/EPEC_GenARM_*help_*harm_tau0.1_k50/
#   has 11 α dirs, each with n=300 generation.json + reward_result.json + mean_result.json.
#
# Prereqs on a100-demo:
#   source ~/common-agency/venv/bin/activate
#   tmux new -s phase2_n300_genarm_fill
#   bash ~/phase2_w2s/run_n300_t512_epec_genarm_fill5.sh 2>&1 | tee ~/phase2_n300_genarm_fill.log

set -euo pipefail
source "$HOME/common-agency/venv/bin/activate"

############# CONFIG #############
REPO="$HOME/common-agency"
BASE_65B="TheBloke/alpaca-lora-65B-GPTQ"

HELP_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp_genarm_help"
HARM_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp_genarm_harm"
ARM_BASE="PKU-Alignment/alpaca-7b-reproduced"

FULL_DATASET="$REPO/data/test_prompt_only.json"
SUBSET_DATASET="$REPO/data/test_prompt_only_300.json"
N_PROMPTS=300
MAX_TOKENS=512
TAU=0.1
TOPK=50

OUT_ROOT="$REPO/results_phase2_n50_t512_epec"   # SAME root as existing 6 α
OUT_GENARM="$OUT_ROOT/genarm"
mkdir -p "$OUT_GENARM"

# ONLY the 5 missing alphas:
ALPHAS=(0.1 0.3 0.5 0.7 0.9)

cd "$REPO/code/evaluation"

############# 1. SUBSET #############
echo "=== [1/3] Building $N_PROMPTS-prompt subset (idempotent) ==="
python "$HOME/phase2_w2s/make_subset.py" \
    --input  "$FULL_DATASET" \
    --output "$SUBSET_DATASET" \
    --n      "$N_PROMPTS"

############# 2. EPEC+GenARM (5 new α × n=300) #############
echo ""
echo "=== [2/3] EPEC+GenARM new α: ${ALPHAS[*]} × $N_PROMPTS prompts ==="
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    echo ""
    echo "--- EPEC_GenARM alpha_help=$AH  alpha_harm=$AS (NEW) ---"
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

############# 3. BEAVER scoring (5 new configs) #############
echo ""
echo "=== [3/3] Beaver-7B scoring (5 new α at n=$N_PROMPTS) ==="
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    GENARM_DIR="$OUT_GENARM/EPEC_GenARM_${AH}help_${AS}harm_tau${TAU}_k${TOPK}"
    echo ""
    echo "--- Scoring $(basename $GENARM_DIR) ---"
    python compute_reward.py --path "$GENARM_DIR"
done

echo ""
echo "=== DONE — fill5 complete: 11-α grid at n=$N_PROMPTS ==="
echo "Results:   $OUT_ROOT"
