#!/bin/bash
# Extend EPEC+GenARM from n=50 -> n=200 (=add 150 new prompts × 6 α).
# Skips PARM (user pivot: GenARM is the winner; PARM EPEC underperformed and
# is parked).
#
# --resume true means existing 6 GenARM dirs (from n=50 run) are reused —
# uids 0-49 already in generation.json are skipped, only uids 50-199 are newly
# generated. Same per-config dirs get topped up to n=200.
#
# Prereqs on a100-demo (paths verified by deploy_n50_epec.sh):
#   source ~/common-agency/venv/bin/activate
#   tmux new -s phase2_n200_genarm
#   bash ~/phase2_w2s/run_n200_t512_epec_genarm_only.sh 2>&1 | tee ~/phase2_n200_genarm.log

set -euo pipefail
source "$HOME/common-agency/venv/bin/activate"

############# CONFIG #############
REPO="$HOME/common-agency"
BASE_65B="TheBloke/alpaca-lora-65B-GPTQ"

HELP_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp_genarm_help"
HARM_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp_genarm_harm"
ARM_BASE="PKU-Alignment/alpaca-7b-reproduced"

FULL_DATASET="$REPO/data/test_prompt_only.json"
SUBSET_DATASET="$REPO/data/test_prompt_only_200.json"
N_PROMPTS=200
MAX_TOKENS=512
TAU=0.1
TOPK=50

OUT_ROOT="$REPO/results_phase2_n50_t512_epec"   # SAME root as n=50 — extend per-config dirs
OUT_GENARM="$OUT_ROOT/genarm"
mkdir -p "$OUT_GENARM"

ALPHAS=(0.0 0.2 0.4 0.6 0.8 1.0)

cd "$REPO/code/evaluation"

############# 1. SUBSET #############
echo "=== [1/3] Building $N_PROMPTS-prompt subset (idempotent) ==="
python "$HOME/phase2_w2s/make_subset.py" \
    --input  "$FULL_DATASET" \
    --output "$SUBSET_DATASET" \
    --n      "$N_PROMPTS"

############# 2. EPEC+GenARM extension (resume per-config) #############
echo ""
echo "=== [2/3] EPEC+GenARM token-level (${#ALPHAS[@]} α × $N_PROMPTS prompts, --resume on) ==="
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

############# 3. BEAVER scoring (re-score on n=200 superset) #############
echo ""
echo "=== [3/3] Beaver-7B reward + cost scoring (6 GenARM configs at n=200) ==="
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    GENARM_DIR="$OUT_GENARM/EPEC_GenARM_${AH}help_${AS}harm_tau${TAU}_k${TOPK}"
    echo ""
    echo "--- Scoring $(basename $GENARM_DIR) ---"
    python compute_reward.py --path "$GENARM_DIR"
done

echo ""
echo "=== DONE — n=$N_PROMPTS, max_tokens=$MAX_TOKENS, ${#ALPHAS[@]} α, GenARM-only ==="
echo "Results:   $OUT_ROOT"
