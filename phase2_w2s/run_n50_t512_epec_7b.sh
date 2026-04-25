#!/bin/bash
# Phase-2 EPEC sweep with 7B SAME-MODEL setup (Tong Zhu's epec-parm-sweep-20260424).
#
# Differs from run_n50_t512_epec.sh (W2S 65B):
#   * base = PKU-Alignment/alpaca-7b-reproduced  (NOT 65B-GPTQ)
#   * uses Tong Zhu's exact scripts from epec-parm-sweep-20260424:
#       generate_outputs_epec_genarm.py  (3 forwards/token, base + help LoRA + harm LoRA)
#       generate_outputs_epec_parm.py    (3 forwards/token, base + PARM with pref_vec switch)
#   * These scripts have NO --resume — we add bash-level skip below: if a config's
#     generation.json already has >= N_PROMPTS records, skip it.
#
# Pipeline:
#   1. Build 50-prompt subset (deterministic)
#   2. For each α in 0.0..1.0 (step 0.1), α_harm = 1 - α_help:
#        a. EPEC+GenARM (Tong Zhu's script)
#        b. EPEC+PARM   (Tong Zhu's script)
#   3. Beaver-7B reward + cost scoring on every per-config dir
#
# Prereqs on a100-demo:
#   source ~/common-agency/venv/bin/activate
#   tmux new -s phase2_n50_epec_7b
#   bash ~/phase2_w2s/run_n50_t512_epec_7b.sh 2>&1 | tee ~/phase2_n50_epec_7b.log

set -euo pipefail
source "$HOME/common-agency/venv/bin/activate"

############# CONFIG #############
REPO="$HOME/common-agency"
BASE_7B="PKU-Alignment/alpaca-7b-reproduced"

PARM_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp/final_checkpoint"
HELP_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp_genarm_help/final_checkpoint"
HARM_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp_genarm_harm/final_checkpoint"

FULL_DATASET="$REPO/data/test_prompt_only.json"
SUBSET_DATASET="$REPO/data/test_prompt_only_50.json"
N_PROMPTS=50
MAX_TOKENS=512
TAU=0.1
TOPK=50

OUT_ROOT="$REPO/results_phase2_n50_t512_epec_7b"
OUT_PARM="$OUT_ROOT/parm"
OUT_GENARM="$OUT_ROOT/genarm"
mkdir -p "$OUT_PARM" "$OUT_GENARM"

ALPHAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

cd "$REPO/code/evaluation"

count_done() {
    # Args: $1 = path to generation.json. Echo number of records, or 0 if missing.
    local p="$1"
    if [[ ! -f "$p" ]]; then
        echo 0
        return
    fi
    python -c "
import json, sys
try:
    d = json.load(open('$p'))
    print(len(d))
except Exception:
    print(0)
"
}

############# 1. SUBSET #############
echo "=== [1/4] Building $N_PROMPTS-prompt subset (idempotent) ==="
python "$HOME/phase2_w2s/make_subset.py" \
    --input  "$FULL_DATASET" \
    --output "$SUBSET_DATASET" \
    --n      "$N_PROMPTS"

############# 2. EPEC+GenARM (11 α) #############
echo ""
echo "=== [2/4] EPEC+GenARM 7B (${#ALPHAS[@]} α × $N_PROMPTS prompts) ==="
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    NAME="EPEC_GenARM_${AH}help_${AS}harm_tau${TAU}_k${TOPK}"
    GEN_PATH="$OUT_GENARM/$NAME/generation.json"
    DONE=$(count_done "$GEN_PATH")
    echo ""
    echo "--- $NAME (existing: $DONE / $N_PROMPTS) ---"
    if [[ "$DONE" -ge "$N_PROMPTS" ]]; then
        echo "  skip (already complete)"
        continue
    fi
    python generate_outputs_epec_genarm.py \
        --base                "$BASE_7B" \
        --help_adapter        "$HELP_ADAPTER" \
        --harm_adapter        "$HARM_ADAPTER" \
        --alpha_helpfulness   "$AH" \
        --alpha_harmlessness  "$AS" \
        --tau                 "$TAU" \
        --k                   "$TOPK" \
        --max_new_tokens      "$MAX_TOKENS" \
        --datasets            "$SUBSET_DATASET" \
        --output_dir          "$OUT_GENARM" \
        --limit               "$N_PROMPTS"
done

############# 3. EPEC+PARM (11 α) #############
echo ""
echo "=== [3/4] EPEC+PARM 7B (${#ALPHAS[@]} α × $N_PROMPTS prompts) ==="
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    NAME="EPEC_PARM_${AH}help_${AS}harm_tau${TAU}_k${TOPK}"
    GEN_PATH="$OUT_PARM/$NAME/generation.json"
    DONE=$(count_done "$GEN_PATH")
    echo ""
    echo "--- $NAME (existing: $DONE / $N_PROMPTS) ---"
    if [[ "$DONE" -ge "$N_PROMPTS" ]]; then
        echo "  skip (already complete)"
        continue
    fi
    python generate_outputs_epec_parm.py \
        --base                "$BASE_7B" \
        --parm_adapter        "$PARM_ADAPTER" \
        --alpha_helpfulness   "$AH" \
        --alpha_harmlessness  "$AS" \
        --tau                 "$TAU" \
        --k                   "$TOPK" \
        --max_new_tokens      "$MAX_TOKENS" \
        --datasets            "$SUBSET_DATASET" \
        --output_dir          "$OUT_PARM" \
        --limit               "$N_PROMPTS"
done

############# 4. BEAVER scoring (all 22 configs) #############
echo ""
echo "=== [4/4] Beaver-7B reward + cost scoring (22 EPEC configs) ==="
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    GENARM_DIR="$OUT_GENARM/EPEC_GenARM_${AH}help_${AS}harm_tau${TAU}_k${TOPK}"
    PARM_DIR="$OUT_PARM/EPEC_PARM_${AH}help_${AS}harm_tau${TAU}_k${TOPK}"
    echo ""
    echo "--- Scoring $(basename $GENARM_DIR) ---"
    python compute_reward.py --path "$GENARM_DIR" || echo "  (compute_reward failed; continuing)"
    echo ""
    echo "--- Scoring $(basename $PARM_DIR) ---"
    python compute_reward.py --path "$PARM_DIR" || echo "  (compute_reward failed; continuing)"
done

echo ""
echo "=== DONE — n=$N_PROMPTS, max_tokens=$MAX_TOKENS, ${#ALPHAS[@]} α × 2 methods, base=$BASE_7B ==="
echo "Results:   $OUT_ROOT"
