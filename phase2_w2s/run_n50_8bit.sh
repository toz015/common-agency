#!/bin/bash
# Phase-2 weak-to-strong 8-bit bnb sanity run:
#   50 prompts × 3 alphas × 2 methods at max_new_tokens=128.
#
# Drop-in alt to run_sanity_w2s.sh: same prompt set / alphas / tokens, but
# swaps the 4-bit GPTQ base (TheBloke/alpaca-lora-65B-GPTQ) for the FP16 HF
# mirror (TheBloke/alpaca-lora-65B-HF) loaded via bitsandbytes 8-bit.
#
# Output: results_phase2_8bit_sanity/ — does NOT clobber the 4-bit sanity or
# n=100 results dirs.
#
# VRAM budget on A100 80GB:
#   PARM   : ~65GB base + tiny LoRA        -> comfortable
#   GenARM : 65GB base + 2×7GB ARM bases  -> ~79GB, tight (KV-cache may OOM)
#
# Prereqs (same A100 instance as sanity / n=100 runs):
#   source ~/common-agency/venv/bin/activate
#   pip install -U bitsandbytes accelerate  # if not already present
#   tmux new -s phase2_8bit
#   bash ~/phase2_w2s/run_n50_8bit.sh 2>&1 | tee ~/phase2_8bit.log

set -euo pipefail
source "$HOME/common-agency/venv/bin/activate"

# Route HF cache to the 375 GB local NVMe. Boot disk (/, 194 GB) doesn't
# have room for the 130 GB FP16 base alongside existing caches. User must
# have mounted /dev/nvme0n1 at /mnt/ssd first; see plan for the mkfs/mount
# commands. Local SSD is ephemeral (wiped on stop), so expect a re-download
# after each instance start.
if [[ ! -d /mnt/ssd ]]; then
    echo "ERROR: /mnt/ssd does not exist. Mount the local NVMe first:" >&2
    echo "  sudo mkfs.ext4 -F /dev/nvme0n1 && sudo mkdir -p /mnt/ssd && \\" >&2
    echo "  sudo mount /dev/nvme0n1 /mnt/ssd && sudo chown -R \$USER:\$USER /mnt/ssd" >&2
    exit 1
fi
mkdir -p /mnt/ssd/huggingface
export HF_HOME=/mnt/ssd/huggingface

############# CONFIG #############
REPO="$HOME/common-agency"
BASE_65B="TheBloke/alpaca-lora-65B-HF"

PARM_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp"
HELP_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp_genarm_help"
HARM_ADAPTER="$REPO/code/training/PKU-SafeRLHF/exp_genarm_harm"

FULL_DATASET="$REPO/data/test_prompt_only.json"
SUBSET_DATASET="$REPO/data/test_prompt_only_50.json"
N_PROMPTS=50
MAX_TOKENS=128
CHECKPOINT_EVERY=10

OUT_ROOT="$REPO/results_phase2_8bit_sanity"
OUT_PARM="$OUT_ROOT/parm"
OUT_GENARM="$OUT_ROOT/genarm"
mkdir -p "$OUT_PARM" "$OUT_GENARM"

ALPHAS=(0.2 0.4 0.8)

GEN_PARM="$HOME/phase2_w2s/generate_outputs_8bit.py"
GEN_GENARM="$HOME/phase2_w2s/generate_outputs_genarm_8bit.py"

cd "$REPO/code/evaluation"

############# 1. SUBSET PROMPTS #############
echo "=== [1/5] Building $N_PROMPTS-prompt subset ==="
python "$HOME/phase2_w2s/make_subset.py" \
    --input  "$FULL_DATASET" \
    --output "$SUBSET_DATASET" \
    --n      "$N_PROMPTS"

############# 2. GENERATE — GenARM #############
echo ""
echo "=== [2/5] GenARM 8-bit generation (3 alphas × $N_PROMPTS prompts, bnb 8-bit 65B + 8-bit ARMs, max_new=$MAX_TOKENS) ==="
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    echo ""
    echo "--- GenARM alpha_help=$AH  alpha_harm=$AS ---"
    python "$GEN_GENARM" \
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
echo "=== [3/5] PARM 8-bit generation (3 alphas × $N_PROMPTS prompts, bnb 8-bit 65B, max_new=$MAX_TOKENS) ==="
for AH in "${ALPHAS[@]}"; do
    AS=$(python -c "print(round(1.0 - $AH, 1))")
    echo ""
    echo "--- PARM alpha_help=$AH  alpha_harm=$AS ---"
    python "$GEN_PARM" \
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
    --output_png "$OUT_ROOT/pareto_8bit_sanity.png" \
    --output_csv "$OUT_ROOT/pareto_8bit_sanity.csv" \
    --n_per_cell "$N_PROMPTS"

echo ""
echo "=== DONE ==="
echo "Results:  $OUT_ROOT"
echo "Plot:     $OUT_ROOT/pareto_8bit_sanity.png"
echo "CSV:      $OUT_ROOT/pareto_8bit_sanity.csv"
