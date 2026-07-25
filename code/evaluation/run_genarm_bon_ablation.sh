#!/usr/bin/env bash
# Rebuttal ablation (Reviewer Uw2G Q6): GenARM Best-of-N reranking vs CAGE.
#
# For each α in the paper's safety sweep, sample N candidates from GenARM(α)
# with temperature=1.0 on the first M prompts, score with beaver-reward +
# beaver-cost, then rerank by α_help * q_help - α_harm * q_harm.
#
# Requires the trained GenARM LoRA adapters at:
#   code/training/PKU-SafeRLHF/exp_genarm_help/
#   code/training/PKU-SafeRLHF/exp_genarm_harm/
#
# Usage (from code/evaluation/):
#   bash run_genarm_bon_ablation.sh          # runs generation + scoring + metrics
#   SKIP_GEN=1 bash run_genarm_bon_ablation.sh   # only re-score / re-rank / re-eval
#
# Environment overrides:
#   N              (default 5)   number of BoN candidates per prompt
#   M              (default 50)  number of test prompts
#   ALPHAS         (default "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8")
#   OUT_ROOT       (default ./results_bon)
#   SKIP_GEN=1     skip generation (assumes candidates.json already exists)
#   SKIP_SCORE=1   skip scoring (assumes reward_result.json already exists)

set -euo pipefail
cd "$(dirname "$0")"

N=${N:-5}
M=${M:-50}
ALPHAS=${ALPHAS:-"0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8"}
OUT_ROOT=${OUT_ROOT:-./results_bon}
BASE=${BASE:-PKU-Alignment/alpaca-7b-reproduced}
ARM_HELP=${ARM_HELP:-../training/PKU-SafeRLHF/exp_genarm_help}
ARM_HARM=${ARM_HARM:-../training/PKU-SafeRLHF/exp_genarm_harm}
DATASET=${DATASET:-../data/PKU-SafeRLHF/test_prompt_only.json}

mkdir -p "$OUT_ROOT"

echo "=== GenARM-BoN@$N ablation | first $M prompts | alphas: $ALPHAS ==="

for ah in $ALPHAS; do
  as=$(python3 -c "print(round(1.0 - $ah, 4))")
  run_name="GenARM_BoN_N${N}_${ah}help_${as}harm"
  run_dir="$OUT_ROOT/$run_name"

  echo ""
  echo "=== alpha_help=$ah | alpha_harm=$as | dir=$run_dir ==="

  # --- 1. sample ---
  if [[ "${SKIP_GEN:-0}" != "1" ]]; then
    python3 generate_outputs_genarm_bon.py \
      --model_base_name_or_path "$BASE" \
      --model_arm_help_path "$ARM_HELP" \
      --model_arm_harm_path "$ARM_HARM" \
      --alpha_helpfulness "$ah" \
      --alpha_harmlessness "$as" \
      --num_candidates "$N" \
      --num_prompts "$M" \
      --datasets "$DATASET" \
      --output_dir "$OUT_ROOT" \
      --resume
  else
    echo "SKIP_GEN=1 -> skipping sampling for alpha=$ah"
  fi

  # --- 2. score + rerank ---
  if [[ "${SKIP_SCORE:-0}" != "1" ]]; then
    python3 score_and_rerank_bon.py --run_dir "$run_dir"
  else
    echo "SKIP_SCORE=1 -> skipping scoring for alpha=$ah"
  fi
done

# --- 3. metrics ---
echo ""
echo "=== Computing final HV/MIP table ==="
python3 compute_safety_metrics_bon.py \
  --first_m "$M" \
  --num_candidates "$N" \
  --bon_root "$OUT_ROOT"

echo ""
echo "=== Done. See metrics/safety_bon_ablation.json for machine-readable results. ==="
