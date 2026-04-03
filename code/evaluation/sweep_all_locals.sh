#!/bin/bash
# Local parallel execution for fine-grained tau between 0 and 0.5 AND re-running 0.05 and 0.1 with swept EPEC
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

INPUT="scored/scored_candidates_test_prompt_only_alpaca-7b-reproduced_N20.json"
OUTDIR="solved_pref_sweep"
mkdir -p "$OUTDIR"

# Finer tau values to analyze the high-agency trade-off regime, plus the ones we need to re-sweep
TAUS="0.01 0.02 0.05 0.08 0.1 0.15"
W_HELPS="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

echo "============================================"
echo "Starting PARALLEL sweep on local M1 Pro"
echo "Taus: $TAUS"
echo "============================================"

for tau in $TAUS; do
    (
        echo "--> [Started Thread for tau=$tau]"
        for w_help in $W_HELPS; do
            w_harm=$(python3 -c "print(round(1.0 - $w_help, 2))")
            logfile="${OUTDIR}/run_tau${tau}_w${w_help}.log"
            
            # OVERWRITE all existing logs for these taus, because they might contain single-point EPEC
            echo "      [tau=$tau] Running w_help=$w_help ..."
            python3 solve_equilibrium.py \
                --input_file "$INPUT" \
                --output_dir "$OUTDIR/" \
                --temperature_tau "$tau" \
                --parm_weights "$w_help" "$w_harm" \
                --normalize_rewards \
                > "$logfile" 2>&1
        done
        echo "--> [Finished Thread for tau=$tau]"
    ) &
done

wait
echo "============================================"
echo "  Local comprehensive sweep complete!"
echo "============================================"
