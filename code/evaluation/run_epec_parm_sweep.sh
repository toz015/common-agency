#!/bin/bash
# EPEC+PARM sweep: 3 alpha points, 1000 prompts, max_new_tokens=512
set -e

ALPHAS="0.2,0.8 0.5,0.5 0.8,0.2"

for pair in $ALPHAS; do
    IFS=',' read -r ah aharm <<< "$pair"
    echo "=========================================="
    echo "Running EPEC+PARM: alpha_help=$ah, alpha_harm=$aharm"
    echo "Started: $(date)"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=0 python generate_outputs_epec_parm.py \
        --alpha_helpfulness $ah \
        --alpha_harmlessness $aharm \
        --limit 1000 \
        --max_new_tokens 512 \
        --output_dir ./results_epec_parm \
        2>&1 | tail -5
    
    echo "Finished alpha=($ah,$aharm) at $(date)"
    echo ""
done

echo "All EPEC+PARM sweeps complete at $(date)"
