#!/bin/bash
# One-time setup on the A100 GCP instance for Phase-2 weak-to-strong sanity check.
#
# Target instance:
#   gcloud compute ssh --zone=us-central1-c a100-demo \
#       --project=llm-applications-490420
#
# Memory note: A100 80GB. Inference stack (4-bit GPTQ 65B + fp16 7B + adapter)
# fits in ~50 GB with KV cache headroom. Beaver scoring (2× 7B fp16 ≈ 28 GB)
# also fits comfortably — runs in a separate process after generation.
#
# Usage (once after SSH'ing into the instance):
#   bash setup_a100.sh

set -euo pipefail

REPO_DIR="$HOME/common-agency"
REPO_URL="https://github.com/toz015/common-agency.git"
REPO_BRANCH="fresh-start"

echo "=== 1. Clone repo (fresh-start branch, with vendored peft submodule) ==="
if [ ! -d "$REPO_DIR" ]; then
    git clone --recursive -b "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
else
    echo "Repo already exists; pulling latest."
    cd "$REPO_DIR" && git fetch && git checkout "$REPO_BRANCH" && git pull \
        && git submodule update --init --recursive
fi

cd "$REPO_DIR"

echo ""
echo "=== 2. Create venv ==="
if [ ! -d venv ]; then
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip wheel

echo ""
echo "=== 3. Install vendored peft (PBLoRA-patched, MUST be before plain peft) ==="
# Official peft does not understand peft_type="PBLORA"; the vendored copy does.
pip install -e peft/

echo ""
echo "=== 4. Install runtime deps (incl. auto-gptq for 4-bit 65B) ==="
pip install \
    "torch>=2.1" \
    "transformers>=4.40,<4.50" \
    accelerate \
    bitsandbytes \
    "auto-gptq>=0.7.1" \
    optimum \
    datasets \
    tqdm \
    matplotlib \
    sentencepiece \
    protobuf \
    google-cloud-storage

echo ""
echo "=== 5. Verify GPU ==="
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("Mem (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
PY

echo ""
echo "=== 6. Verify auto-gptq import (loading TheBloke/alpaca-lora-65B-GPTQ depends on this) ==="
python - <<'PY'
try:
    from auto_gptq import AutoGPTQForCausalLM
    print("auto_gptq import: OK")
except Exception as e:
    print("auto_gptq import FAILED:", e)
PY

echo ""
echo "=== 7. Verify adapters exist (committed in fresh-start branch) ==="
ls -la "$REPO_DIR/code/training/PKU-SafeRLHF/exp/"             || echo "MISSING: PARM PBLoRA adapter (exp/)"
ls -la "$REPO_DIR/code/training/PKU-SafeRLHF/exp_genarm_help/" || echo "MISSING: GenARM help adapter"
ls -la "$REPO_DIR/code/training/PKU-SafeRLHF/exp_genarm_harm/" || echo "MISSING: GenARM harm adapter"

echo ""
echo "=== 8. Verify test data ==="
ls -la "$REPO_DIR/data/test_prompt_only.json" || echo "MISSING: test_prompt_only.json"

echo ""
echo "Setup complete. Next step:"
echo "  tmux new -s phase2"
echo "  bash ~/phase2_w2s/run_sanity_w2s.sh 2>&1 | tee ~/phase2_sanity.log"
