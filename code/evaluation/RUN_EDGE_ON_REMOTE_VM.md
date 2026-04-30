# Run a HH-RLHF edge sweep on another VM

This walks you through running ONE edge sweep (5 prefs × 4 methods × 200 prompts ≈ 2.3 days) on a fresh VM, independent of the (help, harm) edge already running on the primary VM.

Pick which edge:
- **`run_edge_help_humor.sh`** — α_harm = 0; gives the (help, humor) 2D Pareto plot
- **`run_edge_harm_humor.sh`** — α_help = 0; gives the (harm, humor) 2D Pareto plot

The instructions below work for either. Substitute the script name in step 5.

---

## 0. Prerequisites on the target VM

- Single NVIDIA GPU with ≥ 24 GB VRAM (A100, L4, A6000, RTX 6000 Ada all fine; the LLaMA-2-7B forward + PARM/GenARM adapters peak around ~22 GB).
- ≥ 60 GB free disk (model weights ~30 GB once downloaded; outputs are small).
- Conda or Miniconda installed.
- HuggingFace access to the LLaMA-2 family (your `~/.cache/huggingface/token` should be set, or you'll be prompted on first download).

---

## 1. Clone the repo and switch to the working branch

```bash
git clone https://github.com/toz015/common-agency.git ~/PARM/common-agency
cd ~/PARM/common-agency
git fetch --all
git checkout hh-rlhf-3objective-eval
```

If `hh-rlhf-3objective-eval` is not yet on the remote (e.g., the push from the primary VM has not gone through), use `fresh-start` instead and copy `run_edge_help_humor.sh` / `run_edge_harm_humor.sh` over manually.

You also need the sibling repos for the local PEFT and language-model-arithmetic packages (referenced by the generation scripts as relative imports):

```bash
cd ~/PARM
git clone https://github.com/THUDM/peft.git peft || true   # adjust if your fork URL differs
# Or rsync from the primary VM if the local peft fork is custom (recommended):
#   rsync -avz primary-vm:/home/toz015/PARM/peft/ ~/PARM/peft/
#   rsync -avz primary-vm:/home/toz015/PARM/language-model-arithmetic/ ~/PARM/language-model-arithmetic/
#   rsync -avz primary-vm:/home/toz015/PARM/safe-rlhf/ ~/PARM/safe-rlhf/
```

(The `peft` and `language-model-arithmetic` packages are local editable installs with PARM-specific modifications — don't substitute upstream PyPI versions.)

---

## 2. Create the conda env

```bash
conda create -n parm python=3.10 -y
conda activate parm

cd ~/PARM/language-model-arithmetic && pip install -e .
cd ~/PARM/peft                       && pip install -e .
cd ~/PARM/safe-rlhf                  && pip install -e .
cd ~/PARM/common-agency              && pip install -r requirements.txt

conda install -c nvidia cuda-compiler -y
```

Verify:

```bash
python -c "import peft; print(peft.__file__)"
# expect: /home/<user>/PARM/peft/src/peft/__init__.py  (the local fork)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: True <YourGPU>
```

---

## 3. Sync the test prompts and (optionally) the existing 16-pref results

The test prompt file is checked into the repo at `code/evaluation/../data/HH-RLHF/test_prompt_only.json` — should already be present after the clone.

If you also want the existing 16-pref result data on this VM (e.g., to regenerate the global plots locally after the new edge finishes), pull it from the primary VM:

```bash
rsync -avz primary-vm:/home/toz015/PARM/common-agency/code/evaluation/results/HH-RLHF/ \
           ~/PARM/common-agency/code/evaluation/results/HH-RLHF/
```

This is **not required** to run the edge sweep — only required if you want to regenerate the cross-method plots on this VM.

---

## 4. Pre-download the heavy model weights (one-time, optional but recommended)

This avoids surprises mid-run. The HF hub will cache to `~/.cache/huggingface/hub/`.

```bash
huggingface-cli download meta-llama/Llama-2-7b-chat-hf
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0
huggingface-cli download Ray2333/gpt2-large-helpful-reward_model
huggingface-cli download Ray2333/gpt2-large-harmless-reward_model
huggingface-cli download mohameddhiab/humor-no-humor
```

Total ≈ 28 GB.

---

## 5. Launch the edge sweep in the background

Pick ONE of the two scripts based on which edge you want.

```bash
cd ~/PARM/common-agency/code/evaluation
mkdir -p logs

# === Edge: (help, humor), α_harm = 0 ===
nohup bash -c '
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate parm
  echo "=== START $(date) ==="
  python -c "import peft; print(\"peft:\", peft.__file__)"
  bash run_edge_help_humor.sh
  echo "=== DONE $(date) ==="
' > logs/edge_help_humor.log 2>&1 &
echo "Background PID: $!"
```

Or for the other edge:

```bash
# === Edge: (harm, humor), α_help = 0 ===
nohup bash -c '
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate parm
  echo "=== START $(date) ==="
  python -c "import peft; print(\"peft:\", peft.__file__)"
  bash run_edge_harm_humor.sh
  echo "=== DONE $(date) ==="
' > logs/edge_harm_humor.log 2>&1 &
echo "Background PID: $!"
```

Each script:

1. Loops `GenARM → EPEC_GenARM → PARM → EPEC_PARM`.
2. For each method, generates the 5 edge preferences sequentially.
3. Skips any (method, pref) that already has `generation.json` with ≥ 200 prompts.
4. Auto-scores any new `generation.json` at the end.

ETA per edge ≈ **2.3 days** (GenARM ≈ 3 hr, EPEC+GenARM ≈ 4 hr, PARM ≈ 1 hr, EPEC+PARM ≈ 3 hr per pref × 5 prefs).

---

## 6. Monitor progress

```bash
# Last 40 log lines
tail -40 logs/edge_help_humor.log

# Just the milestones
grep -E '^>>>|GenARM:|EPEC_|PARM:|Saved|Saving to|^=== ' logs/edge_help_humor.log

# Process still alive?
ps -p <PID> -o pid,etime,stat,cmd

# How many of the 5 prefs done for the current method?
ls results/HH-RLHF/GenARM_*help_*harm_0.0humor/generation.json 2>/dev/null | wc -l   # (help, humor edge: change to harm_*humor)
```

---

## 7. After completion: pull results back to the primary VM

The edge sweep produces 20 new `results/HH-RLHF/<method>_<h>help_<s>harm_<u>humor/` directories. Pull them back to the primary VM where the cross-method plotting scripts live:

```bash
# From the primary VM
rsync -avz remote-vm:/home/<user>/PARM/common-agency/code/evaluation/results/HH-RLHF/*help_0.0harm_*/ \
           ~/PARM/common-agency/code/evaluation/results/HH-RLHF/
# (adjust the glob for the harm-humor edge: *_0.0help_*)
```

Then on the primary VM, regenerate the metrics and plots:

```bash
cd ~/PARM/common-agency/code/evaluation
python plot_pareto_and_metrics.py
python compute_hv_normalized.py
python plot_epec_delta_heatmap.py
python plot_preference_radar_normalized.py
```

---

## Common pitfalls

- **`ModuleNotFoundError: No module named 'peft'`** → the conda env wasn't activated, or the local PEFT fork isn't installed. Re-run `pip install -e .` inside `~/PARM/peft`.
- **`OutOfMemoryError`** during EPEC generation → the GPU has < 22 GB usable. Try `--max_new_tokens 128` to halve the working memory, or run on a larger GPU.
- **Disk fills mid-run** → each generation is small (~280 KB) but mid-run logs and HF cache can balloon. `pip cache purge` and `huggingface-cli delete-cache` if needed.
- **HF download fails on `meta-llama/Llama-2-7b-chat-hf`** → you need to accept the LLaMA license at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf and `huggingface-cli login` with a token that has the gated-repo access.

---

## What this run produces

5 new preference-vector results per method × 4 methods = 20 `generation.json` + `mean_result.json` + `reward_result.json` triples, all under `code/evaluation/results/HH-RLHF/`. Combined with the existing 16-pref sweep on the primary VM, you'll have enough data for a clean 2D Pareto plot along the chosen edge.
