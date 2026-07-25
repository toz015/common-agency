# Run the GenARM Best-of-N rebuttal ablation on another VM

Everything for this experiment is written and committed. **The only thing left is
executing it on a GPU box with enough disk.** This doc is the complete handoff.

---

## 0. What this experiment is

Rebuttal ablation for **Reviewer Uw2G, Q6**:

> *"Could one simply sample several candidates from GenARM and rerank them using
> the same reward models?"*

The claim to defend is that Best-of-N reranking **does not** recover CAGE's Pareto
front — BoN can only pick among samples the single blended policy already
produces, so it cannot reach the trade-off points CAGE reaches by construction.

**Design.** For each of the 8 preference vectors in the paper's safety sweep
(α_help = 0.1 … 0.8, α_harm = 1 − α_help):

1. Sample `N = 5` candidates per prompt from GenARM(α) at temperature 1.0, on the
   first `M = 50` test prompts.
2. Score every candidate with `beaver-7b-v1.0-reward` (q_help) and
   `beaver-7b-v1.0-cost` (q_harm).
3. Rerank and keep the argmax of `s = α_help · q_help − α_harm · q_harm`.
4. Aggregate the 8 winners' means into HV and MIP, against the same two baselines.

### Current results table — the row we still need

| Method | HV | MIP |
|---|---|---|
| CAGE | 231.68 | 0.801 |
| GenARM greedy | 112.90 | 0.559 |
| **GenARM-BoN@5** | **← this run produces it** | |

The baselines are already computed and committed in
`metrics/safety_bon_ablation.json`. They were subset from the 1000-prompt runs in
`reference_results_1000/`, which were produced on an earlier machine — **do not
regenerate them**, the ablation must compare against exactly these numbers.

---

## 1. Prerequisites on the target VM

- **GPU:** one NVIDIA card. Generation runs `ModelArithmetic` over an
  alpaca-7b base plus two LoRA-adapted copies of it, so plan for **≥ 40 GB VRAM**
  (A100 40/80 GB, H100). ⚠️ The exact peak has never been measured — see
  §5, which tells you to run one α first and watch it.
- **Disk: ≥ 60 GB free.** This is what blocked the original VM. Breakdown:
  - `alpaca-7b-reproduced` 13.5 GB
  - `beaver-7b-v1.0-reward` 13.2 GB
  - `beaver-7b-v1.0-cost` 13.2 GB
  - repo + outputs ≈ 2 GB
  If you only have ~30 GB, see §7 for the staged workaround.
- Conda or Miniconda.
- No gated-model access needed — all three models are public PKU-Alignment repos.

---

## 2. Clone the repo

```bash
git clone https://github.com/toz015/common-agency.git ~/common-agency
cd ~/common-agency
git fetch --all
git checkout rebuttal-genarm-bon-ablation
git submodule update --init --recursive     # pulls safe-rlhf
```

Unlike the HH-RLHF edge sweep, **no rsync from another machine is needed.**
Everything this experiment depends on is in the repo:

| Dependency | Where | Size |
|---|---|---|
| `peft` (PARM fork, `PBLoraConfig`) | `peft/` — tracked in-repo | — |
| `language-model-arithmetic` (PARM fork) | `language-model-arithmetic/` — tracked in-repo | — |
| `safe-rlhf` (for `AutoModelForScore`) | `safe-rlhf/` — git submodule | — |
| GenARM helpfulness adapter | `code/training/PKU-SafeRLHF/exp_genarm_help/` | 26 MB |
| GenARM harmlessness adapter | `code/training/PKU-SafeRLHF/exp_genarm_harm/` | 26 MB |
| Test prompts (1500, we use first 50) | `code/data/PKU-SafeRLHF/test_prompt_only.json` | — |
| CAGE + GenARM-greedy baselines | `code/evaluation/reference_results_1000/` | 34 MB |

The `peft` and `language-model-arithmetic` forks carry PARM-specific changes —
**do not** replace them with the PyPI versions.

---

## 3. Create the conda env

```bash
conda create -n parm python=3.10 -y
conda activate parm

cd ~/common-agency/language-model-arithmetic && pip install -e .
cd ~/common-agency/peft                      && pip install -e .
cd ~/common-agency/safe-rlhf                 && pip install .
cd ~/common-agency                           && pip install -r requirements.txt

conda install -c nvidia cuda-compiler -y
```

Verify before spending GPU hours:

```bash
python -c "import peft; print(peft.__file__)"
# expect: ~/common-agency/peft/src/peft/__init__.py   (the local fork, NOT site-packages)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: True <YourGPU>
python -c "from safe_rlhf.models import AutoModelForScore; print('ok')"
python -c "from model_arithmetic import ModelArithmetic, PromptedLLM; print('ok')"
```

---

## 4. The pipeline

Four files, all already on the branch. Driver script first, then what each stage does:

```
run_genarm_bon_ablation.sh          # loops over the 8 alphas, calls the three below
├── generate_outputs_genarm_bon.py  # samples N candidates -> candidates.json
├── score_and_rerank_bon.py         # beaver scoring + argmax rerank -> reward_result.json
└── compute_safety_metrics_bon.py   # HV/MIP across all 3 methods -> metrics/*.json + MD table
```

Output layout, one directory per α under `results_bon/`:

```
results_bon/GenARM_BoN_N5_0.1help_0.9harm/
├── candidates.json          # 250 rows = 50 prompts x 5 candidates
├── reward_result_all.json   # same 250 rows + q_help, q_harm, rerank_score (audit trail)
├── reward_result.json       # the 50 reranked winners; same schema as compute_reward.py
└── mean_result.json         # {"help": ..., "harm": ...}
```

`reward_result.json` deliberately mirrors the paper's existing schema so the
downstream HV/MIP code reads it unchanged.

---

## 5. Run it — benchmark one α first

**Do not launch all 8 α values blind.** Runtime here is genuinely unmeasured: the
full sweep is 2000 generations (8 α × 50 prompts × 5 candidates) of up to 512
tokens each, through a three-model ensemble. That could be several hours or
several days, and the VRAM peak is likewise unverified.

Run the middle preference alone first:

```bash
cd ~/common-agency/code/evaluation
conda activate parm

ALPHAS="0.5" bash run_genarm_bon_ablation.sh 2>&1 | tee logs/bon_alpha0.5.log
```

Watch `nvidia-smi` during the first minute or two. Then read off the timing the
generation script prints at the end and multiply by 8 to size the full sweep.

If VRAM is tight, the knobs are `--max_new_tokens` (default 512) and running
fewer α per process. If it OOMs outright, that's the three-copies-of-7B problem
and the ensemble needs to be sharded or moved to an 80 GB card.

**Then launch the rest.** `--resume` is passed on every call, so any α whose
`candidates.json` already exists is skipped — the sweep is safely restartable and
re-running it will not redo α=0.5.

```bash
nohup bash run_genarm_bon_ablation.sh > logs/bon_full_sweep.log 2>&1 &
tail -f logs/bon_full_sweep.log
```

### Knobs

| Var | Default | Meaning |
|---|---|---|
| `N` | 5 | candidates per prompt (the "N" in Best-of-N) |
| `M` | 50 | number of test prompts |
| `ALPHAS` | `0.1 … 0.8` | α_help sweep |
| `OUT_ROOT` | `./results_bon` | output root |
| `SKIP_GEN=1` | — | re-score / re-rank only, reusing existing `candidates.json` |
| `SKIP_SCORE=1` | — | generation only |

If you need a bigger N to make the point convincingly (a reviewer may argue N=5
is too small), `N=10 bash run_genarm_bon_ablation.sh` writes to a *separate*
directory name, so it will not clobber the N=5 results.

---

## 6. Finishing up

The driver calls the metrics step itself, but you can rerun it anytime without a GPU:

```bash
python3 compute_safety_metrics_bon.py --first_m 50 --num_candidates 5 --bon_root ./results_bon
```

It prints a Markdown table ready to paste into the rebuttal and rewrites
`metrics/safety_bon_ablation.json` with all three methods.

**Sanity checks before you trust the number:**

- Every α directory has `n: 50` in the per-alpha block — a `MISSING:` line printed
  during the run means that α silently didn't contribute.
- `GenARM-BoN@5`'s HV should land **between** GenARM greedy (112.90) and CAGE
  (231.68). BoN reranking genuinely helps over greedy, so a modest gain is the
  expected and honest result — the argument is that it does not close the gap to
  CAGE. If BoN somehow *beats* CAGE, stop and investigate before writing it up.
- Spot-check a few rows of `reward_result_all.json`: the 5 candidates for one
  prompt should actually differ from each other. If they're identical, sampling
  silently fell back to greedy and the whole ablation is void.

Then commit results and push:

```bash
cd ~/common-agency
git add code/evaluation/results_bon code/evaluation/metrics/safety_bon_ablation.json
git commit -m "Rebuttal Uw2G Q6: GenARM Best-of-N@5 ablation results"
git push origin rebuttal-genarm-bon-ablation
```

---

## 7. If the target VM is also short on disk

Generation and scoring need **disjoint** model sets, so you can stage them and
never hold all 40 GB at once. Peak drops to ~27 GB:

```bash
# Stage A — needs only alpaca-7b (13.5 GB)
SKIP_SCORE=1 bash run_genarm_bon_ablation.sh

# free the base model
rm -rf ~/.cache/huggingface/hub/models--PKU-Alignment--alpaca-7b-reproduced

# Stage B — needs only the two beaver models (26.4 GB)
SKIP_GEN=1 bash run_genarm_bon_ablation.sh
```

Stage B still loads both reward models simultaneously (26.4 GB). To go lower than
that you'd have to split `score_and_rerank_bon.py` into two passes — help first,
then harm — which is a code change, not a flag.

---

## 8. State of the original VM (for reference)

The A100 box where this was authored has a working GPU
(A100-SXM4-40GB, driver 535.261.03, CUDA 12.2) but only ~6 GB free disk, with
another user holding 52 GB of `/home`. That is the sole reason this run moved.
Nothing was ever executed there: `results_bon/` was empty and no candidate was
ever sampled. Note also that on that machine torch lives only in the `parm` env —
base python has none.
