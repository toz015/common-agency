# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PARM (Preference-Aware Autoregressive Reward Model)** implements multi-objective test-time alignment for LLMs (ICML 2025). It trains a reward model with multi-objective support (helpfulness + harmlessness) and uses model arithmetic at inference time to blend objectives via alpha weights.

## Environment Setup

```bash
conda create -n parm python=3.10
conda activate parm

cd language-model-arithmetic/ && pip install -e .
cd ../peft/ && pip install -e .
conda install -c nvidia cuda-compiler
cd ../safe-rlhf && pip install .
cd .. && pip install -r requirements.txt
```

## Common Commands

**Data preparation** (relabels PKU-SafeRLHF-10K with beaver reward models):
```bash
cd code/data && python relabel.py
```

**Training** (multi-GPU via Accelerate):
```bash
cd code/training && bash run.sh
```

**Standard evaluation pipeline:**
```bash
cd code/evaluation
python generate_outputs.py --model_parm_both_name_or_path <path> --alpha_helpfulness 0.5 --alpha_harmlessness 0.5
python compute_reward.py --path <path>
```

**EPEC evaluation pipeline (Common-Agency method):**
```bash
cd code/evaluation
python generate_candidates.py --model_base_name_or_path PKU-Alignment/alpaca-7b-reproduced --datasets ../data/test_prompt_only.json
python score_candidates.py --candidates_file candidates/all_candidates_*.json
```

## Architecture

### Data Flow
```
PKU-SafeRLHF-10K → relabel.py → train/dev/test.json
                                        ↓
                             train_pref_arm.py (PrefARMTrainer)
                                        ↓
                             Trained ARM model
                                        ↓
                generate_outputs.py (ModelArithmetic blending)
                                        ↓
                             compute_reward.py → reward scores
```

### Key Components

**`code/training/train_pref_arm.py` + `pref_arm_trainer.py`**
- `PrefARMTrainer` extends HuggingFace TRL's `DPOTrainer`
- Uses **PBLoRA** (Preference-Based LoRA) from the local `peft/` package — maintains separate LoRA adapters per objective
- During training, samples preference weights from a Dirichlet distribution (`pref_sample_p`)
- Key hyperparameters in `run.sh`: `beta_safe`, `beta_help`, `lora_r`, `lora_r2`

**`code/evaluation/generate_outputs.py`**
- Uses `ModelArithmetic` from `language-model-arithmetic/` to blend logits:
  `M = M_base + alpha_helpfulness × M_help + alpha_harmlessness × M_harm`
- `alpha_helpfulness` and `alpha_harmlessness` are the test-time alignment controls

**`code/data/relabel.py`**
- Assigns `better_response_id` (higher helpfulness score) and `safer_response_id` (lower cost/harm) labels to each sample

### Local Subpackages

| Package | Location | Purpose |
|---|---|---|
| `peft` | `peft/` | Modified PEFT with `PBLoraConfig` for multi-objective LoRA |
| `language-model-arithmetic` | `language-model-arithmetic/` | Logit-level model blending at inference time |
| `safe-rlhf` | `safe-rlhf/` | PKU-Alignment's reward model infrastructure |

### External Models (HuggingFace)

- **Base LLM**: `PKU-Alignment/alpaca-7b-reproduced`
- **Helpfulness reward**: `PKU-Alignment/beaver-7b-v1.0-reward`
- **Harmlessness reward**: `PKU-Alignment/beaver-7b-v1.0-cost`
- **Training data**: `PKU-Alignment/PKU-SafeRLHF-10K`
