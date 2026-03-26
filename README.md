# Common Agency

*Department of Statistics and Data Science, UCLA*

> **Work in progress.** This repository is under active development.

## Overview

This project studies **multi-objective test-time alignment of large language models (LLMs)** through the lens of **common agency** — a game-theoretic framework where multiple principals (e.g., helpfulness, harmlessness) simultaneously offer incentive contracts to a single LLM agent.

Each principal strategically chooses an incentive vector to steer the shared model toward its objective. The agent's best response is a KL-regularized policy over candidate outputs. We formulate the coordination problem as an **Equilibrium Problem with Equilibrium Constraints (EPEC)** and solve it via a Nonlinear Jacobi method, iterating each principal's MPEC until convergence to a Nash equilibrium.

Compared to existing methods like **PARM** (Lin et al., 2025), which fixes preference weights at inference time, our common-agency formulation treats incentive specification as a *strategic decision* — allowing objectives to interact and coordinate through the shared LLM output distribution.

## Baseline

- **PARM**: [Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model](https://github.com/BaijiongLin/PARM) (Lin et al., ICML 2025)

## Repository Structure

```
code/
  data/               # Test prompt preparation
  evaluation/
    generate_candidates.py   # Step 1: sample N candidate responses from base LLM
    score_candidates.py      # Step 2: score candidates with Beaver reward models
    scored/                  # Scored candidate outputs
```

## Evaluation Pipeline

**Step 1** — Generate candidate responses from the base LLM:
```bash
cd code/evaluation
python generate_candidates.py \
  --model_base_name_or_path PKU-Alignment/alpaca-7b-reproduced \
  --datasets ../data/test_prompt_only.json \
  --num_candidates 20
```

**Step 2** — Score each candidate with helpfulness and harmlessness reward models:
```bash
python score_candidates.py \
  --candidates_file candidates/<run_tag>/all_candidates_<run_tag>.json
```

Output fields per candidate: `q_help` (higher = more helpful), `q_harm` (lower = safer).

## Models Used

| Role | Model |
|---|---|
| Base LLM | `PKU-Alignment/alpaca-7b-reproduced` |
| Helpfulness reward | `PKU-Alignment/beaver-7b-v1.0-reward` |
| Harmlessness reward | `PKU-Alignment/beaver-7b-v1.0-cost` |
