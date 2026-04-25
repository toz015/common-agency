# Phase-2 — Logit-sum W2S baseline + Token-level EPEC (W2S 65B)

End-to-end pipeline for two complementary common-agency experiments on
PKU-SafeRLHF, sharing prompts / α grid / Beaver scorer for direct
comparability:

1. **Logit-sum baseline (W2S, 65B base)** — PARM and GenARM at 11 α's
   (0.0 → 1.0), n=300 prompts, max_new_tokens=512. 4-bit GPTQ Llama-65B
   steered at decoding time by 7B reward-model adapters via
   `model_arithmetic`. Full 22-config sweep, Beaver-scored, with Pareto
   plot.
2. **Token-level EPEC (W2S 65B, n=50 probe)** — same model stack as the
   logit-sum baseline (4-bit GPTQ 65B + 7B PARM PBLoRA / 7B GenARM
   help+harm LoRAs), but the per-token aggregator is the Common-Agency
   equilibrium (Nonlinear Jacobi + L-BFGS-B inner step) instead of a
   linear logit sum. Four forwards per token (one 65B base + three on
   the 7B ARM stack), top-k=50, τ=0.1, max_iter=5.

Earlier directions retired (still recoverable from git history):
- **Post-hoc EPEC rerank** over the logit-sum candidate menu — was a
  cheap CPU-side experiment, but doesn't reflect what the paper claims
  as EPEC. Superseded by token-level EPEC.
- **Tong Zhu's 7B same-model EPEC** (`epec-parm-sweep-20260424`) —
  briefly tried by reusing his scripts verbatim with base = 7B
  alpaca-reproduced. Not directly comparable to our 65B logit-sum
  baseline, so reverted.
- **Marlin int4*fp16 GPTQ kernel** — tried for 2-3× fwd speedup; fwd
  time was unchanged but EPEC inner-solve time grew 7× (21 s → 157 s
  per prompt), suspect because Marlin's logit memory layout hits a slow
  numpy/scipy code path. Disabled until diagnosed.

Tong Zhu's reference 7B scripts are kept at
`code/evaluation/generate_outputs_epec_{genarm,parm}.py` for comparison
but are not part of the active pipeline.

---

## 1. Model stack (shared across both experiments)

| Component                              | Version                                              |
|----------------------------------------|------------------------------------------------------|
| Base LM                                | `TheBloke/alpaca-lora-65B-GPTQ` (4-bit, group=128)   |
| ARM backbone (for GenARM / PARM)       | `PKU-Alignment/alpaca-7b-reproduced` (fp16)          |
| PARM PBLoRA adapter                    | `code/training/PKU-SafeRLHF/exp`                     |
| GenARM helpfulness LoRA                | `code/training/PKU-SafeRLHF/exp_genarm_help`         |
| GenARM safety LoRA                     | `code/training/PKU-SafeRLHF/exp_genarm_harm`         |
| Reward model (scoring)                 | `PKU-Alignment/beaver-7b-v1.0-reward`                |
| Cost model (scoring)                   | `PKU-Alignment/beaver-7b-v1.0-cost`                  |
| Eval prompts (1500 total)              | `data/test_prompt_only.json` (PKU-SafeRLHF)          |
| Decoding logit-arithmetic              | `model_arithmetic` 1.1.0                             |
| Quantization runtime                   | `auto_gptq` 0.7.1+cu124                              |
| Transformers                           | 4.36.2 (pinned)                                      |
| PyTorch / CUDA                         | 2.5.1+cu124 / driver 570.211.01                      |

**Method signatures (logit-sum baseline).**
- **PARM** = base 65B + 1 PBLoRA adapter on the 7B backbone with preference
  vector `(α_help, α_harm)` injected per run.
- **GenARM** = base 65B + 2 independent autoregressive reward LoRAs on
  the 7B backbone, linearly combined with `(α_help, α_harm)` at every
  decoding step.

**Method signatures (token-level EPEC).**
- Implicit reward at each step: `q_j = log π_arm_j − log π_arm_base` on
  the 65B base LM's top-50 actions.
- Each principal `j` solves
  `max_{y_j ∈ [0, q_j]} π★ · (q_j − y_j)`
  via L-BFGS-B; aggregator is the Common-Agency equilibrium
  `π★ = softmax(log π_65B_base + Σ_j y_j / τ)`.
- 4 forwards per token: 65B base, 7B ARM-base, 7B+help, 7B+harm.

---

## 2. Results (t=512)

### 2a. Logit-sum baseline (n=300)

Full scored results at `results_n300_t512/{parm,genarm}/*/`.
Initial n=100 sanity run kept at `results_n100_t512/{parm,genarm}/*/`
(first 100 uids are a deterministic prefix of n=300).

![pareto_n300](pareto_n300.png)

- GenARM strictly Pareto-dominates PARM at every α ≥ 0.3.
- GenARM saturates at help ≈ 6.0 (peak α=0.7); PARM saturates at help ≈ 4.95 (α=1.0).
- CSV: `pareto_n300.csv`. Per-config aggregate: `results_n300_t512/*/*/mean_result.json`.
- Earlier n=100 version kept alongside: `pareto_n100.{png,csv}`.

### 2b. Token-level EPEC (W2S 65B, n=50)

Output: `results_phase2_n50_t512_epec/{parm,genarm}/EPEC_<method>_<ah>help_<as>harm_tau0.1_k50/`.
Once Beaver-scored, will be plotted alongside the n=50 prefix of 2a.

### Runtime (actual / projected)

- **Logit-sum fill (11 α × 2 methods × n=100 × 512 tok):** ~12 h on 1× A100-80GB.
  Breakdown in `runtime_analysis_n100_fill.md`.
- **Logit-sum extend n=100 → n=300 (+200 new prompts × 22 configs + Beaver rescore):** ~29 h
  (GenARM ~110 min/config, PARM ~78 min/config, ~3 h Beaver @ n=300).
- **Token-level EPEC W2S 65B (11 α × 2 methods × n=50 × 512 tok):** projected ~50 h
  at 162 s/prompt (4 forwards/token, fwd dominates EPEC ~85/15). Updated post-run.

---

## 3. Layout

```
phase2_w2s/
├── README.md                       <- this file
├── runtime_analysis_n100_fill.md   <- per-α wall-clock, risk assessment
│
├── # Helpers
├── make_subset.py                  <- take first N prompts (deterministic prefix)
├── plot_pareto_w2s.py              <- logit-sum Pareto plot from mean_result.json
├── compute_hv.py                   <- 2D hypervolume (shared ref point)
│
├── # Logit-sum baseline drivers
├── setup_a100.sh                   <- one-time A100 env setup
├── deploy_n100_fill.sh             <- 11-α × n=100 logit-sum sweep deploy
├── deploy_n300_extend.sh           <- n=100 → n=300 logit-sum extension
├── run_n100_t512_fill.sh           <- 11-α × n=100 wrapper
├── run_n300_t512_extend.sh         <- extend +200 per α (resume-friendly)
│
├── # Token-level EPEC (W2S 65B) drivers — generation scripts live at
├── # code/evaluation/generate_outputs_epec_{genarm,parm}.py for parity
├── # with the logit-sum baseline scripts.
├── deploy_n50_epec.sh              <- one-shot deploy
├── run_n50_t512_epec.sh            <- 11-α × 2-method × n=50 wrapper
├── monitor_n50_epec.sh             <- background poll-monitor
│
├── # Data products (logit-sum)
├── pareto_n300.csv                 <- PRIMARY: method, α, help, harm, safety (n=300)
├── pareto_n300.png                 <- PRIMARY Pareto plot
├── pareto_n100.csv                 <- initial n=100 sanity version
├── pareto_n100.png
│
├── results_n300_t512/              <- PRIMARY: 22 dirs × {mean,reward}_result.json
│   ├── parm/PARM_<ah>help_<as>harm/{reward_result,mean_result}.json
│   └── genarm/GenARM_<ah>help_<as>harm/{reward_result,mean_result}.json
│
└── results_n100_t512/              <- initial sanity: 22 dirs × {generation,reward,mean}_result.json
    ├── parm/PARM_<ah>help_<as>harm/{generation,reward_result,mean_result}.json
    └── genarm/GenARM_<ah>help_<as>harm/{generation,reward_result,mean_result}.json
```

Token-level EPEC results land at `../results_phase2_n50_t512_epec/`
(outside `phase2_w2s/`, mirroring the existing logit-sum results dirs).

---

## 4. Reproducing the logit-sum baseline on a fresh A100

Expensive — ~12 h for the 11-α × n=100 sweep, ~36 h for n=300.
See `runtime_analysis_n100_fill.md` for the rate breakdown.

```bash
# From project root:
bash phase2_w2s/deploy_n100_fill.sh       # 11-α × n=100
bash phase2_w2s/deploy_n300_extend.sh     # extend to n=300 (resume-friendly)
```

Both scripts scp into `~/phase2_w2s/` on `a100-demo`, launch a detached
tmux, and tee to `~/phase2_{n100_fill,n300_extend}.log`. The extension
script is idempotent and resumes via uid-matching.

---

## 5. Reproducing the token-level EPEC (W2S 65B) sweep

Projected ~50 h on 1× A100-80GB (4 forwards/token, 65B GPTQ + 3×7B fp16).

```bash
# From project root:
bash phase2_w2s/deploy_n50_epec.sh
# Optionally tail progress in another terminal:
bash phase2_w2s/monitor_n50_epec.sh
```

Drives `run_n50_t512_epec.sh` which loops the 11-α × 2-method grid and
calls `generate_outputs_epec_{genarm,parm}.py` per config. Each script
honors `--resume true`, so re-launching after a crash skips already-
generated uids per (config, prompt).

---

## 6. What's **not** in this branch

- **Tong Zhu's 7B same-model EPEC** scripts (from his
  `epec-parm-sweep-20260424` branch). Briefly tried (overwrote
  `code/evaluation/generate_outputs_epec_*.py`) but reverted because
  base = 7B is not directly comparable to our 65B logit-sum baseline,
  and the shared-backbone `disable_adapter()` / `set_adapter()` pattern
  can't extend to W2S (the 7B LoRAs would dim-mismatch on the 65B
  base). Recoverable from git history at commit `8ce97a5`. Current
  `code/evaluation/generate_outputs_epec_*.py` are the W2S 65B versions.
- **Marlin GPTQ kernel.** Disabled (see commit `48e140c` for the
  enabling patch); EPEC inner-solve time regressed 7× on Marlin output.
- **Post-hoc EPEC rerank** pipeline. Removed once token-level EPEC was
  in place.
- N > 300 sweep results.
- HH-RLHF evaluation. Data is available in the parent repo
  (`code/data/HH-RLHF/test_prompt_only.json`, 1000 prompts), but the
  current subset path is PKU-SafeRLHF only.

---

## 7. Credits

- EPEC algorithm and inner-solver pattern — Tong Zhu's
  `epec-parm-sweep-20260424` branch (originally targeted 7B same-model,
  3 forwards/token), ported into the W2S 65B stack at
  `code/evaluation/generate_outputs_epec_*.py` (4 forwards/token: 65B
  base + separate 7B ARM stack, since 7B LoRAs can't load on 65B).
- Everything in `phase2_w2s/*.sh`, `compute_hv.py`, and
  `plot_pareto_w2s.py` — written for this branch.
