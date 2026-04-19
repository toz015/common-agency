# Phase-2 W2S Sanity-Check Results

Raw outputs from the sanity check defined in
[`../phase2_w2s/`](../phase2_w2s/README.md). Read that README first for
purpose, scope, run instructions, and analysis.

## Layout

```
results_phase2_sanity/
├── pareto_sanity.csv               # raw Pareto means across 6 configs
├── pareto_sanity.png
├── pareto_sanity_filtered.csv      # drift-filtered re-score
├── pareto_sanity_filtered.png
├── parm/{PARM_<αh>help_<αc>harm}/
│   ├── generation.json             # 50 records: uid, prompt, response, model, elapsed
│   ├── reward_result.json          # generation.json + per-response Beaver scores
│   └── mean_result.json            # {"help": float, "harm": float}
└── genarm/{GenARM_<αh>help_<αc>harm}/…   # same structure as parm/
```

## Field names in `reward_result.json`

Per-response scores are stored on each record under these exact keys:

| Key                          | Meaning                                       |
|------------------------------|-----------------------------------------------|
| `help_score (high better)`   | Beaver-7B reward; **higher = more helpful**   |
| `harm_score (low better)`    | Beaver-7B cost;   **lower  = safer**          |

The CSVs flip the harm sign (`safety = -harm`) so every column on the Pareto plot has "higher is better".

## Configs included

3 alphas × 2 methods = 6 cells, 50 prompts each → **300 generations** total.

| Method | α_help | α_harm |
|--------|--------|--------|
| PARM   | 0.2    | 0.8    |
| PARM   | 0.4    | 0.6    |
| PARM   | 0.8    | 0.2    |
| GenARM | 0.2    | 0.8    |
| GenARM | 0.4    | 0.6    |
| GenARM | 0.8    | 0.2    |

## How these were produced

Generated and scored end-to-end on a single A100 80GB; see
[`../phase2_w2s/run_sanity_w2s.sh`](../phase2_w2s/run_sanity_w2s.sh).
Drift-filtered files were produced afterwards on a laptop by
[`../phase2_w2s/filter_drift_rescore.py`](../phase2_w2s/filter_drift_rescore.py)
— no GPU required.

## Headline numbers

See [`../phase2_w2s/README.md`](../phase2_w2s/README.md) §7 for the Pareto
table, §8 for the drift-filter analysis, §9 for adapter-load verification, and
§10 for the help-score-gap stylistic breakdown.
