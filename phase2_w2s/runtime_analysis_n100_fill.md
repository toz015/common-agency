# Runtime analysis — phase2_n100_fill

**Config:** 4-bit 65B GPTQ base + 7B fp16 ARMs, logit-sum aggregator,
n=100 prompts, max_new_tokens=512, α grid fill (8 alphas × 2 methods).
Instance: `a100-demo` (1× A100-80GB, us-central1-c).
Run started: 2026-04-22 04:13:47 UTC.

## Observed per-alpha GenARM wall-clock

Measured from per-dir `generation.json` mtimes (checkpoint-driven; final
write = alpha complete).

| α_help | start (UTC)        | finish (UTC)       | wall-clock   |
|--------|--------------------|--------------------|--------------|
| 0.0    | 04:13:47 (+5 min load) | 05:00:23       | **46.6 min** |
| 0.1    | 05:00:23           | 05:42:48           | **42.4 min** |
| 0.3    | 05:42:48           | 06:32:27           | **49.7 min** |
| 0.5    | 06:32:27           | in flight (40/100 @ 07:01 UTC) | ~46 min projected |

**GenARM per-alpha average: ~46 min** → ~27.6 s/prompt averaged.
tqdm instantaneous: 27–69 s/it, median ~33 s/it. Spikes correlate with
generations that run to the full 512-token cap (no early EOS).

## Extrapolated remaining work

| Phase                                          | Count       | Unit cost | Subtotal |
|------------------------------------------------|-------------|-----------|----------|
| GenARM remaining (0.5 ~60% left, 0.6/0.7/0.9/1.0) | 4.4 alphas | 46 min    | **~3.4 h** |
| PARM fill (8 alphas, 2 fwd/tok vs GenARM's 3)  | 8 alphas    | ~31 min¹  | **~4.1 h** |
| Beaver reward + cost scoring (16 new configs)  | 16          | ~6 min²   | **~1.6 h** |
| Pareto aggregation + plot                       | —           | —         | <2 min   |

¹ PARM has one fewer ARM forward per token than GenARM → expect ~⅔ of
GenARM's time. The original 3-alpha run took ~4-5 h per method for 3
alphas (≈80-100 min/alpha); the fill is running faster because
model-loading cost is amortized across one long tmux process instead of
being paid per-invocation.

² From past W2S runs; Beaver-7B scorer on 100 generations at fp16 is
~5-8 min end-to-end.

## Bottom line

- **Remaining: ~9 h** (3.4 h GenARM + 4.1 h PARM + 1.6 h scoring).
- **ETA full pipeline done: ~16:00 UTC** (≈ 09:00 PT / 12:00 ET on 2026-04-22).
- **Total run length start-to-finish: ~12 h** (started 04:13 UTC).
- **Cost: ~$35-45** at A100-80GB on-demand (~$3.70/hr × 12 h).

## Cadence drivers

- 65B-GPTQ forward ≈ 60-80 ms/token; 2× 7B-fp16 forwards ≈ 25 ms each
  → ~110-130 ms per token × ~250 generated tokens avg = ~30 s/prompt.
  Matches observation.
- Spikes (55-69 s/prompt) are prompts where the model runs to the full
  512-token cap (refusal → continued apology, open-ended narrative);
  ~1 in 6 prompts.
- No per-α trend: aggregation is symmetric in α so cost shouldn't vary
  — variance is prompt-driven, not α-driven.

## Risk assessment

No issues detected at mid-run:
- VRAM 67 GB / 82 GB stable (15 GB headroom).
- GPU util 37-40% steady (bottleneck is sequential forwards, not compute).
- No Traceback / OOM / NaN in log.
- Log fresh (<1 min mtime gap).

Worst-case upside: if α=1.0 or α=0.0 extremes hit pathological
long-generation rates (e.g. refusal loops), add ~20% → ~2 h.

## Follow-on

After fill completes, the natural next step is the EPEC sweep
(`run_epec_n100_t512.sh`, same 100 prompts × 11 alphas × 2 methods but
with EPEC aggregator). Expected **~2-3× slower per token** than
logit-sum (4 forwards/token + per-token L-BFGS-B), so budget ~24-30 h
for the EPEC sweep.
