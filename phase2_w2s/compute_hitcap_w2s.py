"""
Compute cap-hit rate (% responses with >=510 generated tokens) per (method, alpha)
on the W2S 65B + 7B-ARM stack. Hit-cap is a proxy for token-repetition pathology
('refuse then pad to inflate helpfulness' is the canonical pattern at low α; bell-
shaped sentence repetition for GenARM at mid-α; safety-lecture loops for CAGE
at low α).

Reads `generation.json` from each per-(method, alpha) dir, tokenizes the
`response` field with LlamaTokenizer, counts records with token count >=
THRESH (default 510 of max_new_tokens=512). Dumps a single JSON keyed by
method, with per-α counts.

Run on a100-demo (where the generation.json files live):
  python phase2_w2s/compute_hitcap_w2s.py \
      --parm_dir   ~/common-agency/results_n300_t512/parm \
      --genarm_dir ~/common-agency/results_n300_t512/genarm \
      --cage_dir   ~/common-agency/results_phase2_n50_t512_epec/genarm \
      --output     phase2_w2s/hitcap_w2s.json

Tokenizer: PKU-Alignment/alpaca-7b-reproduced (matches the eval script).
"""
import argparse
import json
import re
from pathlib import Path

DIR_RE = re.compile(r"(?:EPEC_)?(?P<method>PARM|GenARM)_(?P<ah>[0-9.]+)help_(?P<as>[0-9.]+)harm")


def count_hits(records, tokenizer, thresh):
    n_total = len(records)
    n_hit = 0
    tot_tok = 0
    max_tok = 0
    for r in records:
        text = r.get("response", "")
        ids = tokenizer(text, add_special_tokens=False).input_ids
        n = len(ids)
        tot_tok += n
        if n > max_tok:
            max_tok = n
        if n >= thresh:
            n_hit += 1
    avg_tok = tot_tok / n_total if n_total else 0.0
    return n_total, n_hit, avg_tok, max_tok


def scan_method(root, method_filter, tokenizer, thresh):
    """Walk root for subdirs matching DIR_RE; return list sorted by α_help."""
    out = []
    if not root.exists():
        print(f"WARN: {root} does not exist, skipping.")
        return out
    for sub in sorted(root.iterdir()):
        m = DIR_RE.search(sub.name)
        if not m or m.group("method") != method_filter:
            continue
        gen = sub / "generation.json"
        if not gen.exists():
            print(f"WARN: no generation.json in {sub}")
            continue
        records = json.loads(gen.read_text())
        n_total, n_hit, avg, mx = count_hits(records, tokenizer, thresh)
        out.append({
            "a": float(m.group("ah")),
            "n_total": n_total,
            "n_hit": n_hit,
            "pct": round(100.0 * n_hit / n_total, 2) if n_total else 0.0,
            "avg_tok": round(avg, 1),
            "max_tok": int(mx),
        })
    out.sort(key=lambda r: r["a"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parm_dir",   required=False, help="logit-sum PARM root, e.g. results_n300_t512/parm")
    p.add_argument("--genarm_dir", required=False, help="logit-sum GenARM root")
    p.add_argument("--cage_dir",   required=False, help="EPEC+GenARM root, e.g. results_phase2_n50_t512_epec/genarm")
    p.add_argument("--threshold",  type=int, default=510, help="token-count threshold for cap-hit (max_new_tokens=512)")
    p.add_argument("--tokenizer",  default="PKU-Alignment/alpaca-7b-reproduced")
    p.add_argument("--output",     required=True, help="output JSON path")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    out = {
        "_about": (
            f"Cap-hit rate (>= {args.threshold} tokens, max_new_tokens=512) per "
            f"(method, alpha_help) on the W2S 65B + 7B-ARM stack."
        ),
        "_threshold_tokens": args.threshold,
        "_tokenizer": args.tokenizer,
    }
    if args.parm_dir:
        out["PARM"] = scan_method(Path(args.parm_dir), "PARM", tokenizer, args.threshold)
    if args.genarm_dir:
        out["GenARM"] = scan_method(Path(args.genarm_dir), "GenARM", tokenizer, args.threshold)
    if args.cage_dir:
        out["CAGE"] = scan_method(Path(args.cage_dir), "GenARM", tokenizer, args.threshold)

    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.output}")
    for k in ("PARM", "GenARM", "CAGE"):
        if k in out:
            print(f"\n{k}:")
            for r in out[k]:
                print(f"  α={r['a']:.1f}  n_hit/n_total = {r['n_hit']}/{r['n_total']}  ({r['pct']}%)")


if __name__ == "__main__":
    main()
