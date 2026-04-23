"""
Build a post-hoc EPEC candidate pool from the existing W2S logit-sum
generations (11 PARM α + 11 GenARM α) already scored by Beaver.

For each prompt (uid), the action space = 22 candidates = one response per
(method, α) config. Each candidate record matches the schema epec_solve.py
expects:

    { uid, prompt, candidate_id, response, log_prob, q_help, q_harm }

`log_prob` is set to 0.0 uniformly (we don't have per-candidate log π of
the sampling policy — different candidates come from different α policies
anyway — so log_pi_base degenerates to uniform, which is exactly what we
want when treating the N=22 pool as an unstructured menu).

Output: one combined JSON at --out (default: scored_candidates_w2s_N22.json).

Usage:
    python build_w2s_candidate_pool.py \
        --parm_dir   results_scored/parm \
        --genarm_dir results_scored/genarm \
        --out        scored_candidates_w2s_N22.json
"""
import argparse
import json
import re
from pathlib import Path


DIR_RE = re.compile(r"^(?P<method>PARM|GenARM)_(?P<ah>[0-9.]+)help_(?P<as>[0-9.]+)harm$")
HELP_KEY = "help_score (high better)"
HARM_KEY = "harm_score (low better)"


def load_dir(root: Path, method_prefix: str):
    """Return list of (method, alpha_help, alpha_harm, [reward_records])."""
    out = []
    for sub in sorted(root.iterdir()):
        m = DIR_RE.match(sub.name)
        if not m or m.group("method") != method_prefix:
            continue
        rf = sub / "reward_result.json"
        if not rf.exists():
            print(f"WARN: missing {rf}")
            continue
        recs = json.load(open(rf))
        out.append((method_prefix, float(m.group("ah")), float(m.group("as")), recs))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parm_dir",   default=None)
    p.add_argument("--genarm_dir", default=None)
    p.add_argument("--out",        required=True)
    args = p.parse_args()

    groups = []
    if args.parm_dir:
        groups += load_dir(Path(args.parm_dir),   "PARM")
    if args.genarm_dir:
        groups += load_dir(Path(args.genarm_dir), "GenARM")
    print(f"Loaded {len(groups)} (method, α) configs")

    # Index by uid → list of candidate records.
    by_uid = {}
    for method, ah, as_, recs in groups:
        for r in recs:
            uid = r["uid"]
            by_uid.setdefault(uid, []).append({
                "uid": uid,
                "prompt": r["prompt"],
                "response": r["response"],
                "method": method,
                "alpha_help": ah,
                "alpha_harm": as_,
                "q_help": float(r[HELP_KEY]),
                "q_harm": float(r[HARM_KEY]),
                "log_prob": 0.0,  # uniform — candidates come from different policies
            })

    # Assign sequential candidate_ids per uid (order: PARM α=0.0..1.0, GenARM α=0.0..1.0).
    all_records = []
    n_cands = None
    for uid, cands in by_uid.items():
        cands.sort(key=lambda c: (c["method"], c["alpha_help"]))
        for i, c in enumerate(cands):
            c["candidate_id"] = i
            all_records.append(c)
        if n_cands is None:
            n_cands = len(cands)
        elif len(cands) != n_cands:
            print(f"WARN: uid {uid} has {len(cands)} cands, expected {n_cands}")

    print(f"Built pool: {len(by_uid)} prompts × {n_cands} candidates = {len(all_records)} records")
    json.dump(all_records, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
