"""
Surface drift-contaminated responses from the n=100 t=512 4-bit run so we can
eyeball the format-leak pattern. Prints each hit with:
  - method, alpha_help, uid
  - help/harm scores
  - char offset where the first ### marker appears
  - response body with the drift marker visually flagged

Run: python phase2_w2s/show_drift_examples.py [--limit N] [--method PARM|GenARM]
"""
import argparse
import json
import re
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "phase2_results_n100" / "results_phase2_n100_t512"
DRIFT_RE = re.compile(r"###\s*(Instruction|Response|Human|Assistant|Input)", re.IGNORECASE)

CONFIGS = [
    ("parm",   "PARM",   "PARM_0.2help_0.8harm", 0.2, 0.8),
    ("parm",   "PARM",   "PARM_0.4help_0.6harm", 0.4, 0.6),
    ("parm",   "PARM",   "PARM_0.8help_0.2harm", 0.8, 0.2),
    ("genarm", "GenARM", "GenARM_0.2help_0.8harm", 0.2, 0.8),
    ("genarm", "GenARM", "GenARM_0.4help_0.6harm", 0.4, 0.6),
    ("genarm", "GenARM", "GenARM_0.8help_0.2harm", 0.8, 0.2),
]


def find_drift(text: str):
    """Return list of (start_idx, matched_str) for every drift marker."""
    return [(m.start(), m.group(0)) for m in DRIFT_RE.finditer(text or "")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    ap.add_argument("--limit", type=int, default=20, help="Max examples to print across all configs.")
    ap.add_argument("--method", type=str, default=None, choices=[None, "PARM", "GenARM"])
    ap.add_argument("--per_config", type=int, default=4, help="Max examples per config.")
    args = ap.parse_args()

    shown = 0
    for sub, method, name, ah, ahm in CONFIGS:
        if args.method and method != args.method:
            continue
        p = args.root / sub / name / "reward_result.json"
        if not p.exists():
            continue
        data = json.load(open(p))
        hits = []
        for rec in data:
            drifts = find_drift(rec.get("response", ""))
            if drifts:
                hits.append((rec, drifts))
        if not hits:
            continue
        print(f"\n{'=' * 70}")
        print(f"[{method}] αh={ah} αc={ahm}  —  {len(hits)}/{len(data)} responses drifted")
        print(f"{'=' * 70}")
        for rec, drifts in hits[: args.per_config]:
            if shown >= args.limit:
                print(f"\n...hit --limit={args.limit}, stopping.")
                return
            shown += 1
            first_idx, first_tok = drifts[0]
            help_s = rec.get("help_score (high better)", float("nan"))
            harm_s = rec.get("harm_score (low better)", float("nan"))
            resp = rec["response"]
            # Insert a visible marker at each drift location.
            chunks = []
            last = 0
            for idx, tok in drifts:
                chunks.append(resp[last:idx])
                chunks.append(f"\n----[DRIFT @ char {idx} — '{tok}']----\n")
                last = idx
            chunks.append(resp[last:])
            annotated = "".join(chunks)

            print(f"\n--- uid={rec['uid']}  help={help_s:+.3f}  harm={harm_s:+.3f}  first_drift_at={first_idx} ---")
            print(f"[PROMPT] {rec.get('prompt', '(missing)')[:200]}")
            print(f"[RESPONSE | len={len(resp)} chars]")
            print(annotated)
    print(f"\n\nShown {shown} examples total.")


if __name__ == "__main__":
    main()
