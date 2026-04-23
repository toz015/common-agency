"""
Dump every drift-contaminated response from the n=100 t=512 run verbatim,
grouped by config. No stats, no interpretation — just the raw samples.
Writes to phase2_w2s/drift_samples.txt.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "phase2_results_n100" / "results_phase2_n100_t512"
OUT = Path(__file__).resolve().parent / "drift_samples.txt"

CONFIGS = [
    ("parm",   "PARM_0.2help_0.8harm"),
    ("parm",   "PARM_0.4help_0.6harm"),
    ("parm",   "PARM_0.8help_0.2harm"),
    ("genarm", "GenARM_0.2help_0.8harm"),
    ("genarm", "GenARM_0.4help_0.6harm"),
    ("genarm", "GenARM_0.8help_0.2harm"),
]
PATTERNS = re.compile(
    r"###\s*(Instruction|Response|Human|Assistant|Input|Examples?)"
    r"|\bUSER:\s|\bASSISTANT:\s|\bHuman:\s"
    r"|\n\s*Q:\s|\n\s*A:\s",
    re.IGNORECASE,
)

lines = []
for sub, name in CONFIGS:
    p = ROOT / sub / name / "reward_result.json"
    if not p.exists():
        continue
    data = json.load(open(p))
    hits = [rec for rec in data if PATTERNS.search(rec.get("response", ""))]
    if not hits:
        continue
    lines.append("\n" + "#" * 78)
    lines.append(f"# {name}  —  {len(hits)}/{len(data)} drifted")
    lines.append("#" * 78 + "\n")
    for rec in hits:
        h = rec.get("help_score (high better)", float("nan"))
        c = rec.get("harm_score (low better)", float("nan"))
        lines.append(f"---- uid={rec['uid']}  help={h:+.3f}  harm={c:+.3f}  len={len(rec['response'])} ----")
        lines.append(f"PROMPT:\n{rec['prompt']}\n")
        lines.append(f"RESPONSE:\n{rec['response']}\n")
OUT.write_text("\n".join(lines))
print(f"wrote {OUT}  ({len(lines)} lines)")
