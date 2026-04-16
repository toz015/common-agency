"""
Prepare HH-RLHF test prompts for the PARM/EPEC HH-RLHF evaluation.

Source : Dahoas/full-hh-rlhf  (test split, ~12k samples)
         Each row already has a clean `prompt` field that contains the
         multi-turn dialogue ending in "\n\nAssistant: " — exactly the
         format the argsearch SFT base model expects.

Output : code/data/hh_rlhf_test_prompts.json
         List of {uid, prompt} dicts — same schema as test_prompt_only.json,
         so generate_candidates_hh.py can load it the same way.

The full 12k test set is saved (not subsampled) so it can be reused later
for additional experiments. The candidate-generation script chooses how
many to actually run via --num_prompts.
"""

import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    out_path = Path(__file__).parent / "hh_rlhf_test_prompts.json"

    print("Loading Dahoas/full-hh-rlhf (test split)...")
    ds = load_dataset("Dahoas/full-hh-rlhf", split="test")
    print(f"Loaded {len(ds)} rows")

    records = [
        {"uid": f"hh_eval{i}", "prompt": row["prompt"]}
        for i, row in enumerate(ds)
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(records)} prompts to {out_path}")
    print("\nFirst prompt preview:")
    print(records[0]["prompt"][:300] + ("..." if len(records[0]["prompt"]) > 300 else ""))


if __name__ == "__main__":
    main()
