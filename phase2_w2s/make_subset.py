"""Take the first N prompts from test_prompt_only.json for the Phase-2 sanity check."""
import json
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to full test_prompt_only.json")
    p.add_argument("--output", required=True, help="Path to write the N-prompt subset")
    p.add_argument("--n", type=int, default=50, help="How many prompts to keep (default 50)")
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {args.input}, got {type(data).__name__}")

    subset = data[: args.n]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(subset)} prompts -> {args.output}")


if __name__ == "__main__":
    main()
