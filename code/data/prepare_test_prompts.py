"""
Download PKU-SafeRLHF-10K from HuggingFace and extract test-split prompts.

Produces test_prompt_only.json in the same format as relabel.py:
    [{"uid": "eval0", "prompt": "..."}, ...]

No reward model loading required — prompts only.
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', default='PKU-Alignment/PKU-SafeRLHF-10K')
    parser.add_argument('--output', default='./test_prompt_only.json')
    parser.add_argument('--test_size', default=1500, type=int,
                        help='Number of prompts to keep as test split.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Shuffle seed (use same seed as relabel.py run to match splits).')
    return parser.parse_args()


def main():
    args = parse_args()

    print(f'Loading {args.dataset} ...')
    dataset = load_dataset(args.dataset, split='train', num_proc=2)
    print(f'Loaded {len(dataset)} examples.')

    prompts = [{'prompt': d['prompt']} for d in dataset]

    # Mirror relabel.py: shuffle then take the last test_size as test split
    random.seed(args.seed)
    random.shuffle(prompts)
    test_prompts = prompts[-args.test_size:]

    output = [
        {'uid': f'eval{i}', 'prompt': p['prompt']}
        for i, p in enumerate(test_prompts)
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f'Saved {len(output)} prompts to {out_path}')


if __name__ == '__main__':
    main()
