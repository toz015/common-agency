"""
Step 2 of Common-Agency EPEC method:
    Score each candidate response with both Beaver reward models.
    Produces q^1_n (helpfulness) and q^2_n (harmlessness) for every candidate.

Reward models and scoring logic follow PARM's compute_reward.py exactly:
    - model:    PKU-Alignment/beaver-7b-v1.0-reward  (helpfulness, higher is better)
    - model:    PKU-Alignment/beaver-7b-v1.0-cost    (harmlessness, lower cost = safer)
    - template: 'BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:{response}'
    - score:    model(**input_ids)['end_scores'][0][0].item()

Input:  all_candidates_<run_tag>.json  produced by generate_candidates.py (Step 1)
Output: scored_candidates_<run_tag>.json  — same entries with two new fields added:
            q_help  (float)  helpfulness score  — higher is better
            q_harm  (float)  harmlessness cost  — lower is safer (matches PARM convention)
"""

import sys, types, importlib.util, importlib.abc, importlib.machinery

class _DeepSpeedStubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'deepspeed' or fullname.startswith('deepspeed.'):
            return importlib.machinery.ModuleSpec(fullname, _DeepSpeedStubLoader())
        return None

class _DeepSpeedStubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []
        mod.__package__ = spec.name
        return mod
    def exec_module(self, module): pass

sys.meta_path.insert(0, _DeepSpeedStubFinder())

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from safe_rlhf.models import AutoModelForScore

# Identical to compute_reward.py and relabel.py in PARM
REWARD_TEMPLATE = 'BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:{response}'

MODEL_PATH_HELPFUL  = 'PKU-Alignment/beaver-7b-v1.0-reward'
MODEL_PATH_HARMLESS = 'PKU-Alignment/beaver-7b-v1.0-cost'


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Score candidate responses with Beaver helpfulness and harmlessness models.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--candidates_file',
        required=True,
        type=str,
        help='Path to all_candidates_<run_tag>.json produced by generate_candidates.py.',
    )
    parser.add_argument(
        '--output_dir',
        default='./scored',
        type=str,
        help='Directory where the scored output file will be saved.',
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Output filename mirrors the input filename with "scored_" prefix
    candidates_path = Path(args.candidates_file)
    run_tag = candidates_path.stem.replace('all_candidates_', '')  # e.g. test_prompt_only_alpaca-7b-reproduced_N20
    output_path = Path(args.output_dir) / f'scored_candidates_{run_tag}.json'
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'\nScored output will be saved to: {output_path}\n')

    # ------------------------------------------------------------------ #
    # Load reward models — dtype and loading follow compute_reward.py
    # ------------------------------------------------------------------ #
    print(f'Loading helpfulness model: {MODEL_PATH_HELPFUL}')
    model_helpful = AutoModelForScore.from_pretrained(
        MODEL_PATH_HELPFUL, torch_dtype=torch.bfloat16, device_map='auto'
    )
    print(f'Loading harmlessness model: {MODEL_PATH_HARMLESS}')
    model_harmless = AutoModelForScore.from_pretrained(
        MODEL_PATH_HARMLESS, torch_dtype=torch.bfloat16, device_map='auto'
    )
    # Use the helpful model's tokenizer — same choice as compute_reward.py line 25
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_HELPFUL)

    model_helpful.eval()
    model_harmless.eval()

    # ------------------------------------------------------------------ #
    # Load candidates
    # ------------------------------------------------------------------ #
    with open(candidates_path, 'r') as f:
        candidates = json.load(f)
    print(f'Loaded {len(candidates)} candidates from {candidates_path}')

    # ------------------------------------------------------------------ #
    # Score each candidate
    # Same logic as compute_reward.py lines 39-52
    # ------------------------------------------------------------------ #
    scored = []
    with torch.no_grad():
        for d in tqdm(candidates):
            text = REWARD_TEMPLATE.format(input=d['prompt'], response=d['response'])
            input_ids = tokenizer(text, return_tensors='pt').to(model_helpful.device)

            q_help = model_helpful(**input_ids)['end_scores'][0][0].item()
            q_harm = model_harmless(**input_ids)['end_scores'][0][0].item()

            entry = dict(d)   # preserve all fields from Step 1 (uid, prompt, candidate_id, response, log_prob)
            entry['q_help'] = q_help   # helpfulness  — higher is better
            entry['q_harm'] = q_harm   # harmlessness cost — lower is safer
            scored.append(entry)

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)

    total = len(scored)
    mean_help = sum(e['q_help'] for e in scored) / total
    mean_harm = sum(e['q_harm'] for e in scored) / total
    print(
        f'\nDone. {total} candidates scored.'
        f'\nMean q_help: {mean_help:.4f}  |  Mean q_harm: {mean_harm:.4f}'
        f'\nSaved to: {output_path}'
    )


if __name__ == '__main__':
    main()
