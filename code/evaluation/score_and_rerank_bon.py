'''
Score GenARM-BoN candidates and rerank to pick the best per prompt.

Input:
    <run_dir>/candidates.json  (from generate_outputs_genarm_bon.py)

Output:
    <run_dir>/reward_result_all.json      -- all N * M candidates with q_help, q_harm
    <run_dir>/reward_result.json          -- the M reranked winners (mirrors paper's schema)
    <run_dir>/mean_result.json            -- {"help": <mean>, "harm": <mean>} of the winners

Reranking rule:
    winner = argmax over the N candidates of
        s = alpha_help * q_help - alpha_harm * q_harm
    (q_harm is beaver-cost, so subtracting rewards lower cost.)

Reward models and template match compute_reward.py / score_candidates.py exactly.
'''
import sys, types, importlib.abc, importlib.machinery


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

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _DeepSpeedStubFinder())

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from safe_rlhf.models import AutoModelForScore

REWARD_TEMPLATE = 'BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:{response}'
MODEL_PATH_HELPFUL = 'PKU-Alignment/beaver-7b-v1.0-reward'
MODEL_PATH_HARMLESS = 'PKU-Alignment/beaver-7b-v1.0-cost'


def parse_arguments():
    p = argparse.ArgumentParser(
        description='Score BoN candidates with Beaver reward models and rerank.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--run_dir', required=True, type=str,
                   help='Directory containing candidates.json (from generate_outputs_genarm_bon.py).')
    return p.parse_args()


def main():
    args = parse_arguments()
    run_dir = Path(args.run_dir)
    cand_path = run_dir / 'candidates.json'
    assert cand_path.exists(), f'candidates.json not found in {run_dir}'

    with open(cand_path, 'r') as f:
        candidates = json.load(f)
    print(f'Loaded {len(candidates)} candidates from {cand_path}')

    # Sanity: all candidates should share the same (alpha_help, alpha_harm).
    alphas = {(c['alpha_help'], c['alpha_harm']) for c in candidates}
    assert len(alphas) == 1, f'candidates.json mixes multiple alphas: {alphas}'
    alpha_help, alpha_harm = next(iter(alphas))
    print(f'alpha = ({alpha_help}, {alpha_harm})')

    print(f'\nLoading helpfulness model: {MODEL_PATH_HELPFUL}')
    model_helpful = AutoModelForScore.from_pretrained(
        MODEL_PATH_HELPFUL, torch_dtype=torch.bfloat16, device_map='auto'
    )
    print(f'Loading harmlessness model: {MODEL_PATH_HARMLESS}')
    model_harmless = AutoModelForScore.from_pretrained(
        MODEL_PATH_HARMLESS, torch_dtype=torch.bfloat16, device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_HELPFUL)
    model_helpful.eval()
    model_harmless.eval()

    scored = []
    with torch.no_grad():
        for c in tqdm(candidates, desc='scoring'):
            text = REWARD_TEMPLATE.format(input=c['prompt'], response=c['response'])
            input_ids = tokenizer(text, return_tensors='pt').to(model_helpful.device)
            q_help = model_helpful(**input_ids)['end_scores'][0][0].item()
            q_harm = model_harmless(**input_ids)['end_scores'][0][0].item()

            entry = dict(c)
            entry['q_help'] = q_help
            entry['q_harm'] = q_harm
            entry['rerank_score'] = alpha_help * q_help - alpha_harm * q_harm
            scored.append(entry)

    # Write full per-candidate table (all N * M rows) for auditing.
    with open(run_dir / 'reward_result_all.json', 'w') as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)

    # Group by uid and pick the argmax rerank_score.
    by_uid = defaultdict(list)
    for e in scored:
        by_uid[e['uid']].append(e)

    winners = []
    for uid, group in by_uid.items():
        # Sort stable by candidate_id to make ties deterministic.
        group.sort(key=lambda e: e['candidate_id'])
        best = max(group, key=lambda e: e['rerank_score'])
        # Emit exactly the schema of compute_reward.py's reward_result.json so
        # downstream HV/MIP scripts can read the file unchanged.
        winners.append({
            'uid': uid,
            'prompt': best['prompt'],
            'response': best['response'],
            'model': f'GenARM_BoN_N{len(group)}_{alpha_help}help_{alpha_harm}harm',
            'winning_candidate_id': best['candidate_id'],
            'help_score (high better)': best['q_help'],
            'harm_score (low better)': best['q_harm'],
        })

    # Sort winners by uid ordinal so subsetting/plotting is stable.
    winners.sort(key=lambda w: int(w['uid'].replace('eval', '')) if w['uid'].startswith('eval') else w['uid'])

    with open(run_dir / 'reward_result.json', 'w') as f:
        json.dump(winners, f, ensure_ascii=False, indent=2)

    mean_help = sum(w['help_score (high better)'] for w in winners) / len(winners)
    mean_harm = sum(w['harm_score (low better)'] for w in winners) / len(winners)
    with open(run_dir / 'mean_result.json', 'w') as f:
        json.dump({'help': mean_help, 'harm': mean_harm}, f, indent=2)

    print(f'\nDone. {len(winners)} winners chosen out of {len(scored)} candidates.')
    print(f'Mean help: {mean_help:+.4f}   Mean harm (lower=safer): {mean_harm:+.4f}')
    print(f'Winners written to: {run_dir / "reward_result.json"}')
    print(f'Full audit file:    {run_dir / "reward_result_all.json"}')


if __name__ == '__main__':
    main()
