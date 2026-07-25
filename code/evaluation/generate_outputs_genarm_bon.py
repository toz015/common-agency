'''
Rebuttal ablation for Reviewer Uw2G Q6:
    "Could one simply sample several candidates from GenARM and rerank them
     using the same reward models?"

For each preference vector (alpha_help, alpha_harm), sample N candidates from
GenARM(alpha) with temperature=1.0 on the first M test prompts. Candidates
are later scored with the two Beaver reward models and reranked by
    s = alpha_help * q_help - alpha_harm * q_harm
in score_and_rerank_bon.py.

Reuses the same ModelArithmetic formula as generate_outputs_genarm.py:
    M = M_base + alpha_help * M_arm_help + alpha_harm * M_arm_harm
with temperature=1.0 (i.e. samples from softmax of the blended log-probs).

Output layout (per alpha):
    <output_dir>/GenARM_BoN_N{N}_{alpha_h}help_{alpha_s}harm/candidates.json
which contains 1 entry per (prompt, candidate_id) with fields
    uid, prompt, candidate_id, response, alpha_help, alpha_harm.
'''
import argparse
import json
import os
import time
from pathlib import Path
from tqdm import tqdm

from transformers import AutoTokenizer, set_seed
from model_arithmetic import ModelArithmetic, PromptedLLM

PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'
PROMPT_INPUT_ALPACA: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='GenARM Best-of-N sampling (temperature=1.0) for rebuttal ablation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model_base_name_or_path', default='PKU-Alignment/alpaca-7b-reproduced', type=str)
    parser.add_argument('--model_arm_help_path', required=True, type=str)
    parser.add_argument('--model_arm_harm_path', required=True, type=str)
    parser.add_argument('--alpha_helpfulness', type=float, required=True)
    parser.add_argument('--alpha_harmlessness', type=float, required=True)
    parser.add_argument('--num_candidates', type=int, default=5,
                        help='N in Best-of-N (samples per prompt).')
    parser.add_argument('--num_prompts', type=int, default=50,
                        help='Take the first M prompts from the dataset.')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--datasets', type=str,
                        default='../data/PKU-SafeRLHF/test_prompt_only.json')
    parser.add_argument('--output_dir', type=str, default='./results_bon')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true',
                        help='Skip alpha directories that already contain candidates.json.')
    return parser.parse_args()


def get_model_arithmetic(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_base_name_or_path)
    prompt_template = lambda system_prompt, input_string: PROMPT_INPUT_ALPACA.format(input=input_string)

    M_base = PromptedLLM(system_prompt='Not used', prompt_template=prompt_template,
                         model=args.model_base_name_or_path, tokenizer=tokenizer)
    M_help = PromptedLLM(system_prompt='Not used', prompt_template=prompt_template,
                         model=args.model_arm_help_path, tokenizer=tokenizer)
    M_harm = PromptedLLM(system_prompt='Not used', prompt_template=prompt_template,
                         model=args.model_arm_harm_path, tokenizer=tokenizer)

    formula = M_base + args.alpha_helpfulness * M_help + args.alpha_harmlessness * M_harm
    model = ModelArithmetic(formula, max_length=args.max_length)
    return model, tokenizer


if __name__ == '__main__':
    args = parse_arguments()
    set_seed(args.seed)

    run_name = f'GenARM_BoN_N{args.num_candidates}_{args.alpha_helpfulness}help_{args.alpha_harmlessness}harm'
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / 'candidates.json'
    print(f'\nRun: {run_name}\nCandidates will be saved to: {out_path}\n')

    if args.resume and out_path.exists():
        print(f'--resume set and {out_path} exists. Skipping.')
        raise SystemExit(0)

    with open(args.datasets, 'r') as f:
        data = json.load(f)
    data = data[:args.num_prompts]
    print(f'Sampling {args.num_candidates} candidates per prompt on {len(data)} prompts.')

    model, tokenizer = get_model_arithmetic(args)
    model.eval()

    # Sampling: temperature=1.0 gives samples from softmax of the blended log-probs
    # (same formula GenARM uses at decoding time, but sampled instead of argmax).
    generate = lambda prompt: model.generate_text(
        prompt,
        max_new_tokens=args.max_new_tokens,
        batch_size=None,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        num_return_sequences=args.num_candidates,
        do_speculation=False,
    )

    all_candidates = []
    t0 = time.time()
    for i in tqdm(range(len(data))):
        uid = data[i]['uid']
        prompt = data[i]['prompt']

        t_start = time.time()
        responses = generate(prompt)  # list of len num_candidates
        elapsed = time.time() - t_start

        assert len(responses) == args.num_candidates, \
            f'Expected {args.num_candidates} samples, got {len(responses)} for uid={uid}'

        for cand_id, resp in enumerate(responses):
            all_candidates.append({
                'uid': uid,
                'prompt': prompt,
                'candidate_id': cand_id,
                'response': resp.removesuffix(tokenizer.eos_token),
                'alpha_help': args.alpha_helpfulness,
                'alpha_harm': args.alpha_harmlessness,
                'elapsed_per_prompt_s': elapsed,   # total for all N samples of this prompt
            })

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_candidates, f, ensure_ascii=False, indent=2)

    total_h = (time.time() - t0) / 3600
    print(f'\nDone. {len(all_candidates)} candidates saved to {out_path}')
    print(f'Total time: {total_h:.2f} hours ({total_h * 60:.1f} min)')
