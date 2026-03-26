"""
Step 1 of Common-Agency EPEC method:
    For each test prompt, sample N candidate responses from the base LLM
    and collect log π_base(a_n) for each candidate.

Dataset : PKU-SafeRLHF-10K  (test split)
Base LLM: PKU-Alignment/alpaca-7b-reproduced  (Alpaca-7B)

Prompt template and test data path follow PARM's generate_outputs.py exactly.
Output: candidates/<uid>.json  per prompt, each containing a list of dicts:
    {
        "uid":          str,
        "prompt":       str,
        "candidate_id": int,          # 0 … N-1
        "response":     str,
        "log_prob":     float,        # sum of per-token log π_base
    }
A single consolidated file candidates/all_candidates.json is also written.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------- #
# Prompt template — identical to generate_outputs.py / relabel.py in PARM
# --------------------------------------------------------------------------- #
PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'  # no trailing space
PROMPT_TEMPLATE: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Sample N candidates from base LLM and collect log-probs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--model_base_name_or_path',
        default='PKU-Alignment/alpaca-7b-reproduced',
        type=str,
        help='HuggingFace model name or local path of the base LLM.',
    )
    parser.add_argument(
        '--datasets',
        default='../data/test_prompt_only.json',
        type=str,
        help='Path to the test prompt file (same format as PARM).',
    )
    parser.add_argument(
        '--output_dir',
        default='./candidates',
        type=str,
        help='Directory where candidate files will be saved.',
    )
    parser.add_argument(
        '--num_candidates',
        default=20,
        type=int,
        help='Number of candidate responses to sample per prompt (N).',
    )
    parser.add_argument(
        '--max_new_tokens',
        default=1024,
        type=int,
        help='Maximum number of new tokens to generate.',
    )
    parser.add_argument(
        '--temperature',
        default=1.0,
        type=float,
        help='Sampling temperature for the base LLM.',
    )
    parser.add_argument(
        '--top_p',
        default=1.0,
        type=float,
    )
    parser.add_argument(
        '--top_k',
        default=0,
        type=int,
    )
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip prompts whose output file already exists.',
    )
    parser.add_argument(
        '--num_prompts',
        default=None,
        type=int,
        help='If set, only process the first N prompts (for testing).',
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Log-probability computation
# --------------------------------------------------------------------------- #

def compute_sequence_logprob(
    scores: tuple,               # tuple of (vocab_size,) raw logit tensors, one per new token
    generated_token_ids: torch.Tensor,  # (new_len,) ids of the generated tokens
) -> float:
    """
    Return sum_t log π_base(a_t | context) over the newly generated tokens.

    Follows the same logic as PromptedLLM.run() in PARM's runnable_operators.py:
        logprobs = torch.log_softmax(model_output.logits, dim=-1)
    We apply log_softmax to each step's raw logits (scores), then gather
    the log-prob of the token that was actually sampled.
    """
    log_prob = 0.0
    for token_id, step_logits in zip(generated_token_ids, scores):
        # step_logits: (1, vocab_size) — raw logits at this generation step
        step_logprobs = torch.log_softmax(step_logits, dim=-1)  # same op as PARM
        log_prob += step_logprobs[0, token_id].item()
    return log_prob


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_arguments()
    torch.manual_seed(args.seed)

    # Build an informative run tag:  <dataset>_<model>_N<num_candidates>
    # e.g. "PKU-SafeRLHF_alpaca-7b-reproduced_N20"
    dataset_tag = Path(args.datasets).stem            # e.g. "test_prompt_only"
    model_tag   = Path(args.model_base_name_or_path).name  # e.g. "alpaca-7b-reproduced"
    run_tag     = f"{dataset_tag}_{model_tag}_N{args.num_candidates}"

    run_dir = Path(args.output_dir) / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'\nOutput directory: {run_dir}\n')

    # ------------------------------------------------------------------ #
    # Load base model — follow PARM's dtype / loading convention
    # ------------------------------------------------------------------ #
    print(f'\nLoading base model: {args.model_base_name_or_path}\n')
    tokenizer = AutoTokenizer.from_pretrained(args.model_base_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_base_name_or_path,
        torch_dtype=torch.float16,
        device_map='auto',
        low_cpu_mem_usage=True,
    )
    model.eval()

    # ------------------------------------------------------------------ #
    # Load test prompts (same format as PARM: list of {uid, prompt})
    # ------------------------------------------------------------------ #
    with open(args.datasets, 'r') as f:
        data_evaluation = json.load(f)
    print(f'Loaded {len(data_evaluation)} test prompts from {args.datasets}')

    if args.num_prompts is not None:
        data_evaluation = data_evaluation[:args.num_prompts]
        print(f'Limiting to first {args.num_prompts} prompts.')

    # ------------------------------------------------------------------ #
    # Generation loop
    # ------------------------------------------------------------------ #
    all_candidates = []

    for item in tqdm(data_evaluation):
        uid = item['uid']
        prompt_text = item['prompt']

        # Per-uid output path — allows resuming
        uid_path = run_dir / f'{uid}.json'
        if args.resume and uid_path.exists():
            with open(uid_path, 'r') as f:
                all_candidates.extend(json.load(f))
            continue

        # Apply the same Alpaca prompt template used throughout PARM
        prompt_formatted = PROMPT_TEMPLATE.format(input=prompt_text)
        input_ids = tokenizer(
            prompt_formatted, return_tensors='pt', truncation=True, max_length=512
        ).input_ids.to(model.device)

        uid_candidates = []

        with torch.no_grad():
            for cand_id in range(args.num_candidates):
                # Sample one response
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,  # HF treats top_k=0 as disabled (no filtering)
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

                # Decode only the newly generated tokens (strip the prompt)
                generated_token_ids = outputs.sequences[0, input_ids.shape[1]:]

                # Sequence log-prob: same log_softmax(logits) logic as PARM's
                # PromptedLLM.run() in runnable_operators.py line 516
                log_prob = compute_sequence_logprob(outputs.scores, generated_token_ids)
                response = tokenizer.decode(
                    generated_token_ids, skip_special_tokens=True
                ).strip()

                uid_candidates.append({
                    'uid': uid,
                    'prompt': prompt_text,
                    'candidate_id': cand_id,
                    'response': response,
                    'log_prob': log_prob,
                })

        # Save per-uid file
        with open(uid_path, 'w', encoding='utf-8') as f:
            json.dump(uid_candidates, f, ensure_ascii=False, indent=2)

        all_candidates.extend(uid_candidates)

    # ------------------------------------------------------------------ #
    # Save consolidated file — name carries the same run_tag
    # ------------------------------------------------------------------ #
    consolidated_path = run_dir / f'all_candidates_{run_tag}.json'
    with open(consolidated_path, 'w', encoding='utf-8') as f:
        json.dump(all_candidates, f, ensure_ascii=False, indent=2)

    print(
        f'\nDone. {len(all_candidates)} total candidates saved to {run_dir}/'
        f'\nConsolidated file: {consolidated_path}'
    )


if __name__ == '__main__':
    main()
