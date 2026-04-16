"""
Step 1 of EPEC for the HH-RLHF benchmark:
    For each test prompt, sample N candidate responses from the SFT base LLM
    and collect log pi_base(a_n) for each candidate.

Dataset : Dahoas/full-hh-rlhf  (test split, prepared by code/data/prepare_hh_rlhf_prompts.py)
Base LLM: argsearch/llama-7b-sft-float32  (SFT-on-HH-RLHF, public, used by GenARM/ARGS)

The Dahoas prompt field already contains the multi-turn dialogue formatted as
    "\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant:"
which is exactly what the SFT base model was trained on, so NO chat template
or wrapping is applied.  The prompt is passed through verbatim.

Output: candidates/<run_tag>/<uid>.json  per prompt  +  one consolidated
        all_candidates_<run_tag>.json file containing the same records.
        Each record:
            {
                "uid":          str,
                "prompt":       str,
                "candidate_id": int,    # 0 .. N-1
                "response":     str,
                "log_prob":     float,  # sum of per-token log pi_base
            }
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample N candidates from the HH-RLHF SFT base LLM and collect log-probs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_base_name_or_path",
        default="argsearch/llama-7b-sft-float32",
        type=str,
        help="HuggingFace model name or local path of the SFT base LLM.",
    )
    parser.add_argument(
        "--datasets",
        default="../data/hh_rlhf_test_prompts.json",
        type=str,
        help="Path to the HH-RLHF test prompt JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        default="./candidates",
        type=str,
    )
    parser.add_argument(
        "--num_candidates",
        default=20,
        type=int,
        help="Number of candidate responses to sample per prompt (N).",
    )
    parser.add_argument(
        "--num_prompts",
        default=1000,
        type=int,
        help="Number of prompts to actually generate for (matches PARM's 1k test split).",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=128,
        type=int,
        help="Maximum number of new tokens to generate per response.",
    )
    parser.add_argument(
        "--max_prompt_length",
        default=2048,
        type=int,
        help="Filter out prompts whose tokenized length exceeds this.",
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--top_p",
        default=1.0,
        type=float,
    )
    parser.add_argument(
        "--top_k",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip prompts whose per-uid output file already exists.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Sequence log-probability computation (identical to generate_candidates.py)
# --------------------------------------------------------------------------- #

def compute_sequence_logprob(scores: tuple, generated_token_ids: torch.Tensor) -> float:
    """
    Sum_t log pi_base(a_t | context) over the newly generated tokens.
    Mirrors PARM's PromptedLLM.run() in language-model-arithmetic/runnable_operators.py.
    """
    log_prob = 0.0
    for token_id, step_logits in zip(generated_token_ids, scores):
        step_logprobs = torch.log_softmax(step_logits, dim=-1)
        log_prob += step_logprobs[0, token_id].item()
    return log_prob


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_arguments()
    torch.manual_seed(args.seed)

    dataset_tag = Path(args.datasets).stem            # e.g. "hh_rlhf_test_prompts"
    model_tag   = Path(args.model_base_name_or_path).name  # e.g. "llama-7b-sft-float32"
    run_tag     = f"{dataset_tag}_{model_tag}_N{args.num_candidates}"

    run_dir = Path(args.output_dir) / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {run_dir}\n")

    # ------------------------------------------------------------------ #
    # Load base SFT model
    # ------------------------------------------------------------------ #
    print(f"Loading base model: {args.model_base_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_base_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_base_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    # ------------------------------------------------------------------ #
    # Load + filter prompts
    # ------------------------------------------------------------------ #
    with open(args.datasets, "r") as f:
        all_prompts = json.load(f)
    print(f"Loaded {len(all_prompts)} prompts from {args.datasets}")

    # Filter out prompts that exceed max_prompt_length when tokenized
    kept = []
    dropped = 0
    for item in all_prompts:
        n_tok = len(tokenizer(item["prompt"], add_special_tokens=True)["input_ids"])
        if n_tok <= args.max_prompt_length:
            kept.append(item)
        else:
            dropped += 1
    print(f"Filtered prompts > {args.max_prompt_length} tokens: kept {len(kept)}, dropped {dropped}")

    # Take the first num_prompts after filtering
    data_evaluation = kept[: args.num_prompts]
    print(f"Will generate for {len(data_evaluation)} prompts × {args.num_candidates} candidates "
          f"= {len(data_evaluation) * args.num_candidates} generations\n")

    # ------------------------------------------------------------------ #
    # Generation loop
    # ------------------------------------------------------------------ #
    all_candidates = []

    for item in tqdm(data_evaluation):
        uid = item["uid"]
        prompt_text = item["prompt"]

        uid_path = run_dir / f"{uid}.json"
        if args.resume and uid_path.exists():
            with open(uid_path, "r") as f:
                all_candidates.extend(json.load(f))
            continue

        # Pass-through: the SFT base model already understands this format
        input_ids = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=args.max_prompt_length
        ).input_ids.to(model.device)

        uid_candidates = []

        with torch.no_grad():
            for cand_id in range(args.num_candidates):
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

                generated_token_ids = outputs.sequences[0, input_ids.shape[1]:]
                log_prob = compute_sequence_logprob(outputs.scores, generated_token_ids)
                response = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

                uid_candidates.append({
                    "uid": uid,
                    "prompt": prompt_text,
                    "candidate_id": cand_id,
                    "response": response,
                    "log_prob": log_prob,
                })

        with open(uid_path, "w", encoding="utf-8") as f:
            json.dump(uid_candidates, f, ensure_ascii=False, indent=2)

        all_candidates.extend(uid_candidates)

    # ------------------------------------------------------------------ #
    # Consolidated file
    # ------------------------------------------------------------------ #
    consolidated_path = run_dir / f"all_candidates_{run_tag}.json"
    with open(consolidated_path, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, ensure_ascii=False, indent=2)

    print(
        f"\nDone. {len(all_candidates)} total candidates saved to {run_dir}/"
        f"\nConsolidated file: {consolidated_path}"
    )


if __name__ == "__main__":
    main()
