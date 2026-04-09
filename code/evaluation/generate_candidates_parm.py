"""
Generate N candidate responses per prompt from the PARM-blended policy
(M_base + alpha_help * M_help + alpha_harm * M_harm) and record log pi_PARM
of each sampled response, so EPEC can be applied on top of PARM.

For each prompt:
  - Sample N candidates via ModelArithmetic.generate_text(temperature=1.0)
  - Re-score each candidate by running model.forward(prompt+response) and
    summing log_softmax over the response tokens.

Output: candidates_parm/PARM_<ah>help_<as>harm_N<N>/<uid>.json   (per prompt)
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from generate_outputs import get_model_arithmetic, PROMPT_INPUT_ALPACA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_base_name_or_path", default="PKU-Alignment/alpaca-7b-reproduced")
    p.add_argument("--model_parm_both_name_or_path", required=True)
    p.add_argument("--alpha_helpfulness", type=float, required=True)
    p.add_argument("--alpha_harmlessness", type=float, required=True)
    p.add_argument("--datasets", default="../data/test_prompt_only.json")
    p.add_argument("--output_dir", default="./candidates_parm")
    p.add_argument("--cache_dir", default="./cache_parm")
    p.add_argument("--num_candidates", type=int, default=5)
    p.add_argument("--num_prompts", type=int, default=0, help="0 = all")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--normalize_logit", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    tag = f"PARM_{args.alpha_helpfulness}help_{args.alpha_harmlessness}harm_N{args.num_candidates}"
    run_dir = Path(args.output_dir) / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {run_dir}\n")

    # Patch adapter config with this preference vector (same as generate_outputs.py)
    cache_path = Path(args.cache_dir) / tag
    cache_path.mkdir(parents=True, exist_ok=True)
    with open(f"{args.model_parm_both_name_or_path}/adapter_config.json") as f:
        cfg = json.load(f)
    cfg["pref_vec_init"] = [args.alpha_harmlessness, args.alpha_helpfulness]
    with open(cache_path / "adapter_config.json", "w") as f:
        json.dump(cfg, f, indent=4)
    shutil.copyfile(f"{args.model_parm_both_name_or_path}/adapter_model.safetensors",
                    cache_path / "adapter_model.safetensors")

    model, tokenizer, temperature_arith = get_model_arithmetic(
        model_pth_base=args.model_base_name_or_path,
        model_pth_reward_both=str(cache_path),
        args=args,
    )
    model.eval()
    if args.normalize_logit:
        temperature_arith = 1.0
    # Note: temperature_arith from get_model_arithmetic is unused here;
    # sampling temperature is passed directly to generate_text via args.temperature.

    with open(args.datasets) as f:
        prompts = json.load(f)
    if args.num_prompts > 0:
        prompts = prompts[: args.num_prompts]
    print(f"Will generate for {len(prompts)} prompts × {args.num_candidates} candidates")

    for item in tqdm(prompts):
        uid = item["uid"]
        prompt_text = item["prompt"]
        out_path = run_dir / f"{uid}.json"
        if args.resume and out_path.exists():
            continue

        # Sample N candidates and capture per-token blended log pi_PARM in one pass.
        responses, per_token_lps = model.generate_text(
            prompt_text,
            max_new_tokens=args.max_new_tokens,
            batch_size=None,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_return_sequences=args.num_candidates,
            do_speculation=False,
            return_token_logprobs=True,
        )
        responses = [r.removesuffix(tokenizer.eos_token) for r in responses]

        records = []
        for cand_id, (response, lps) in enumerate(zip(responses, per_token_lps)):
            lp = float(sum(lps))
            records.append({
                "uid": uid,
                "prompt": prompt_text,
                "candidate_id": cand_id,
                "response": response,
                "log_prob": lp,
            })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved per-uid files to {run_dir}")


if __name__ == "__main__":
    main()
