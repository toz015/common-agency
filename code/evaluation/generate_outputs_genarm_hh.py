"""
GenARM generation for HH-RLHF (3 objectives: help, harm, humor).

Memory-efficient: loads base LLM + TinyLLaMA with adapter switching
(avoids loading 3 separate TinyLLaMA instances).

Per token: forward base + 3 ARM forwards (switching adapters),
then linear blend: logit_final = log_base + α_help*r_help + α_harm*r_harm + α_humor*r_humor
where r_j = log π_ARM_j - log π_base (implicit reward).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--arm_base", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--help_adapter", default="../training/HH-RLHF/exp_genarm_help")
    p.add_argument("--harm_adapter", default="../training/HH-RLHF/exp_genarm_harm")
    p.add_argument("--humor_adapter", default="../training/HH-RLHF/exp_genarm_humor")
    p.add_argument("--alpha_helpfulness", type=float, required=True)
    p.add_argument("--alpha_harmlessness", type=float, required=True)
    p.add_argument("--alpha_humor", type=float, required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--datasets", default="../data/HH-RLHF/test_prompt_only.json")
    p.add_argument("--output_dir", default="./results_hh")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def load_models(args, device):
    """Load base LLM and TinyLLaMA with 3 adapters."""
    # Base LLM
    base_tok = AutoTokenizer.from_pretrained(args.base)
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map=device
    )
    base_model.eval()

    # ARM model (TinyLLaMA + 3 adapters)
    arm_tok = AutoTokenizer.from_pretrained(args.arm_base)
    if arm_tok.pad_token is None:
        arm_tok.pad_token = arm_tok.eos_token
    arm_model = AutoModelForCausalLM.from_pretrained(
        args.arm_base, torch_dtype=torch.bfloat16, device_map=device
    )
    arm_model = PeftModel.from_pretrained(arm_model, args.help_adapter, adapter_name="help")
    arm_model.load_adapter(args.harm_adapter, adapter_name="harm")
    arm_model.load_adapter(args.humor_adapter, adapter_name="humor")
    arm_model.eval()

    return base_model, base_tok, arm_model, arm_tok


@torch.no_grad()
def generate_genarm(base_model, base_tok, arm_model, arm_tok,
                    prompt_text, alpha_help, alpha_harm, alpha_humor,
                    max_new_tokens=256, device="cuda"):
    """Generate tokens via GenARM linear logit blending."""
    base_ids = base_tok(prompt_text, return_tensors="pt").input_ids.to(device)
    arm_ids = arm_tok(prompt_text, return_tensors="pt").input_ids.to(device)

    base_cur, arm_cur = base_ids, arm_ids
    pkv_base = pkv_arm_base = pkv_help = pkv_harm = pkv_humor = None
    out_ids = []
    eos_id = base_tok.eos_token_id

    for _ in range(max_new_tokens):
        # Base LLM forward
        out = base_model(input_ids=base_cur, past_key_values=pkv_base, use_cache=True)
        log_base = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        pkv_base = out.past_key_values

        # ARM base forward (no adapter)
        with arm_model.disable_adapter():
            out = arm_model(input_ids=arm_cur, past_key_values=pkv_arm_base, use_cache=True)
        log_arm_base = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        pkv_arm_base = out.past_key_values

        # ARM help
        arm_model.set_adapter("help")
        out = arm_model(input_ids=arm_cur, past_key_values=pkv_help, use_cache=True)
        log_help = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        pkv_help = out.past_key_values

        # ARM harm
        arm_model.set_adapter("harm")
        out = arm_model(input_ids=arm_cur, past_key_values=pkv_harm, use_cache=True)
        log_harm = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        pkv_harm = out.past_key_values

        # ARM humor
        arm_model.set_adapter("humor")
        out = arm_model(input_ids=arm_cur, past_key_values=pkv_humor, use_cache=True)
        log_humor = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        pkv_humor = out.past_key_values

        # Implicit rewards (on base LLM vocab)
        vocab_min = min(log_base.shape[-1], log_arm_base.shape[-1])
        r_help = log_help[:vocab_min] - log_arm_base[:vocab_min]
        r_harm = log_harm[:vocab_min] - log_arm_base[:vocab_min]
        r_humor = log_humor[:vocab_min] - log_arm_base[:vocab_min]

        # GenARM linear blend
        blended = log_base[:vocab_min] + alpha_help * r_help + alpha_harm * r_harm + alpha_humor * r_humor
        tok_id = int(torch.argmax(blended))

        if tok_id == eos_id:
            break
        out_ids.append(tok_id)

        base_cur = torch.tensor([[tok_id]], device=device)
        # Map token back to ARM tokenizer
        tok_text = base_tok.decode([tok_id])
        arm_cur = arm_tok(tok_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if arm_cur.numel() == 0:
            arm_cur = torch.tensor([[arm_tok.unk_token_id or 0]], device=device)

    return base_tok.decode(out_ids, skip_special_tokens=True)


def main():
    args = parse_args()
    device = "cuda"

    model_name = f"GenARM_{args.alpha_helpfulness}help_{args.alpha_harmlessness}harm_{args.alpha_humor}humor"
    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generation.json"
    print(f"Saving to {out_path}")

    with open(args.datasets) as f:
        data = json.load(f)
    if args.limit > 0:
        data = data[:args.limit]

    base_model, base_tok, arm_model, arm_tok = load_models(args, device)
    print(f"\nModel: {model_name}, Prompts: {len(data)}")

    results = []
    t0 = time.time()
    for row in tqdm(data):
        start = time.time()
        response = generate_genarm(
            base_model, base_tok, arm_model, arm_tok,
            row["prompt"], args.alpha_helpfulness, args.alpha_harmlessness, args.alpha_humor,
            max_new_tokens=args.max_new_tokens, device=device,
        )
        results.append({
            "uid": row["uid"], "prompt": row["prompt"],
            "response": response, "model": model_name,
            "elapsed": time.time() - start,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Done. {len(results)} prompts in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
