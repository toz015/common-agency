"""
PARM generation for HH-RLHF (3 objectives: safe, help, humor).

Uses ModelArithmetic: M = M_base + M_parm(pref_vec)
The PBLoRA adapter's pref_vec_init controls the preference weights.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
from model_arithmetic import ModelArithmetic, PromptedLLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_base", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--model_parm_path", default="../training/HH-RLHF/exp")
    p.add_argument("--alpha_helpfulness", type=float, required=True)
    p.add_argument("--alpha_harmlessness", type=float, required=True)
    p.add_argument("--alpha_humor", type=float, required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--datasets", default="../data/HH-RLHF/test_prompt_only.json")
    p.add_argument("--output_dir", default="./results_hh")
    p.add_argument("--cache_dir", default="./cache")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    model_name = f"PARM_{args.alpha_helpfulness}help_{args.alpha_harmlessness}harm_{args.alpha_humor}humor"
    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generation.json"
    print(f"Saving to {out_path}")

    # Load test prompts
    with open(args.datasets) as f:
        data = json.load(f)
    if args.limit > 0:
        data = data[:args.limit]

    # Copy adapter with modified pref_vec_init
    # PBLoRA obj_key order during training: safe, help, humor
    cache_path = os.path.join(args.cache_dir, model_name)
    os.makedirs(cache_path, exist_ok=True)

    with open(f"{args.model_parm_path}/adapter_config.json") as f:
        config = json.load(f)
    config["pref_vec_init"] = [args.alpha_harmlessness, args.alpha_helpfulness, args.alpha_humor]

    with open(f"{cache_path}/adapter_config.json", "w") as f:
        json.dump(config, f, indent=4)
    shutil.copyfile(
        f"{args.model_parm_path}/adapter_model.safetensors",
        f"{cache_path}/adapter_model.safetensors",
    )

    # HH-RLHF prompts are pre-formatted — pass through directly
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    prompt_template = lambda system_prompt, input_string: input_string

    M_base = PromptedLLM(system_prompt="", prompt_template=prompt_template,
                         model=args.model_base, tokenizer=tokenizer)
    M_reward = PromptedLLM(system_prompt="", prompt_template=prompt_template,
                           model=cache_path, tokenizer=tokenizer)

    model = ModelArithmetic(M_base + M_reward, max_length=args.max_length)
    model.eval()

    generate = lambda prompt: model.generate_text(
        prompt, max_new_tokens=args.max_new_tokens, batch_size=None,
        temperature=0, top_p=1, top_k=0, do_speculation=False
    )[0].removesuffix(tokenizer.eos_token)

    print(f"\nModel: {model_name}, Prompts: {len(data)}")

    results = []
    t0 = time.time()
    for row in tqdm(data):
        start = time.time()
        response = generate(row["prompt"])
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
