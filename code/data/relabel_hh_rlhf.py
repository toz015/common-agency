"""
Relabel HH-RLHF data with three reward models for multi-objective PARM training.

Reward models (matching PARM paper's HH-RLHF setup):
  - Helpfulness: Ray2333/gpt2-large-helpful-reward_model
  - Harmlessness: Ray2333/gpt2-large-harmless-reward_model
  - Humor:        mohameddhiab/humor-no-humor (text-classification pipeline)

Randomly samples 12K from Dahoas/full-hh-rlhf train split, scores each
(prompt, chosen) and (prompt, rejected) pair, assigns per-objective labels,
then splits into 10K train / 1K dev / 1K test.
"""

import gc
import json
import os
import random
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_samples", type=int, default=12000)
    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--dev_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./HH-RLHF")
    return parser.parse_args()


def get_humor_score(humor_pipe, text, max_length=512):
    """Get humor probability for a text snippet."""
    result = humor_pipe(text[:max_length], truncation=True)
    r = result[0]
    if r["label"] == "HUMOR":
        return r["score"]
    else:
        return 1.0 - r["score"]


def score_with_ray2333(model_path, samples, score_key):
    """Load a Ray2333 reward model, score all samples, then free GPU memory."""
    print(f"Loading {score_key} reward model: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    with torch.no_grad():
        for sample in tqdm(samples, desc=f"Scoring {score_key}"):
            for idx, resp_key in enumerate(["response_0", "response_1"]):
                # HH-RLHF format: prompt already ends with "\n\nAssistant:"
                # Response is the assistant's reply
                text = sample["prompt"] + sample[resp_key]
                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=1024
                ).to("cuda:0")
                score = model(**inputs).logits[0][0].item()
                sample[f"{score_key}_score_{idx}"] = score

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Done scoring {score_key}, freed GPU memory.\n")


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load dataset ----
    print("Loading Dahoas/full-hh-rlhf (train split)...")
    ds = load_dataset("Dahoas/full-hh-rlhf", split="train")
    print(f"Loaded {len(ds)} rows. Sampling {args.total_samples}...")

    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[: args.total_samples]
    ds = ds.select(indices)

    # Build sample dicts
    samples = []
    for row in ds:
        samples.append({
            "prompt": row["prompt"],
            "response_0": row["chosen"],
            "response_1": row["rejected"],
        })

    # ---- Score helpfulness (Ray2333, ~800MB, fits easily on GPU) ----
    score_with_ray2333(
        "Ray2333/gpt2-large-helpful-reward_model", samples, "help"
    )

    # ---- Score harmlessness (Ray2333, ~800MB) ----
    score_with_ray2333(
        "Ray2333/gpt2-large-harmless-reward_model", samples, "harm"
    )

    # ---- Score humor (small model, runs on CPU) ----
    print("Loading humor model...")
    humor_pipe = pipeline(
        "text-classification",
        model="mohameddhiab/humor-no-humor",
        device="cpu",
    )
    for sample in tqdm(samples, desc="Scoring humor"):
        for idx, resp_key in enumerate(["response_0", "response_1"]):
            sample[f"humor_score_{idx}"] = get_humor_score(
                humor_pipe, sample[resp_key]
            )
    del humor_pipe
    print("Done scoring humor.\n")

    # ---- Assign labels ----
    for d in samples:
        d["better_response_id"] = (
            0 if d["help_score_0"] > d["help_score_1"] else 1
        )
        # Higher harmless score = safer (Ray2333 convention: higher = more harmless)
        d["safer_response_id"] = (
            0 if d["harm_score_0"] > d["harm_score_1"] else 1
        )
        d["funnier_response_id"] = (
            0 if d["humor_score_0"] > d["humor_score_1"] else 1
        )

    # ---- Save all ----
    with open(f"{args.output_dir}/all.json", "w") as f:
        json.dump(samples, f)
    print(f"Saved {len(samples)} samples to {args.output_dir}/all.json")

    # ---- Split ----
    random.shuffle(samples)

    train_data = samples[: args.train_size]
    dev_data = samples[args.train_size : args.train_size + args.dev_size]
    test_data = samples[args.train_size + args.dev_size :]

    with open(f"{args.output_dir}/train.json", "w") as f:
        json.dump(train_data, f)
    with open(f"{args.output_dir}/dev.json", "w") as f:
        json.dump(dev_data, f)
    with open(f"{args.output_dir}/test.json", "w") as f:
        json.dump(test_data, f)

    print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")

    # ---- Test prompts ----
    test_prompts = [
        {"uid": f"hh_eval{i}", "prompt": d["prompt"]}
        for i, d in enumerate(test_data)
    ]
    with open(f"{args.output_dir}/test_prompt_only.json", "w", encoding="utf-8") as f:
        json.dump(test_prompts, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(test_prompts)} test prompts")


if __name__ == "__main__":
    main()
