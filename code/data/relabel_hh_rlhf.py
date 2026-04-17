"""
Relabel HH-RLHF data with three reward models for multi-objective PARM training.

Reward models (loaded one at a time to fit on a single 24GB GPU):
  - Helpfulness: PKU-Alignment/beaver-7b-v1.0-reward
  - Harmlessness: PKU-Alignment/beaver-7b-v1.0-cost
  - Humor:        mohameddhiab/humor-no-humor (text-classification pipeline)

Randomly samples 12K from Dahoas/full-hh-rlhf train split, scores each
(prompt, chosen) and (prompt, rejected) pair, assigns per-objective labels,
then splits into 10K train / 1K dev / 1K test.

Output format (per sample):
  {
    "prompt": str,          # multi-turn conversation context
    "response_0": str,      # chosen response
    "response_1": str,
    "help_score_0/1": float,
    "harm_score_0/1": float,
    "humor_score_0/1": float,
    "better_response_id": 0|1,   # higher helpfulness
    "safer_response_id":  0|1,   # lower cost (more harmless)
    "funnier_response_id": 0|1,  # higher humor
  }
"""

import gc
import json
import os
import random
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from safe_rlhf.models import AutoModelForScore
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


def score_with_beaver(model_path, tokenizer, samples, template, score_key):
    """Load a Beaver reward model, score all samples, then free GPU memory."""
    print(f"Loading {score_key} reward model: {model_path}")
    model = AutoModelForScore.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    with torch.no_grad():
        for sample in tqdm(samples, desc=f"Scoring {score_key}"):
            for idx, resp_key in enumerate(["response_0", "response_1"]):
                text = template.format(
                    input=sample["prompt"].strip(), response=sample[resp_key]
                )
                input_ids = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=2048
                ).to("cuda:0")
                sample[f"{score_key}_score_{idx}"] = (
                    model(**input_ids)["end_scores"][0][0].item()
                )

    # Free GPU memory before loading next model
    del model
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

    # ---- Beaver reward model template ----
    template = "BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:{response}"
    tokenizer = AutoTokenizer.from_pretrained("PKU-Alignment/beaver-7b-v1.0-reward")

    # ---- Score helpfulness (load, score, free) ----
    score_with_beaver(
        "PKU-Alignment/beaver-7b-v1.0-reward", tokenizer, samples, template, "help"
    )

    # ---- Score harmlessness (load, score, free) ----
    score_with_beaver(
        "PKU-Alignment/beaver-7b-v1.0-cost", tokenizer, samples, template, "harm"
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
        # Lower harm cost = safer
        d["safer_response_id"] = (
            0 if d["harm_score_0"] < d["harm_score_1"] else 1
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
