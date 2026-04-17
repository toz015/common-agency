"""
Score generated HH-RLHF outputs with 3 reward models (loaded one at a time).

Reward models:
  - Helpfulness: PKU-Alignment/beaver-7b-v1.0-reward
  - Harmlessness: PKU-Alignment/beaver-7b-v1.0-cost
  - Humor:        mohameddhiab/humor-no-humor
"""
import gc
import json
import argparse
import torch
from transformers import AutoTokenizer, pipeline
from safe_rlhf.models import AutoModelForScore
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="Directory containing generation.json")
    return p.parse_args()


def score_beaver(generation, model_path, score_key, template):
    """Score with a Beaver reward model, then free GPU."""
    print(f"Loading {score_key} model: {model_path}")
    model = AutoModelForScore.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    with torch.no_grad():
        for d in tqdm(generation, desc=f"Scoring {score_key}"):
            text = template.format(input=d["prompt"].strip(), response=d["response"])
            input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to("cuda:0")
            d[f"{score_key}_score"] = model(**input_ids)["end_scores"][0][0].item()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Done {score_key}.\n")


def score_humor(generation):
    """Score humor with the classification pipeline (CPU)."""
    print("Loading humor model...")
    humor_pipe = pipeline("text-classification", model="mohameddhiab/humor-no-humor", device="cpu")

    for d in tqdm(generation, desc="Scoring humor"):
        text = d["response"][:512]
        result = humor_pipe(text, truncation=True)[0]
        d["humor_score"] = result["score"] if result["label"] == "HUMOR" else 1.0 - result["score"]

    del humor_pipe
    print("Done humor.\n")


def main():
    args = parse_args()

    with open(f"{args.path}/generation.json") as f:
        generation = json.load(f)

    template = "BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:{response}"

    # Score sequentially to fit on single GPU
    score_beaver(generation, "PKU-Alignment/beaver-7b-v1.0-reward", "help", template)
    score_beaver(generation, "PKU-Alignment/beaver-7b-v1.0-cost", "harm", template)
    score_humor(generation)

    # Save full results
    with open(f"{args.path}/reward_result.json", "w") as f:
        json.dump(generation, f, ensure_ascii=False, indent=2)

    # Save mean results
    n = len(generation)
    mean_result = {
        "help": sum(d["help_score"] for d in generation) / n,
        "harm": sum(d["harm_score"] for d in generation) / n,
        "humor": sum(d["humor_score"] for d in generation) / n,
        "n_prompts": n,
    }
    with open(f"{args.path}/mean_result.json", "w") as f:
        json.dump(mean_result, f, indent=2)

    print(f"Results for {args.path}:")
    print(f"  Help:  {mean_result['help']:.4f}")
    print(f"  Harm:  {mean_result['harm']:.4f}")
    print(f"  Humor: {mean_result['humor']:.4f}")


if __name__ == "__main__":
    main()
