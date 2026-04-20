"""
Score generated HH-RLHF outputs with 3 reward models (matching PARM paper).

Reward models:
  - Helpfulness: Ray2333/gpt2-large-helpful-reward_model
  - Harmlessness: Ray2333/gpt2-large-harmless-reward_model
  - Humor:        mohameddhiab/humor-no-humor
"""
import gc
import json
import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True, help="Directory containing generation.json")
    return p.parse_args()


def score_ray2333(generation, model_path, score_key):
    """Score with a Ray2333 reward model, then free GPU."""
    print(f"Loading {score_key} model: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    with torch.no_grad():
        for d in tqdm(generation, desc=f"Scoring {score_key}"):
            # HH-RLHF format: prompt + response
            text = d["prompt"] + d["response"]
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=1024
            ).to("cuda:0")
            d[f"{score_key}_score"] = model(**inputs).logits[0][0].item()

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

    # Score with Ray2333 models (matching PARM paper)
    score_ray2333(generation, "Ray2333/gpt2-large-helpful-reward_model", "help")
    score_ray2333(generation, "Ray2333/gpt2-large-harmless-reward_model", "harm")
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
