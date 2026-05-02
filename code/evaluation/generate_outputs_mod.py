'''
MOD (Multi-Objective Decoding) inference: logit-level fusion of two DPO models
trained on separate objectives (helpfulness and harmlessness).

    logits = alpha_help * logits_dpo_help + alpha_harm * logits_dpo_harm

Each DPO model is base + LoRA adapter. We load the base model once and switch
between adapters using PEFT's adapter API, then fuse logits at each step.

Reference: Shi et al., "Decoding-Time Language Model Alignment with Multiple
Objectives", NeurIPS 2024.
'''
import argparse
import json
import time
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'
PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='MOD generation via logit fusion of two DPO adapters.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model_base_name_or_path',
                        default="PKU-Alignment/alpaca-7b-reproduced", type=str)
    parser.add_argument('--model_dpo_help_path', required=True, type=str,
                        help='path to the helpfulness DPO LoRA adapter')
    parser.add_argument('--model_dpo_harm_path', required=True, type=str,
                        help='path to the harmlessness DPO LoRA adapter')
    parser.add_argument('--alpha_helpfulness', type=float, required=True)
    parser.add_argument('--alpha_harmlessness', type=float, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--datasets', type=str, default="../data/test_prompt_only.json")
    parser.add_argument('--output_dir', type=str, default="./results_mod")
    return parser.parse_args()


class MODFusionModel:
    """Fuse logits from two LoRA adapters loaded on a shared base model."""

    def __init__(self, base_model_path, help_adapter_path, harm_adapter_path,
                 alpha_help, alpha_harm, device="cuda"):
        self.alpha_help = alpha_help
        self.alpha_harm = alpha_harm
        self.device = device

        print(f"Loading base model: {base_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.float16, device_map=device,
        )
        self.model.eval()

        print(f"Loading help adapter: {help_adapter_path}")
        self.model = PeftModel.from_pretrained(
            self.model, help_adapter_path, adapter_name="help",
        )
        print(f"Loading harm adapter: {harm_adapter_path}")
        self.model.load_adapter(harm_adapter_path, adapter_name="harm")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def generate(self, prompt_text, max_new_tokens=512):
        input_text = PROMPT_INPUT.format(input=prompt_text)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)

        generated_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            self.model.set_adapter("help")
            logits_help = self.model(generated_ids).logits[:, -1, :]

            self.model.set_adapter("harm")
            logits_harm = self.model(generated_ids).logits[:, -1, :]

            fused_logits = self.alpha_help * logits_help + self.alpha_harm * logits_harm
            next_token = fused_logits.argmax(dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        response_ids = generated_ids[0, input_ids.shape[1]:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response


if __name__ == '__main__':
    args = parse_arguments()

    model_name = f'MOD_{args.alpha_helpfulness}help_{args.alpha_harmlessness}harm'
    out_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = Path(os.path.join(out_dir, 'generation.json'))
    print(f'Generation results will be saved in {out_path}')

    with open(args.datasets, 'r') as f:
        data_evaluation = json.load(f)

    fusion = MODFusionModel(
        args.model_base_name_or_path,
        args.model_dpo_help_path,
        args.model_dpo_harm_path,
        args.alpha_helpfulness,
        args.alpha_harmlessness,
    )

    output_set = []
    start_time = time.time()

    for i in tqdm(range(len(data_evaluation))):
        prompt = data_evaluation[i]['prompt']
        t0 = time.time()
        response = fusion.generate(prompt, max_new_tokens=args.max_new_tokens)
        elapsed = time.time() - t0
        output_set.append({
            "uid": data_evaluation[i]['uid'],
            "prompt": prompt,
            "response": response,
            "model": model_name,
            "elapsed": elapsed,
        })

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_set, f, ensure_ascii=False, indent=4)

    total_hours = (time.time() - start_time) / 3600
    print(f'Done! Saved to {out_path}')
    print(f'Time: {total_hours:.2f} hours for {len(output_set)} outputs')

    timing = {
        "method": model_name,
        "n_prompts": len(output_set),
        "total_seconds": time.time() - start_time,
        "mean_per_prompt": (time.time() - start_time) / len(output_set),
        "alpha_helpfulness": args.alpha_helpfulness,
        "alpha_harmlessness": args.alpha_harmlessness,
        "max_new_tokens": args.max_new_tokens,
    }
    with open(os.path.join(out_dir, 'timing.json'), 'w') as f:
        json.dump(timing, f, indent=2)
