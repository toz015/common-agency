'''
GenARM generation on a bnb 8-bit FP16 HF base + 8-bit 7B ARMs (vs the
4-bit GPTQ base + FP16 ARMs setup in generate_outputs_genarm.py). Copy —
do NOT modify the original.

M = M_base + alpha_help * M_arm_help + alpha_harm * M_arm_harm
with every AutoModelForCausalLM.from_pretrained monkey-patched to go
through BitsAndBytesConfig(load_in_8bit=True). VRAM budget on A100 80GB:
65GB base + 2 × 7GB ARMs = 79GB; tight, may OOM with long KV cache.
'''
import argparse
import json
import os
import time
from pathlib import Path

import transformers
import transformers.modeling_utils
import accelerate
import accelerate.big_modeling
from transformers import AutoTokenizer, BitsAndBytesConfig

# Patch 1: inject 8-bit quantization into every AutoModelForCausalLM.from_pretrained.
_orig_automodel_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained
_BNB_8BIT_CFG = BitsAndBytesConfig(load_in_8bit=True)


def _from_pretrained_8bit(*args, **kwargs):
    kwargs.setdefault('quantization_config', _BNB_8BIT_CFG)
    kwargs['device_map'] = 'auto'
    return _orig_automodel_from_pretrained(*args, **kwargs)


transformers.AutoModelForCausalLM.from_pretrained = _from_pretrained_8bit

# Patch 2: short-circuit accelerate.dispatch_model for bnb-quantized models.
# transformers 4.36.2 + accelerate 1.x + bitsandbytes 0.49 has an API mismatch
# where dispatch_model falls through to `model.to(device)`, which bnb rejects
# ("`.to` is not supported for 4-bit or 8-bit bitsandbytes models"). bnb places
# weights on the correct device during from_pretrained, so dispatch is a no-op
# for quantized models — just return the model as-is.
_orig_dispatch_model = accelerate.big_modeling.dispatch_model


def _dispatch_model_bnb_safe(model, *args, **kwargs):
    if (getattr(model, 'is_quantized', False)
            or getattr(model, 'is_loaded_in_8bit', False)
            or getattr(model, 'is_loaded_in_4bit', False)):
        return model
    return _orig_dispatch_model(model, *args, **kwargs)


accelerate.big_modeling.dispatch_model = _dispatch_model_bnb_safe
accelerate.dispatch_model = _dispatch_model_bnb_safe
if hasattr(transformers.modeling_utils, 'dispatch_model'):
    transformers.modeling_utils.dispatch_model = _dispatch_model_bnb_safe

from tqdm import tqdm
from model_arithmetic import ModelArithmetic, PromptedLLM

PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'
PROMPT_INPUT_ALPACA: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT


def str2bool(s: str) -> bool:
    if s.lower() in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if s.lower() in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    return bool(s)


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='GenARM sequential generation with 8-bit bnb base + 8-bit ARMs and checkpointing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model_base_name_or_path', default="TheBloke/alpaca-lora-65B-HF", type=str)
    p.add_argument('--model_arm_help_path', required=True, type=str)
    p.add_argument('--model_arm_harm_path', required=True, type=str)
    p.add_argument('--alpha_helpfulness', type=float, required=True)
    p.add_argument('--alpha_harmlessness', type=float, required=True)
    p.add_argument('--max_new_tokens', type=int, default=128)
    p.add_argument('--max_length', type=int, default=2048)
    p.add_argument('--normalize_logit', type=str2bool, default=False)
    p.add_argument('--datasets', type=str, default="../data/test_prompt_only.json")
    p.add_argument('--output_dir', type=str, default="./results")
    p.add_argument('--resume', type=str2bool, default=True,
                   help='If True (default), load existing generation.json and skip done uids.')
    p.add_argument('--checkpoint_every', type=int, default=10,
                   help='Write generation.json every N completed prompts.')
    return p.parse_args()


def get_model_arithmetic(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_base_name_or_path)
    if 'alpaca' in args.model_base_name_or_path.lower() and '65b' in args.model_base_name_or_path.lower():
        print('\nUsing Alpaca-65B base template.\n')
        prompt_template_base = lambda sp, x: (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{x}\n\n### Response:\n"
        )
    else:
        prompt_template_base = lambda sp, x: PROMPT_INPUT_ALPACA.format(input=x)

    prompt_template_arm = lambda sp, x: PROMPT_INPUT_ALPACA.format(input=x)

    M_base = PromptedLLM(system_prompt="Not used", prompt_template=prompt_template_base,
                         model=args.model_base_name_or_path, tokenizer=tokenizer)
    M_help = PromptedLLM(system_prompt="Not used", prompt_template=prompt_template_arm,
                         model=args.model_arm_help_path, tokenizer=tokenizer)
    M_harm = PromptedLLM(system_prompt="Not used", prompt_template=prompt_template_arm,
                         model=args.model_arm_harm_path, tokenizer=tokenizer)
    formula = M_base + args.alpha_helpfulness * M_help + args.alpha_harmlessness * M_harm
    model = ModelArithmetic(formula, max_length=args.max_length)
    return model, tokenizer, 0


if __name__ == '__main__':
    args = parse_arguments()

    model_name = f'GenARM_{args.alpha_helpfulness}help_{args.alpha_harmlessness}harm'
    out_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = Path(os.path.join(out_dir, 'generation.json'))
    print(f'Generation results will be saved in {out_path}')

    with open(args.datasets, 'r') as f:
        data_evaluation = json.load(f)

    output_set = []
    done_uids = set()
    if args.resume and out_path.exists():
        try:
            with open(out_path, 'r', encoding='utf-8') as f:
                output_set = json.load(f)
            done_uids = {r['uid'] for r in output_set}
            print(f'[resume] loaded {len(output_set)} existing records ({len(done_uids)} unique uids)')
        except Exception as e:
            print(f'[resume] could not load {out_path}: {e}; starting fresh')
            output_set = []
            done_uids = set()

    todo = [d for d in data_evaluation if d['uid'] not in done_uids]
    if not todo:
        print(f'[resume] all {len(data_evaluation)} prompts already done — nothing to do.')
        raise SystemExit(0)
    print(f'[resume] {len(todo)} prompts remaining (out of {len(data_evaluation)} total)')

    model, tokenizer, temperature = get_model_arithmetic(args)
    model.eval()
    if args.normalize_logit:
        print('\nEnforcing temperature=1.0; logit weights normalized.\n')
        temperature = 1.0
        model_name += '_NormalizedLogit'
    print(f'\nModel Name: {model_name}')

    t0 = time.time()
    for i, d in enumerate(tqdm(todo)):
        s = time.time()
        response = model.generate_text(
            d['prompt'],
            max_new_tokens=args.max_new_tokens,
            batch_size=None,
            temperature=temperature,
            top_p=1, top_k=0,
            do_speculation=False,
        )[0]
        response = response.removesuffix(tokenizer.eos_token)
        elapsed = time.time() - s
        output_set.append({
            'uid': d['uid'],
            'prompt': d['prompt'],
            'response': response,
            'model': model_name,
            'elapsed': elapsed,
        })
        if (i + 1) % args.checkpoint_every == 0 or (i + 1) == len(todo):
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(output_set, f, ensure_ascii=False, indent=4)
            print(f'[ckpt] saved {len(output_set)}/{len(data_evaluation)} → {out_path}')

    print(f'Done!  {(time.time() - t0) / 3600:.2f}h for {len(todo)} new outputs.')
