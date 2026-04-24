'''
GenARM inference: blend base LM logits with two independent autoregressive
reward models (one per objective) via model-arithmetic.

    M = M_base + alpha_helpfulness * M_arm_help + alpha_harmlessness * M_arm_harm

Each ARM is a plain LoRA adapter directory — PromptedLLM loads it directly
(basic_model_loader.py detects adapter_config.json and wraps with PeftModel).
'''
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import time
import os

from transformers import AutoTokenizer
from model_arithmetic import ModelArithmetic, PromptedLLM

PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'
PROMPT_INPUT_ALPACA: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT


def str2bool(string: str) -> bool:
    if string.lower() in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if string.lower() in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    return bool(string)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='GenARM generation via logit blending of two independent ARMs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    model_parser = parser.add_argument_group('model')
    model_parser.add_argument('--model_base_name_or_path', default="PKU-Alignment/alpaca-7b-reproduced", type=str)
    model_parser.add_argument('--model_arm_help_path', type=str,
                              help='path to the helpfulness LoRA adapter directory',
                              default="../training/PKU-SafeRLHF/exp_genarm_help/final_checkpoint")
    model_parser.add_argument('--model_arm_harm_path', type=str,
                              help='path to the harmlessness LoRA adapter directory',
                              default="../training/PKU-SafeRLHF/exp_genarm_harm/final_checkpoint")
    model_parser.add_argument('--alpha_helpfulness', type=float, required=True)
    model_parser.add_argument('--alpha_harmlessness', type=float, required=True)
    model_parser.add_argument('--max_new_tokens', type=int, default=512)
    model_parser.add_argument('--max_length', type=int, default=512)
    model_parser.add_argument('--normalize_logit', type=str2bool, default=False,
                              help='If True, force temperature=1.0 so logit weights are normalized; else temperature=1/(1+alpha_help+alpha_harm).')

    dataset_parser = parser.add_argument_group('dataset')
    dataset_parser.add_argument('--datasets', type=str, default="../data/PKU-SafeRLHF/test_prompt_only.json")
    dataset_parser.add_argument(
    '--limit',
    type=int,
    default=100,
    help='Number of test prompts to run. If None, run all prompts.',
)
    logging_parser = parser.add_argument_group('logging')
    logging_parser.add_argument('--output_dir', type=str, default="./results_genarm")
    logging_parser.add_argument('--resume', type=str2bool, default=False)

    return parser.parse_args()


def get_model_arithmetic(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_base_name_or_path)
    prompt_template = lambda system_prompt, input_string: PROMPT_INPUT_ALPACA.format(input=input_string)

    M_base = PromptedLLM(system_prompt="Not used", prompt_template=prompt_template,
                         model=args.model_base_name_or_path, tokenizer=tokenizer)
    M_help = PromptedLLM(system_prompt="Not used", prompt_template=prompt_template,
                         model=args.model_arm_help_path, tokenizer=tokenizer)
    M_harm = PromptedLLM(system_prompt="Not used", prompt_template=prompt_template,
                         model=args.model_arm_harm_path, tokenizer=tokenizer)

    formula = M_base + args.alpha_helpfulness * M_help + args.alpha_harmlessness * M_harm

    model = ModelArithmetic(formula, max_length=args.max_length)
    temperature = 0
    return model, tokenizer, temperature


if __name__ == '__main__':
    args = parse_arguments()

    model_name = f'GenARM_{args.alpha_helpfulness}help_{args.alpha_harmlessness}harm'

    out_path = Path(os.path.join(args.output_dir, model_name, 'generation') + ".json")
    os.makedirs(os.path.join(args.output_dir, model_name), exist_ok=True)
    print(f'Generation results will be saved in {out_path}')

    with open(args.datasets, 'r') as f:
        data_evaluation = json.load(f)

    if args.limit is not None:
        data_evaluation = data_evaluation[:args.limit]

    model, tokenizer, temperature = get_model_arithmetic(args)
    model.eval()
    if args.normalize_logit:
        print('\nEnforcing temperature=1.0 in model_arithmetic generation; logit weights are normalized.\n')
        temperature = 1.0
    generate = lambda prompt: model.generate_text(
        prompt, max_new_tokens=args.max_new_tokens, batch_size=None,
        temperature=temperature, top_p=1, top_k=0, do_speculation=False,use_cache=False,
    )[0].removesuffix(tokenizer.eos_token)

    if args.normalize_logit:
        model_name += '_NormalizedLogit'
    print(f'\nModel Name: {model_name}')

    output_set = []
    start_time_script = time.time()
    for i in tqdm(range(len(data_evaluation))):
        prompt = data_evaluation[i]['prompt']
        start = time.time()
        response = generate(prompt)
        elapsed = time.time() - start
        output_set.append({
            "uid": data_evaluation[i]['uid'],
            "prompt": prompt,
            "response": response,
            "model": model_name,
            "elapsed": elapsed,
        })

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output_set, f, ensure_ascii=False, indent=4)
    print(f'Done!\nSaving to {out_path}\nTime:{(time.time()-start_time_script)/3600} hours for {len(output_set)} outputs')
