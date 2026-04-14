'''
GenARM baseline: train a single-objective autoregressive reward model via stock
TRL DPOTrainer + plain LoRA. Run twice (--objective help, --objective harm) to
produce two independent ARMs. Hyperparameters and target modules match PARM's
PBLoRA setup (train_pref_arm.py + run.sh) for fair comparison.
'''
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOTrainer, DPOConfig
from trl.commands.cli_utils import TrlParser

import wandb
wandb.init(mode="disabled")


@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(
        default="PKU-Alignment/alpaca-7b-reproduced",
    )
    preference_dataset: Optional[str] = field(default="PKU_SafeRLHF")
    objective: Optional[str] = field(
        default="help",
        metadata={"help": "which single objective to train on: 'help' uses better_response_id, 'harm' uses safer_response_id"},
    )
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit")
    lora_alpha: Optional[float] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_r: Optional[int] = field(default=8)
    load_in_4bit: Optional[bool] = field(default=True)
    model_dtype: Optional[str] = field(default="float16")
    sanity_check: Optional[bool] = field(default=False)
    ignore_bias_buffers: Optional[bool] = field(default=False)


def get_PKU_SafeRLHF(objective: str, sanity_check: bool = False, num_proc: int = 4):
    assert objective in {"help", "harm"}, f"objective must be help|harm, got {objective}"

    train_dataset = load_dataset("json", data_files="../data/train.json", split="train", num_proc=num_proc)
    eval_dataset = load_dataset("json", data_files="../data/dev.json", split="train", num_proc=num_proc)
    original_columns = train_dataset.column_names

    if sanity_check:
        train_dataset = train_dataset.select(range(min(len(train_dataset), 1000)))

    PROMPT_BEGIN = "BEGINNING OF CONVERSATION: "
    PROMPT_USER = "USER: {input} "
    PROMPT_ASSISTANT = "ASSISTANT:"

    def format_prompt(input_str: str) -> str:
        return PROMPT_BEGIN + PROMPT_USER.format(input=input_str) + PROMPT_ASSISTANT

    label_key = "better_response_id" if objective == "help" else "safer_response_id"

    def map_sample(sample) -> Dict[str, str]:
        chosen_id = sample[label_key]
        rejected_id = 1 - chosen_id
        return {
            "prompt": format_prompt(sample["prompt"]),
            "chosen": sample[f"response_{chosen_id}"],
            "rejected": sample[f"response_{rejected_id}"],
        }

    return (
        train_dataset.map(map_sample, batched=False, num_proc=num_proc, remove_columns=original_columns),
        eval_dataset.map(map_sample, batched=False, num_proc=num_proc, remove_columns=original_columns),
    )


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig))
    script_args, training_args = parser.parse_args_and_config()

    print(f"\nPreference dataset: {script_args.preference_dataset}  |  objective: {script_args.objective}\n")

    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    set_seed(training_args.seed)

    if script_args.preference_dataset != "PKU_SafeRLHF":
        raise ValueError(f"Invalid preference dataset: {script_args.preference_dataset}")
    train_dataset, eval_dataset = get_PKU_SafeRLHF(
        objective=script_args.objective, sanity_check=script_args.sanity_check
    )

    print(f"\nBefore filtering. Train: {train_dataset.num_rows}, Eval: {eval_dataset.num_rows}\n")
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= training_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= training_args.max_length
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= training_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= training_args.max_length
    )
    print(f"After filtering. Train: {train_dataset.num_rows}, Eval: {eval_dataset.num_rows}\n")

    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    print(f"\n model_name_or_path: {script_args.model_name_or_path}\n")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        load_in_4bit=script_args.load_in_4bit,
        device_map={"": Accelerator().local_process_index},
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
