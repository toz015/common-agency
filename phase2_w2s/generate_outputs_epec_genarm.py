"""
EPEC+GenARM token-level decoding for W2S Phase-2 (PKU-SafeRLHF, 2 principals).

Per token:
  1. Forward 4-bit 65B base LLM                           → log_base (vocab 32000)
  2. Forward 7B ARM base (alpaca-7b-reproduced, no LoRA)  → log_arm_base
  3. Switch to "help" LoRA, forward                       → log_help
  4. Switch to "harm" LoRA, forward                       → log_harm
  5. Compute implicit rewards on top-k base actions:
        q_j = log_arm_j - log_arm_base    (j ∈ {help, harm})
  6. Scale by user preference α_j, shift to non-negative
  7. Solve 2-principal Common-Agency EPEC (Nonlinear Jacobi)
  8. Greedy select from π★

Ported from: code/evaluation/generate_outputs_epec_hh.py @ origin/fresh-start
Differences vs original (HH-RLHF, 3 principal):
  • Drop humor → J=2 principals (help, harm)
  • Base LM: TheBloke/alpaca-lora-65B-GPTQ (4-bit) instead of LLaMA-2-7B-Chat
  • ARM base: PKU-Alignment/alpaca-7b-reproduced instead of TinyLLaMA
  • Adapters: phase2 GenARM help/harm only (no humor)
  • Prompt template: Alpaca-65B (matches W2S logit-sum baseline for direct comparison)
  • Single tokenizer for both forwards (LLaMA-1 family, vocab matches)
  • --resume support for sweep restartability
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from scipy.optimize import minimize
from scipy.special import softmax
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------- Prompt template (matches W2S logit-sum baseline) ----------

ALPACA_65B_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{input}\n\n### Response:\n"
)


def format_prompt(prompt: str) -> str:
    return ALPACA_65B_TEMPLATE.format(input=prompt)


# ---------- EPEC solver (k-dim, J=2 principals) ----------

def best_response(log_pi_base: np.ndarray, Y: np.ndarray, tau: float) -> np.ndarray:
    return softmax(log_pi_base + Y / tau)


def solve_principal(i, q, y, log_pi_base, tau):
    qi = q[i]
    y_other = sum(y[j] for j in range(len(y)) if j != i)

    def neg_obj(yi):
        Y = y_other + yi
        pi = best_response(log_pi_base, Y, tau)
        return -float(pi @ (qi - yi))

    bounds = [(0.0, max(float(qi[a]), 0.0)) for a in range(len(qi))]
    upper = np.array([b[1] for b in bounds])
    x0 = np.clip(y[i], 0.0, upper)
    try:
        res = minimize(neg_obj, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 8, "ftol": 1e-4})
        return res.x
    except ValueError:
        return x0


def epec_one_step(log_pi_base, q_list, tau=0.1, eps=1e-3, max_iter=5):
    """Solve EPEC for J principals via Nonlinear Jacobi."""
    J = len(q_list)
    y = [np.zeros_like(qj) for qj in q_list]
    Y = sum(y)
    pi = best_response(log_pi_base, Y, tau)
    for _ in range(max_iter):
        y_new = [solve_principal(i, q_list, y, log_pi_base, tau) for i in range(J)]
        Y_new = sum(y_new)
        pi_new = best_response(log_pi_base, Y_new, tau)
        dy = max(np.max(np.abs(y_new[i] - y[i])) for i in range(J))
        dp = float(np.max(np.abs(pi_new - pi)))
        y, pi = y_new, pi_new
        if dy + dp <= eps:
            break
    return pi


# ---------- Model loading ----------

def load_models(args, device):
    """Load 4-bit 65B base + 7B ARM with help/harm LoRA adapters."""
    # Base LLM (4-bit GPTQ via auto_gptq, bypasses broken optimum.gptq path
    # in this venv's optimum/auto_gptq combo — same idiom as
    # language-model-arithmetic/src/model_arithmetic/basic_model_loader.py)
    print(f"Loading base LM: {args.base}")
    base_tok = AutoTokenizer.from_pretrained(args.base)
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token
    if args.base.endswith("GPTQ") or args.base.endswith("GGML"):
        from auto_gptq import AutoGPTQForCausalLM
        base_model = AutoGPTQForCausalLM.from_quantized(
            args.base,
            use_safetensors=True,
            trust_remote_code=True,
            quantize_config=None,
            device_map={"": 0},
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, device_map=device,
        )
    base_model.eval()

    # ARM base (alpaca-7b-reproduced, fp16/bf16) + 2 LoRA adapters
    print(f"Loading ARM base: {args.arm_base}")
    arm_tok = AutoTokenizer.from_pretrained(args.arm_base)
    if arm_tok.pad_token is None:
        arm_tok.pad_token = arm_tok.eos_token
    arm_model = AutoModelForCausalLM.from_pretrained(
        args.arm_base, torch_dtype=torch.bfloat16, device_map=device
    )
    print(f"Loading help adapter: {args.help_adapter}")
    arm_model = PeftModel.from_pretrained(
        arm_model, args.help_adapter, adapter_name="help"
    )
    print(f"Loading harm adapter: {args.harm_adapter}")
    arm_model.load_adapter(args.harm_adapter, adapter_name="harm")
    arm_model.eval()

    return base_model, base_tok, arm_model, arm_tok


@torch.no_grad()
def generate_epec(base_model, base_tok, arm_model, arm_tok,
                  prompt_text, alpha_help, alpha_harm,
                  max_new_tokens=512, k=50, tau=0.1, device="cuda"):
    """Generate tokens via EPEC equilibrium decoding (2 principals)."""
    formatted = format_prompt(prompt_text)
    base_ids = base_tok(formatted, return_tensors="pt").input_ids.to(device)
    arm_ids = arm_tok(formatted, return_tensors="pt").input_ids.to(device)

    base_cur, arm_cur = base_ids, arm_ids
    pkv_base = pkv_arm_base = pkv_help = pkv_harm = None
    out_ids = []
    eos_id = base_tok.eos_token_id
    fwd_time = epec_time = 0.0

    for _ in range(max_new_tokens):
        t0 = time.time()

        # 1. Base LLM forward
        out = base_model(input_ids=base_cur, past_key_values=pkv_base, use_cache=True)
        log_base = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_base = out.past_key_values

        # 2. ARM base forward (no adapter)
        with arm_model.disable_adapter():
            out = arm_model(input_ids=arm_cur, past_key_values=pkv_arm_base, use_cache=True)
        log_arm_base = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_arm_base = out.past_key_values

        # 3. ARM help
        arm_model.set_adapter("help")
        out = arm_model(input_ids=arm_cur, past_key_values=pkv_help, use_cache=True)
        log_help = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_help = out.past_key_values

        # 4. ARM harm
        arm_model.set_adapter("harm")
        out = arm_model(input_ids=arm_cur, past_key_values=pkv_harm, use_cache=True)
        log_harm = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_harm = out.past_key_values

        fwd_time += time.time() - t0

        # 5. EPEC solve on top-k actions from base LLM
        t0 = time.time()
        topk = np.argpartition(log_base, -k)[-k:]

        # Defensive: handle vocab size mismatch (no-op for LLaMA-1 family)
        vocab_min = min(len(log_base), len(log_arm_base))
        topk_valid = topk[topk < vocab_min]
        if len(topk_valid) < len(topk):
            topk = topk_valid
        lb = log_base[topk]

        # Implicit rewards
        q_help = log_help[topk] - log_arm_base[topk]
        q_harm = log_harm[topk] - log_arm_base[topk]

        # Scale by alpha and shift to non-negative
        q_help = np.clip(q_help - q_help.min(), 0.0, None) * alpha_help
        q_harm = np.clip(q_harm - q_harm.min(), 0.0, None) * alpha_harm

        pi_star = epec_one_step(lb, [q_help, q_harm], tau=tau)
        a = int(np.argmax(pi_star))
        tok_id = int(topk[a])
        epec_time += time.time() - t0

        if tok_id == eos_id:
            break
        out_ids.append(tok_id)

        base_cur = torch.tensor([[tok_id]], device=device)
        # Re-tokenize the new token's surface form for the ARM (LLaMA-1 family;
        # in practice the arm tokenizer matches base, so this is a single id).
        tok_text = base_tok.decode([tok_id])
        arm_cur = arm_tok(tok_text, return_tensors="pt",
                          add_special_tokens=False).input_ids.to(device)
        if arm_cur.numel() == 0:
            arm_cur = torch.tensor([[arm_tok.unk_token_id or 0]], device=device)

    return base_tok.decode(out_ids, skip_special_tokens=True), fwd_time, epec_time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="TheBloke/alpaca-lora-65B-GPTQ",
                   help="Base LM (4-bit GPTQ supported via auto_gptq)")
    p.add_argument("--arm_base", default="PKU-Alignment/alpaca-7b-reproduced",
                   help="ARM base model (LoRA adapters live on top)")
    p.add_argument("--help_adapter", required=True,
                   help="Path to GenARM help LoRA adapter")
    p.add_argument("--harm_adapter", required=True,
                   help="Path to GenARM harm LoRA adapter")
    p.add_argument("--alpha_helpfulness", type=float, required=True)
    p.add_argument("--alpha_harmlessness", type=float, required=True)
    p.add_argument("--tau", type=float, default=0.1,
                   help="EPEC temperature")
    p.add_argument("--k", type=int, default=50,
                   help="Top-k base actions for EPEC solve")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--datasets", required=True,
                   help="JSON file: list of {uid, prompt}")
    p.add_argument("--output_dir", required=True,
                   help="Per-config dir created under here")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only first N prompts")
    p.add_argument("--resume", type=str, default="false",
                   help="If true, skip uids already in generation.json")
    return p.parse_args()


def str2bool(s: str) -> bool:
    return s.lower() in {"1", "true", "t", "yes", "y", "on"}


def main():
    args = parse_args()
    args.resume = str2bool(args.resume)
    device = "cuda"

    model_name = (
        f"EPEC_GenARM_{args.alpha_helpfulness}help_"
        f"{args.alpha_harmlessness}harm_tau{args.tau}_k{args.k}"
    )
    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generation.json"
    timing_path = out_dir / "timing.json"
    print(f"Saving to {out_path}")

    with open(args.datasets) as f:
        data = json.load(f)
    if args.limit > 0:
        data = data[:args.limit]

    # Resume: load existing outputs, skip uids already present
    existing = {}
    if args.resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for r in json.load(f):
                existing[r["uid"]] = r
        print(f"Resume: {len(existing)} uids already present, will be skipped.")
    todo = [row for row in data if row["uid"] not in existing]
    print(f"Total prompts: {len(data)}, todo: {len(todo)}, "
          f"existing: {len(existing)}")

    base_model, base_tok, arm_model, arm_tok = load_models(args, device)
    print(f"\nModel: {model_name}")

    results = list(existing.values())
    tot_fwd = tot_epec = 0.0
    t0_all = time.time()
    for row in tqdm(todo):
        start = time.time()
        response, fwd_t, epec_t = generate_epec(
            base_model, base_tok, arm_model, arm_tok,
            row["prompt"], args.alpha_helpfulness, args.alpha_harmlessness,
            max_new_tokens=args.max_new_tokens, k=args.k, tau=args.tau,
            device=device,
        )
        tot_fwd += fwd_t
        tot_epec += epec_t
        results.append({
            "uid": row["uid"], "prompt": row["prompt"],
            "response": response, "model": model_name,
            "elapsed": time.time() - start,
            "fwd_time": fwd_t, "epec_time": epec_t,
        })
        # Checkpoint after every prompt (cheap) — robust to interrupt
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    total = time.time() - t0_all
    timing = {
        "method": model_name, "n_prompts_new": len(todo),
        "n_prompts_total": len(results),
        "total_seconds": total, "fwd_seconds": tot_fwd,
        "epec_seconds": tot_epec, "tau": args.tau, "k": args.k,
    }
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)

    print(f"Done. {len(todo)} new prompts in {total/60:.1f} min "
          f"(total now {len(results)})")
    print(f"Forward: {tot_fwd:.0f}s, EPEC solve: {tot_epec:.0f}s")


if __name__ == "__main__":
    main()
