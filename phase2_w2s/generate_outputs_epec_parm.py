"""
EPEC+PARM token-level decoding for W2S Phase-2 (PKU-SafeRLHF, 2 principals).

Uses PARM's PBLoRA adapter (obj_num=2) as a multi-objective reward source —
extracts per-objective rewards by setting pref_vec to unit vectors
(0,1)=help, (1,0)=harm, then applies game-theoretic EPEC aggregation.

Per token:
  1. Forward 4-bit 65B base LLM                         → log_base
  2. Forward 7B PARM base (no adapter active)           → log_arm_base
  3. Set pref_vec=(0,1), forward PARM                   → log_help
  4. Set pref_vec=(1,0), forward PARM                   → log_harm
  5. Compute implicit rewards: q_j = log_j - log_arm_base on top-k
  6. Scale by user preference α_j, shift to non-negative
  7. Solve 2-principal Common-Agency EPEC (Nonlinear Jacobi)
  8. Greedy select from π★

Ported from: code/evaluation/generate_outputs_epec_parm_hh.py @ origin/fresh-start
Differences vs original (HH-RLHF, 3 principal):
  • Drop humor → J=2 principals (help, harm)
  • PARM adapter trained with obj_num=2; pref_vec ordering = [harm, help]
    (matches W2S logit-sum: see generate_outputs.py line ~153)
  • Base LM: TheBloke/alpaca-lora-65B-GPTQ (4-bit) instead of LLaMA-2-7B-Chat
  • PARM base: PKU-Alignment/alpaca-7b-reproduced instead of TinyLLaMA
  • Prompt template: Alpaca-65B (matches W2S logit-sum baseline)
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


# ---------- PBLoRA pref_vec helpers ----------

def set_pref_vec(model, pref_vec):
    """In-place set every pref_vec parameter inside a PBLoRA-wrapped model."""
    pref = torch.tensor(pref_vec)
    for n, p in model.named_parameters():
        if "pref_vec" in n:
            # cast to the param's dtype/device
            p.data = pref.to(device=p.device, dtype=p.dtype)
            p.requires_grad = False


# Pref_vec ordering for PARM trained with obj_num=2:
#   slot 0 = harmlessness, slot 1 = helpfulness
# (matches W2S logit-sum: pref_vec_init = [alpha_harm, alpha_help])
PREF_HELP = [0.0, 1.0]   # extract pure-help reward
PREF_HARM = [1.0, 0.0]   # extract pure-harm reward


# ---------- Model loading ----------

def load_models(args, device):
    """Load 4-bit 65B base + 7B PARM (alpaca-7b + PBLoRA)."""
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
            use_marlin=True,  # 2-3x faster int4*fp16 kernel on A100
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, device_map=device,
        )
    base_model.eval()

    print(f"Loading PARM base: {args.parm_base}")
    parm_tok = AutoTokenizer.from_pretrained(args.parm_base)
    if parm_tok.pad_token is None:
        parm_tok.pad_token = parm_tok.eos_token
    parm_model = AutoModelForCausalLM.from_pretrained(
        args.parm_base, torch_dtype=torch.bfloat16, device_map=device
    )
    print(f"Loading PARM PBLoRA adapter: {args.parm_adapter}")
    parm_model = PeftModel.from_pretrained(parm_model, args.parm_adapter)
    parm_model.eval()

    return base_model, base_tok, parm_model, parm_tok


@torch.no_grad()
def generate_epec(base_model, base_tok, parm_model, parm_tok,
                  prompt_text, alpha_help, alpha_harm,
                  max_new_tokens=512, k=50, tau=0.1, device="cuda"):
    """Generate via EPEC_PARM (2 principals): PARM rewards + EPEC equilibrium."""
    formatted = format_prompt(prompt_text)
    base_ids = base_tok(formatted, return_tensors="pt").input_ids.to(device)
    parm_ids = parm_tok(formatted, return_tensors="pt").input_ids.to(device)

    base_cur, parm_cur = base_ids, parm_ids
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

        # 2. PARM base (no adapter)
        with parm_model.disable_adapter():
            out = parm_model(input_ids=parm_cur, past_key_values=pkv_arm_base, use_cache=True)
        log_arm_base = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_arm_base = out.past_key_values

        # 3. PARM with pref_vec=(0,1) → helpfulness reward
        set_pref_vec(parm_model, PREF_HELP)
        out = parm_model(input_ids=parm_cur, past_key_values=pkv_help, use_cache=True)
        log_help = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_help = out.past_key_values

        # 4. PARM with pref_vec=(1,0) → harmlessness reward
        set_pref_vec(parm_model, PREF_HARM)
        out = parm_model(input_ids=parm_cur, past_key_values=pkv_harm, use_cache=True)
        log_harm = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_harm = out.past_key_values

        fwd_time += time.time() - t0

        # 5. EPEC solve on top-k actions from base LLM
        t0 = time.time()
        topk = np.argpartition(log_base, -k)[-k:]

        # Defensive vocab-size guard (no-op when both are LLaMA-1)
        vocab_min = min(len(log_base), len(log_arm_base))
        topk_valid = topk[topk < vocab_min]
        if len(topk_valid) < len(topk):
            topk = topk_valid
        lb = log_base[topk]

        q_help = log_help[topk] - log_arm_base[topk]
        q_harm = log_harm[topk] - log_arm_base[topk]

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
        tok_text = base_tok.decode([tok_id])
        parm_cur = parm_tok(tok_text, return_tensors="pt",
                            add_special_tokens=False).input_ids.to(device)
        if parm_cur.numel() == 0:
            parm_cur = torch.tensor([[parm_tok.unk_token_id or 0]], device=device)

    return base_tok.decode(out_ids, skip_special_tokens=True), fwd_time, epec_time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="TheBloke/alpaca-lora-65B-GPTQ")
    p.add_argument("--parm_base", default="PKU-Alignment/alpaca-7b-reproduced")
    p.add_argument("--parm_adapter", required=True,
                   help="Path to PARM PBLoRA adapter (obj_num=2)")
    p.add_argument("--alpha_helpfulness", type=float, required=True)
    p.add_argument("--alpha_harmlessness", type=float, required=True)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--datasets", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", type=str, default="false")
    return p.parse_args()


def str2bool(s: str) -> bool:
    return s.lower() in {"1", "true", "t", "yes", "y", "on"}


def main():
    args = parse_args()
    args.resume = str2bool(args.resume)
    device = "cuda"

    model_name = (
        f"EPEC_PARM_{args.alpha_helpfulness}help_"
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

    existing = {}
    if args.resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for r in json.load(f):
                existing[r["uid"]] = r
        print(f"Resume: {len(existing)} uids already present, will be skipped.")
    todo = [row for row in data if row["uid"] not in existing]
    print(f"Total prompts: {len(data)}, todo: {len(todo)}, existing: {len(existing)}")

    base_model, base_tok, parm_model, parm_tok = load_models(args, device)
    print(f"\nModel: {model_name}")
    print(f"PARM adapter: {args.parm_adapter}")

    results = list(existing.values())
    tot_fwd = tot_epec = 0.0
    t0_all = time.time()
    for row in tqdm(todo):
        start = time.time()
        response, fwd_t, epec_t = generate_epec(
            base_model, base_tok, parm_model, parm_tok,
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
        # Per-prompt checkpoint
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
