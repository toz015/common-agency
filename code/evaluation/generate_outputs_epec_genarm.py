"""
Token-level EPEC decoding with GenARM ingredients.

Per token: forward base + ARM_help + ARM_harm, compute per-objective implicit
rewards q^j = log π_ARM_j - log π_base on the top-k actions from the base LM,
then solve the Common-Agency EPEC (Nonlinear Jacobi) on the k-dim action space
and greedy-sample from π★.

Replaces GenARM's linear logit-sum `log π_base + α_h·r_h + α_s·r_s` with a
game-equilibrium aggregation using the *same* three models.
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

PROMPT_INPUT_ALPACA = "BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:"


# ---------- EPEC solver (k-dim, 2 principals) ----------

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


# ---------- Model loading (base + two LoRA adapters, shared backbone) ----------

def load_models(base_path, help_adapter, harm_adapter, device):
    tok = AutoTokenizer.from_pretrained(base_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map=device
    )
    pm = PeftModel.from_pretrained(base, help_adapter, adapter_name="help")
    pm.load_adapter(harm_adapter, adapter_name="harm")
    pm.eval()
    return pm, tok


@torch.no_grad()
def three_forwards(pm, input_ids, pkv_base, pkv_help, pkv_harm):
    """One forward per adapter state; returns last-position log-softmax for each."""
    # base: disable adapters
    with pm.disable_adapter():
        out = pm(input_ids=input_ids, past_key_values=pkv_base, use_cache=True)
    log_b = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
    pkv_base = out.past_key_values
    # help
    pm.set_adapter("help")
    out = pm(input_ids=input_ids, past_key_values=pkv_help, use_cache=True)
    log_h = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
    pkv_help = out.past_key_values
    # harm
    pm.set_adapter("harm")
    out = pm(input_ids=input_ids, past_key_values=pkv_harm, use_cache=True)
    log_s = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
    pkv_harm = out.past_key_values
    return log_b, log_h, log_s, pkv_base, pkv_help, pkv_harm


@torch.no_grad()
def generate_epec(pm, tok, prompt_text, alpha_help, alpha_harm,
                  max_new_tokens=512, k=50, tau=0.1, device="cuda"):
    prompt = PROMPT_INPUT_ALPACA.format(input=prompt_text)
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    cur = input_ids
    pkv_base = pkv_help = pkv_harm = None
    out_ids = []
    eos = tok.eos_token_id
    fwd_time = 0.0
    epec_time = 0.0
    n_tokens = 0
    for _ in range(max_new_tokens):
        t0 = time.time()
        log_b, log_h, log_s, pkv_base, pkv_help, pkv_harm = three_forwards(
            pm, cur, pkv_base, pkv_help, pkv_harm
        )
        fwd_time += time.time() - t0
        t0 = time.time()
        topk = np.argpartition(log_b, -k)[-k:]
        lb = log_b[topk]
        q1 = log_h[topk] - log_b[topk]
        q2 = log_s[topk] - log_b[topk]
        q1 = np.clip(q1 - q1.min(), 0.0, None) * alpha_help
        q2 = np.clip(q2 - q2.min(), 0.0, None) * alpha_harm
        pi_star = epec_one_step(lb, [q1, q2], tau=tau)
        a = int(np.argmax(pi_star))
        tok_id = int(topk[a])
        epec_time += time.time() - t0
        n_tokens += 1
        if tok_id == eos:
            break
        out_ids.append(tok_id)
        cur = torch.tensor([[tok_id]], device=device)
    return tok.decode(out_ids, skip_special_tokens=True), fwd_time, epec_time, n_tokens


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="PKU-Alignment/alpaca-7b-reproduced")
    p.add_argument("--help_adapter", default="../training/PKU-SafeRLHF/exp_genarm_help/final_checkpoint")
    p.add_argument("--harm_adapter", default="../training/PKU-SafeRLHF/exp_genarm_harm/final_checkpoint")
    p.add_argument("--alpha_helpfulness", type=float, required=True)
    p.add_argument("--alpha_harmlessness", type=float, required=True)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--datasets", default="../data/PKU-SafeRLHF/test_prompt_only.json")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--limit", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda"
    model_name = f"EPEC_GenARM_{args.alpha_helpfulness}help_{args.alpha_harmlessness}harm_tau{args.tau}_k{args.k}"
    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generation.json"
    print(f"Saving to {out_path}")

    pm, tok = load_models(args.base, args.help_adapter, args.harm_adapter, device)

    data = json.load(open(args.datasets))
    if args.limit > 0:
        data = data[: args.limit]

    torch.cuda.reset_peak_memory_stats()
    results = []
    tot_fwd = tot_epec = tot_tokens = 0.0
    t0 = time.time()
    for row in tqdm(data):
        t = time.time()
        resp, fwd_t, epec_t, n_tok = generate_epec(
            pm, tok, row["prompt"],
            alpha_help=args.alpha_helpfulness, alpha_harm=args.alpha_harmlessness,
            max_new_tokens=args.max_new_tokens, k=args.k, tau=args.tau, device=device,
        )
        tot_fwd += fwd_t
        tot_epec += epec_t
        tot_tokens += n_tok
        results.append({
            "uid": row["uid"], "prompt": row["prompt"], "response": resp,
            "model": model_name, "elapsed": time.time() - t,
            "fwd_time": fwd_t, "epec_time": epec_t, "n_tokens": n_tok,
        })
    total = time.time() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    timing = {
        "method": model_name,
        "n_prompts": len(results),
        "total_seconds": total,
        "mean_per_prompt": total / len(results),
        "fwd_fraction": tot_fwd / (tot_fwd + tot_epec),
        "epec_fraction": tot_epec / (tot_fwd + tot_epec),
        "total_tokens": int(tot_tokens),
        "sec_per_token": (tot_fwd + tot_epec) / max(tot_tokens, 1),
        "peak_gpu_memory_gb": peak_mem_gb,
        "tau": args.tau, "k": args.k,
        "alpha_helpfulness": args.alpha_helpfulness,
        "alpha_harmlessness": args.alpha_harmlessness,
        "max_new_tokens": args.max_new_tokens,
    }
    json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)
    json.dump(timing, open(out_dir / "timing.json", "w"), indent=2)
    print(f"Done. {len(results)} prompts in {total/60:.1f} min")
    print(f"Peak GPU mem: {peak_mem_gb:.2f} GB")
    print(f"Forward fraction: {timing['fwd_fraction']:.2%}  EPEC fraction: {timing['epec_fraction']:.2%}")


if __name__ == "__main__":
    main()
