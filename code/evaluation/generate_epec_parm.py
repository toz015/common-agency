"""
EPEC_PARM generation for PKU-SafeRLHF (2 principals: help, harm).

Uses PARM's PBLoRA adapter as reward source — extracts per-objective rewards
by setting pref_vec to unit vectors (0,1) and (1,0), then applies
game-theoretic EPEC (Nonlinear Jacobi) aggregation.

Per token:
  1. Forward base (alpaca-7b, no adapter) → log_base
  2. Forward PARM with pref_vec=(0,1) → log_parm_help
  3. Forward PARM with pref_vec=(1,0) → log_parm_harm
  4. Compute implicit rewards: q_j = log_parm_j - log_base on top-k
  5. Scale by user preference α_j
  6. Solve EPEC (Nonlinear Jacobi)
  7. Greedy select from π★

Since base LLM = PARM base (both alpaca-7b-reproduced), we use a single
PeftModel and switch between disable_adapter (base) and set pref_vec
(per-objective PARM).
"""
import argparse
import json
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


# ---------- EPEC solver (k-dim, J principals) ----------

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


# ---------- Model loading ----------

def set_pref_vec(model, pref_vec):
    """Set PBLoRA pref_vec parameter to given values."""
    pref = torch.tensor(pref_vec)
    for n, p in model.named_parameters():
        if 'pref_vec' in n:
            p.data = pref.to(p.device)
            p.requires_grad = False


def load_models(args, device):
    """Load alpaca-7b with PBLoRA adapter."""
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map=device
    )
    pm = PeftModel.from_pretrained(base, args.parm_adapter)
    pm.eval()
    return pm, tok


@torch.no_grad()
def generate_epec(pm, tok, prompt_text, alpha_help, alpha_harm,
                  max_new_tokens=256, k=50, tau=0.1, device="cuda"):
    prompt = PROMPT_INPUT_ALPACA.format(input=prompt_text)
    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    cur = input_ids
    pkv_base = None
    pkv_parm_help = None
    pkv_parm_harm = None

    out_ids = []
    eos = tok.eos_token_id
    fwd_time = epec_time = 0.0
    n_tokens = 0

    # PBLoRA pref_vec order from PARM training: [safe, help]
    pref_help = [0.0, 1.0]
    pref_harm = [1.0, 0.0]

    for _ in range(max_new_tokens):
        t0 = time.time()

        # 1. Base forward (no adapter)
        with pm.disable_adapter():
            out = pm(input_ids=cur, past_key_values=pkv_base, use_cache=True)
        log_base = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_base = out.past_key_values

        # 2. PARM with pref=(0,1) → helpfulness
        set_pref_vec(pm, pref_help)
        out = pm(input_ids=cur, past_key_values=pkv_parm_help, use_cache=True)
        log_help = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_parm_help = out.past_key_values

        # 3. PARM with pref=(1,0) → harmlessness
        set_pref_vec(pm, pref_harm)
        out = pm(input_ids=cur, past_key_values=pkv_parm_harm, use_cache=True)
        log_harm = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_parm_harm = out.past_key_values

        fwd_time += time.time() - t0

        # 4. EPEC solve on top-k from base
        t0 = time.time()
        topk = np.argpartition(log_base, -k)[-k:]
        lb = log_base[topk]

        q_h = log_help[topk] - log_base[topk]
        q_s = log_harm[topk] - log_base[topk]
        q_h = np.clip(q_h - q_h.min(), 0.0, None) * alpha_help
        q_s = np.clip(q_s - q_s.min(), 0.0, None) * alpha_harm

        pi_star = epec_one_step(lb, [q_h, q_s], tau=tau)
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
    p.add_argument("--parm_adapter", default="../training/PKU-SafeRLHF/exp/final_checkpoint")
    p.add_argument("--alpha_helpfulness", type=float, required=True)
    p.add_argument("--alpha_harmlessness", type=float, required=True)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--datasets", default="../data/PKU-SafeRLHF/test_prompt_only.json")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--limit", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda"

    model_name = f"EPEC_PARM_{args.alpha_helpfulness}help_{args.alpha_harmlessness}harm_tau{args.tau}_k{args.k}"
    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generation.json"
    print(f"Saving to {out_path}")

    with open(args.datasets) as f:
        data = json.load(f)
    if args.limit > 0:
        data = data[:args.limit]

    pm, tok = load_models(args, device)
    print(f"\nModel: {model_name}, Prompts: {len(data)}")
    print(f"PARM adapter: {args.parm_adapter}")

    torch.cuda.reset_peak_memory_stats()
    results = []
    tot_fwd = tot_epec = tot_tokens = 0.0
    t0_all = time.time()
    for row in tqdm(data):
        start = time.time()
        response, fwd_t, epec_t, n_tok = generate_epec(
            pm, tok, row["prompt"],
            alpha_help=args.alpha_helpfulness, alpha_harm=args.alpha_harmlessness,
            max_new_tokens=args.max_new_tokens, k=args.k, tau=args.tau, device=device,
        )
        tot_fwd += fwd_t
        tot_epec += epec_t
        tot_tokens += n_tok
        results.append({
            "uid": row["uid"], "prompt": row["prompt"],
            "response": response, "model": model_name,
            "elapsed": time.time() - start,
            "fwd_time": fwd_t, "epec_time": epec_t, "n_tokens": n_tok,
        })

    total = time.time() - t0_all
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    timing = {
        "method": model_name, "n_prompts": len(results),
        "total_seconds": total, "mean_per_prompt": total / len(results),
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
    with open(out_dir / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    print(f"Done. {len(results)} prompts in {total/60:.1f} min")
    print(f"Peak GPU mem: {peak_mem_gb:.2f} GB")
    print(f"Forward: {tot_fwd/(tot_fwd+tot_epec):.1%}, EPEC: {tot_epec/(tot_fwd+tot_epec):.1%}")


if __name__ == "__main__":
    main()