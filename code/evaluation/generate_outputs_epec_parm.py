"""
EPEC_PARM generation for PKU-SafeRLHF (2 principals: help, safe).

This script uses a PKU-trained PARM / PBLoRA adapter as the reward source.
For each token:
  1. Forward base LLM (Alpaca-7B) -> log_base
  2. Forward PARM base with adapter disabled -> log_parm_base
  3. Forward PARM with pref_vec for helpfulness -> log_help
  4. Forward PARM with pref_vec for safety -> log_safe
  5. Compute implicit rewards on top-k actions:
         q_help = log_help - log_parm_base
         q_safe = log_safe - log_parm_base
  6. Scale by user preferences alpha_help / alpha_safe
  7. Solve Common-Agency EPEC (Nonlinear Jacobi)
  8. Greedy select from pi_star

Compared with the HH-RLHF 3-objective version:
  - remove humor
  - use PKU dataset
  - use Alpaca-7B family for both base and PARM base
  - use PKU prompt template
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


# ============================================================
# EPEC solver (k-dim, J principals)
# ============================================================

def best_response(log_pi_base: np.ndarray, Y: np.ndarray, tau: float) -> np.ndarray:
    return softmax(log_pi_base + Y / tau)


def solve_principal(i, q, y, log_pi_base, tau):
    """
    Best response of principal i given others' incentives.
    Feasible set: 0 <= y_i(a) <= q_i(a) for each action a.
    """
    qi = q[i]
    y_other = sum(y[j] for j in range(len(y)) if j != i)

    def neg_obj(yi):
        Y = y_other + yi
        pi = best_response(log_pi_base, Y, tau)
        return -float(pi @ (qi - yi))

    bounds = [(0.0, max(float(qi[a]), 0.0)) for a in range(len(qi))]
    upper = np.array([b[1] for b in bounds], dtype=np.float64)
    x0 = np.clip(y[i], 0.0, upper)

    try:
        res = minimize(
            neg_obj,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 8, "ftol": 1e-4},
        )
        return res.x
    except ValueError:
        return x0


def epec_one_step(log_pi_base, q_list, tau=0.1, eps=1e-3, max_iter=5):
    """
    Solve EPEC for J principals via Nonlinear Jacobi.
    Returns equilibrium policy pi_star on the restricted action space.
    """
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


# ============================================================
# PARM preference-vector interface
# ============================================================

def set_pref_vec(model, pref_vec):
    """
    Set PBLoRA pref_vec parameter to the given values.
    """
    pref = torch.tensor(pref_vec, dtype=torch.float32)
    found = False
    for name, param in model.named_parameters():
        if "pref_vec" in name:
            param.data = pref.to(param.device, dtype=param.dtype)
            param.requires_grad = False
            found = True
    if not found:
        raise ValueError("No parameter containing 'pref_vec' was found in the PARM model.")


# ============================================================
# Model loading
# ============================================================

def load_models(args, device):
    """
    Load:
      - base generation model
      - PARM base model + PKU-trained PBLoRA adapter
    For PKU, both are Alpaca-7B family to keep tokenizer / vocab aligned.
    """
    # Base generation model
    base_tok = AutoTokenizer.from_pretrained(args.base)
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    base_model.eval()

    # PARM model
    parm_tok = AutoTokenizer.from_pretrained(args.parm_base)
    if parm_tok.pad_token is None:
        parm_tok.pad_token = parm_tok.eos_token

    parm_model = AutoModelForCausalLM.from_pretrained(
        args.parm_base,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    parm_model = PeftModel.from_pretrained(parm_model, args.parm_adapter)
    parm_model.eval()

    return base_model, base_tok, parm_model, parm_tok


# ============================================================
# Token-level generation
# ============================================================

@torch.no_grad()
def generate_epec_parm_pku(
    base_model,
    base_tok,
    parm_model,
    parm_tok,
    prompt_text,
    alpha_help,
    alpha_safe,
    max_new_tokens=256,
    k=50,
    tau=0.1,
    device="cuda",
):
    """
    Generate a response via EPEC_PARM on PKU-SafeRLHF.
    """
    prompt = PROMPT_INPUT_ALPACA.format(input=prompt_text)

    base_ids = base_tok(prompt, return_tensors="pt").input_ids.to(device)
    parm_ids = parm_tok(prompt, return_tensors="pt").input_ids.to(device)

    base_cur = base_ids
    parm_cur = parm_ids

    pkv_base = None
    pkv_parm_base = None
    pkv_parm_help = None
    pkv_parm_safe = None

    out_ids = []
    eos_id = base_tok.eos_token_id

    fwd_time = 0.0
    epec_time = 0.0
    n_tokens = 0

    # IMPORTANT:
    # Confirm the order of pref_vec in your PKU PARM checkpoint.
    # Here we assume the PKU training order is [safe, help].
    pref_help = [0.0, 1.0]
    pref_safe = [1.0, 0.0]

    for _ in range(max_new_tokens):
        t0 = time.time()

        # 1. Base forward
        out = base_model(input_ids=base_cur, past_key_values=pkv_base, use_cache=True)
        log_base = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_base = out.past_key_values

        # 2. PARM base forward (adapter disabled)
        with parm_model.disable_adapter():
            out = parm_model(input_ids=parm_cur, past_key_values=pkv_parm_base, use_cache=True)
        log_parm_base = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_parm_base = out.past_key_values

        # 3. PARM helpfulness forward
        set_pref_vec(parm_model, pref_help)
        out = parm_model(input_ids=parm_cur, past_key_values=pkv_parm_help, use_cache=True)
        log_help = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_parm_help = out.past_key_values

        # 4. PARM safety forward
        set_pref_vec(parm_model, pref_safe)
        out = parm_model(input_ids=parm_cur, past_key_values=pkv_parm_safe, use_cache=True)
        log_safe = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_parm_safe = out.past_key_values

        fwd_time += time.time() - t0

        # 5. Restrict to top-k actions from base model
        t0 = time.time()

        k_eff = min(k, len(log_base))
        topk = np.argpartition(log_base, -k_eff)[-k_eff:]
        topk = topk[np.argsort(log_base[topk])[::-1]]

        # Since PKU version should use aligned Alpaca tokenizers, the vocab ids should match.
        # Still guard against rare mismatch.
        vocab_min = min(len(log_base), len(log_parm_base), len(log_help), len(log_safe))
        topk = topk[topk < vocab_min]
        if len(topk) == 0:
            break

        lb = log_base[topk]

        # 6. Implicit rewards from PARM
        q_help = log_help[topk] - log_parm_base[topk]
        q_safe = log_safe[topk] - log_parm_base[topk]

        # Shift to non-negative, then scale
        q_help = np.clip(q_help - q_help.min(), 0.0, None) * alpha_help
        q_safe = np.clip(q_safe - q_safe.min(), 0.0, None) * alpha_safe

        # 7. Solve EPEC
        pi_star = epec_one_step(lb, [q_help, q_safe], tau=tau)

        # 8. Greedy decode from equilibrium policy
        a = int(np.argmax(pi_star))
        tok_id = int(topk[a])

        epec_time += time.time() - t0
        n_tokens += 1

        if tok_id == eos_id:
            break

        out_ids.append(tok_id)

        # advance one token
        base_cur = torch.tensor([[tok_id]], device=device)
        parm_cur = torch.tensor([[tok_id]], device=device)

    response = base_tok.decode(out_ids, skip_special_tokens=True)
    return response, fwd_time, epec_time, n_tokens


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    # PKU generation base
    p.add_argument("--base", default="PKU-Alignment/alpaca-7b-reproduced")

    # PKU PARM base + adapter
    p.add_argument("--parm_base", default="PKU-Alignment/alpaca-7b-reproduced")
    p.add_argument("--parm_adapter", default="../training/PKU-SafeRLHF/exp/final_checkpoint")

    # two-objective weights
    p.add_argument("--alpha_helpfulness", type=float, required=True)
    p.add_argument("--alpha_harmlessness", type=float, required=True)

    # decoding / equilibrium
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=256)

    # data / output
    p.add_argument("--datasets", default="../data/PKU-SafeRLHF/test_prompt_only.json")
    p.add_argument("--output_dir", default="./results")

    # runtime
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda")

    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    device = args.device

    model_name = (
        f"EPEC_PARM_"
        f"{args.alpha_helpfulness}help_"
        f"{args.alpha_harmlessness}harm_"
        f"tau{args.tau}_k{args.k}"
    )

    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generation.json"

    print(f"Saving to {out_path}")

    with open(args.datasets, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit > 0:
        data = data[:args.limit]

    base_model, base_tok, parm_model, parm_tok = load_models(args, device)

    print(f"\nModel: {model_name}")
    print(f"Prompts: {len(data)}")
    print(f"PARM adapter: {args.parm_adapter}")

    results = []
    tot_fwd = 0.0
    tot_epec = 0.0
    tot_tokens = 0
    t0_all = time.time()

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for row in tqdm(data):
        start = time.time()

        response, fwd_t, epec_t, n_tok = generate_epec_parm_pku(
            base_model=base_model,
            base_tok=base_tok,
            parm_model=parm_model,
            parm_tok=parm_tok,
            prompt_text=row["prompt"],
            alpha_help=args.alpha_helpfulness,
            alpha_safe=args.alpha_harmlessness,
            max_new_tokens=args.max_new_tokens,
            k=args.k,
            tau=args.tau,
            device=device,
        )

        tot_fwd += fwd_t
        tot_epec += epec_t
        tot_tokens += n_tok

        results.append(
            {
                "uid": row["uid"],
                "prompt": row["prompt"],
                "response": response,
                "model": model_name,
                "elapsed": time.time() - start,
                "fwd_time": fwd_t,
                "epec_time": epec_t,
                "n_tokens": n_tok,
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total = time.time() - t0_all
    peak_mem_gb = None
    if str(device).startswith("cuda") and torch.cuda.is_available():
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    denom = max(tot_fwd + tot_epec, 1e-12)
    timing = {
        "method": model_name,
        "n_prompts": len(results),
        "total_seconds": total,
        "mean_per_prompt": total / max(len(results), 1),
        "fwd_seconds": tot_fwd,
        "epec_seconds": tot_epec,
        "fwd_fraction": tot_fwd / denom,
        "epec_fraction": tot_epec / denom,
        "total_tokens": int(tot_tokens),
        "sec_per_token": (tot_fwd + tot_epec) / max(tot_tokens, 1),
        "peak_gpu_memory_gb": peak_mem_gb,
        "tau": args.tau,
        "k": args.k,
        "alpha_helpfulness": args.alpha_helpfulness,
        "alpha_harmlessness": args.alpha_harmlessness,
        "max_new_tokens": args.max_new_tokens,
        "datasets": args.datasets,
        "parm_adapter": args.parm_adapter,
    }

    with open(out_dir / "timing.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)

    print(f"Done. {len(results)} prompts in {total / 60:.1f} min")
    print(f"Forward: {tot_fwd:.1f}s, EPEC solve: {tot_epec:.1f}s")
    if peak_mem_gb is not None:
        print(f"Peak GPU mem: {peak_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()