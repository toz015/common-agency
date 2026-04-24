"""
EPEC_PARM generation for HH-RLHF (3 principals: help, harm, humor).

Uses PARM's PBLoRA adapter as reward source — extracts per-objective rewards
by setting pref_vec to unit vectors (1,0,0), (0,1,0), (0,0,1), then applies
game-theoretic EPEC (Nonlinear Jacobi) aggregation.

Combines PARM's jointly-trained cross-objective rewards with equilibrium
aggregation for better multi-objective alignment.

Per token:
  1. Forward base LLM (LLaMA-2-7B-Chat) → log_base
  2. Forward TinyLLaMA (no adapter) → log_tiny_base
  3. Forward PARM with pref_vec=(1,0,0) → log_parm_help
  4. Forward PARM with pref_vec=(0,1,0) → log_parm_harm
  5. Forward PARM with pref_vec=(0,0,1) → log_parm_humor
  6. Compute implicit rewards: q_j = log_parm_j - log_tiny_base on top-k
  7. Scale by user preference α_j
  8. Solve EPEC (Nonlinear Jacobi)
  9. Greedy select from π★
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from scipy.optimize import minimize
from scipy.special import softmax
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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

def set_pref_vec(model, pref_vec):
    """Set PBLoRA pref_vec parameter to given values."""
    pref = torch.tensor(pref_vec)
    for n, p in model.named_parameters():
        if 'pref_vec' in n:
            p.data = pref.to(p.device)
            p.requires_grad = False


def load_models(args, device):
    """Load base LLM and TinyLLaMA with PBLoRA adapter."""
    # Base LLM
    base_tok = AutoTokenizer.from_pretrained(args.base)
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map=device
    )
    base_model.eval()

    # PARM model (TinyLLaMA + PBLoRA)
    parm_tok = AutoTokenizer.from_pretrained(args.parm_base)
    if parm_tok.pad_token is None:
        parm_tok.pad_token = parm_tok.eos_token
    parm_model = AutoModelForCausalLM.from_pretrained(
        args.parm_base, torch_dtype=torch.bfloat16, device_map=device
    )
    parm_model = PeftModel.from_pretrained(parm_model, args.parm_adapter)
    parm_model.eval()

    return base_model, base_tok, parm_model, parm_tok


@torch.no_grad()
def generate_ecpc(base_model, base_tok, parm_model, parm_tok,
                  prompt_text, alpha_help, alpha_harm, alpha_humor,
                  max_new_tokens=256, k=50, tau=0.1, device="cuda"):
    """Generate tokens via EPEC_PARM: PARM rewards + EPEC equilibrium."""
    base_ids = base_tok(prompt_text, return_tensors="pt").input_ids.to(device)
    parm_ids = parm_tok(prompt_text, return_tensors="pt").input_ids.to(device)

    base_cur, parm_cur = base_ids, parm_ids
    # Separate KV caches for each model state
    pkv_base = None          # LLaMA-2-7B base
    pkv_tiny_base = None     # TinyLLaMA no adapter
    pkv_parm_help = None     # PARM with pref=(1,0,0)
    pkv_parm_harm = None     # PARM with pref=(0,1,0)
    pkv_parm_humor = None    # PARM with pref=(0,0,1)

    out_ids = []
    eos_id = base_tok.eos_token_id
    fwd_time = epec_time = 0.0

    # Unit vectors for per-objective reward extraction
    # Order matches training: safe, help, humor
    pref_help = [0.0, 1.0, 0.0]   # pure helpfulness
    pref_harm = [1.0, 0.0, 0.0]   # pure harmlessness (safe)
    pref_humor = [0.0, 0.0, 1.0]  # pure humor

    for _ in range(max_new_tokens):
        t0 = time.time()

        # 1. Base LLM forward
        out = base_model(input_ids=base_cur, past_key_values=pkv_base, use_cache=True)
        log_base = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_base = out.past_key_values

        # 2. TinyLLaMA base forward (no adapter)
        with parm_model.disable_adapter():
            out = parm_model(input_ids=parm_cur, past_key_values=pkv_tiny_base, use_cache=True)
        log_tiny = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_tiny_base = out.past_key_values

        # 3. PARM with pref=(0,1,0) → helpfulness reward
        set_pref_vec(parm_model, pref_help)
        out = parm_model(input_ids=parm_cur, past_key_values=pkv_parm_help, use_cache=True)
        log_help = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_parm_help = out.past_key_values

        # 4. PARM with pref=(1,0,0) → harmlessness reward
        set_pref_vec(parm_model, pref_harm)
        out = parm_model(input_ids=parm_cur, past_key_values=pkv_parm_harm, use_cache=True)
        log_harm = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_parm_harm = out.past_key_values

        # 5. PARM with pref=(0,0,1) → humor reward
        set_pref_vec(parm_model, pref_humor)
        out = parm_model(input_ids=parm_cur, past_key_values=pkv_parm_humor, use_cache=True)
        log_humor = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_parm_humor = out.past_key_values

        fwd_time += time.time() - t0

        # 6. EPEC solve on top-k actions from base LLM
        t0 = time.time()
        topk = np.argpartition(log_base, -k)[-k:]

        # Handle vocab size mismatch
        vocab_min = min(len(log_base), len(log_tiny))
        topk_valid = topk[topk < vocab_min]
        if len(topk_valid) < len(topk):
            topk = topk_valid
        lb = log_base[topk]

        # Implicit rewards from PARM
        q_help = log_help[topk] - log_tiny[topk]
        q_harm = log_harm[topk] - log_tiny[topk]
        q_humor = log_humor[topk] - log_tiny[topk]

        # Scale by alpha and shift to non-negative
        q_help = np.clip(q_help - q_help.min(), 0.0, None) * alpha_help
        q_harm = np.clip(q_harm - q_harm.min(), 0.0, None) * alpha_harm
        q_humor = np.clip(q_humor - q_humor.min(), 0.0, None) * alpha_humor

        pi_star = epec_one_step(lb, [q_help, q_harm, q_humor], tau=tau)
        a = int(np.argmax(pi_star))
        tok_id = int(topk[a])
        epec_time += time.time() - t0

        if tok_id == eos_id:
            break
        out_ids.append(tok_id)

        base_cur = torch.tensor([[tok_id]], device=device)
        tok_text = base_tok.decode([tok_id])
        parm_cur = parm_tok(tok_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if parm_cur.numel() == 0:
            parm_cur = torch.tensor([[parm_tok.unk_token_id or 0]], device=device)

    return base_tok.decode(out_ids, skip_special_tokens=True), fwd_time, epec_time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--parm_base", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--parm_adapter", default="../training/HH-RLHF/exp")
    p.add_argument("--alpha_helpfulness", type=float, required=True)
    p.add_argument("--alpha_harmlessness", type=float, required=True)
    p.add_argument("--alpha_humor", type=float, required=True)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--datasets", default="../data/HH-RLHF/test_prompt_only.json")
    p.add_argument("--output_dir", default="./results/HH-RLHF")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda"

    model_name = f"EPEC_PARM_{args.alpha_helpfulness}help_{args.alpha_harmlessness}harm_{args.alpha_humor}humor"
    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generation.json"
    print(f"Saving to {out_path}")

    with open(args.datasets) as f:
        data = json.load(f)
    if args.limit > 0:
        data = data[:args.limit]

    base_model, base_tok, parm_model, parm_tok = load_models(args, device)
    print(f"\nModel: {model_name}, Prompts: {len(data)}")
    print(f"PARM adapter: {args.parm_adapter}")

    results = []
    tot_fwd = tot_epec = 0.0
    t0_all = time.time()
    for row in tqdm(data):
        start = time.time()
        response, fwd_t, epec_t = generate_ecpc(
            base_model, base_tok, parm_model, parm_tok,
            row["prompt"], args.alpha_helpfulness, args.alpha_harmlessness, args.alpha_humor,
            max_new_tokens=args.max_new_tokens, k=args.k, tau=args.tau, device=device,
        )
        tot_fwd += fwd_t
        tot_epec += epec_t
        results.append({
            "uid": row["uid"], "prompt": row["prompt"],
            "response": response, "model": model_name,
            "elapsed": time.time() - start,
            "fwd_time": fwd_t, "epec_time": epec_t,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total = time.time() - t0_all
    timing = {
        "method": model_name, "n_prompts": len(results),
        "total_seconds": total, "fwd_seconds": tot_fwd, "epec_seconds": tot_epec,
        "tau": args.tau, "k": args.k,
    }
    with open(out_dir / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    print(f"Done. {len(results)} prompts in {total/60:.1f} min")
    print(f"Forward: {tot_fwd:.0f}s, EPEC solve: {tot_epec:.0f}s")


if __name__ == "__main__":
    main()
