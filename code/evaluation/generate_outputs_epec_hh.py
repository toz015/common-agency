"""
EPEC (Common-Agency) generation for HH-RLHF (3 principals: help, harm, humor).

Per token: forward base LLM + 3 ARM forwards (TinyLLaMA with adapter switching),
compute implicit rewards q^j = log π_ARM_j - log π_base on top-k actions,
solve the 3-principal Common-Agency EPEC via Nonlinear Jacobi iteration,
then greedy-sample from π★.
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

def load_models(args, device):
    """Load base LLM and TinyLLaMA with 3 adapters."""
    base_tok = AutoTokenizer.from_pretrained(args.base)
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map=device
    )
    base_model.eval()

    arm_tok = AutoTokenizer.from_pretrained(args.arm_base)
    if arm_tok.pad_token is None:
        arm_tok.pad_token = arm_tok.eos_token
    arm_model = AutoModelForCausalLM.from_pretrained(
        args.arm_base, torch_dtype=torch.bfloat16, device_map=device
    )
    arm_model = PeftModel.from_pretrained(arm_model, args.help_adapter, adapter_name="help")
    arm_model.load_adapter(args.harm_adapter, adapter_name="harm")
    arm_model.load_adapter(args.humor_adapter, adapter_name="humor")
    arm_model.eval()

    return base_model, base_tok, arm_model, arm_tok


@torch.no_grad()
def generate_epec(base_model, base_tok, arm_model, arm_tok,
                  prompt_text, alpha_help, alpha_harm, alpha_humor,
                  max_new_tokens=256, k=50, tau=0.1, device="cuda"):
    """Generate tokens via EPEC equilibrium decoding."""
    base_ids = base_tok(prompt_text, return_tensors="pt").input_ids.to(device)
    arm_ids = arm_tok(prompt_text, return_tensors="pt").input_ids.to(device)

    base_cur, arm_cur = base_ids, arm_ids
    pkv_base = pkv_arm_base = pkv_help = pkv_harm = pkv_humor = None
    out_ids = []
    eos_id = base_tok.eos_token_id
    fwd_time = epec_time = 0.0

    for _ in range(max_new_tokens):
        t0 = time.time()

        # Base LLM forward
        out = base_model(input_ids=base_cur, past_key_values=pkv_base, use_cache=True)
        log_base = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_base = out.past_key_values

        # ARM base forward (no adapter)
        with arm_model.disable_adapter():
            out = arm_model(input_ids=arm_cur, past_key_values=pkv_arm_base, use_cache=True)
        log_arm_base = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_arm_base = out.past_key_values

        # ARM help
        arm_model.set_adapter("help")
        out = arm_model(input_ids=arm_cur, past_key_values=pkv_help, use_cache=True)
        log_help = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_help = out.past_key_values

        # ARM harm
        arm_model.set_adapter("harm")
        out = arm_model(input_ids=arm_cur, past_key_values=pkv_harm, use_cache=True)
        log_harm = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_harm = out.past_key_values

        # ARM humor
        arm_model.set_adapter("humor")
        out = arm_model(input_ids=arm_cur, past_key_values=pkv_humor, use_cache=True)
        log_humor = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
        pkv_humor = out.past_key_values

        fwd_time += time.time() - t0

        # EPEC solve on top-k actions from base LLM
        t0 = time.time()
        topk = np.argpartition(log_base, -k)[-k:]
        lb = log_base[topk]

        # Implicit rewards (ARM vocab may differ from base vocab)
        vocab_min = min(len(log_base), len(log_arm_base))
        topk_valid = topk[topk < vocab_min]
        if len(topk_valid) < len(topk):
            topk = topk_valid
            lb = log_base[topk]

        q_help = (log_help[topk] - log_arm_base[topk])
        q_harm = (log_harm[topk] - log_arm_base[topk])
        q_humor = (log_humor[topk] - log_arm_base[topk])

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
        arm_cur = arm_tok(tok_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        if arm_cur.numel() == 0:
            arm_cur = torch.tensor([[arm_tok.unk_token_id or 0]], device=device)

    return base_tok.decode(out_ids, skip_special_tokens=True), fwd_time, epec_time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument("--arm_base", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--help_adapter", default="../training/HH-RLHF/exp_genarm_help")
    p.add_argument("--harm_adapter", default="../training/HH-RLHF/exp_genarm_harm")
    p.add_argument("--humor_adapter", default="../training/HH-RLHF/exp_genarm_humor")
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

    model_name = f"EPEC_{args.alpha_helpfulness}help_{args.alpha_harmlessness}harm_{args.alpha_humor}humor"
    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generation.json"
    print(f"Saving to {out_path}")

    with open(args.datasets) as f:
        data = json.load(f)
    if args.limit > 0:
        data = data[:args.limit]

    base_model, base_tok, arm_model, arm_tok = load_models(args, device)
    print(f"\nModel: {model_name}, Prompts: {len(data)}")

    results = []
    tot_fwd = tot_epec = 0.0
    t0_all = time.time()
    for row in tqdm(data):
        start = time.time()
        response, fwd_t, epec_t = generate_epec(
            base_model, base_tok, arm_model, arm_tok,
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
