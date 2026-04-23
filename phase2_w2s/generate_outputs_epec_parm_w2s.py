"""
EPEC-PARM, weak-to-strong variant.

Token-level Common-Agency EPEC decoding with:
  - strong 65B-4bit GPTQ base (π_base)
  - weak 7B fp16 backbone (alpaca-7b-reproduced), serving BOTH as the
    reference for the implicit reward (via pm.disable_adapter()) and as
    the trunk onto which PARM's PBLoRA is attached
  - a single PARM PBLoRA adapter whose `pref_vec` parameter is mutated
    between forwards to extract per-objective rewards

Convention: our SafeRLHF PBLoRA's adapter_config.json uses
`pref_vec_init = [alpha_harmlessness, alpha_helpfulness]` (index 0 = harm,
index 1 = help). Per-objective rewards therefore use unit vectors
  pref_harm = [1.0, 0.0]   pref_help = [0.0, 1.0]

Per decoding step (4 forwards):
  1. 65B GPTQ base                                          → log_b       (π_base)
  2. 7B backbone, pm.disable_adapter()                      → log_tiny    (reference)
  3. 7B backbone + PBLoRA, pref_vec=[1,0]                   → log_harm_arm
  4. 7B backbone + PBLoRA, pref_vec=[0,1]                   → log_help_arm

Top-k (default 50) actions come from log_b. Per-objective q_j is
  q_help = clip(log_help_arm[topk] − log_tiny[topk] − min, 0, None) * alpha_help
  q_harm = clip(log_harm_arm[topk] − log_tiny[topk] − min, 0, None) * alpha_harm
which enter epec_one_step(log_b[topk], [q_help, q_harm], tau); π★ → argmax.

Replaces the linear logit-sum aggregator from generate_outputs.py
(model_arithmetic formula M_base + M_reward with pref_vec_init baked in)
with the EPEC equilibrium aggregator, same base + same PBLoRA.
"""
import argparse
import json
import os
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

PROMPT_TEMPLATE_BASE_65B = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{input}\n\n### Response:\n"
)
PROMPT_TEMPLATE_ARM_7B = "BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:"


# =============================================================================
# EPEC solver — ported verbatim from fresh-start/generate_outputs_epec_genarm.py
# =============================================================================

def best_response(log_pi_base, Y, tau):
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


# =============================================================================
# PBLoRA pref_vec mutation.
# =============================================================================

def set_pref_vec(model, pref_vec):
    """Overwrite the PBLoRA `pref_vec` parameter(s) in-place."""
    pref = torch.tensor(pref_vec)
    for n, p in model.named_parameters():
        if "pref_vec" in n:
            p.data = pref.to(p.device, dtype=p.dtype)
            p.requires_grad = False


# =============================================================================
# Model loading.
# =============================================================================

def load_base_65b(base_path):
    tok = AutoTokenizer.from_pretrained(base_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_path, device_map="auto", trust_remote_code=False,
        )
    except Exception as e:
        print(f"[base-65B] AutoModel path failed ({type(e).__name__}: {e}); "
              f"falling back to AutoGPTQForCausalLM.from_quantized")
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(
            base_path, device="cuda:0", use_safetensors=True,
            trust_remote_code=False, use_triton=False,
        )
    model.eval()
    return model, tok


def load_parm(backbone_path, parm_adapter_path, device):
    """7B backbone fp16 + PBLoRA adapter (single PeftModel).

    We copy the adapter to a cache dir with pref_vec_init=[0,0] as a sentinel —
    it's mutated at runtime by set_pref_vec() anyway. This mirrors the baseline
    generate_outputs.py pattern (which also copies + rewrites adapter_config).
    """
    tok = AutoTokenizer.from_pretrained(backbone_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        backbone_path, torch_dtype=torch.bfloat16, device_map=device,
    )
    pm = PeftModel.from_pretrained(backbone, parm_adapter_path)
    pm.eval()
    return pm, tok


def stage_parm_cache(parm_adapter_path, cache_root, model_name):
    """Copy the PBLoRA adapter to a private cache dir and set a placeholder
    pref_vec_init (we'll overwrite it via set_pref_vec each step)."""
    cache_path = Path(cache_root) / model_name
    cache_path.mkdir(parents=True, exist_ok=True)
    with open(Path(parm_adapter_path) / "adapter_config.json") as f:
        config = json.load(f)
    # Placeholder; runtime mutation overwrites the pref_vec parameter directly.
    config["pref_vec_init"] = [0.0, 0.0]
    with open(cache_path / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    shutil.copyfile(
        Path(parm_adapter_path) / "adapter_model.safetensors",
        cache_path / "adapter_model.safetensors",
    )
    return str(cache_path)


# =============================================================================
# Per-token forwards.
# =============================================================================

@torch.no_grad()
def forward_base_65b(base_model, input_ids, pkv):
    out = base_model(input_ids=input_ids, past_key_values=pkv, use_cache=True)
    logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
    return logp, out.past_key_values


@torch.no_grad()
def forward_parm(pm, input_ids, pkv, pref_vec_or_none):
    """pref_vec_or_none ∈ {None, list[float]}. None ⇒ disable adapter (tiny backbone)."""
    if pref_vec_or_none is None:
        with pm.disable_adapter():
            out = pm(input_ids=input_ids, past_key_values=pkv, use_cache=True)
    else:
        set_pref_vec(pm, pref_vec_or_none)
        out = pm(input_ids=input_ids, past_key_values=pkv, use_cache=True)
    logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
    return logp, out.past_key_values


# =============================================================================
# Generation loop.
# =============================================================================

@torch.no_grad()
def generate_epec(base_model, base_tok, pm, arm_tok,
                  prompt_text, alpha_help, alpha_harm,
                  max_new_tokens=512, k=50, tau=0.1,
                  eps=1e-3, max_iter=5, device="cuda", debug_first_n=0):
    base_prompt = PROMPT_TEMPLATE_BASE_65B.format(input=prompt_text)
    arm_prompt  = PROMPT_TEMPLATE_ARM_7B.format(input=prompt_text)
    base_ids = base_tok(base_prompt, return_tensors="pt").input_ids.to(device)
    arm_ids  = arm_tok(arm_prompt,   return_tensors="pt").input_ids.to(device)

    base_cur, arm_cur = base_ids, arm_ids
    pkv_base = pkv_tiny = pkv_help = pkv_harm = None

    out_ids = []
    eos_id = base_tok.eos_token_id
    fwd_time = epec_time = 0.0

    # Convention: pref_vec = [harm, help].
    PREF_HARM = [1.0, 0.0]
    PREF_HELP = [0.0, 1.0]

    for step in range(max_new_tokens):
        t0 = time.time()

        log_b,         pkv_base = forward_base_65b(base_model, base_cur, pkv_base)
        log_tiny,      pkv_tiny = forward_parm(pm, arm_cur, pkv_tiny, pref_vec_or_none=None)
        log_harm_arm,  pkv_harm = forward_parm(pm, arm_cur, pkv_harm, pref_vec_or_none=PREF_HARM)
        log_help_arm,  pkv_help = forward_parm(pm, arm_cur, pkv_help, pref_vec_or_none=PREF_HELP)

        fwd_time += time.time() - t0
        t0 = time.time()

        topk = np.argpartition(log_b, -k)[-k:]
        vocab_min = min(len(log_b), len(log_tiny), len(log_help_arm), len(log_harm_arm))
        topk = topk[topk < vocab_min]
        lb = log_b[topk]

        # Implicit rewards referenced against the 7B backbone (log_tiny).
        q_help = log_help_arm[topk] - log_tiny[topk]
        q_harm = log_harm_arm[topk] - log_tiny[topk]
        q_help = np.clip(q_help - q_help.min(), 0.0, None) * alpha_help
        q_harm = np.clip(q_harm - q_harm.min(), 0.0, None) * alpha_harm

        pi_star = epec_one_step(lb, [q_help, q_harm],
                                tau=tau, eps=eps, max_iter=max_iter)
        a = int(np.argmax(pi_star))
        tok_id = int(topk[a])
        epec_time += time.time() - t0

        if debug_first_n and step < debug_first_n:
            print(f"[dbg step={step}] lb[max]={lb.max():+.3f} "
                  f"q_h[max]={q_help.max():+.3f} q_s[max]={q_harm.max():+.3f} "
                  f"pi*_max={pi_star.max():.4f} "
                  f"pi*_finite={np.isfinite(pi_star).all()} "
                  f"tok={tok_id!r}→{base_tok.decode([tok_id])!r}")

        if tok_id == eos_id:
            break
        out_ids.append(tok_id)

        base_cur = torch.tensor([[tok_id]], device=device)
        arm_cur  = torch.tensor([[tok_id]], device=device)

    return base_tok.decode(out_ids, skip_special_tokens=True), fwd_time, epec_time, len(out_ids)


# =============================================================================
# CLI + main loop.
# =============================================================================

def str2bool(s):
    if s.lower() in {"1", "true", "t", "yes", "y", "on"}:  return True
    if s.lower() in {"0", "false", "f", "no", "n", "off"}: return False
    return bool(s)


def parse_args():
    p = argparse.ArgumentParser(
        description="EPEC-PARM W2S (65B GPTQ base + 7B backbone + PBLoRA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_base_name_or_path", default="TheBloke/alpaca-lora-65B-GPTQ")
    p.add_argument("--model_backbone_name_or_path", default="PKU-Alignment/alpaca-7b-reproduced",
                   help="Weak 7B backbone the PBLoRA was trained on.")
    p.add_argument("--model_parm_both_name_or_path", required=True,
                   help="Path to the PARM PBLoRA adapter dir.")
    p.add_argument("--alpha_helpfulness", type=float, required=True)
    p.add_argument("--alpha_harmlessness", type=float, required=True)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--max_iter", type=int, default=5)
    p.add_argument("--datasets", default="../data/test_prompt_only.json")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--cache_dir", default="./cache")
    p.add_argument("--resume", type=str2bool, default=True)
    p.add_argument("--checkpoint_every", type=int, default=10)
    p.add_argument("--debug_first_n", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda"

    model_name = (f"PARM_EPEC_{args.alpha_helpfulness}help_"
                  f"{args.alpha_harmlessness}harm_tau{args.tau}_k{args.k}")
    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generation.json"
    print(f"Generation results → {out_path}")

    with open(args.datasets) as f:
        data = json.load(f)

    # Resume.
    output_set, done = [], set()
    if args.resume and out_path.exists():
        try:
            output_set = json.load(open(out_path))
            done = {r["uid"] for r in output_set}
            print(f"[resume] {len(output_set)} records, {len(done)} unique uids already done")
        except Exception as e:
            print(f"[resume] could not load {out_path}: {e}; starting fresh")
            output_set, done = [], set()

    todo = [d for d in data if d["uid"] not in done]
    if not todo:
        print(f"[resume] all {len(data)} prompts already done — nothing to do.")
        return
    print(f"[resume] {len(todo)} / {len(data)} prompts remaining")

    # Stage PBLoRA adapter to cache (matches baseline generate_outputs.py pattern).
    parm_cache = stage_parm_cache(args.model_parm_both_name_or_path,
                                  args.cache_dir, model_name)
    print(f"PBLoRA cache: {parm_cache}")

    # Load models.
    print(f"Loading 65B base: {args.model_base_name_or_path}")
    base_model, base_tok = load_base_65b(args.model_base_name_or_path)
    print(f"Loading 7B backbone + PBLoRA: {args.model_backbone_name_or_path}")
    pm, arm_tok = load_parm(args.model_backbone_name_or_path, parm_cache, device)

    torch.cuda.reset_peak_memory_stats()
    tot_fwd = tot_epec = tot_tokens = 0.0
    t_script = time.time()
    for i, row in enumerate(tqdm(todo)):
        t0 = time.time()
        resp, fwd_t, epec_t, n_tok = generate_epec(
            base_model, base_tok, pm, arm_tok,
            row["prompt"],
            alpha_help=args.alpha_helpfulness,
            alpha_harm=args.alpha_harmlessness,
            max_new_tokens=args.max_new_tokens,
            k=args.k, tau=args.tau,
            eps=args.eps, max_iter=args.max_iter,
            device=device,
            debug_first_n=args.debug_first_n if i == 0 else 0,
        )
        tot_fwd += fwd_t; tot_epec += epec_t; tot_tokens += n_tok
        output_set.append({
            "uid": row["uid"], "prompt": row["prompt"], "response": resp,
            "model": model_name, "elapsed": time.time() - t0,
            "fwd_time": fwd_t, "epec_time": epec_t, "n_tokens": n_tok,
        })
        if (i + 1) % args.checkpoint_every == 0 or (i + 1) == len(todo):
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output_set, f, ensure_ascii=False, indent=2)
            print(f"[ckpt] {len(output_set)} / {len(data)} → {out_path}")

    total = time.time() - t_script
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    timing = {
        "method": model_name,
        "n_prompts_done_this_run": len(todo),
        "n_prompts_total": len(data),
        "total_seconds": total,
        "fwd_seconds": tot_fwd, "epec_seconds": tot_epec,
        "total_tokens": int(tot_tokens),
        "sec_per_token": (tot_fwd + tot_epec) / max(tot_tokens, 1),
        "peak_gpu_memory_gb": peak_gb,
        "tau": args.tau, "k": args.k,
        "alpha_helpfulness": args.alpha_helpfulness,
        "alpha_harmlessness": args.alpha_harmlessness,
        "max_new_tokens": args.max_new_tokens,
    }
    json.dump(timing, open(out_dir / "timing.json", "w"), indent=2)
    print(f"Done. {len(todo)} new prompts in {total/60:.1f} min. "
          f"peak_mem={peak_gb:.1f} GB  fwd/epec = {tot_fwd:.0f}s / {tot_epec:.0f}s")


if __name__ == "__main__":
    main()
