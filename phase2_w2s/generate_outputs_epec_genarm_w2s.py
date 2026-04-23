"""
EPEC-GenARM, weak-to-strong variant.

Token-level Common-Agency EPEC decoding with:
  - strong 65B-4bit GPTQ base (π_base)
  - weak 7B fp16 backbone (alpaca-7b-reproduced), which serves as the
    reference for the implicit reward q_j = log π_ARM_j − log π_backbone
  - two LoRA adapters (help, harm) trained on the 7B backbone, loaded onto
    a SHARED backbone via PeftModel.load_adapter / pm.set_adapter

Per decoding step (4 forwards):
  1. 65B GPTQ base                         → log_b       (π_base)
  2. 7B backbone, pm.disable_adapter()     → log_tiny    (reference for q_j)
  3. 7B backbone + LoRA_help  (set_adapter)→ log_h
  4. 7B backbone + LoRA_harm  (set_adapter)→ log_s

Top-k (default 50) actions come from log_b. Per-objective q_j is
clip(log_ARM_j[topk] − log_tiny[topk] − min, 0, None) * alpha_j; these
enter epec_one_step(log_b[topk], [q_help, q_harm], tau) and π★ is argmax'd
to pick the next token.

Replaces the linear logit-sum aggregator from generate_outputs_genarm.py
(model_arithmetic) with the EPEC equilibrium aggregator, on the SAME
base + ARM stack, for direct apples-to-apples comparison.
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

# Prompt templates — base (65B alpaca) uses Stanford-Alpaca instruction format;
# ARM (7B alpaca-7b-reproduced) uses PKU-SafeRLHF Beaver conversation format.
# (Matches phase2_w2s/generate_outputs_genarm.py and generate_outputs_8bit.py.)
PROMPT_TEMPLATE_BASE_65B = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{input}\n\n### Response:\n"
)
PROMPT_TEMPLATE_ARM_7B = "BEGINNING OF CONVERSATION: USER: {input} ASSISTANT:"


# =============================================================================
# EPEC solver (k-dim, 2 principals) — ported verbatim from fresh-start.
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
# Model loading — 65B GPTQ base + 7B backbone with two adapters.
# =============================================================================

def load_base_65b(base_path):
    """Load the 65B GPTQ quantized base.

    transformers 4.36.2 auto-detects GPTQ from the checkpoint's
    quantization_config and routes through auto_gptq. If that path
    fails we fall back to AutoGPTQForCausalLM.from_quantized.
    """
    tok = AutoTokenizer.from_pretrained(base_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            device_map="auto",
            trust_remote_code=False,
        )
    except Exception as e:
        print(f"[base-65B] AutoModel path failed ({type(e).__name__}: {e}); "
              f"falling back to AutoGPTQForCausalLM.from_quantized")
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(
            base_path,
            device="cuda:0",
            use_safetensors=True,
            trust_remote_code=False,
            use_triton=False,
        )
    model.eval()
    return model, tok


def load_backbone_with_adapters(backbone_path, help_adapter, harm_adapter, device):
    """Load 7B backbone in fp16 and attach both LoRA adapters onto it.

    Loaded as ONE PeftModel, adapters switched in-place via
    pm.set_adapter(name) / pm.disable_adapter(). This collapses the
    backbone weights to a single copy (~14 GB) instead of duplicating.
    """
    tok = AutoTokenizer.from_pretrained(backbone_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        backbone_path, torch_dtype=torch.bfloat16, device_map=device,
    )
    pm = PeftModel.from_pretrained(backbone, help_adapter, adapter_name="help")
    pm.load_adapter(harm_adapter, adapter_name="harm")
    pm.eval()
    return pm, tok


# =============================================================================
# Per-token forwards.
# =============================================================================

@torch.no_grad()
def forward_base_65b(base_model, input_ids, pkv):
    out = base_model(input_ids=input_ids, past_key_values=pkv, use_cache=True)
    logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1).cpu().numpy()
    return logp, out.past_key_values


@torch.no_grad()
def forward_arm_7b(pm, input_ids, pkv, adapter):
    """adapter ∈ {None, 'help', 'harm'}.  None ⇒ pm.disable_adapter()."""
    if adapter is None:
        with pm.disable_adapter():
            out = pm(input_ids=input_ids, past_key_values=pkv, use_cache=True)
    else:
        pm.set_adapter(adapter)
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
    # Encode prompt with each template/tokenizer.
    base_prompt = PROMPT_TEMPLATE_BASE_65B.format(input=prompt_text)
    arm_prompt  = PROMPT_TEMPLATE_ARM_7B.format(input=prompt_text)
    base_ids = base_tok(base_prompt, return_tensors="pt").input_ids.to(device)
    arm_ids  = arm_tok(arm_prompt,   return_tensors="pt").input_ids.to(device)

    base_cur, arm_cur = base_ids, arm_ids
    pkv_base = pkv_tiny = pkv_help = pkv_harm = None

    out_ids = []
    eos_id = base_tok.eos_token_id
    fwd_time = epec_time = 0.0

    for step in range(max_new_tokens):
        t0 = time.time()

        log_b,    pkv_base = forward_base_65b(base_model, base_cur,  pkv_base)
        log_tiny, pkv_tiny = forward_arm_7b(pm, arm_cur,  pkv_tiny, adapter=None)
        log_h,    pkv_help = forward_arm_7b(pm, arm_cur,  pkv_help, adapter="help")
        log_s,    pkv_harm = forward_arm_7b(pm, arm_cur,  pkv_harm, adapter="harm")

        fwd_time += time.time() - t0
        t0 = time.time()

        # Top-k actions from 65B base logits.
        topk = np.argpartition(log_b, -k)[-k:]
        # Vocab-mismatch guard (defensive — should be a no-op for LLaMA ↔ LLaMA).
        vocab_min = min(len(log_b), len(log_tiny), len(log_h), len(log_s))
        topk = topk[topk < vocab_min]
        lb = log_b[topk]

        # Implicit per-objective rewards from 7B ARMs, referenced against the
        # 7B backbone (log_tiny). NOT against log_b — log_b is 65B, which lives
        # in a different space; using it here would conflate policies.
        q_help = log_h[topk] - log_tiny[topk]
        q_harm = log_s[topk] - log_tiny[topk]
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
                  f"pi*_nanOrinf={not np.isfinite(pi_star).all()} "
                  f"tok={tok_id!r}→{base_tok.decode([tok_id])!r}")

        if tok_id == eos_id:
            break
        out_ids.append(tok_id)

        base_cur = torch.tensor([[tok_id]], device=device)
        arm_cur  = torch.tensor([[tok_id]], device=device)

    return base_tok.decode(out_ids, skip_special_tokens=True), fwd_time, epec_time, len(out_ids)


# =============================================================================
# CLI + main loop (resume / checkpoint).
# =============================================================================

def str2bool(s):
    if s.lower() in {"1", "true", "t", "yes", "y", "on"}:  return True
    if s.lower() in {"0", "false", "f", "no", "n", "off"}: return False
    return bool(s)


def parse_args():
    p = argparse.ArgumentParser(
        description="EPEC-GenARM W2S (65B GPTQ base + two 7B LoRA ARMs).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_base_name_or_path", default="TheBloke/alpaca-lora-65B-GPTQ")
    p.add_argument("--model_backbone_name_or_path", default="PKU-Alignment/alpaca-7b-reproduced",
                   help="Weak 7B backbone the ARMs were fine-tuned on.")
    p.add_argument("--model_arm_help_path", required=True)
    p.add_argument("--model_arm_harm_path", required=True)
    p.add_argument("--alpha_helpfulness", type=float, required=True)
    p.add_argument("--alpha_harmlessness", type=float, required=True)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--max_iter", type=int, default=5)
    p.add_argument("--datasets", default="../data/test_prompt_only.json")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--resume", type=str2bool, default=True)
    p.add_argument("--checkpoint_every", type=int, default=10)
    p.add_argument("--debug_first_n", type=int, default=0,
                   help="Print pi_star debug for the first N tokens of the first prompt.")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda"

    model_name = (f"GenARM_EPEC_{args.alpha_helpfulness}help_"
                  f"{args.alpha_harmlessness}harm_tau{args.tau}_k{args.k}")
    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generation.json"
    print(f"Generation results → {out_path}")

    with open(args.datasets) as f:
        data = json.load(f)

    # Resume: skip already-done uids.
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

    # Load models.
    print(f"Loading 65B base: {args.model_base_name_or_path}")
    base_model, base_tok = load_base_65b(args.model_base_name_or_path)
    print(f"Loading 7B backbone + adapters: {args.model_backbone_name_or_path}")
    print(f"  help={args.model_arm_help_path}")
    print(f"  harm={args.model_arm_harm_path}")
    pm, arm_tok = load_backbone_with_adapters(
        args.model_backbone_name_or_path,
        args.model_arm_help_path,
        args.model_arm_harm_path,
        device,
    )

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
