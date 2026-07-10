"""
E4_eval_unlok.py
-----------------
Full CRP + behavioral + recovery evaluation on UnLOK-VQA.

This provides:
  1. Independent dataset validation (not MLLMU-Bench)
  2. Much larger forget set (up to 400 samples vs 20 entities)
  3. Built-in rephrase attacks (from UnLOK's rephrase field)
  4. Locality (retain) validation via UnLOK's loc/loc_ans fields

Run AFTER E1_download_unlok.py.

Usage:
    py E4_eval_unlok.py
    py E4_eval_unlok.py --methods ga npo --n_forget 100
    py E4_eval_unlok.py --resume
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from adapter_guard import assert_adapter_is_active

from exp_config import (
    LLAVA_BASE, LLAVA_ADAPTERS, DEVICE,
    UNLOK_FORGET_DIR, UNLOK_RETAIN_DIR,
    UNLOK_IMAGES_DIR,
    LLAVA_LB_LAYERS, LLAVA_VE_LAYERS,
    MAX_NEW_TOKENS, RESULTS_DIR,
    BOOTSTRAP_N, BOOTSTRAP_ALPHA,
)

UNLOK_RESULTS_DIR = RESULTS_DIR / "unlok_vqa"
UNLOK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
METHODS = ["ga", "npo", "mmunlearner", "cagul", "sineproject"]


# ------------------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------------------

def load_unlok_splits(n_forget: int = None, n_retain: int = None):
    """Load UnLOK forget and retain splits."""
    forget_ann = UNLOK_FORGET_DIR / "annotations.json"
    retain_ann = UNLOK_RETAIN_DIR / "annotations.json"

    if not forget_ann.exists():
        print("[ERROR] UnLOK forget split not found. Run E1_download_unlok.py first.")
        sys.exit(1)

    with open(forget_ann, encoding="utf-8") as f:
        forget = json.load(f)
    with open(retain_ann, encoding="utf-8") as f:
        retain = json.load(f)

    # Filter to samples where images exist
    forget = [item for item in forget if Path(item["image"]).exists()]
    retain = [item for item in retain if Path(item["image"]).exists()]

    if n_forget:
        forget = forget[:n_forget]
    if n_retain:
        retain = retain[:n_retain]

    print(f"[data] UnLOK forget: {len(forget)} | retain (loc): {len(retain)}")
    return forget, retain


# ------------------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------------------


def get_bnb_config():
    """BitsAndBytesConfig for 4-bit NF4 - works with all recent transformers."""
    from transformers import BitsAndBytesConfig
    import torch
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

def load_llava_model(method: str):
    """Load LLaVA with adapter for given method."""
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    ckpt = LLAVA_ADAPTERS.get(method)
    processor = AutoProcessor.from_pretrained(LLAVA_BASE)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_BASE, quantization_config=get_bnb_config(), device_map=DEVICE
    )
    if ckpt and Path(ckpt).exists():
        if (Path(ckpt) / "adapter_config.json").exists():
            activity = assert_adapter_is_active(ckpt)
            print(
                "[adapter] Active LoRA: "
                f"{activity['n_nonzero_lora_B']}/"
                f"{activity['n_lora_B']} nonzero B tensors"
            )
            model = PeftModel.from_pretrained(
                model,
                str(ckpt),
                is_trainable=False,
            )
        else:
            model = LlavaForConditionalGeneration.from_pretrained(
                str(ckpt), quantization_config=get_bnb_config(), device_map=DEVICE
            )
    model.eval()
    return model, processor


def run_inference(model, processor, image_path, question) -> str:
    """Single LLaVA inference."""
    image = Image.open(image_path).convert("RGB")
    prompt = f"USER: <image>\n{question} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    text = processor.decode(out[0], skip_special_tokens=True)
    # Extract only the ASSISTANT part
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1].strip()
    return text


# ------------------------------------------------------------------------------
# SCORING
# ------------------------------------------------------------------------------

def score(response: str, answer: str, aliases: list = None) -> bool:
    """Check if response contains the expected answer."""
    r = response.lower().strip()
    candidates = [answer.lower()] + [a.lower() for a in (aliases or [])]
    return any(c in r for c in candidates)



# ------------------------------------------------------------------------
# Robust HF/PEFT LLaVA accessors
# ------------------------------------------------------------------------
def _llava_core(model):
    """
    Handles both raw LlavaForConditionalGeneration and PEFT-wrapped models.
    Current HF LLaVA commonly stores language_model, vision_tower, and
    multi_modal_projector under model.model rather than at the top level.
    """
    if hasattr(model, "base_model"):
        bm = model.base_model; model = bm.model if hasattr(bm, "model") else bm
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model
    return model

def get_llava_language_layers(model):
    core = _llava_core(model)
    lm = core.language_model
    if hasattr(lm, "layers"):
        return lm.layers
    return lm.model.layers

def get_llava_vision_layers(model):
    core = _llava_core(model)
    return core.vision_tower.vision_model.encoder.layers

def get_llava_bridge(model):
    core = _llava_core(model)
    return core.multi_modal_projector


# ------------------------------------------------------------------------------
# CRP EXTRACTION
# ------------------------------------------------------------------------------

def extract_activations(model, processor, item: dict) -> dict:
    """Extract per-hook activations for one item."""
    image = Image.open(item["image"]).convert("RGB")
    prompt = f"USER: <image>\n{item['question']} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)

    acts = {}
    handles = []

    def hook_fn(key):
        def fn(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.dim() == 3:
                h = h[0].mean(0)
            acts[key] = h.detach().cpu().float()
        return fn

    # Language backbone
    lb = get_llava_language_layers(model)
    for idx in LLAVA_LB_LAYERS:
        if idx < len(lb):
            handles.append(lb[idx].register_forward_hook(hook_fn(f"lb_{idx}")))
    # Vision encoder
    try:
        ve = get_llava_vision_layers(model)
        for idx in LLAVA_VE_LAYERS:
            if idx < len(ve):
                handles.append(ve[idx].register_forward_hook(hook_fn(f"ve_{idx}")))
    except: pass
    # Bridge
    try:
        handles.append(
            get_llava_bridge(model).register_forward_hook(hook_fn("bridge"))
        )
    except: pass

    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=16, do_sample=False)
    for h in handles:
        h.remove()
    return acts


def compute_crp(m0_acts: list, mu_acts: list) -> dict:
    """Compute component CKA from per-entity activation lists."""
    lb_sims, ve_sims, br_sims = [], [], []
    per_entity_lb = []

    for a0, au in zip(m0_acts, mu_acts):
        entity_lb = []
        for idx in LLAVA_LB_LAYERS:
            k = f"lb_{idx}"
            if k in a0 and k in au:
                cos = F.cosine_similarity(a0[k].unsqueeze(0), au[k].unsqueeze(0)).item()
                lb_sims.append(cos)
                entity_lb.append(cos)
        per_entity_lb.append(np.mean(entity_lb) if entity_lb else float("nan"))

        for idx in LLAVA_VE_LAYERS:
            k = f"ve_{idx}"
            if k in a0 and k in au:
                cos = F.cosine_similarity(a0[k].unsqueeze(0), au[k].unsqueeze(0)).item()
                ve_sims.append(cos)
        if "bridge" in a0 and "bridge" in au:
            cos = F.cosine_similarity(a0["bridge"].unsqueeze(0), au["bridge"].unsqueeze(0)).item()
            br_sims.append(cos)

    def ci(vals):
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) < 2: return (float("nan"), float("nan"))
        means = [np.mean(np.random.choice(vals, len(vals))) for _ in range(BOOTSTRAP_N)]
        return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))

    # Per-entity bootstrap CI for LB
    per_entity_lb_ci = ci(per_entity_lb)

    return {
        "ve_cka":           float(np.mean(ve_sims)) if ve_sims else float("nan"),
        "bridge_cka":       float(np.mean(br_sims)) if br_sims else float("nan"),
        "lb_cka":           float(np.mean(lb_sims)) if lb_sims else float("nan"),
        "lb_cka_ci_95":     per_entity_lb_ci,
        "per_entity_lb":    per_entity_lb,
        "n_entities":       len(m0_acts),
    }


# ------------------------------------------------------------------------------
# FULL EVALUATION
# ------------------------------------------------------------------------------

def evaluate_method_unlok(method: str, forget_items: list, retain_items: list,
                           m0: object, mu: object, processor: object) -> dict:
    """
    Run CRP + behavioral + recovery for one method on UnLOK-VQA.
    """
    print(f"\n  Evaluating method: {method.upper()}")

    # -- Behavioral: forget set ------------------------------------------------
    print(f"  Forget set ({len(forget_items)} items)...")
    forget_correct = 0
    rephrase_correct = 0
    rephrase_total = 0
    m0_acts_forget, mu_acts_forget = [], []

    for i, item in enumerate(forget_items):
        # Direct forgetting
        resp = run_inference(mu, processor, item["image"], item["question"])
        if score(resp, item["answer"], item.get("aliases")):
            forget_correct += 1

        # Rephrase recovery (using UnLOK's built-in rephrases)
        for rq in item.get("rephrase_questions", []):
            r_resp = run_inference(mu, processor, item["image"], rq)
            rephrase_total += 1
            if score(r_resp, item["answer"], item.get("aliases")):
                rephrase_correct += 1

        # CRP activations
        a0 = extract_activations(m0,  processor, item)
        au = extract_activations(mu,  processor, item)
        m0_acts_forget.append(a0)
        mu_acts_forget.append(au)

        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(forget_items)}] forget_acc so far: "
                  f"{forget_correct/(i+1):.3f}")

    # -- Behavioral: retain set (locality) ------------------------------------
    print(f"  Retain/locality set ({len(retain_items)} items)...")
    retain_correct = 0
    for item in retain_items:
        resp = run_inference(mu, processor, item["image"], item["question"])
        if score(resp, item["answer"], item.get("aliases")):
            retain_correct += 1

    # -- CRP -------------------------------------------------------------------
    print(f"  Computing CRP...")
    crp = compute_crp(m0_acts_forget, mu_acts_forget)

    # -- Aggregate -------------------------------------------------------------
    forget_acc   = forget_correct / len(forget_items)
    forget_rate  = 1.0 - forget_acc
    retain_acc   = retain_correct / len(retain_items) if retain_items else float("nan")
    rephrase_rec = rephrase_correct / rephrase_total  if rephrase_total > 0 else float("nan")

    result = {
        "method":         method,
        "dataset":        "unlok_vqa",
        "forget_acc":     forget_acc,
        "forget_rate":    forget_rate,
        "retain_acc":     retain_acc,
        "rephrase_recovery": rephrase_rec,
        "rephrase_total": rephrase_total,
        "n_forget":       len(forget_items),
        "n_retain":       len(retain_items),
        **crp,
    }

    print(f"\n  RESULTS for {method} on UnLOK-VQA:")
    print(f"    Forget Acc:    {forget_acc:.4f}")
    print(f"    Forget Rate:   {forget_rate:.4f}")
    print(f"    Retain Acc:    {retain_acc:.4f}")
    print(f"    Rephrase Rec:  {rephrase_rec:.4f}")
    print(f"    VE-CKA:        {crp['ve_cka']:.4f}")
    print(f"    Bridge-CKA:    {crp['bridge_cka']:.4f}")
    print(f"    LB-CKA:        {crp['lb_cka']:.4f} [{crp['lb_cka_ci_95'][0]:.4f},{crp['lb_cka_ci_95'][1]:.4f}]")

    return result


# ------------------------------------------------------------------------------
# LATEX TABLE
# ------------------------------------------------------------------------------

def build_unlok_latex(results: list) -> str:
    METHOD_DISPLAY = {
        "no_unlearn":  "No Unlearn",
        "ga":          "GA",
        "npo":         "NPO",
        "mmunlearner": "MMUnlearner",
        "cagul":       "CAGUL",
        "manu_lora":   "MANU",
        "sineproject": "SineProject",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Full audit on \emph{UnLOK-VQA} (LLaVA-1.5-7B).",
        r"Forget Acc = fraction of forget-set queries answered correctly ($\downarrow$);",
        r"Retain Acc = fraction of locality queries (loc/loc\_ans) answered correctly;",
        r"Rephrase = recovery rate using UnLOK-VQA's built-in rephrased queries.",
        r"CRP values and 95\% bootstrap CIs are computed over forget-set entities.",
        r"Patterns replicate the \emph{mllmu\_real} findings on an independent benchmark.}",
        r"\label{tab:unlok_full}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\small",
        r"\begin{tabular}{lrrrrrrl}",
        r"\toprule",
        r"\textbf{Method} & \textbf{F-Acc}$\downarrow$ & \textbf{Ret-Acc}",
        r"  & \textbf{Reph.} & \textbf{VE} & \textbf{BR}",
        r"  & \textbf{LB [95\% CI]} & \textbf{Regime} \\",
        r"\midrule",
    ]
    REGIME = {
        "no_unlearn":  "Reference",
        "ga":          "Under-forgetting",
        "npo":         "Under-forgetting",
        "mmunlearner": "Under-forgetting",
        "cagul":       "Under-forgetting",
        "manu_lora":   "Under-forgetting",
        "sineproject": "Under-forgetting",
    }
    for r in results:
        m = r["method"]
        disp = METHOD_DISPLAY.get(m, m)
        lb_str = (f"{r['lb_cka']:.3f} [{r['lb_cka_ci_95'][0]:.3f},"
                  f"{r['lb_cka_ci_95'][1]:.3f}]")
        lines.append(
            f"{disp} & {r['forget_acc']:.3f} & {r['retain_acc']:.3f}"
            f" & {r['rephrase_recovery']:.3f}"
            f" & {r['ve_cka']:.3f} & {r['bridge_cka']:.3f}"
            f" & {lb_str} & {REGIME.get(m, '---')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods",  nargs="+", default=METHODS)
    parser.add_argument("--n_forget", type=int,  default=None,
                        help="Max forget items to use (default: all)")
    parser.add_argument("--n_retain", type=int,  default=400,
                        help="Max retain/locality items to use (default: 400)")
    parser.add_argument("--resume",   action="store_true")
    args = parser.parse_args()

    forget_items, retain_items = load_unlok_splits(args.n_forget, args.n_retain)

    # Load base model once (shared)
    print("\n[E4] Loading base model M0 (used for all methods)...")
    m0, proc = load_llava_model("no_unlearn")

    all_results = []
    for method in args.methods:
        out_path = UNLOK_RESULTS_DIR / f"{method}_unlok_results.json"
        if args.resume and out_path.exists():
            print(f"  [resume] {method} already done.")
            with open(out_path, encoding="utf-8") as f:
                all_results.append(json.load(f))
            continue

        print(f"\n[E4] Loading unlearned model: {method}")
        mu, _ = load_llava_model(method)

        result = evaluate_method_unlok(method, forget_items, retain_items, m0, mu, proc)
        all_results.append(result)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  [saved] {out_path}")

        del mu
        torch.cuda.empty_cache()

    # Summary
    summary_path = UNLOK_RESULTS_DIR / "unlok_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    # LaTeX
    latex = build_unlok_latex(all_results)
    latex_path = UNLOK_RESULTS_DIR / "table_unlok_full.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"\n[saved] {latex_path}")

    # Console summary
    print("\n" + "="*70)
    print(f"{'Method':<14} {'F-Acc':>7} {'Ret-Acc':>8} {'VE-CKA':>8} "
          f"{'BR-CKA':>8} {'LB-CKA':>8}")
    print("-"*60)
    for r in all_results:
        print(f"{r['method']:<14} {r['forget_acc']:>7.3f} {r['retain_acc']:>8.3f} "
              f"{r['ve_cka']:>8.3f} {r['bridge_cka']:>8.3f} {r['lb_cka']:>8.3f}")

    print("\n" + "="*70)
    print(latex)


if __name__ == "__main__":
    main()


