"""
stage4_mllmu_bench.py
──────────────────────
Two-phase script for MLLMU-Bench evaluation.

Phase 1: Preflight (always runs first)
  Checks:
    - Data split paths exist
    - Exact forget/retain item counts
    - Image availability (counts missing images)
    - All requested method checkpoints exist
    - Hook resolution (dry-run with one sample)
    - Estimated runtime
  Produces: preflight_report.json + PASS/WARN/FAIL
  Does NOT proceed to evaluation if any FAIL.

Phase 2: Evaluation (only if preflight passed)
  - Behavioural evaluation (ForgetAcc, RetainAcc)
  - CRP extraction (VE-CKA, BR-CKA, LB-CKA)
  - Save after each method
  - Resume support
  - Does NOT overwrite mllmu_real results

Usage:
    py stage4_mllmu_bench.py --preflight_only    # check only
    py stage4_mllmu_bench.py                     # preflight then eval
    py stage4_mllmu_bench.py --resume            # skip completed methods
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from exp_config import (
    LLAVA_BASE, LLAVA_ADAPTERS, DEVICE, ROOT, RESULTS_DIR,
    LLAVA_VE_LAYERS, LLAVA_LB_LAYERS, MAX_NEW_TOKENS,
)

BENCH_OUT = RESULTS_DIR / "mllmu_bench"
BENCH_OUT.mkdir(parents=True, exist_ok=True)

VALID_METHODS = ["npo", "mmunlearner", "cagul", "sineproject", "graddiff"]
CHECKS        = []

# Known MLLMU-Bench data locations
DATA_SEARCH = [
    ROOT / "data" / "mllmu_bench",
    ROOT / "data" / "mllmu" / "bench",
    ROOT / "data" / "MLLMU-Bench",
    ROOT / "datasets" / "mllmu_bench",
    ROOT / "data" / "mllmu_bench_dataset",
]


def chk(label, verdict, detail):
    CHECKS.append({"check": label, "verdict": verdict, "detail": str(detail)})
    icon = {"PASS": "OK", "WARN": "!!", "FAIL": "XX"}.get(verdict, "??")
    print(f"  [{icon} {verdict}] {label}: {detail}")
    return verdict


def save_report(tag=""):
    n_p = sum(1 for c in CHECKS if c["verdict"] == "PASS")
    n_w = sum(1 for c in CHECKS if c["verdict"] == "WARN")
    n_f = sum(1 for c in CHECKS if c["verdict"] == "FAIL")
    rp  = BENCH_OUT / f"preflight_report{tag}.json"
    with open(rp, "w", encoding="utf-8") as f:
        json.dump({"checks": CHECKS, "n_pass": n_p,
                   "n_warn": n_w, "n_fail": n_f}, f, indent=2)
    print(f"\n  Checks: {n_p} PASS  {n_w} WARN  {n_f} FAIL  → {rp}")
    return n_f == 0


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def find_split(base: Path, names: list):
    for name in names:
        p = base / name
        if p.exists(): return p
    return None


def find_bench_data():
    for base in DATA_SEARCH:
        if not base.exists(): continue
        forget = find_split(base, ["forget","forget10","forget_set","Forget"])
        retain = find_split(base, ["retain","retain15","retain_set","Retain"])
        if forget:
            return base, forget, retain
    return None, None, None


def load_split_with_stats(split_path: Path) -> tuple:
    """Load items and return (items, n_total, n_missing_images)."""
    if split_path is None:
        return [], 0, 0

    ann = split_path / "annotations.json"
    raw = []
    if ann.exists():
        with open(ann, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = [raw]
    else:
        # Subdirectory layout
        for d in sorted(split_path.iterdir()):
            if not d.is_dir(): continue
            imgs  = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            jsons = list(d.glob("*.json"))
            if not imgs or not jsons: continue
            with open(jsons[0], encoding="utf-8") as f:
                qa = json.load(f)
            for q in (qa if isinstance(qa, list) else [qa])[:2]:
                raw.append({"entity": d.name, "image": str(imgs[0]),
                             "question": q["question"],
                             "answer":   q.get("answer", q.get("gt",""))})

    items   = []
    missing = 0
    for item in raw:
        p = Path(item.get("image",""))
        if not p.is_absolute(): p = split_path / p
        if p.exists():
            item["image"] = p
            items.append(item)
        else:
            missing += 1

    return items, len(raw), missing


# ══════════════════════════════════════════════════════════════════════════════
# PREFLIGHT
# ══════════════════════════════════════════════════════════════════════════════

def preflight(methods: list) -> dict:
    print("\n=== STAGE 4 PREFLIGHT ===")

    # Data
    base, forget_path, retain_path = find_bench_data()
    if base is None:
        chk("MLLMU-Bench data", "FAIL",
            f"Not found in any of: {[str(p) for p in DATA_SEARCH]}")
        return None

    chk("MLLMU-Bench base", "PASS", str(base))

    forget_items, f_total, f_missing = load_split_with_stats(forget_path)
    retain_items, r_total, r_missing = load_split_with_stats(retain_path)

    if forget_path:
        chk("forget split path", "PASS", str(forget_path))
        if f_missing > 0:
            chk("forget images", "WARN",
                f"{f_missing}/{f_total} images missing — "
                f"only {len(forget_items)} usable")
        else:
            chk("forget images", "PASS",
                f"{len(forget_items)} items, all images present")
    else:
        chk("forget split", "FAIL", "No forget subdirectory found")
        return None

    if retain_path:
        chk("retain split path", "PASS", str(retain_path))
        if r_missing > 0:
            chk("retain images", "WARN",
                f"{r_missing}/{r_total} images missing")
        else:
            chk("retain images", "PASS", f"{len(retain_items)} items")
    else:
        chk("retain split", "WARN",
            "No retain subdirectory found — RetainAcc cannot be computed")

    if not forget_items:
        chk("forget items usable", "FAIL", "Zero usable forget items")
        return None

    chk("forget items usable", "PASS", f"{len(forget_items)}")

    # Checkpoints
    valid_methods = []
    for method in methods:
        ckpt = LLAVA_ADAPTERS.get(method)
        if ckpt and Path(str(ckpt)).exists():
            chk(f"checkpoint {method}", "PASS", str(ckpt))
            valid_methods.append(method)
        else:
            chk(f"checkpoint {method}", "FAIL",
                f"Not found: {ckpt}")

    if not valid_methods:
        chk("valid checkpoints", "FAIL", "No valid checkpoints")
        return None

    # Hook resolution dry run (one sample)
    print("\n  Dry-run hook resolution on one sample...")
    try:
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        from transformers import BitsAndBytesConfig
        bnb  = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16)
        proc = AutoProcessor.from_pretrained(LLAVA_BASE)
        m0   = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_BASE, quantization_config=bnb, device_map=DEVICE)
        m0.eval()

        test_item = forget_items[0]
        image  = Image.open(test_item["image"]).convert("RGB")
        prompt = f"USER: <image>\n{test_item['question']} ASSISTANT:"
        inputs = proc(text=prompt, images=image, return_tensors="pt").to(DEVICE)

        hooks_fired = []
        def make_hook(k):
            def fn(m,inp,out):
                h = out[0] if isinstance(out,tuple) else out
                if torch.is_tensor(h) and h.numel() > 0:
                    hooks_fired.append(k)
            return fn

        handles = []
        core = m0.base_model.model if hasattr(m0,"base_model") else m0
        for path in ["model.vision_tower.vision_model.encoder.layers",
                     "vision_tower.vision_model.encoder.layers"]:
            try:
                obj = core
                for p in path.split("."): obj = getattr(obj, p)
                for idx in LLAVA_VE_LAYERS[:2]:
                    if idx < len(obj):
                        handles.append(obj[idx].register_forward_hook(
                            make_hook(f"ve_{idx}")))
                break
            except AttributeError: pass

        with torch.no_grad():
            m0(**inputs, use_cache=False)
        for h in handles: h.remove()
        del m0; torch.cuda.empty_cache()

        if hooks_fired:
            chk("hook dry run", "PASS",
                f"Fired: {hooks_fired}")
        else:
            chk("hook dry run", "WARN",
                "No hooks fired — hook paths may need updating")
    except Exception as e:
        chk("hook dry run", "WARN", f"Could not complete: {e}")

    # Runtime estimate
    # Rough estimate: ~8s per item per model load per method
    n_f = len(forget_items)
    n_r = len(retain_items)
    n_m = len(valid_methods)
    est_behav = n_m * (n_f + n_r) * 4 / 60   # minutes
    est_crp   = n_m * 2 * n_f * 8 / 60        # minutes (M0 + Mu)
    est_total = est_behav + est_crp
    chk("runtime estimate", "PASS",
        f"~{est_total:.0f} min for {n_m} methods "
        f"({n_f} forget + {n_r} retain samples). "
        f"Start this as an overnight job.")

    print(f"\n  Preflight complete.")
    print(f"  Methods to run: {valid_methods}")
    print(f"  Forget items:   {len(forget_items)}")
    print(f"  Retain items:   {len(retain_items)}")
    print(f"  Est. runtime:   {est_total:.0f} minutes")

    return {
        "forget_items": forget_items,
        "retain_items": retain_items,
        "valid_methods": valid_methods,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def get_bnb():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True)


def load_model(ckpt_path=None):
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    proc = AutoProcessor.from_pretrained(LLAVA_BASE)
    bnb  = get_bnb()
    if ckpt_path is None:
        model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_BASE, quantization_config=bnb, device_map=DEVICE)
    else:
        ckpt = Path(str(ckpt_path))
        if (ckpt / "adapter_config.json").exists():
            base  = LlavaForConditionalGeneration.from_pretrained(
                LLAVA_BASE, quantization_config=bnb, device_map=DEVICE)
            model = PeftModel.from_pretrained(base, str(ckpt))
        else:
            model = LlavaForConditionalGeneration.from_pretrained(
                str(ckpt), quantization_config=bnb, device_map=DEVICE)
    model.eval()
    return model, proc


def score_items(model, proc, items):
    correct = 0
    for item in items:
        image  = Image.open(item["image"]).convert("RGB")
        prompt = f"USER: <image>\n{item['question']} ASSISTANT:"
        inputs = proc(text=prompt, images=image, return_tensors="pt").to(DEVICE)
        plen   = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False)
        resp = proc.decode(out[0][plen:], skip_special_tokens=True).strip()
        if item.get("answer","").lower().strip() in resp.lower():
            correct += 1
    return correct / len(items) if items else float("nan")


def debiased_cka(X, Y):
    if X is None or Y is None or X.shape[0] < 4: return float("nan")
    X = X.float()-X.float().mean(0,keepdim=True)
    Y = Y.float()-Y.float().mean(0,keepdim=True)
    n=X.shape[0]; K=X@X.T; L=Y@Y.T
    Kt=K-torch.diag(torch.diag(K)); Lt=L-torch.diag(torch.diag(L))
    c=1./(n*(n-3))
    def hsic(A,B): return c*((A*B).sum()-(2./(n-2))*(A.sum(1)*B.sum(1)).sum()
                              +A.sum()*B.sum()/((n-1)*(n-2)))
    v=torch.clamp(hsic(Kt,Lt)/torch.sqrt(torch.clamp(hsic(Kt,Kt)*hsic(Lt,Lt),
                                                      min=1e-10)),0.,1.)
    return float("nan") if torch.isnan(v) else float(v.item())


def eval_method(method, ckpt_path, forget_items, retain_items, resume):
    out_path = BENCH_OUT / f"{method}_bench.json"
    if resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            r = json.load(f)
        print(f"  [cached] {method}: FA={r.get('forget_acc','?')}  "
              f"RA={r.get('retain_acc','?')}")
        return

    print(f"\n  Method: {method.upper()}")
    result = {"method": method, "dataset": "mllmu_bench"}
    t0     = time.time()

    # Load both models
    m0, proc = load_model(None)
    mu, _    = load_model(ckpt_path)

    # Behavioural
    print("  Scoring forget set...")
    fa = score_items(mu, proc, forget_items)
    print("  Scoring retain set...")
    ra = score_items(mu, proc, retain_items) if retain_items else float("nan")
    result.update({"forget_acc": fa, "forget_rate": 1.0-fa,
                   "retain_acc": ra, "forget_n": len(forget_items),
                   "retain_n": len(retain_items)})
    print(f"  ForgetAcc={fa:.4f}  RetainAcc={ra:.4f}")

    # CRP
    print("  Computing CRP...")
    ve_keys = [f"ve_{i}" for i in LLAVA_VE_LAYERS]
    lb_keys = [f"lb_{i}" for i in LLAVA_LB_LAYERS]

    def extract(model):
        core = model.base_model.model if hasattr(model,"base_model") else model
        captured = {}
        handles  = []
        def hook(k):
            def fn(m,inp,out):
                h=out[0] if isinstance(out,tuple) else out
                if not torch.is_tensor(h): return
                h=h.detach().float().cpu()
                if h.dim()==3: h=h[0].mean(0)
                elif h.dim()==2: h=h.mean(0)
                if not torch.isnan(h).any(): captured[k]=h
            return fn
        for path in ["model.vision_tower.vision_model.encoder.layers",
                     "vision_tower.vision_model.encoder.layers"]:
            try:
                obj=core
                for p in path.split("."): obj=getattr(obj,p)
                for idx in LLAVA_VE_LAYERS:
                    if idx<len(obj): handles.append(obj[idx].register_forward_hook(hook(f"ve_{idx}")))
                break
            except AttributeError: pass
        for path in ["model.multi_modal_projector","multi_modal_projector"]:
            try:
                obj=core
                for p in path.split("."): obj=getattr(obj,p)
                handles.append(obj.register_forward_hook(hook("bridge")))
                break
            except AttributeError: pass
        for path in ["model.language_model.model.layers","model.language_model.layers",
                     "language_model.model.layers"]:
            try:
                obj=core
                for p in path.split("."): obj=getattr(obj,p)
                for idx in LLAVA_LB_LAYERS:
                    if idx<len(obj): handles.append(obj[idx].register_forward_hook(hook(f"lb_{idx}")))
                break
            except AttributeError: pass
        return captured, handles

    m0_acts=[]; mu_acts=[]
    for i, item in enumerate(forget_items):
        image  = Image.open(item["image"]).convert("RGB")
        prompt = f"USER: <image>\n{item['question']} ASSISTANT:"
        inputs = proc(text=prompt, images=image, return_tensors="pt").to(DEVICE)
        c0,h0  = extract(m0); cu,hu = extract(mu)
        with torch.no_grad():
            m0(**inputs, use_cache=False)
            mu(**inputs, use_cache=False)
        for h in h0+hu: h.remove()
        m0_acts.append(c0); mu_acts.append(cu)
        if (i+1)%10==0: print(f"    CRP [{i+1}/{len(forget_items)}]")

    def comp(keys):
        vals=[]
        for k in keys:
            X=torch.stack([d[k] for d in m0_acts if k in d])
            Y=torch.stack([d[k] for d in mu_acts if k in d])
            if X.shape==Y.shape:
                v=debiased_cka(X,Y)
                if not np.isnan(v): vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    result["ve_mean"]     = comp(ve_keys)
    result["bridge_mean"] = comp(["bridge"])
    result["lb_mean"]     = comp(lb_keys)
    result["runtime_min"] = round((time.time()-t0)/60, 1)

    print(f"  VE={result['ve_mean']:.4f}  "
          f"BR={result['bridge_mean']:.4f}  "
          f"LB={result['lb_mean']:.4f}")

    del m0, mu; torch.cuda.empty_cache()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    chk(f"{method} eval", "PASS",
        f"saved → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods",        nargs="+", default=VALID_METHODS)
    parser.add_argument("--resume",         action="store_true", default=True)
    parser.add_argument("--preflight_only", action="store_true")
    args = parser.parse_args()

    # Always run preflight first
    preflight_data = preflight(args.methods)
    preflight_ok   = save_report("_preflight")

    if not preflight_ok:
        print("\n  PREFLIGHT FAILED. Fix the issues above before evaluating.")
        sys.exit(1)

    if args.preflight_only:
        print("\n  Preflight passed. Run without --preflight_only to evaluate.")
        return

    print("\n=== STAGE 4 EVALUATION ===")
    print("  Preflight passed. Starting evaluation.")
    print("  Results are saved after each method.")
    print("  Existing mllmu_real results are NOT modified.")

    forget_items  = preflight_data["forget_items"]
    retain_items  = preflight_data["retain_items"]
    valid_methods = preflight_data["valid_methods"]

    for method in valid_methods:
        ckpt = LLAVA_ADAPTERS[method]
        try:
            eval_method(method, ckpt, forget_items, retain_items, args.resume)
        except Exception as e:
            chk(f"{method} eval", "FAIL", str(e))
            print(f"  Continuing to next method.")

    # Print summary table
    print("\n" + "="*70)
    print("MLLMU-BENCH RESULTS SUMMARY")
    print("="*70)
    print(f"  {'Method':<15} {'F-Acc':>8} {'F-Rate':>8} "
          f"{'Ret-Acc':>9} {'VE':>8} {'BR':>8} {'LB':>8}")
    print("  " + "-"*66)
    for method in valid_methods:
        p = BENCH_OUT / f"{method}_bench.json"
        if not p.exists(): continue
        with open(p, encoding="utf-8") as f: r = json.load(f)
        print(f"  {method:<15} "
              f"{r.get('forget_acc',float('nan')):>8.4f} "
              f"{r.get('forget_rate',float('nan')):>8.4f} "
              f"{r.get('retain_acc',float('nan')):>9.4f} "
              f"{r.get('ve_mean',float('nan')):>8.4f} "
              f"{r.get('bridge_mean',float('nan')):>8.4f} "
              f"{r.get('lb_mean',float('nan')):>8.4f}")

    save_report("_eval_complete")


if __name__ == "__main__":
    main()
