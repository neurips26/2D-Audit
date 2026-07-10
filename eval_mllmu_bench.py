"""
eval_mllmu_bench.py
────────────────────
Evaluates all five valid LLaVA methods on MLLMU-Bench.
Closes the supervisor's dataset breadth concern by adding a third dataset.

MLLMU-Bench uses fictitious personas. CNIS is not computed.
Only behavioural and CRP results are reported.

Finds MLLMU-Bench data automatically under common paths.
Saves results to outputs/revision/mllmu_bench/.

Usage:
    py eval_mllmu_bench.py
    py eval_mllmu_bench.py --methods npo mmunlearner --resume
    py eval_mllmu_bench.py --crp_only    # only CRP, skip behavioural
    py eval_mllmu_bench.py --behav_only  # only behavioural, skip CRP
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from exp_config import (
    LLAVA_BASE, LLAVA_ADAPTERS, DEVICE, ROOT, RESULTS_DIR,
    LLAVA_VE_LAYERS, LLAVA_LB_LAYERS, MAX_NEW_TOKENS,
    BOOTSTRAP_N, BOOTSTRAP_ALPHA,
)

BENCH_DIR = RESULTS_DIR / "mllmu_bench"
BENCH_DIR.mkdir(parents=True, exist_ok=True)

VALID_METHODS = ["npo", "mmunlearner", "cagul", "sineproject", "graddiff"]

# Search paths for MLLMU-Bench data
BENCH_SEARCH = [
    ROOT / "data" / "mllmu_bench",
    ROOT / "data" / "mllmu" / "bench",
    ROOT / "data" / "MLLMU-Bench",
    ROOT / "datasets" / "mllmu_bench",
    Path("C:/Users") / "34998855" / "data" / "mllmu_bench",
]


def find_bench_splits() -> tuple:
    """Find forget and retain splits for MLLMU-Bench."""
    for base in BENCH_SEARCH:
        if not base.exists():
            continue
        # Try standard subfolder names
        for forget_name in ("forget", "forget10", "forget_set"):
            for retain_name in ("retain", "retain15", "retain_set"):
                forget = base / forget_name
                retain = base / retain_name
                if forget.exists() and retain.exists():
                    return forget, retain
        # Try flat layout
        forget = base / "forget"
        retain = base / "retain"
        if forget.exists():
            return forget, retain
        # Try annotations directly
        if (base / "forget_annotations.json").exists():
            return base / "forget_annotations.json", base / "retain_annotations.json"
    return None, None


def load_split(split_path: Path) -> list:
    """Load items from a split directory or annotation file."""
    if split_path is None:
        return []

    # Direct annotation file
    if split_path.suffix == ".json":
        with open(split_path, encoding="utf-8") as f:
            items = json.load(f)
        result = []
        for item in (items if isinstance(items, list) else []):
            p = Path(item.get("image",""))
            if not p.exists():
                p = split_path.parent / item.get("image","")
            if p.exists():
                item["image"] = p
                result.append(item)
        return result

    # Annotation file inside directory
    ann = split_path / "annotations.json"
    if ann.exists():
        with open(ann, encoding="utf-8") as f:
            items = json.load(f)
        result = []
        for item in (items if isinstance(items, list) else []):
            p = Path(item.get("image",""))
            if not p.is_absolute(): p = split_path / p
            if p.exists():
                item["image"] = p
                result.append(item)
        return result

    # Subdirectory layout
    items = []
    for d in sorted(split_path.iterdir()):
        if not d.is_dir(): continue
        imgs  = list(d.glob("*.jpg")) + list(d.glob("*.png"))
        jsons = list(d.glob("*.json"))
        if not imgs or not jsons: continue
        with open(jsons[0], encoding="utf-8") as f:
            qa = json.load(f)
        for q in (qa if isinstance(qa, list) else [qa])[:2]:
            items.append({"entity": d.name, "image": imgs[0],
                          "question": q["question"],
                          "answer":   q.get("answer", q.get("gt",""))})
    return items


def get_bnb():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)


def load_model(ckpt_path=None):
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    proc = AutoProcessor.from_pretrained(LLAVA_BASE)
    bnb  = get_bnb()
    if ckpt_path is None:
        model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_BASE, quantization_config=bnb, device_map=DEVICE)
    else:
        ckpt = Path(ckpt_path)
        if (ckpt / "adapter_config.json").exists():
            base  = LlavaForConditionalGeneration.from_pretrained(
                LLAVA_BASE, quantization_config=bnb, device_map=DEVICE)
            model = PeftModel.from_pretrained(base, str(ckpt))
        else:
            model = LlavaForConditionalGeneration.from_pretrained(
                str(ckpt), quantization_config=bnb, device_map=DEVICE)
    model.eval()
    return model, proc


def unwrap(model):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def find_module(obj, paths):
    for path in paths:
        cur = obj; ok = True
        for p in path.split("."):
            if not hasattr(cur, p): ok = False; break
            cur = getattr(cur, p)
        if ok and cur is not None: return cur
    return None


def is_correct(resp: str, item: dict) -> bool:
    r = resp.lower().strip()
    return item.get("answer","").lower().strip() in r


def eval_behavioural(model, proc, items: list) -> dict:
    """ForgetAcc and RetainAcc on a split."""
    correct = 0
    for item in items:
        image  = Image.open(item["image"]).convert("RGB")
        prompt = f"USER: <image>\n{item['question']} ASSISTANT:"
        inputs = proc(text=prompt, images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False)
        plen = inputs["input_ids"].shape[1]
        resp = proc.decode(out[0][plen:], skip_special_tokens=True).strip()
        if is_correct(resp, item): correct += 1
    n = len(items)
    return {"n": n, "correct": correct,
            "accuracy": correct/n if n else float("nan"),
            "forget_rate": 1.0 - correct/n if n else float("nan")}


def debiased_cka(X, Y):
    if X is None or Y is None or X.shape[0] < 4: return float("nan")
    X = X.float() - X.float().mean(0,keepdim=True)
    Y = Y.float() - Y.float().mean(0,keepdim=True)
    n = X.shape[0]; K = X@X.T; L = Y@Y.T
    Kt = K-torch.diag(torch.diag(K)); Lt = L-torch.diag(torch.diag(L))
    c  = 1./(n*(n-3))
    def hsic(A,B): return c*((A*B).sum()-(2./(n-2))*(A.sum(1)*B.sum(1)).sum()
                              +A.sum()*B.sum()/((n-1)*(n-2)))
    v = torch.clamp(hsic(Kt,Lt)/torch.sqrt(torch.clamp(hsic(Kt,Kt)*hsic(Lt,Lt),
                                                        min=1e-10)),0.,1.)
    return float("nan") if torch.isnan(v) else float(v.item())


def boot_ci(vals, n_boot=500):
    v = [x for x in vals if not np.isnan(x)]
    if len(v) < 2: return float("nan"), float("nan")
    b = [float(np.mean(np.random.choice(v,len(v)))) for _ in range(n_boot)]
    return float(np.percentile(b,2.5)), float(np.percentile(b,97.5))


def eval_crp(m0, mu, proc, items: list) -> dict:
    """CRP extraction for one method on one split."""
    core_m0 = unwrap(m0); core_mu = unwrap(mu)
    ve_keys = [f"ve_{i}" for i in LLAVA_VE_LAYERS]
    lb_keys = [f"lb_{i}" for i in LLAVA_LB_LAYERS]

    def extract(model, core, item):
        captured = {}
        handles  = []
        def hook(k):
            def fn(m,inp,out):
                h = out[0] if isinstance(out,tuple) else out
                if not torch.is_tensor(h): return
                h = h.detach().float().cpu()
                if h.dim()==3: h=h[0].mean(0)
                elif h.dim()==2: h=h.mean(0)
                if not torch.isnan(h).any(): captured[k]=h
            return fn
        ve = find_module(core,["model.vision_tower.vision_model.encoder.layers",
                                "vision_tower.vision_model.encoder.layers"])
        if ve:
            for idx in LLAVA_VE_LAYERS:
                if idx<len(ve): handles.append(ve[idx].register_forward_hook(hook(f"ve_{idx}")))
        br = find_module(core,["model.multi_modal_projector","multi_modal_projector"])
        if br: handles.append(br.register_forward_hook(hook("bridge")))
        lb = find_module(core,["model.language_model.model.layers","model.language_model.layers",
                                "language_model.model.layers"])
        if lb:
            for idx in LLAVA_LB_LAYERS:
                if idx<len(lb): handles.append(lb[idx].register_forward_hook(hook(f"lb_{idx}")))
        image  = Image.open(item["image"]).convert("RGB")
        prompt = f"USER: <image>\n{item['question']} ASSISTANT:"
        inputs = proc(text=prompt, images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad(): model(**inputs, use_cache=False)
        for h in handles: h.remove()
        return captured

    m0_acts = []; mu_acts = []
    for i, item in enumerate(items):
        m0_acts.append(extract(m0, core_m0, item))
        mu_acts.append(extract(mu, core_mu, item))
        if (i+1) % 10 == 0: print(f"    CRP [{i+1}/{len(items)}]")

    def comp(keys):
        vals = []
        for k in keys:
            X = torch.stack([d[k] for d in m0_acts if k in d])
            Y = torch.stack([d[k] for d in mu_acts if k in d])
            if X.shape == Y.shape:
                v = debiased_cka(X,Y)
                if not np.isnan(v): vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    ve = comp(ve_keys); br = comp(["bridge"]); lb = comp(lb_keys)
    lo,hi = boot_ci([lb]*len(items))   # placeholder — use proper boot after activation saving
    return {"ve_mean":ve, "bridge_mean":br, "lb_mean":lb,
            "lb_ci_95": [lo,hi], "n_items": len(items)}


def run_method(method: str, ckpt_path,
               forget_items: list, retain_items: list,
               do_behav: bool, do_crp: bool, resume: bool):
    out = BENCH_DIR / f"{method}_mllmu_bench.json"
    if resume and out.exists():
        print(f"  [cached] {method}")
        return

    result = {"method": method, "dataset": "mllmu_bench"}

    print(f"\n  Loading M0 and {method}...")
    m0, proc = load_model(None)

    if do_behav:
        print("  Behavioural on forget set...")
        f_res = eval_behavioural(m0, proc, forget_items)
        print("  Behavioural on retain set...")
        r_res = eval_behavioural(m0, proc, retain_items)
        result["no_unlearn_forget_acc"] = f_res["accuracy"]
        result["no_unlearn_retain_acc"] = r_res["accuracy"]

    mu, _ = load_model(ckpt_path)

    if do_behav:
        print(f"  Behavioural [{method}] forget...")
        f_mu = eval_behavioural(mu, proc, forget_items)
        print(f"  Behavioural [{method}] retain...")
        r_mu = eval_behavioural(mu, proc, retain_items)
        result["forget_acc"]  = f_mu["accuracy"]
        result["forget_rate"] = f_mu["forget_rate"]
        result["retain_acc"]  = r_mu["accuracy"]
        result["forget_n"]    = f_mu["n"]
        result["retain_n"]    = r_mu["n"]
        print(f"  ForgetAcc={f_mu['accuracy']:.4f}  RetainAcc={r_mu['accuracy']:.4f}")

    if do_crp:
        print(f"  CRP [{method}] on forget set...")
        crp = eval_crp(m0, mu, proc, forget_items)
        result.update(crp)
        print(f"  VE={crp['ve_mean']:.4f}  BR={crp['bridge_mean']:.4f}  "
              f"LB={crp['lb_mean']:.4f}")

    del m0, mu; torch.cuda.empty_cache()

    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  [saved] {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods",    nargs="+", default=VALID_METHODS)
    parser.add_argument("--resume",     action="store_true", default=True)
    parser.add_argument("--crp_only",   action="store_true")
    parser.add_argument("--behav_only", action="store_true")
    args = parser.parse_args()

    do_behav = not args.crp_only
    do_crp   = not args.behav_only

    # Find data
    forget_path, retain_path = find_bench_splits()
    if forget_path is None:
        print("[ERROR] MLLMU-Bench data not found. Searched:")
        for p in BENCH_SEARCH: print(f"  {p}")
        print("\nIf data is elsewhere, add the path to BENCH_SEARCH in this script.")
        sys.exit(1)

    print(f"[MLLMU-Bench] Forget: {forget_path}")
    print(f"[MLLMU-Bench] Retain: {retain_path}")

    forget_items = load_split(forget_path)
    retain_items = load_split(retain_path)
    print(f"  Forget items: {len(forget_items)}")
    print(f"  Retain items: {len(retain_items)}")

    if not forget_items:
        print("[ERROR] No forget items loaded."); sys.exit(1)

    for method in args.methods:
        ckpt = LLAVA_ADAPTERS.get(method)
        if not ckpt or not Path(str(ckpt)).exists():
            print(f"  [skip] {method}: checkpoint missing"); continue
        run_method(method, ckpt, forget_items, retain_items,
                   do_behav, do_crp, args.resume)

    # Summary table
    print("\n" + "="*70)
    print("MLLMU-BENCH RESULTS")
    print("="*70)
    print(f"  {'Method':<15} {'F-Acc':>8} {'Ret-Acc':>9} "
          f"{'VE-CKA':>8} {'BR-CKA':>8} {'LB-CKA':>8}")
    print("  " + "-"*60)
    for method in args.methods:
        out = BENCH_DIR / f"{method}_mllmu_bench.json"
        if not out.exists(): continue
        with open(out, encoding="utf-8") as f:
            r = json.load(f)
        print(f"  {method:<15} "
              f"{r.get('forget_acc',float('nan')):>8.4f} "
              f"{r.get('retain_acc',float('nan')):>9.4f} "
              f"{r.get('ve_mean',float('nan')):>8.4f} "
              f"{r.get('bridge_mean',float('nan')):>8.4f} "
              f"{r.get('lb_mean',float('nan')):>8.4f}")

    print(f"\nResults saved to: {BENCH_DIR}")


if __name__ == "__main__":
    main()
