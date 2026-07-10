"""
E9_fp16_sensitivity.py
───────────────────────
Preliminary quantization-sensitivity check for CRP.

Loads M0 in FP16 (sequential, not simultaneous), extracts and saves
activations, unloads, then loads Mu in FP16 and extracts matching
activations. Compares FP16 CKA to 4-bit NF4 CKA on the same subset.

This is described as a PRELIMINARY check, not a full-precision
replication. The subset is small (default 20 items) and is the
same 20 items used in E2 for a fair comparison.

FP16 memory note:
  LLaVA-1.5-7B in FP16 requires ~14 GB. Only ONE model is loaded
  at a time (sequential extraction), so peak VRAM is ~14 GB.
  The RTX 4090 has 24 GB. This should fit.
  If it OOMs, reduce --n_items or use --dtype bfloat16.

Usage:
    py E9_fp16_sensitivity.py
    py E9_fp16_sensitivity.py --n_items 10
    py E9_fp16_sensitivity.py --method ga_retrained
    py E9_fp16_sensitivity.py --dtype bfloat16
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from adapter_guard import assert_adapter_is_active

from exp_config import (
    LLAVA_BASE, LLAVA_ADAPTERS, DEVICE,
    MLLMU_REAL_FORGET, RESULTS_DIR,
    LLAVA_VE_LAYERS, LLAVA_LB_LAYERS,
    PER_ENTITY_CRP_DIR,
)

FP16_DIR = RESULTS_DIR / "fp16_sensitivity"
FP16_DIR.mkdir(parents=True, exist_ok=True)


def load_fp16(ckpt_path, dtype_str: str, label: str):
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    print(f"  Loading {label} in {dtype_str}...")
    proc  = AutoProcessor.from_pretrained(LLAVA_BASE)
    ckpt  = Path(ckpt_path) if ckpt_path else None
    if ckpt is None:
        model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_BASE, torch_dtype=dtype, device_map=DEVICE)
    elif (ckpt / "adapter_config.json").exists():
        base  = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_BASE, torch_dtype=dtype, device_map=DEVICE)
        assert_adapter_is_active(ckpt)
        model = PeftModel.from_pretrained(
            base,
            str(ckpt),
            is_trainable=False,
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            str(ckpt), torch_dtype=dtype, device_map=DEVICE)
    model.eval()
    print(f"  {label}: {type(model).__name__}, dtype={dtype_str}")
    return model, proc


def unwrap(model):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def find_module(obj, paths, label):
    for path in paths:
        cur = obj; ok = True
        for part in path.split("."):
            if not hasattr(cur, part): ok = False; break
            cur = getattr(cur, part)
        if ok and cur is not None:
            return cur, path
    return None, None


def extract_save(model, proc, items, save_path: Path, label: str) -> list[dict]:
    """Extract activations and save to disk (sequential strategy)."""
    core = unwrap(model)
    all_acts = []

    def make_hook(key, d):
        def fn(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(h): return
            h = h.detach().float().cpu()
            if h.dim() == 3:   h = h[0].mean(0)
            elif h.dim() == 2: h = h.mean(0)
            if not torch.isnan(h).any():
                d[key] = h
        return fn

    for i, item in enumerate(items):
        captured = {}
        handles  = []
        image    = Image.open(item["image"]).convert("RGB")
        prompt   = f"USER: <image>\n{item['question']} ASSISTANT:"
        inputs   = proc(text=prompt, images=image, return_tensors="pt").to(DEVICE)

        ve, _ = find_module(core, [
            "model.vision_tower.vision_model.encoder.layers",
            "vision_tower.vision_model.encoder.layers"], "VE")
        if ve:
            for idx in LLAVA_VE_LAYERS:
                if idx < len(ve):
                    handles.append(ve[idx].register_forward_hook(
                        make_hook(f"ve_{idx}", captured)))

        br, _ = find_module(core, [
            "model.multi_modal_projector", "multi_modal_projector"], "Bridge")
        if br:
            handles.append(br.register_forward_hook(make_hook("bridge", captured)))

        lb, _ = find_module(core, [
            "model.language_model.model.layers", "model.language_model.layers",
            "language_model.model.layers"], "LB")
        if lb:
            for idx in LLAVA_LB_LAYERS:
                if idx < len(lb):
                    handles.append(lb[idx].register_forward_hook(
                        make_hook(f"lb_{idx}", captured)))

        with torch.no_grad():
            model(**inputs, use_cache=False)

        for h in handles: h.remove()
        all_acts.append(captured)
        print(f"  [{i+1:02d}/{len(items)}] {item.get('entity','?')}  "
              f"keys={sorted(captured.keys())}" if i == 0 else
              f"  [{i+1:02d}/{len(items)}]")

    # Save as numpy arrays
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np_acts = [{k: v.numpy() for k, v in d.items()} for d in all_acts]
    torch.save(np_acts, str(save_path))
    print(f"  [saved] {save_path}")
    return all_acts


def debiased_cka(X, Y):
    if X is None or Y is None or X.shape[0] < 4: return float("nan")
    X = X.float() - X.float().mean(0, keepdim=True)
    Y = Y.float() - Y.float().mean(0, keepdim=True)
    n = X.shape[0]; K = X@X.T; L = Y@Y.T
    Kt = K-torch.diag(torch.diag(K)); Lt = L-torch.diag(torch.diag(L))
    c = 1./(n*(n-3))
    def hsic(A,B): return c*((A*B).sum()-(2./(n-2))*(A.sum(1)*B.sum(1)).sum()
                               +A.sum()*B.sum()/((n-1)*(n-2)))
    v = torch.clamp(hsic(Kt,Lt)/torch.sqrt(torch.clamp(hsic(Kt,Kt)*hsic(Lt,Lt),
                                                        min=1e-10)), 0., 1.)
    return float("nan") if torch.isnan(v) else float(v.item())


def compute_crp(acts_m0: list, acts_mu: list) -> dict:
    """Compute component CKA from two activation lists."""
    def stack(acts, key):
        vecs = [torch.tensor(d[key]) for d in acts if key in d]
        return torch.stack(vecs) if vecs else None

    lb_keys = [f"lb_{i}" for i in LLAVA_LB_LAYERS]
    ve_keys = [f"ve_{i}" for i in LLAVA_VE_LAYERS]

    def comp_cka(keys):
        vals = []
        for k in keys:
            X = stack(acts_m0, k); Y = stack(acts_mu, k)
            if X is not None and Y is not None:
                c = debiased_cka(X, Y)
                if not np.isnan(c): vals.append(c)
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "ve_cka":     comp_cka(ve_keys),
        "bridge_cka": comp_cka(["bridge"]),
        "lb_cka":     comp_cka(lb_keys),
    }


def load_nf4_crp(method: str) -> dict:
    """Load existing NF4 CRP result for comparison."""
    p = PER_ENTITY_CRP_DIR / f"{method}_per_entity_crp.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return {
        "ve_cka":     float(d.get("ve_mean", float("nan"))),
        "bridge_cka": float(d.get("bridge_mean", float("nan"))),
        "lb_cka":     float(d.get("lb_mean", float("nan"))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",   default="ga_retrained")
    parser.add_argument("--n_items",  type=int, default=20)
    parser.add_argument("--dtype",    default="float16",
                        choices=["float16", "bfloat16"])
    args = parser.parse_args()

    ckpt = LLAVA_ADAPTERS.get(args.method)
    if not ckpt or not Path(ckpt).exists():
        print(f"[ERROR] Checkpoint missing for '{args.method}': {ckpt}")
        sys.exit(1)

    # Load forget items (same first N as E2)
    ann = MLLMU_REAL_FORGET / "annotations.json"
    if ann.exists():
        with open(ann, encoding="utf-8") as f:
            items = json.load(f)[:args.n_items]
    else:
        items = []
        for entity_dir in sorted(MLLMU_REAL_FORGET.iterdir()):
            if not entity_dir.is_dir(): continue
            imgs  = list(entity_dir.glob("*.jpg"))
            jsons = list(entity_dir.glob("*.json"))
            if not imgs or not jsons: continue
            with open(jsons[0], encoding="utf-8") as f:
                qa = json.load(f)
            for q in (qa if isinstance(qa, list) else [qa])[:1]:
                items.append({"entity": entity_dir.name, "image": str(imgs[0]),
                               "question": q["question"]})
            if len(items) >= args.n_items: break

    print(f"\n[E9] FP16 Sensitivity Check")
    print(f"  Method: {args.method}  |  N: {len(items)}  |  dtype: {args.dtype}")
    print("  Strategy: sequential load (one model at a time)")
    print("  Peak VRAM ~ 14 GB for LLaVA-1.5-7B in {args.dtype}")

    m0_save = FP16_DIR / f"m0_{args.dtype}_{len(items)}items.pt"
    mu_save = FP16_DIR / f"{args.method}_{args.dtype}_{len(items)}items.pt"

    # STEP 1: Load M0, extract, UNLOAD
    if m0_save.exists():
        print(f"\n[E9] M0 activations cached: {m0_save}")
        acts_m0 = torch.load(str(m0_save), weights_only=False)
    else:
        print(f"\n[E9] STEP 1: Extracting M0 ({args.dtype})...")
        m0, proc = load_fp16(None, args.dtype, "M0")
        acts_m0  = extract_save(m0, proc, items, m0_save, "M0")
        del m0; torch.cuda.empty_cache()
        print(f"  M0 unloaded. CUDA cache cleared.")

    # STEP 2: Load Mu, extract, UNLOAD
    if mu_save.exists():
        print(f"\n[E9] Mu activations cached: {mu_save}")
        acts_mu = torch.load(str(mu_save), weights_only=False)
    else:
        print(f"\n[E9] STEP 2: Extracting Mu-{args.method} ({args.dtype})...")
        mu, proc = load_fp16(ckpt, args.dtype, f"Mu-{args.method}")
        acts_mu  = extract_save(mu, proc, items, mu_save, f"Mu-{args.method}")
        del mu; torch.cuda.empty_cache()
        print(f"  Mu unloaded. CUDA cache cleared.")

    # STEP 3: Compute CRP
    print(f"\n[E9] STEP 3: Computing CRP from {args.dtype} activations...")
    fp16_crp = compute_crp(acts_m0, acts_mu)
    nf4_crp  = load_nf4_crp(args.method)

    print(f"\n  CRP COMPARISON ({args.method}):")
    print(f"  {'Metric':<12} {'NF4 (E2)':>12} {args.dtype.upper():>12}  "
          f"{'|diff|':>8}")
    for key in ["ve_cka", "bridge_cka", "lb_cka"]:
        nf4 = nf4_crp.get(key, float("nan"))
        fp  = fp16_crp.get(key, float("nan"))
        try:
            diff = abs(float(nf4) - float(fp))
            diff_str = f"{diff:.4f}"
        except (TypeError, ValueError):
            diff_str = "nan"
        print(f"  {key:<12} {nf4:>12.4f} {fp:>12.4f}  {diff_str:>8}")

    result = {
        "method":   args.method,
        "n_items":  len(items),
        "dtype":    args.dtype,
        "fp16_crp": fp16_crp,
        "nf4_crp":  nf4_crp,
        "interpretation": (
            "PRELIMINARY quantization-sensitivity check. "
            "Not a full-precision replication. "
            "If |diff| < 0.01 for all components, NF4 results are "
            "unlikely to be quantization artefacts. "
            "This does not resolve the frozen-component anomaly question."
        ),
    }
    out = FP16_DIR / f"fp16_sensitivity_{args.method}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[saved] {out}")
    print("\n  NOTE: This is a preliminary check, not a full-precision replication.")
    print("  Report as: 'We verify on N=X samples that NF4 and FP16 CRP values")
    print("  differ by at most Y, suggesting quantization is not the primary driver.'")


if __name__ == "__main__":
    main()
