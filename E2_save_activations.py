"""
E2_save_activations.py
───────────────────────
Re-runs the CRP extraction for all valid methods on mllmu_real
and saves raw paired activation matrices to disk.

This enables Fix 1 to compute VALID sample-level bootstrap CIs
by resampling the 40 forget examples (not layers).

Saves per method:
  outputs/revision/activations/{method}_m0_ve.pt   shape [40, d_ve]
  outputs/revision/activations/{method}_mu_ve.pt   shape [40, d_ve]
  outputs/revision/activations/{method}_m0_br.pt   shape [40, d_br]
  outputs/revision/activations/{method}_mu_br.pt   shape [40, d_br]
  outputs/revision/activations/{method}_m0_lb.pt   shape [40, d_lb]
  outputs/revision/activations/{method}_mu_lb.pt   shape [40, d_lb]
  outputs/revision/activations/{method}_meta.json

After running, re-run fix1_bootstrap_ci_crp.py — it will find these
files and compute valid CIs automatically.

Usage:
    py E2_save_activations.py
    py E2_save_activations.py --methods npo mmunlearner --resume
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
    LLAVA_BASE, LLAVA_ADAPTERS, DEVICE,
    MLLMU_REAL_FORGET, RESULTS_DIR,
    LLAVA_VE_LAYERS, LLAVA_LB_LAYERS,
)

ACT_DIR = RESULTS_DIR / "activations"
ACT_DIR.mkdir(parents=True, exist_ok=True)

VALID_METHODS = ["npo", "mmunlearner", "cagul", "sineproject", "graddiff"]


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
        for part in path.split("."):
            if not hasattr(cur, part): ok = False; break
            cur = getattr(cur, part)
        if ok and cur is not None:
            return cur
    return None


def extract_one(model, proc, item: dict) -> dict:
    """Extract mean-pooled activations for one item. Returns {comp: tensor(d,)}."""
    core    = unwrap(model)
    image   = Image.open(item["image"]).convert("RGB")
    prompt  = f"USER: <image>\n{item['question']} ASSISTANT:"
    inputs  = proc(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    captured = {}

    def hook(key):
        def fn(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(h): return
            h = h.detach().float().cpu()
            if h.dim() == 3:   h = h[0].mean(0)
            elif h.dim() == 2: h = h.mean(0)
            if not torch.isnan(h).any():
                captured[key] = h
        return fn

    handles = []
    ve = find_module(core, [
        "model.vision_tower.vision_model.encoder.layers",
        "vision_tower.vision_model.encoder.layers"])
    if ve:
        for idx in LLAVA_VE_LAYERS:
            if idx < len(ve):
                handles.append(ve[idx].register_forward_hook(hook(f"ve_{idx}")))

    br = find_module(core, [
        "model.multi_modal_projector", "multi_modal_projector"])
    if br:
        handles.append(br.register_forward_hook(hook("bridge")))

    lb = find_module(core, [
        "model.language_model.model.layers", "model.language_model.layers",
        "language_model.model.layers"])
    if lb:
        for idx in LLAVA_LB_LAYERS:
            if idx < len(lb):
                handles.append(lb[idx].register_forward_hook(hook(f"lb_{idx}")))

    with torch.no_grad():
        model(**inputs, use_cache=False)
    for h in handles: h.remove()
    return captured


def stack_component(all_acts: list, keys: list) -> torch.Tensor:
    """
    Stack per-sample mean-pooled activations across a component.
    Averages over the hook layers within the component.
    Returns tensor of shape [n_samples, d].
    """
    per_sample = []
    for acts in all_acts:
        layer_vecs = []
        for k in keys:
            if k in acts:
                layer_vecs.append(acts[k])
        if layer_vecs:
            per_sample.append(torch.stack(layer_vecs).mean(0))
    if not per_sample:
        return None
    return torch.stack(per_sample)   # [n, d]


def load_forget_items():
    ann = MLLMU_REAL_FORGET / "annotations.json"
    if ann.exists():
        with open(ann, encoding="utf-8") as f:
            items = json.load(f)
        result = []
        for item in items:
            p = Path(item["image"])
            if not p.is_absolute(): p = MLLMU_REAL_FORGET / p
            if p.exists():
                item["image"] = p
                result.append(item)
        return result

    items = []
    for d in sorted(MLLMU_REAL_FORGET.iterdir()):
        if not d.is_dir(): continue
        imgs  = list(d.glob("*.jpg")) + list(d.glob("*.png"))
        jsons = list(d.glob("*.json"))
        if not imgs or not jsons: continue
        with open(jsons[0], encoding="utf-8") as f:
            qa = json.load(f)
        for q in (qa if isinstance(qa, list) else [qa])[:1]:
            items.append({"entity": d.name, "image": imgs[0],
                          "question": q["question"],
                          "answer": q.get("answer", "")})
    return items


def run_method(method: str, ckpt_path, items: list, resume: bool):
    """Extract and save activations for M0 and Mu for one method."""
    meta_path = ACT_DIR / f"{method}_meta.json"
    if resume and meta_path.exists():
        print(f"  [cached] {method}")
        return

    ve_keys = [f"ve_{i}" for i in LLAVA_VE_LAYERS]
    br_keys = ["bridge"]
    lb_keys = [f"lb_{i}" for i in LLAVA_LB_LAYERS]

    for model_label, ckpt in [("m0", None), ("mu", ckpt_path)]:
        print(f"\n  Loading {model_label} for {method}...")
        model, proc = load_model(ckpt)

        all_acts = []
        for i, item in enumerate(items):
            acts = extract_one(model, proc, item)
            all_acts.append(acts)
            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(items)}]  hooks={sorted(acts.keys())[:4]}")

        # Stack by component
        ve_mat = stack_component(all_acts, ve_keys)
        br_mat = stack_component(all_acts, br_keys)
        lb_mat = stack_component(all_acts, lb_keys)

        for comp, mat in [("ve", ve_mat), ("br", br_mat), ("lb", lb_mat)]:
            if mat is not None:
                out = ACT_DIR / f"{method}_{model_label}_{comp}.pt"
                torch.save(mat, str(out))
                print(f"    saved {out}  shape={mat.shape}")
            else:
                print(f"    [warn] {comp} matrix is None")

        del model
        torch.cuda.empty_cache()

    # Save metadata
    meta = {
        "method":      method,
        "checkpoint":  str(ckpt_path),
        "n_items":     len(items),
        "ve_layers":   LLAVA_VE_LAYERS,
        "lb_layers":   LLAVA_LB_LAYERS,
        "pooling":     "mean over tokens then mean over layers",
        "base_model":  LLAVA_BASE,
        "note": ("Raw activation matrices saved for valid sample-level bootstrap CIs. "
                 "Run fix1_bootstrap_ci_crp.py to compute CIs.")
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  [saved] {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=VALID_METHODS)
    parser.add_argument("--resume",  action="store_true", default=True)
    args = parser.parse_args()

    items = load_forget_items()
    if not items:
        print("[ERROR] No forget items found."); sys.exit(1)
    print(f"[E2_save] Forget items: {len(items)}")
    print(f"  Saving activations to: {ACT_DIR}")

    for method in args.methods:
        ckpt = LLAVA_ADAPTERS.get(method)
        if not ckpt or not Path(str(ckpt)).exists():
            print(f"\n  [skip] {method}: checkpoint missing"); continue
        print(f"\n  Method: {method.upper()}")
        run_method(method, ckpt, items, args.resume)

    print("\n[E2_save] Done. Now run:")
    print("  py fix1_bootstrap_ci_crp.py")
    print("  This will find the saved matrices and compute valid CIs.")


if __name__ == "__main__":
    main()
