"""
stage2_save_activations.py
───────────────────────────
Saves paired [n_samples, d] activation matrices for the five existing
valid LLaVA checkpoints. No new training. No overwriting of existing CRP
point estimates.

After each method: validates the saved matrices are non-degenerate
before proceeding to the next method.

Stop conditions:
  - M0 activations produce NaN → STOP
  - Saved matrix shape is wrong → STOP
  - GPU OOM → STOP with message

After all methods complete, immediately re-runs fix1_bootstrap_ci_crp.py
to verify valid CIs are produced.

Outputs go to: outputs/revision/activations/
Existing results in: outputs/crp_per_entity/ are NOT touched.

Usage:
    py stage2_save_activations.py
    py stage2_save_activations.py --methods npo mmunlearner --resume
    py stage2_save_activations.py --verify_only  # check saved files only
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
    LLAVA_BASE, LLAVA_ADAPTERS, DEVICE,
    MLLMU_REAL_FORGET, RESULTS_DIR,
    LLAVA_VE_LAYERS, LLAVA_LB_LAYERS,
)

ACT_DIR = RESULTS_DIR / "activations"
ACT_DIR.mkdir(parents=True, exist_ok=True)

VALID_METHODS = ["npo", "mmunlearner", "cagul", "sineproject", "graddiff"]
CHECKS = []


def chk(label, verdict, detail):
    CHECKS.append({"check": label, "verdict": verdict, "detail": str(detail)})
    icon = {"PASS": "OK", "WARN": "!!", "FAIL": "XX"}.get(verdict, "??")
    print(f"  [{icon} {verdict}] {label}: {detail}")
    if verdict == "FAIL":
        print(f"\n  HARD STOP: {label} failed. Fix before continuing.")
    return verdict


def save_report():
    n_p = sum(1 for c in CHECKS if c["verdict"] == "PASS")
    n_w = sum(1 for c in CHECKS if c["verdict"] == "WARN")
    n_f = sum(1 for c in CHECKS if c["verdict"] == "FAIL")
    rp  = ACT_DIR / "stage2_report.json"
    with open(rp, "w", encoding="utf-8") as f:
        json.dump({"checks": CHECKS,
                   "n_pass": n_p, "n_warn": n_w, "n_fail": n_f}, f, indent=2)
    print(f"\n  Report: {n_p} PASS  {n_w} WARN  {n_f} FAIL  → {rp}")
    return n_f == 0


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


def unwrap(m):
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        return m.base_model.model
    return m


def find_module(obj, paths):
    for path in paths:
        cur = obj; ok = True
        for p in path.split("."):
            if not hasattr(cur, p): ok = False; break
            cur = getattr(cur, p)
        if ok and cur is not None: return cur
    return None


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
                item["image"] = p; result.append(item)
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
                          "question": q["question"], "answer": q.get("answer","")})
    return items


def extract_sample(model, proc, item):
    core     = unwrap(model)
    image    = Image.open(item["image"]).convert("RGB")
    prompt   = f"USER: <image>\n{item['question']} ASSISTANT:"
    inputs   = proc(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    captured = {}

    def hook(key):
        def fn(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(h): return
            h = h.detach().float().cpu()
            if h.dim() == 3:   h = h[0].mean(0)
            elif h.dim() == 2: h = h.mean(0)
            if not torch.isnan(h).any(): captured[key] = h
        return fn

    handles = []
    ve = find_module(core, ["model.vision_tower.vision_model.encoder.layers",
                             "vision_tower.vision_model.encoder.layers"])
    if ve:
        for idx in LLAVA_VE_LAYERS:
            if idx < len(ve):
                handles.append(ve[idx].register_forward_hook(hook(f"ve_{idx}")))
    br = find_module(core, ["model.multi_modal_projector","multi_modal_projector"])
    if br: handles.append(br.register_forward_hook(hook("bridge")))
    lb = find_module(core, ["model.language_model.model.layers",
                             "model.language_model.layers","language_model.model.layers"])
    if lb:
        for idx in LLAVA_LB_LAYERS:
            if idx < len(lb):
                handles.append(lb[idx].register_forward_hook(hook(f"lb_{idx}")))

    with torch.no_grad():
        model(**inputs, use_cache=False)
    for h in handles: h.remove()
    return captured


def stack_comp(all_acts, keys):
    per_sample = []
    for acts in all_acts:
        vecs = [acts[k] for k in keys if k in acts]
        if vecs: per_sample.append(torch.stack(vecs).mean(0))
    return torch.stack(per_sample) if per_sample else None


def validate_matrix(mat, label, n_expected):
    """Hard checks on a saved activation matrix."""
    if mat is None:
        return chk(f"{label} matrix", "FAIL", "Matrix is None — hooks did not fire")
    if mat.shape[0] != n_expected:
        return chk(f"{label} matrix", "FAIL",
                   f"Shape {mat.shape} — expected {n_expected} samples on axis 0")
    if mat.dim() != 2:
        return chk(f"{label} matrix", "FAIL", f"Expected 2D, got {mat.dim()}D")
    if torch.isnan(mat).any():
        return chk(f"{label} matrix", "FAIL", "Contains NaN values")
    if mat.abs().max().item() < 1e-10:
        return chk(f"{label} matrix", "WARN", "All values near zero — possible dead hook")
    chk(f"{label} matrix", "PASS", f"shape={tuple(mat.shape)}  "
        f"max={mat.abs().max().item():.4f}  mean={mat.abs().mean().item():.6f}")
    return "PASS"


def extract_and_save(model_label, ckpt_path, items, method):
    """Extract activations for one model and save matrices."""
    ve_keys = [f"ve_{i}" for i in LLAVA_VE_LAYERS]
    br_keys = ["bridge"]
    lb_keys = [f"lb_{i}" for i in LLAVA_LB_LAYERS]

    print(f"    Loading {model_label}...")
    try:
        model, proc = load_model(ckpt_path)
    except Exception as e:
        chk(f"{method}/{model_label} load", "FAIL", str(e))
        return False

    all_acts = []
    for i, item in enumerate(items):
        try:
            acts = extract_sample(model, proc, item)
            all_acts.append(acts)
        except torch.cuda.OutOfMemoryError:
            chk(f"{method}/{model_label} GPU", "FAIL",
                f"OOM at sample {i}. Reduce batch or free GPU memory.")
            del model; torch.cuda.empty_cache()
            return False
        except Exception as e:
            chk(f"{method}/{model_label} sample {i}", "WARN", str(e))
            all_acts.append({})
        if (i+1) % 10 == 0:
            print(f"      [{i+1}/{len(items)}]")

    ve_mat = stack_comp(all_acts, ve_keys)
    br_mat = stack_comp(all_acts, br_keys)
    lb_mat = stack_comp(all_acts, lb_keys)

    # Validate before saving
    n = len(items)
    ve_ok = validate_matrix(ve_mat, f"{method}/{model_label}/VE", n)
    br_ok = validate_matrix(br_mat, f"{method}/{model_label}/BR", n)
    lb_ok = validate_matrix(lb_mat, f"{method}/{model_label}/LB", n)

    if "FAIL" in [ve_ok, br_ok, lb_ok]:
        del model; torch.cuda.empty_cache()
        return False

    # Save
    for comp, mat in [("ve", ve_mat), ("br", br_mat), ("lb", lb_mat)]:
        if mat is not None:
            out = ACT_DIR / f"{method}_{model_label}_{comp}.pt"
            torch.save(mat, str(out))

    del model; torch.cuda.empty_cache()
    return True


def verify_saved(method, n_expected):
    """Check that all six saved files exist and are valid."""
    all_ok = True
    for model_label in ("m0", "mu"):
        for comp in ("ve", "br", "lb"):
            p = ACT_DIR / f"{method}_{model_label}_{comp}.pt"
            if not p.exists():
                chk(f"{method} verify {model_label}/{comp}", "FAIL", "File missing")
                all_ok = False
                continue
            try:
                mat = torch.load(str(p), weights_only=True)
                v   = validate_matrix(mat, f"{method}/{model_label}/{comp}", n_expected)
                if v == "FAIL": all_ok = False
            except Exception as e:
                chk(f"{method} verify {model_label}/{comp}", "FAIL", str(e))
                all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods",     nargs="+", default=VALID_METHODS)
    parser.add_argument("--resume",      action="store_true", default=True)
    parser.add_argument("--verify_only", action="store_true")
    args = parser.parse_args()

    print("\n=== STAGE 2: ACTIVATION SAVING ===")
    print(f"  Output: {ACT_DIR}")
    print("  Existing CRP results will NOT be modified.")

    items = load_forget_items()
    if not items:
        chk("forget items", "FAIL", "No forget items found")
        save_report(); sys.exit(1)
    chk("forget items", "PASS", f"{len(items)} items loaded")

    if args.verify_only:
        print("\n  Verify mode — checking existing files only")
        for method in args.methods:
            verify_saved(method, len(items))
        save_report(); return

    for method in args.methods:
        print(f"\n  --- Method: {method.upper()} ---")
        ckpt = LLAVA_ADAPTERS.get(method)
        if not ckpt or not Path(str(ckpt)).exists():
            chk(f"{method} checkpoint", "FAIL", f"Not found: {ckpt}")
            print("  Skipping this method.")
            continue
        chk(f"{method} checkpoint", "PASS", str(ckpt))

        # Check if already done
        meta_path = ACT_DIR / f"{method}_meta.json"
        if args.resume and meta_path.exists():
            print(f"  [cached] All files exist. Verifying...")
            ok = verify_saved(method, len(items))
            if ok:
                chk(f"{method} cached", "PASS", "All matrices valid")
                continue
            else:
                print("  Cached files invalid — re-extracting.")

        t0 = time.time()

        # M0 first
        ok_m0 = extract_and_save(None, None, items, method)
        if not ok_m0:
            chk(f"{method} M0 extraction", "FAIL",
                "M0 extraction failed — stopping this method")
            continue
        chk(f"{method} M0 extraction", "PASS",
            f"{time.time()-t0:.0f}s")

        # Mu
        ok_mu = extract_and_save("mu", ckpt, items, method)
        if not ok_mu:
            chk(f"{method} Mu extraction", "FAIL",
                "Mu extraction failed — stopping this method")
            continue
        chk(f"{method} Mu extraction", "PASS",
            f"{time.time()-t0:.0f}s total")

        # Rename m0 files (shared across methods)
        for comp in ("ve","br","lb"):
            src = ACT_DIR / f"{method}_None_{comp}.pt"
            dst = ACT_DIR / f"{method}_m0_{comp}.pt"
            if src.exists(): src.rename(dst)

        # Save metadata
        meta = {"method": method, "checkpoint": str(ckpt),
                "n_items": len(items), "ve_layers": LLAVA_VE_LAYERS,
                "lb_layers": LLAVA_LB_LAYERS, "pooling": "mean_token_then_mean_layer"}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Final verification
        ok_verify = verify_saved(method, len(items))
        if not ok_verify:
            chk(f"{method} final verify", "FAIL", "Post-save validation failed")

    # Run Fix 1 automatically if all checks passed
    clean = save_report()
    if clean:
        print("\n  All methods saved. Running Fix 1 to compute valid CIs...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "fix1_bootstrap_ci_crp.py"],
            capture_output=True, text=True)
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.returncode != 0:
            print("[WARN] Fix 1 returned non-zero. Check output above.")
    else:
        print("\n  Some methods failed. Fix before running Fix 1.")


if __name__ == "__main__":
    main()
