from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapter_guard import assert_adapter_is_active
from exp_config import DEVICE, LLAVA_ADAPTERS, LLAVA_BASE, LLAVA_LB_LAYERS, LLAVA_VE_LAYERS, MLLMU_REAL_FORGET, RESULTS_DIR
from E2_run_audit_per_entity import debiased_cka, find_module, load_split, unwrap

OUTDIR = RESULTS_DIR / "full_precision_replication"
OUTDIR.mkdir(parents=True, exist_ok=True)


def load_model(ckpt, dtype):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from peft import PeftModel
    processor = AutoProcessor.from_pretrained(LLAVA_BASE, use_fast=False)
    if ckpt is None:
        model = LlavaForConditionalGeneration.from_pretrained(LLAVA_BASE, torch_dtype=dtype, device_map=DEVICE)
    else:
        ckpt = Path(ckpt)
        if (ckpt / "adapter_config.json").exists():
            assert_adapter_is_active(ckpt)
            base = LlavaForConditionalGeneration.from_pretrained(LLAVA_BASE, torch_dtype=dtype, device_map=DEVICE)
            model = PeftModel.from_pretrained(base, str(ckpt), is_trainable=False)
        else:
            model = LlavaForConditionalGeneration.from_pretrained(str(ckpt), torch_dtype=dtype, device_map=DEVICE)
    model.eval()
    return model, processor


def extract(model, processor, items, pooling: str):
    core = unwrap(model)
    outputs = []

    def reducer(h):
        h = h.detach().float().cpu()
        if h.dim() == 3:
            h = h[0]
        if h.dim() == 2:
            return h.mean(0) if pooling == "mean" else h[-1]
        return h.flatten()

    for i, item in enumerate(items, 1):
        captured, handles = {}, []
        def hook(key):
            def fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(h):
                    captured[key] = reducer(h)
            return fn

        lb, _ = find_module(core, ["model.language_model.model.layers", "model.language_model.layers", "language_model.model.layers", "language_model.layers"], "LB")
        ve, _ = find_module(core, ["model.vision_tower.vision_model.encoder.layers", "vision_tower.vision_model.encoder.layers"], "VE")
        br, _ = find_module(core, ["model.multi_modal_projector", "multi_modal_projector"], "Bridge")
        if lb is not None:
            for idx in LLAVA_LB_LAYERS:
                if idx < len(lb): handles.append(lb[idx].register_forward_hook(hook(f"lb_{idx}")))
        if ve is not None:
            for idx in LLAVA_VE_LAYERS:
                if idx < len(ve): handles.append(ve[idx].register_forward_hook(hook(f"ve_{idx}")))
        if br is not None: handles.append(br.register_forward_hook(hook("bridge")))

        image = Image.open(item["image"]).convert("RGB")
        prompt = f"USER: <image>\n{item['question']} ASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad(): model(**inputs, use_cache=False)
        for h in handles: h.remove()
        outputs.append(captured)
        print(f"[{pooling}] {i}/{len(items)}")
    return outputs


def crp(a, b):
    def comp(keys):
        vals = []
        for key in keys:
            xa = [r[key] for r in a if key in r]
            xb = [r[key] for r in b if key in r]
            if len(xa) == len(a) == len(xb):
                v = debiased_cka(torch.stack(xa), torch.stack(xb))
                if math.isfinite(v): vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")
    return {
        "ve": comp([f"ve_{i}" for i in LLAVA_VE_LAYERS]),
        "bridge": comp(["bridge"]),
        "lb": comp([f"lb_{i}" for i in LLAVA_LB_LAYERS]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="ga_retrained")
    parser.add_argument("--n_items", type=int, default=40)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--max_pool_diff", type=float, default=0.10)
    args = parser.parse_args()

    checks = []
    def check(name, passed, detail): checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    ckpt = LLAVA_ADAPTERS.get(args.method)
    check("Checkpoint exists", ckpt is not None and Path(ckpt).exists(), ckpt)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    try:
        items = load_split(MLLMU_REAL_FORGET)[:args.n_items]
        check("Adequate matched sample", len(items) >= 20, f"n={len(items)}")
        results = {}
        for pooling in ("mean", "last"):
            m0, proc = load_model(None, dtype)
            a0 = extract(m0, proc, items, pooling)
            del m0; gc.collect(); torch.cuda.empty_cache()
            mu, proc = load_model(ckpt, dtype)
            au = extract(mu, proc, items, pooling)
            del mu; gc.collect(); torch.cuda.empty_cache()
            results[pooling] = crp(a0, au)
            check(f"Finite {pooling} CRP", all(math.isfinite(v) for v in results[pooling].values()), results[pooling])

        pool_diff = {k: abs(results["mean"][k] - results["last"][k]) for k in results["mean"]}
        check("Pooling sensitivity bounded", max(pool_diff.values()) <= args.max_pool_diff, pool_diff)
    except Exception as exc:
        results = {}
        check("Execution", False, f"{type(exc).__name__}: {exc}")

    overall = "PASS" if checks and all(x["passed"] for x in checks) else "FAIL"
    payload = {"overall": overall, "method": args.method, "dtype": args.dtype, "n_items": args.n_items, "results": results, "checks": checks}
    out = OUTDIR / f"full_precision_{args.method}_{args.dtype}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + "=" * 88)
    for x in checks: print(f"[{'PASS' if x['passed'] else 'FAIL'}] {x['name']}: {x['detail']}")
    print(f"OVERALL VERDICT: {overall}")
    print(f"REPORT: {out}")
    return 0 if overall == "PASS" else 1

if __name__ == "__main__": raise SystemExit(main())
