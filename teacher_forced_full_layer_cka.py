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

from exp_config import DEVICE, LLAVA_ADAPTERS, LLAVA_BASE, MLLMU_REAL_FORGET, RESULTS_DIR
from E2_run_audit_per_entity import debiased_cka, find_module, load_llava, load_split, unwrap

OUTDIR = RESULTS_DIR / "teacher_forced_full_layer"
OUTDIR.mkdir(parents=True, exist_ok=True)
DEFAULT_METHODS = ["ga_retrained", "npo", "mmunlearner", "cagul", "sineproject"]


def extract(model, processor, item, layer_ids):
    core = unwrap(model)
    layers, path = find_module(core, ["model.language_model.model.layers", "model.language_model.layers", "language_model.model.layers", "language_model.layers"], "LB")
    if layers is None: raise RuntimeError("Language layers not found")
    captured, handles = {}, []
    for idx in layer_ids:
        if idx >= len(layers): continue
        def make_hook(key):
            def fn(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(h): captured[key] = h.detach().float().cpu()[0].mean(0)
            return fn
        handles.append(layers[idx].register_forward_hook(make_hook(f"lb_{idx}")))

    image = Image.open(item["image"]).convert("RGB")
    prompt = f"USER: <image>\n{item['question']} ASSISTANT: {item.get('answer', '')}"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    labels = inputs["input_ids"].clone()
    with torch.no_grad(): model(**inputs, labels=labels, use_cache=False)
    for h in handles: h.remove()
    return captured, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--n_items", type=int, default=20)
    args = parser.parse_args()

    checks = []
    def check(name, passed, detail): checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    items = load_split(MLLMU_REAL_FORGET)[:args.n_items]
    check("Adequate sample", len(items) >= 18, f"n={len(items)}")

    m0, proc = load_llava(None)
    core = unwrap(m0)
    layers, _ = find_module(core, ["model.language_model.model.layers", "model.language_model.layers", "language_model.model.layers", "language_model.layers"], "LB")
    layer_ids = list(range(len(layers))) if layers is not None else []
    check("Full language layer list found", len(layer_ids) >= 24, f"layers={len(layer_ids)}")
    base = []
    for item in items:
        acts, _ = extract(m0, proc, item, layer_ids)
        base.append(acts)
    del m0; gc.collect(); torch.cuda.empty_cache()

    results = {}
    for method in args.methods:
        ckpt = LLAVA_ADAPTERS.get(method)
        if ckpt is None or not Path(ckpt).exists():
            check(f"{method} checkpoint", False, ckpt)
            continue
        mu, proc = load_llava(ckpt)
        rows = [extract(mu, proc, item, layer_ids)[0] for item in items]
        per_layer = {}
        for idx in layer_ids:
            key = f"lb_{idx}"
            if all(key in r for r in base) and all(key in r for r in rows):
                per_layer[str(idx)] = debiased_cka(torch.stack([r[key] for r in base]), torch.stack([r[key] for r in rows]))
        agg = float(np.mean([v for v in per_layer.values() if math.isfinite(v)])) if per_layer else float("nan")
        results[method] = {"aggregate_lb_cka": agg, "per_layer": per_layer, "n_items": len(items)}
        check(f"{method} full-layer result", math.isfinite(agg) and len(per_layer) == len(layer_ids), f"aggregate={agg}, layers={len(per_layer)}")
        del mu; gc.collect(); torch.cuda.empty_cache()

    overall = "PASS" if checks and all(x["passed"] for x in checks) else "FAIL"
    payload = {"overall": overall, "layers": layer_ids, "entities": [x.get("entity") for x in items], "results": results, "checks": checks}
    out = OUTDIR / "teacher_forced_full_layer_results.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for x in checks: print(f"[{'PASS' if x['passed'] else 'FAIL'}] {x['name']}: {x['detail']}")
    print(f"OVERALL VERDICT: {overall}\nREPORT: {out}")
    return 0 if overall == "PASS" else 1

if __name__ == "__main__": raise SystemExit(main())
