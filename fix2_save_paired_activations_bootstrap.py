"""
fix2_save_paired_activations_bootstrap.py
-----------------------------------------
Regenerates paired M0/Mu activations for the 40-item mllmu_real forget split,
saves raw activation tensors, and computes valid example-level bootstrap CIs.

Bootstrap unit: examples (rows), never layers.
Paired indices: identical bootstrap row indices applied to M0 and Mu.

Methods:
    npo, mmunlearner, cagul, sineproject, graddiff

Outputs:
    outputs/revision/paired_activation_bootstrap/
        base/
            base_activations.pt
            base_manifest.json
        <method>/
            paired_activations.pt
            activation_manifest.json
            bootstrap_ci.json
            bootstrap_samples.npz
        crp_bootstrap_summary.json
        crp_bootstrap_summary.csv
        table_crp_bootstrap.tex
        stage2_report.json

Usage:
    py .\fix2_save_paired_activations_bootstrap.py --resume
    py .\fix2_save_paired_activations_bootstrap.py --methods npo graddiff --resume
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter_guard import assert_adapter_is_active
from exp_config import (
    LLAVA_BASE,
    LLAVA_ADAPTERS,
    DEVICE,
    MLLMU_REAL_FORGET,
    RESULTS_DIR,
    BOOTSTRAP_N,
    BOOTSTRAP_ALPHA,
    LLAVA_LB_LAYERS,
    LLAVA_VE_LAYERS,
)

SCRIPT_VERSION = "fix2_paired_activation_bootstrap_v1.0"
SEED = 42
EXPECTED_N = 40
METHODS = ["npo", "mmunlearner", "cagul", "sineproject", "graddiff"]

OUT_DIR = RESULTS_DIR / "paired_activation_bootstrap"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKS: list[dict[str, Any]] = []


def chk(label: str, verdict: str, detail: Any) -> str:
    CHECKS.append({
        "check": label,
        "verdict": verdict,
        "detail": str(detail),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    icon = {"PASS": "OK", "WARN": "!!", "FAIL": "XX"}.get(verdict, "??")
    print(f"  [{icon} {verdict}] {label}: {detail}")
    return verdict


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"type": "base_model", "base_model": LLAVA_BASE}

    path = path.resolve()
    files = []
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
    ):
        p = path / name
        if p.exists() and p.is_file():
            st = p.stat()
            files.append({
                "path": str(p.resolve()),
                "size_bytes": st.st_size,
                "mtime_ns": st.st_mtime_ns,
                "sha256": sha256_file(p),
            })
    if not files:
        raise RuntimeError(f"No recognised checkpoint files in {path}")
    return {"type": "checkpoint", "path": str(path), "files": files}


def load_split(split_dir: Path) -> list[dict[str, Any]]:
    ann = split_dir / "annotations.json"
    items: list[dict[str, Any]] = []

    if ann.exists():
        raw = json.loads(ann.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise RuntimeError(f"{ann} must contain a list")
        for item in raw:
            p = Path(item["image"])
            if not p.is_absolute():
                p = split_dir / p
            if p.exists():
                copied = dict(item)
                copied["image"] = p.resolve()
                items.append(copied)
    else:
        for entity_dir in sorted(split_dir.iterdir()):
            if not entity_dir.is_dir():
                continue
            images = (
                list(entity_dir.glob("*.jpg"))
                + list(entity_dir.glob("*.jpeg"))
                + list(entity_dir.glob("*.png"))
            )
            jsons = list(entity_dir.glob("*.json"))
            if not images or not jsons:
                continue
            qa = json.loads(jsons[0].read_text(encoding="utf-8"))
            qa_list = qa if isinstance(qa, list) else [qa]
            for q in qa_list:
                items.append({
                    "entity": q.get("entity", entity_dir.name),
                    "image": images[0].resolve(),
                    "question": q["question"],
                    "answer": q.get("answer", q.get("gt", "")),
                    "aliases": q.get(
                        "aliases",
                        [q.get("entity", entity_dir.name).lower()],
                    ),
                })

    for i, item in enumerate(items):
        payload = json.dumps({
            "index": i,
            "entity": item.get("entity"),
            "image": str(item["image"]),
            "question": item.get("question"),
            "answer": item.get("answer"),
            "aliases": item.get("aliases", []),
        }, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        item["_item_id"] = f"forget_{i:03d}_{digest}"

    return items


def get_bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_llava(ckpt_path: Path | None):
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(LLAVA_BASE, use_fast=False)
    bnb = get_bnb_config()

    if ckpt_path is None:
        model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_BASE,
            quantization_config=bnb,
            device_map=DEVICE,
        )
    elif (ckpt_path / "adapter_config.json").exists():
        activity = assert_adapter_is_active(ckpt_path)
        print(
            f"  [adapter] {activity['n_nonzero_lora_B']}/"
            f"{activity['n_lora_B']} nonzero LoRA-B tensors"
        )
        base = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_BASE,
            quantization_config=bnb,
            device_map=DEVICE,
        )
        model = PeftModel.from_pretrained(
            base,
            str(ckpt_path),
            is_trainable=False,
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            str(ckpt_path),
            quantization_config=bnb,
            device_map=DEVICE,
        )

    model.eval()
    return model, processor


def unwrap(model):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def find_module(obj, paths: list[str], label: str):
    for path in paths:
        cur = obj
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and cur is not None:
            return cur, path
    raise RuntimeError(f"{label} hook path not found. Tried: {paths}")


def get_activations(model, processor, item: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    core = unwrap(model)
    image = Image.open(item["image"]).convert("RGB")
    prompt = f"USER: <image>\n{item['question']} ASSISTANT:"
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(DEVICE)

    captured: dict[str, torch.Tensor] = {}

    def make_hook(key):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(h):
                return
            h = h.detach().float().cpu()
            if h.dim() == 3:
                h = h[0].mean(dim=0)
            elif h.dim() == 2:
                h = h.mean(dim=0)
            else:
                h = h.flatten()
            if not torch.isfinite(h).all():
                raise RuntimeError(f"Non-finite activation at {key}")
            captured[key] = h.contiguous()
        return fn

    handles = []
    hook_paths = {}

    lb_layers, lb_path = find_module(
        core,
        [
            "model.language_model.model.layers",
            "model.language_model.layers",
            "language_model.model.layers",
            "language_model.layers",
            "model.language_model.decoder.layers",
            "language_model.decoder.layers",
        ],
        "LB",
    )
    hook_paths["lb"] = lb_path
    for idx in LLAVA_LB_LAYERS:
        if idx >= len(lb_layers):
            raise RuntimeError(f"LB layer {idx} missing; total={len(lb_layers)}")
        handles.append(lb_layers[idx].register_forward_hook(make_hook(f"lb_{idx}")))

    ve_layers, ve_path = find_module(
        core,
        [
            "model.vision_tower.vision_model.encoder.layers",
            "vision_tower.vision_model.encoder.layers",
            "model.vision_tower.encoder.layers",
            "vision_tower.encoder.layers",
            "model.vision_model.encoder.layers",
            "vision_model.encoder.layers",
        ],
        "VE",
    )
    hook_paths["ve"] = ve_path
    for idx in LLAVA_VE_LAYERS:
        if idx >= len(ve_layers):
            raise RuntimeError(f"VE layer {idx} missing; total={len(ve_layers)}")
        handles.append(ve_layers[idx].register_forward_hook(make_hook(f"ve_{idx}")))

    bridge, bridge_path = find_module(
        core,
        [
            "model.multi_modal_projector",
            "multi_modal_projector",
            "model.projector",
            "projector",
            "model.mm_projector",
            "mm_projector",
        ],
        "Bridge",
    )
    hook_paths["bridge"] = bridge_path
    handles.append(bridge.register_forward_hook(make_hook("bridge")))

    try:
        with torch.no_grad():
            model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    expected = (
        {f"ve_{i}" for i in LLAVA_VE_LAYERS}
        | {f"lb_{i}" for i in LLAVA_LB_LAYERS}
        | {"bridge"}
    )
    missing = sorted(expected - set(captured))
    if missing:
        raise RuntimeError(f"Missing captured hooks: {missing}")

    return captured, hook_paths


def stack_activations(records: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = sorted(set.intersection(*(set(r.keys()) for r in records)))
    output = {}
    for key in keys:
        shapes = [tuple(r[key].shape) for r in records]
        if len(set(shapes)) != 1:
            raise RuntimeError(f"Shape mismatch within {key}: {sorted(set(shapes))}")
        tensor = torch.stack([r[key] for r in records], dim=0).contiguous()
        if tensor.shape[0] != EXPECTED_N:
            raise RuntimeError(f"{key}: first dimension {tensor.shape[0]} != {EXPECTED_N}")
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{key}: NaN/Inf found")
        output[key] = tensor
    return output


def debiased_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    if X.shape != Y.shape:
        raise RuntimeError(f"CKA shape mismatch: {tuple(X.shape)} vs {tuple(Y.shape)}")
    if X.shape[0] < 4:
        raise RuntimeError("Debiased CKA requires at least 4 samples")

    X = X.float()
    Y = Y.float()
    if not torch.isfinite(X).all() or not torch.isfinite(Y).all():
        raise RuntimeError("Non-finite tensor passed to CKA")

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    n = X.shape[0]

    K = X @ X.T
    L = Y @ Y.T
    Kt = K - torch.diag(torch.diag(K))
    Lt = L - torch.diag(torch.diag(L))
    c = 1.0 / (n * (n - 3))

    def hsic(A, B):
        t1 = (A * B).sum()
        t2 = (2.0 / (n - 2)) * (A.sum(dim=1) * B.sum(dim=1)).sum()
        t3 = A.sum() * B.sum() / ((n - 1) * (n - 2))
        return c * (t1 - t2 + t3)

    h_kl = hsic(Kt, Lt)
    h_kk = hsic(Kt, Kt)
    h_ll = hsic(Lt, Lt)
    denom = torch.sqrt(torch.clamp(h_kk * h_ll, min=1e-10))
    value = h_kl / denom
    if not torch.isfinite(value):
        raise RuntimeError("CKA returned non-finite value")
    return float(torch.clamp(value, 0.0, 1.0).item())


def component_keys(component: str) -> list[str]:
    if component == "ve":
        return [f"ve_{i}" for i in LLAVA_VE_LAYERS]
    if component == "bridge":
        return ["bridge"]
    if component == "lb":
        return [f"lb_{i}" for i in LLAVA_LB_LAYERS]
    raise ValueError(component)


def component_point(base: dict[str, torch.Tensor], mu: dict[str, torch.Tensor], component: str) -> float:
    values = []
    for key in component_keys(component):
        values.append(debiased_cka(base[key], mu[key]))
    return float(np.mean(values))


def bootstrap_component(
    base: dict[str, torch.Tensor],
    mu: dict[str, torch.Tensor],
    component: str,
    rng: np.random.Generator,
    n_boot: int,
) -> np.ndarray:
    values = np.empty(n_boot, dtype=np.float64)
    keys = component_keys(component)
    for b in range(n_boot):
        idx_np = rng.integers(0, EXPECTED_N, size=EXPECTED_N, endpoint=False)
        idx = torch.from_numpy(idx_np.astype(np.int64))
        hook_values = [
            debiased_cka(base[key].index_select(0, idx), mu[key].index_select(0, idx))
            for key in keys
        ]
        values[b] = float(np.mean(hook_values))
    return values


def activation_cache_valid(
    path: Path,
    manifest_path: Path,
    item_ids: list[str],
    checkpoint_info: dict[str, Any],
) -> bool:
    if not path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("script_version") != SCRIPT_VERSION:
            return False
        if manifest.get("item_ids") != item_ids:
            return False
        if manifest.get("checkpoint") != checkpoint_info:
            return False
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("item_ids") != item_ids:
            return False
        acts = data.get("activations", {})
        expected = (
            {f"ve_{i}" for i in LLAVA_VE_LAYERS}
            | {f"lb_{i}" for i in LLAVA_LB_LAYERS}
            | {"bridge"}
        )
        if set(acts) != expected:
            return False
        for tensor in acts.values():
            if tensor.shape[0] != EXPECTED_N or not torch.isfinite(tensor).all():
                return False
        return True
    except Exception:
        return False


def extract_and_save(
    label: str,
    checkpoint: Path | None,
    items: list[dict[str, Any]],
    item_ids: list[str],
    resume: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    method_dir = OUT_DIR / label
    method_dir.mkdir(parents=True, exist_ok=True)
    activation_path = method_dir / f"{label}_activations.pt"
    manifest_path = method_dir / "activation_manifest.json"
    checkpoint_info = checkpoint_manifest(checkpoint)

    if resume and activation_cache_valid(
        activation_path,
        manifest_path,
        item_ids,
        checkpoint_info,
    ):
        payload = torch.load(activation_path, map_location="cpu", weights_only=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chk(f"{label} activations", "PASS", "valid cache reused")
        return payload["activations"], manifest

    print("\n" + "=" * 72)
    print(f"EXTRACT: {label.upper()}")
    print("=" * 72)

    model, processor = load_llava(checkpoint)
    records = []
    resolved_hook_paths = None
    start = time.time()

    for i, item in enumerate(items):
        acts, hook_paths = get_activations(model, processor, item)
        if resolved_hook_paths is None:
            resolved_hook_paths = hook_paths
        elif resolved_hook_paths != hook_paths:
            raise RuntimeError("Hook paths changed during extraction")
        records.append(acts)
        print(f"  [{i + 1:02d}/{EXPECTED_N}] {item.get('entity', '?')}")

    stacked = stack_activations(records)
    payload = {
        "script_version": SCRIPT_VERSION,
        "label": label,
        "item_ids": item_ids,
        "activations": stacked,
    }
    torch.save(payload, activation_path)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "label": label,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_info,
        "base_model": LLAVA_BASE,
        "dataset_path": str(MLLMU_REAL_FORGET.resolve()),
        "n_samples": EXPECTED_N,
        "item_ids": item_ids,
        "paired_order_sha256": hashlib.sha256(
            json.dumps(item_ids).encode("utf-8")
        ).hexdigest(),
        "hook_paths": resolved_hook_paths,
        "hook_layers": {
            "ve": LLAVA_VE_LAYERS,
            "bridge": ["projector_output"],
            "lb": LLAVA_LB_LAYERS,
        },
        "tensor_shapes": {k: list(v.shape) for k, v in stacked.items()},
        "tensor_dtypes": {k: str(v.dtype) for k, v in stacked.items()},
        "all_finite": all(torch.isfinite(v).all().item() for v in stacked.values()),
        "runtime_minutes": round((time.time() - start) / 60, 2),
        "activation_file": str(activation_path.resolve()),
        "activation_file_sha256": sha256_file(activation_path),
    }
    save_json(manifest_path, manifest)
    chk(f"{label} activations", "PASS", activation_path)

    del model
    torch.cuda.empty_cache()
    return stacked, manifest


def evaluate_method(
    method: str,
    base: dict[str, torch.Tensor],
    base_manifest: dict[str, Any],
    items: list[dict[str, Any]],
    item_ids: list[str],
    resume: bool,
) -> dict[str, Any] | None:
    checkpoint = LLAVA_ADAPTERS.get(method)
    if checkpoint is None or not Path(checkpoint).exists():
        chk(f"{method} checkpoint", "FAIL", checkpoint)
        return None
    checkpoint = Path(checkpoint).resolve()

    mu, mu_manifest = extract_and_save(
        method,
        checkpoint,
        items,
        item_ids,
        resume,
    )

    if base_manifest["item_ids"] != mu_manifest["item_ids"]:
        chk(f"{method} pairing", "FAIL", "item IDs/order mismatch")
        return None
    if base_manifest["paired_order_sha256"] != mu_manifest["paired_order_sha256"]:
        chk(f"{method} pairing", "FAIL", "pairing hash mismatch")
        return None

    for key in base:
        if key not in mu or base[key].shape != mu[key].shape:
            chk(f"{method} shapes", "FAIL", f"{key}: mismatch")
            return None

    chk(f"{method} pairing", "PASS", "paired_indices_verified=true, n_samples=40")

    method_dir = OUT_DIR / method
    ci_path = method_dir / "bootstrap_ci.json"
    samples_path = method_dir / "bootstrap_samples.npz"

    rng = np.random.default_rng(SEED)
    point = {
        comp: component_point(base, mu, comp)
        for comp in ("ve", "bridge", "lb")
    }

    bootstrap = {
        comp: bootstrap_component(base, mu, comp, rng, BOOTSTRAP_N)
        for comp in ("ve", "bridge", "lb")
    }

    alpha = BOOTSTRAP_ALPHA
    ci = {
        comp: [
            float(np.percentile(values, 100 * alpha / 2)),
            float(np.percentile(values, 100 * (1 - alpha / 2))),
        ]
        for comp, values in bootstrap.items()
    }

    np.savez_compressed(
        samples_path,
        ve=bootstrap["ve"],
        bridge=bootstrap["bridge"],
        lb=bootstrap["lb"],
        seed=np.array([SEED], dtype=np.int64),
        n_samples=np.array([EXPECTED_N], dtype=np.int64),
        n_boot=np.array([BOOTSTRAP_N], dtype=np.int64),
    )

    result = {
        "script_version": SCRIPT_VERSION,
        "method": method,
        "metric": "debiased_linear_cka",
        "bootstrap_unit": "examples",
        "n_samples": EXPECTED_N,
        "paired_indices_verified": True,
        "seed": SEED,
        "n_bootstrap": BOOTSTRAP_N,
        "alpha": BOOTSTRAP_ALPHA,
        "base_activation_manifest": str(
            (OUT_DIR / "base" / "activation_manifest.json").resolve()
        ),
        "method_activation_manifest": str(
            (method_dir / "activation_manifest.json").resolve()
        ),
        "checkpoint": mu_manifest["checkpoint"],
        "hook_paths": mu_manifest["hook_paths"],
        "point_estimates": point,
        "ci_95": ci,
        "bootstrap_samples_file": str(samples_path.resolve()),
        "bootstrap_samples_sha256": sha256_file(samples_path),
    }
    save_json(ci_path, result)

    chk(
        f"{method} bootstrap",
        "PASS",
        f"VE={point['ve']:.4f} [{ci['ve'][0]:.4f},{ci['ve'][1]:.4f}] | "
        f"BR={point['bridge']:.4f} [{ci['bridge'][0]:.4f},{ci['bridge'][1]:.4f}] | "
        f"LB={point['lb']:.4f} [{ci['lb'][0]:.4f},{ci['lb'][1]:.4f}]",
    )
    return result


def build_latex(results: list[dict[str, Any]]) -> str:
    display = {
        "npo": "NPO",
        "mmunlearner": "MMUnlearner",
        "cagul": "CAGUL",
        "sineproject": "SineProject",
        "graddiff": "GradDiff",
    }

    def fmt(result, comp):
        mean = result["point_estimates"][comp]
        lo, hi = result["ci_95"][comp]
        return f"{mean:.4f} [{lo:.4f},{hi:.4f}]"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{LLaVA CRP on \emph{mllmu\_real} with paired example-level "
        r"bootstrap 95\% confidence intervals (1000 resamples, $n=40$).}",
        r"\label{tab:crp_bootstrap}",
        r"\setlength{\tabcolsep}{3pt}\small",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"\textbf{Method} & \textbf{VE-CKA [95\% CI]} & "
        r"\textbf{BR-CKA [95\% CI]} & \textbf{LB-CKA [95\% CI]} \\",
        r"\midrule",
    ]
    for r in results:
        lines.append(
            f"{display.get(r['method'], r['method'])} & "
            f"{fmt(r, 've')} & {fmt(r, 'bridge')} & {fmt(r, 'lb')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def save_report() -> bool:
    n_pass = sum(c["verdict"] == "PASS" for c in CHECKS)
    n_warn = sum(c["verdict"] == "WARN" for c in CHECKS)
    n_fail = sum(c["verdict"] == "FAIL" for c in CHECKS)
    payload = {
        "script_version": SCRIPT_VERSION,
        "checks": CHECKS,
        "n_pass": n_pass,
        "n_warn": n_warn,
        "n_fail": n_fail,
        "overall_verdict": "FAIL" if n_fail else ("WARN" if n_warn else "PASS"),
    }
    save_json(OUT_DIR / "stage2_report.json", payload)
    print(f"\n  Checks: {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL")
    return n_fail == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    items = load_split(MLLMU_REAL_FORGET)
    if len(items) != EXPECTED_N:
        chk("forget split", "FAIL", f"{len(items)} items, expected exactly 40")
        save_report()
        return 1

    item_ids = [item["_item_id"] for item in items]
    if len(set(item_ids)) != EXPECTED_N:
        chk("item IDs", "FAIL", "duplicate item IDs")
        save_report()
        return 1
    chk("forget split", "PASS", "40 unique items")

    base, base_manifest = extract_and_save(
        "base",
        None,
        items,
        item_ids,
        args.resume,
    )

    results = []
    for method in args.methods:
        if method not in METHODS:
            chk(method, "FAIL", "not in valid five-method set")
            continue
        result = evaluate_method(
            method,
            base,
            base_manifest,
            items,
            item_ids,
            args.resume,
        )
        if result is not None:
            results.append(result)

    if not results:
        chk("results", "FAIL", "no methods completed")
        save_report()
        return 1

    rows = []
    for r in results:
        rows.append({
            "method": r["method"],
            "n_samples": r["n_samples"],
            "bootstrap_unit": r["bootstrap_unit"],
            "paired_indices_verified": r["paired_indices_verified"],
            "ve_mean": r["point_estimates"]["ve"],
            "ve_ci_low": r["ci_95"]["ve"][0],
            "ve_ci_high": r["ci_95"]["ve"][1],
            "bridge_mean": r["point_estimates"]["bridge"],
            "bridge_ci_low": r["ci_95"]["bridge"][0],
            "bridge_ci_high": r["ci_95"]["bridge"][1],
            "lb_mean": r["point_estimates"]["lb"],
            "lb_ci_low": r["ci_95"]["lb"][0],
            "lb_ci_high": r["ci_95"]["lb"][1],
        })

    csv_path = OUT_DIR / "crp_bootstrap_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    chk("summary CSV", "PASS", csv_path)

    summary_path = OUT_DIR / "crp_bootstrap_summary.json"
    save_json(summary_path, {
        "script_version": SCRIPT_VERSION,
        "bootstrap_unit": "examples",
        "n_samples": EXPECTED_N,
        "paired_indices_verified": True,
        "seed": SEED,
        "n_bootstrap": BOOTSTRAP_N,
        "methods": results,
    })
    chk("summary JSON", "PASS", summary_path)

    latex_path = OUT_DIR / "table_crp_bootstrap.tex"
    latex_path.write_text(build_latex(results), encoding="utf-8")
    chk("LaTeX table", "PASS", latex_path)

    print("\n" + "=" * 92)
    print("FINAL PAIRED EXAMPLE-LEVEL BOOTSTRAP SUMMARY")
    print("=" * 92)
    for r in results:
        print(
            f"{r['method']:<15} "
            f"VE={r['point_estimates']['ve']:.4f} "
            f"[{r['ci_95']['ve'][0]:.4f},{r['ci_95']['ve'][1]:.4f}]  "
            f"BR={r['point_estimates']['bridge']:.4f} "
            f"[{r['ci_95']['bridge'][0]:.4f},{r['ci_95']['bridge'][1]:.4f}]  "
            f"LB={r['point_estimates']['lb']:.4f} "
            f"[{r['ci_95']['lb'][0]:.4f},{r['ci_95']['lb'][1]:.4f}]"
        )

    return 0 if save_report() else 1


if __name__ == "__main__":
    raise SystemExit(main())
