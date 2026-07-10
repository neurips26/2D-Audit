"""
E2_run_audit_per_entity.py
--------------------------
Re-runs CRP on mllmu_real with proper bootstrap CIs.

KEY FIXES:
  - Stacks ALL forget-set activations into N Ã— d matrices.
  - Computes debiased linear CKA on the full activation matrix.
  - Bootstrap CI matches the component mean estimator:
      resample rows -> recompute CKA for every hook -> average hooks.
  - Uses a single forward pass instead of generate(), avoiding hook overwrite.
  - Handles PEFT/LoRA wrapped LLaVA models.
  - Handles HuggingFace LLaVA module nesting under model.*.
  - Auto-skips PEFT adapters where all LoRA_B tensors are zero.
  - Supports retrained GA as ga_retrained.
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
    LLAVA_BASE,
    LLAVA_ADAPTERS,
    DEVICE,
    MLLMU_REAL_FORGET,
    PER_ENTITY_CRP_DIR,
    BOOTSTRAP_N,
    BOOTSTRAP_ALPHA,
    LLAVA_LB_LAYERS,
    LLAVA_VE_LAYERS,
)

PER_ENTITY_CRP_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["ga_retrained", "npo", "mmunlearner", "cagul", "sineproject", "graddiff"]


def is_adapter_dead(ckpt_path) -> bool:
    """
    Return True if a PEFT LoRA adapter exists but all LoRA_B tensors are zero.

    Full-model checkpoints are not treated as dead adapters.
    """
    if ckpt_path is None:
        return False

    ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        return False

    adapter_file = ckpt_path / "adapter_model.safetensors"

    # Full model checkpoint or non-safetensors adapter; do not auto-skip here.
    if not adapter_file.exists():
        return False

    try:
        from safetensors.torch import load_file
    except ImportError:
        print("  [warn] safetensors not installed; cannot check adapter health.")
        return False

    sd = load_file(str(adapter_file))

    b_total = 0
    b_nonzero = 0

    for name, tensor in sd.items():
        if "lora_b" in name.lower():
            b_total += 1
            max_abs = float(tensor.float().abs().max())
            if max_abs > 1e-8:
                b_nonzero += 1

    if b_total > 0 and b_nonzero == 0:
        print(f"  [skip] Dead adapter detected: {ckpt_path}")
        print(f"         LoRA_B nonzero: 0/{b_total}")
        return True

    return False


# ------------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------------

def load_split(split_dir: Path) -> list:
    ann = split_dir / "annotations.json"

    if ann.exists():
        with open(ann, encoding="utf-8") as f:
            items = json.load(f)

        result = []

        for item in items:
            p = Path(item["image"])

            if not p.is_absolute():
                p = split_dir / p

            if p.exists():
                item["image"] = p
                result.append(item)
            else:
                print(f"  [warn] image not found, skipped: {p}")

        return result

    items = []

    for entity_dir in sorted(split_dir.iterdir()):
        if not entity_dir.is_dir():
            continue

        imgs = (
            list(entity_dir.glob("*.jpg"))
            + list(entity_dir.glob("*.jpeg"))
            + list(entity_dir.glob("*.png"))
        )
        jsons = list(entity_dir.glob("*.json"))

        if not imgs or not jsons:
            continue

        with open(jsons[0], encoding="utf-8") as f:
            qa = json.load(f)

        qa_list = qa if isinstance(qa, list) else [qa]

        for q in qa_list:
            items.append({
                "entity": q.get("entity", entity_dir.name),
                "image": imgs[0],
                "question": q["question"],
                "answer": q.get("answer", q.get("gt", "")),
                "aliases": [q.get("entity", entity_dir.name).lower()],
            })

    return items


# ------------------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------------------

def get_bnb_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def print_lora_diagnostics(model):
    print("  [debug] PEFT adapter loaded.")

    try:
        print(f"  [debug] Active adapters: {model.active_adapters}")
    except Exception:
        pass

    total_lora = 0
    nonzero_lora = 0

    component_stats = {
        "vision": {"total": 0, "nonzero": 0},
        "language": {"total": 0, "nonzero": 0},
        "projector": {"total": 0, "nonzero": 0},
        "other": {"total": 0, "nonzero": 0},
    }

    examples = []

    for name, param in model.named_parameters():
        if "lora" not in name.lower():
            continue

        total_lora += 1

        with torch.no_grad():
            mean_abs = float(param.detach().float().abs().mean().cpu())
            max_abs = float(param.detach().float().abs().max().cpu())

        is_nonzero = max_abs > 1e-8

        if is_nonzero:
            nonzero_lora += 1

        lname = name.lower()

        if "vision_tower" in lname or "vision_model" in lname:
            comp = "vision"
        elif "language_model" in lname:
            comp = "language"
        elif "projector" in lname or "multi_modal" in lname or "mm_projector" in lname:
            comp = "projector"
        else:
            comp = "other"

        component_stats[comp]["total"] += 1

        if is_nonzero:
            component_stats[comp]["nonzero"] += 1

        if len(examples) < 12:
            examples.append((name, mean_abs, max_abs))

    print(f"  [debug] Number of LoRA tensors: {total_lora}")
    print(f"  [debug] Nonzero LoRA tensors: {nonzero_lora}/{total_lora}")

    for comp, stats in component_stats.items():
        print(
            f"  [debug] LoRA {comp}: "
            f"{stats['nonzero']}/{stats['total']} nonzero"
        )

    print("  [debug] First LoRA tensor examples:")
    for name, mean_abs, max_abs in examples:
        print(f"    {name}: mean_abs={mean_abs:.10f}, max_abs={max_abs:.10f}")


def load_llava(ckpt_path, debug: bool = False):
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    print(f"  Loading: {ckpt_path or 'base'}")

    bnb = get_bnb_config()

    processor = AutoProcessor.from_pretrained(
        LLAVA_BASE,
        use_fast=False,
    )

    ckpt_path = Path(ckpt_path) if ckpt_path else None

    if ckpt_path is None:
        model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_BASE,
            quantization_config=bnb,
            device_map=DEVICE,
        )

    elif (ckpt_path / "adapter_config.json").exists():
        base = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_BASE,
            quantization_config=bnb,
            device_map=DEVICE,
        )

        activity = assert_adapter_is_active(ckpt_path)
        print(
            "  [adapter] Active LoRA: "
            f"{activity['n_nonzero_lora_B']}/"
            f"{activity['n_lora_B']} nonzero B tensors"
        )

        model = PeftModel.from_pretrained(
            base,
            str(ckpt_path),
            is_trainable=False,
        )

        if debug:
            print_lora_diagnostics(model)

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


def find_module(obj, paths: list, label: str):
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

    print(f"  [warn] {label}: none of the tried paths found.")
    for path in paths:
        print(f"      - {path}")

    return None, None


# ------------------------------------------------------------------------
# ACTIVATION EXTRACTION
# ------------------------------------------------------------------------

def get_activations(model, processor, item: dict, debug: bool = False) -> dict:
    core = unwrap(model)

    image = Image.open(item["image"]).convert("RGB")
    prompt = f"USER: <image>\n{item['question']} ASSISTANT:"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(DEVICE)

    captured = {}

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

            if torch.isnan(h).any():
                print(f"  [warn] NaN activation skipped: {key}")
                return

            captured[key] = h

        return fn

    handles = []

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

    if lb_layers is not None:
        if debug:
            print(f"  [debug] LB path: {lb_path}")
            print(f"  [debug] LB layers: {len(lb_layers)}")

        for idx in LLAVA_LB_LAYERS:
            if idx < len(lb_layers):
                handles.append(
                    lb_layers[idx].register_forward_hook(make_hook(f"lb_{idx}"))
                )

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

    if ve_layers is not None:
        if debug:
            print(f"  [debug] VE path: {ve_path}")
            print(f"  [debug] VE layers: {len(ve_layers)}")

        for idx in LLAVA_VE_LAYERS:
            if idx < len(ve_layers):
                handles.append(
                    ve_layers[idx].register_forward_hook(make_hook(f"ve_{idx}"))
                )

    bridge, br_path = find_module(
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

    if bridge is not None:
        if debug:
            print(f"  [debug] Bridge path: {br_path}")

        handles.append(
            bridge.register_forward_hook(make_hook("bridge"))
        )

    if not handles:
        print("\n[debug] Available top-level core attributes:")
        names = [n for n in dir(core) if not n.startswith("_")]
        print("  " + ", ".join(names[:120]))

        if hasattr(core, "model"):
            print("\n[debug] Available core.model attributes:")
            names2 = [n for n in dir(core.model) if not n.startswith("_")]
            print("  " + ", ".join(names2[:120]))

        raise RuntimeError("No hooks registered. Check LLaVA module paths.")

    with torch.no_grad():
        model(
            **inputs,
            use_cache=False,
        )

    for h in handles:
        h.remove()

    if not captured:
        raise RuntimeError(
            "Hooks were registered, but no activations were captured."
        )

    if debug:
        print(f"  [debug] Captured keys: {sorted(captured.keys())}")

    return captured


# ------------------------------------------------------------------------
# CKA
# ------------------------------------------------------------------------

def debiased_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    if X is None or Y is None:
        return float("nan")

    if X.shape[0] != Y.shape[0]:
        return float("nan")

    if X.shape[0] < 4:
        return float("nan")

    X = X.float()
    Y = Y.float()

    if torch.isnan(X).any() or torch.isnan(Y).any():
        return float("nan")

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

    val = h_kl / denom

    if torch.isnan(val):
        return float("nan")

    val = torch.clamp(val, min=0.0, max=1.0)

    return float(val.item())


# ------------------------------------------------------------------------
# METHOD EVALUATION
# ------------------------------------------------------------------------

def evaluate_method(method: str, forget_items: list, resume: bool, debug: bool) -> dict:
    out_path = PER_ENTITY_CRP_DIR / f"{method}_per_entity_crp.json"

    if resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            old = json.load(f)

        lb = old.get("lb_mean", float("nan"))

        try:
            lb = float(lb)
        except Exception:
            lb = float("nan")

        if not np.isnan(lb) and abs(lb - 1.0) > 0.000001:
            print(f"  [resume] {method} - existing result OK.")
            return old

        print(f"  [resume ignored] {method} - existing result invalid/suspicious. Recomputing.")

    print(f"\n{'=' * 60}")
    print(f"  Per-entity CRP: {method.upper()}")
    print(f"{'=' * 60}")

    ckpt = LLAVA_ADAPTERS.get(method)

    print("  Loading M0 base...")
    m0, proc = load_llava(None, debug=debug)

    print(f"  Loading Mu {method}...")
    mu, _ = load_llava(ckpt, debug=debug)

    if debug:
        print(f"  [debug] M0 class: {type(m0)}")
        print(f"  [debug] Mu class: {type(mu)}")
        print(f"  [debug] Mu unwrapped class: {type(unwrap(mu))}")

    m0_acts = {}
    mu_acts = {}

    for i, item in enumerate(forget_items):
        show_debug = debug and i == 0

        print(f"  [{i + 1:02d}/{len(forget_items)}] {item['entity']}")

        a0 = get_activations(m0, proc, item, debug=show_debug)
        au = get_activations(mu, proc, item, debug=show_debug)

        for k, v in a0.items():
            m0_acts.setdefault(k, []).append(v)

        for k, v in au.items():
            mu_acts.setdefault(k, []).append(v)

    def stack(acts: dict, key: str):
        if key not in acts or not acts[key]:
            return None

        try:
            return torch.stack(acts[key])
        except Exception as e:
            print(f"  [warn] Could not stack {key}: {e}")
            return None

    def component_cka(keys):
        vals = []

        for k in keys:
            X0 = stack(m0_acts, k)
            Xu = stack(mu_acts, k)

            if X0 is None or Xu is None:
                continue

            if X0.shape != Xu.shape:
                print(f"  [warn] Shape mismatch for {k}: {X0.shape} vs {Xu.shape}")
                continue

            c = debiased_cka(X0, Xu)

            if not np.isnan(c):
                vals.append(c)

        return float(np.mean(vals)) if vals else float("nan")

    def component_boot_ci(keys):
        valid_pairs = []

        for k in keys:
            X0 = stack(m0_acts, k)
            Xu = stack(mu_acts, k)

            if X0 is not None and Xu is not None and X0.shape == Xu.shape:
                valid_pairs.append((k, X0, Xu))

        if not valid_pairs:
            return float("nan"), float("nan")

        N = valid_pairs[0][1].shape[0]

        if N < 4:
            return float("nan"), float("nan")

        boot_vals = []

        for _ in range(BOOTSTRAP_N):
            idx_np = np.random.choice(N, N, replace=True)
            idx = torch.tensor(idx_np, dtype=torch.long)

            vals = []

            for k, X0, Xu in valid_pairs:
                c = debiased_cka(X0[idx], Xu[idx])

                if not np.isnan(c):
                    vals.append(c)

            if vals:
                boot_vals.append(float(np.mean(vals)))

        if len(boot_vals) < 2:
            return float("nan"), float("nan")

        return (
            float(np.percentile(boot_vals, 100 * BOOTSTRAP_ALPHA / 2)),
            float(np.percentile(boot_vals, 100 * (1 - BOOTSTRAP_ALPHA / 2))),
        )

    lb_keys = [
        f"lb_{idx}"
        for idx in LLAVA_LB_LAYERS
        if f"lb_{idx}" in m0_acts and f"lb_{idx}" in mu_acts
    ]

    ve_keys = [
        f"ve_{idx}"
        for idx in LLAVA_VE_LAYERS
        if f"ve_{idx}" in m0_acts and f"ve_{idx}" in mu_acts
    ]

    br_keys = ["bridge"] if "bridge" in m0_acts and "bridge" in mu_acts else []

    print(f"\n  Computing CKA with N={len(forget_items)} items...")
    print(f"  VE hooks: {ve_keys}")
    print(f"  BR hooks: {br_keys}")
    print(f"  LB hooks: {lb_keys}")

    ve_mean = component_cka(ve_keys)
    br_mean = component_cka(br_keys)
    lb_mean = component_cka(lb_keys)

    print("  Computing bootstrap CIs...")
    ve_ci = component_boot_ci(ve_keys)
    br_ci = component_boot_ci(br_keys)
    lb_ci = component_boot_ci(lb_keys)

    ve_per_layer = {}
    lb_per_layer = {}
    all_hook_cka = {}

    for k in ve_keys + br_keys + lb_keys:
        X0 = stack(m0_acts, k)
        Xu = stack(mu_acts, k)

        if X0 is None or Xu is None or X0.shape != Xu.shape:
            continue

        c = debiased_cka(X0, Xu)
        all_hook_cka[k] = c

        if k.startswith("ve_"):
            ve_per_layer[int(k.split("_")[1])] = c

        if k.startswith("lb_"):
            lb_per_layer[int(k.split("_")[1])] = c

    result = {
        "method": method,
        "metric": "debiased_linear_cka_matrix_forward_pass",
        "n_items": len(forget_items),
        "ve_mean": ve_mean,
        "ve_ci_95": list(ve_ci),
        "bridge_mean": br_mean,
        "bridge_ci_95": list(br_ci),
        "lb_mean": lb_mean,
        "lb_ci_95": list(lb_ci),
        "ve_per_layer": ve_per_layer,
        "lb_per_layer": lb_per_layer,
        "all_hook_cka": all_hook_cka,
        "entity_names": [item["entity"] for item in forget_items],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"  [saved] {out_path}")

    print(f"\n  SUMMARY - {method}  N={len(forget_items)}")
    print(f"    VE-CKA  : {ve_mean:.6f}  [{ve_ci[0]:.6f}, {ve_ci[1]:.6f}]")
    print(f"    BR-CKA  : {br_mean:.6f}  [{br_ci[0]:.6f}, {br_ci[1]:.6f}]")
    print(f"    LB-CKA  : {lb_mean:.6f}  [{lb_ci[0]:.6f}, {lb_ci[1]:.6f}]")

    if ve_per_layer:
        print("    VE layers:", "  ".join(f"L{k}={v:.4f}" for k, v in sorted(ve_per_layer.items())))

    if lb_per_layer:
        print("    LB layers:", "  ".join(f"L{k}={v:.4f}" for k, v in sorted(lb_per_layer.items())))

    del m0, mu

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ------------------------------------------------------------------------
# LATEX
# ------------------------------------------------------------------------

def build_latex(results: list) -> str:
    display = {
        "ga_retrained": "GA",
        "ga": "GA",
        "npo": "NPO",
        "mmunlearner": "MMUnlearner",
        "cagul": "CAGUL",
        "sineproject": "SineProject",
        "manu_lora": "MANU",
        "manu_full": "MANU",
    }

    n = results[0]["n_items"] if results else 0

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{CRP on \emph{{mllmu\_real}} (LLaVA-1.5-7B) with bootstrap 95\% CIs "
        rf"(1000 resamples, $n={n}$ forget samples).}}",
        r"\label{tab:crp_ci_main}",
        r"\setlength{\tabcolsep}{3pt}\small",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"\textbf{Method} & \textbf{VE-CKA [95\% CI]} & \textbf{BR-CKA [95\% CI]} & \textbf{LB-CKA [95\% CI]} \\",
        r"\midrule",
    ]

    def fmt(mean, ci):
        try:
            return f"{float(mean):.4f} [{float(ci[0]):.4f},{float(ci[1]):.4f}]"
        except Exception:
            return "nan [nan,nan]"

    for r in results:
        method = r["method"]
        disp = display.get(method, method)

        lines.append(
            f"{disp} & "
            f"{fmt(r['ve_mean'], r['ve_ci_95'])} & "
            f"{fmt(r['bridge_mean'], r['bridge_ci_95'])} & "
            f"{fmt(r['lb_mean'], r['lb_ci_95'])} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHODS,
        help="Methods to evaluate.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing result unless NaN or suspicious all-1 result.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print hook and LoRA diagnostics.",
    )

    args = parser.parse_args()

    if not MLLMU_REAL_FORGET.exists():
        print(f"[ERROR] {MLLMU_REAL_FORGET} not found. Check exp_config.py")
        sys.exit(1)

    forget_items = load_split(MLLMU_REAL_FORGET)

    print(f"Forget set: {len(forget_items)} items from {MLLMU_REAL_FORGET}")

    if not forget_items:
        print("[ERROR] No items found.")
        sys.exit(1)

    all_results = []
    skipped = []

    for method in args.methods:
        if method not in LLAVA_ADAPTERS:
            print(f"[skip] unknown method: {method}")
            skipped.append({"method": method, "reason": "unknown method"})
            continue

        ckpt = LLAVA_ADAPTERS[method]

        if method != "no_unlearn":
            if ckpt is None:
                pass
            elif not Path(ckpt).exists():
                print(f"[skip] checkpoint missing: {ckpt}")
                skipped.append({"method": method, "reason": f"missing checkpoint: {ckpt}"})
                continue
            elif is_adapter_dead(ckpt):
                skipped.append({"method": method, "reason": "dead LoRA adapter, all LoRA_B tensors zero"})
                continue

        result = evaluate_method(
            method=method,
            forget_items=forget_items,
            resume=args.resume,
            debug=args.debug,
        )

        all_results.append(result)

    if skipped:
        skipped_path = PER_ENTITY_CRP_DIR / "skipped_methods.json"
        with open(skipped_path, "w", encoding="utf-8") as f:
            json.dump(skipped, f, indent=2)
        print(f"\n[saved] skipped methods: {skipped_path}")

    if not all_results:
        print("[warn] No results.")
        return

    summary = PER_ENTITY_CRP_DIR / "mllmu_real_crp_with_ci.json"

    with open(summary, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n[saved] {summary}")

    latex = build_latex(all_results)
    latex_path = PER_ENTITY_CRP_DIR / "table_crp_ci.tex"

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"[saved] {latex_path}")

    print("\n" + "=" * 60)
    print(latex)

    print("\n" + "=" * 60)
    print(f"{'Method':<15} {'VE':>10} {'BR':>10} {'LB':>10}  LB 95% CI")
    print("-" * 70)

    for r in all_results:
        ci = r["lb_ci_95"]

        print(
            f"{r['method']:<15} "
            f"{float(r['ve_mean']):>10.6f} "
            f"{float(r['bridge_mean']):>10.6f} "
            f"{float(r['lb_mean']):>10.6f}  "
            f"[{float(ci[0]):.6f}, {float(ci[1]):.6f}]"
        )


if __name__ == "__main__":
    main()



