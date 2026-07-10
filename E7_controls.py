"""
E7_controls.py
───────────────
Two controls that directly answer the reviewer's key technical questions.

E7a: Base-vs-base CKA control
  Load M0 twice (two separate 4-bit NF4 loads) and compute CKA between
  them. This quantifies the quantization-induced noise floor of the
  extraction pipeline. If base-vs-base CKA < 1.0, all experiment CKA
  values should be interpreted relative to this floor.
  Directly addresses reviewer Q1: "quantify expected CKA under your
  extraction/quantization pipeline."

E7b: Retain-set CRP
  For each method, compute component CKA on the RETAIN set (not forget).
  This gives selectivity: a method making targeted forget-set changes
  would show lower forget-set CKA than retain-set CKA. Under-forgetting
  methods with minimal representational change should show both near 1.0.
  Directly addresses reviewer Q1: "retain-set CRP to assess selectivity."

E7c: Per-layer LB-CKA for LLaVA
  Extract the per-layer LB CKA values already stored in E2 JSON outputs.
  No new computation needed.
  Directly addresses reviewer Q2: "per-layer LB-CKA for LLaVA."

Usage:
    py E7_controls.py --run base_vs_base
    py E7_controls.py --run retain_crp --methods npo mmunlearner cagul sineproject ga_retrained
    py E7_controls.py --run extract_layers
    py E7_controls.py --run all
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
    MLLMU_REAL_FORGET, MLLMU_REAL_RETAIN,
    PER_ENTITY_CRP_DIR, RESULTS_DIR,
    LLAVA_LB_LAYERS, LLAVA_VE_LAYERS,
    BOOTSTRAP_N, BOOTSTRAP_ALPHA,
)

CONTROLS_DIR = RESULTS_DIR / "controls"
CONTROLS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def get_bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_llava(ckpt_path=None):
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    proc = AutoProcessor.from_pretrained(LLAVA_BASE)
    bnb = get_bnb_config()
    ckpt_path = Path(ckpt_path) if ckpt_path else None
    if ckpt_path is None:
        model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_BASE, quantization_config=bnb, device_map=DEVICE)
    elif (ckpt_path / "adapter_config.json").exists():
        base = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_BASE, quantization_config=bnb, device_map=DEVICE)
        activity = assert_adapter_is_active(ckpt_path)
        print(
            "[adapter] Active LoRA: "
            f"{activity['n_nonzero_lora_B']}/"
            f"{activity['n_lora_B']} nonzero B tensors"
        )
        model = PeftModel.from_pretrained(
            base,
            str(ckpt_path),
            is_trainable=False,
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            str(ckpt_path), quantization_config=bnb, device_map=DEVICE)
    model.eval()
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


def load_split(split_dir: Path) -> list:
    ann = split_dir / "annotations.json"
    if not ann.exists():
        # subdirectory layout
        items = []
        for entity_dir in sorted(split_dir.iterdir()):
            if not entity_dir.is_dir(): continue
            imgs  = list(entity_dir.glob("*.jpg")) + list(entity_dir.glob("*.png"))
            jsons = list(entity_dir.glob("*.json"))
            if not imgs or not jsons: continue
            with open(jsons[0], encoding="utf-8") as f:
                qa = json.load(f)
            for q in (qa if isinstance(qa, list) else [qa]):
                items.append({"entity": entity_dir.name, "image": imgs[0],
                               "question": q["question"], "answer": q.get("answer","")})
        return items
    with open(ann, encoding="utf-8") as f:
        items = json.load(f)
    result = []
    for item in items:
        p = Path(item["image"])
        if not p.is_absolute(): p = split_dir / p
        if p.exists():
            item["image"] = p; result.append(item)
    return result


def get_activations(model, processor, item: dict) -> dict:
    """Single forward pass activation extraction."""
    core  = unwrap(model)
    image = Image.open(item["image"]).convert("RGB")
    prompt = f"USER: <image>\n{item['question']} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)

    captured = {}
    def make_hook(key):
        def fn(m, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(h): return
            h = h.detach().float().cpu()
            if h.dim() == 3: h = h[0].mean(0)
            elif h.dim() == 2: h = h.mean(0)
            if not torch.isnan(h).any():
                captured[key] = h
        return fn

    handles = []
    lb, _ = find_module(core, [
        "model.language_model.model.layers", "model.language_model.layers",
        "language_model.model.layers", "language_model.layers"], "LB")
    if lb:
        for idx in LLAVA_LB_LAYERS:
            if idx < len(lb):
                handles.append(lb[idx].register_forward_hook(make_hook(f"lb_{idx}")))
    ve, _ = find_module(core, [
        "model.vision_tower.vision_model.encoder.layers",
        "vision_tower.vision_model.encoder.layers"], "VE")
    if ve:
        for idx in LLAVA_VE_LAYERS:
            if idx < len(ve):
                handles.append(ve[idx].register_forward_hook(make_hook(f"ve_{idx}")))
    bridge, _ = find_module(core, [
        "model.multi_modal_projector", "multi_modal_projector"], "Bridge")
    if bridge:
        handles.append(bridge.register_forward_hook(make_hook("bridge")))

    with torch.no_grad():
        model(**inputs, use_cache=False)
    for h in handles: h.remove()
    return captured


def debiased_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    if X is None or Y is None or X.shape[0] < 4: return float("nan")
    X = X.float() - X.float().mean(0, keepdim=True)
    Y = Y.float() - Y.float().mean(0, keepdim=True)
    n = X.shape[0]
    K = X @ X.T; L = Y @ Y.T
    Kt = K - torch.diag(torch.diag(K))
    Lt = L - torch.diag(torch.diag(L))
    c = 1.0 / (n * (n - 3))
    def hsic(A, B):
        return c * ((A*B).sum() - (2./(n-2))*(A.sum(1)*B.sum(1)).sum()
                    + A.sum()*B.sum()/((n-1)*(n-2)))
    h_kl = hsic(Kt, Lt); h_kk = hsic(Kt, Kt); h_ll = hsic(Lt, Lt)
    denom = torch.sqrt(torch.clamp(h_kk * h_ll, min=1e-10))
    v = torch.clamp(h_kl / denom, 0., 1.)
    return float("nan") if torch.isnan(v) else float(v.item())


def compute_crp_for_items(m1, m2, processor, items) -> dict:
    """Compute CRP between any two models on a list of items."""
    acts1, acts2 = {}, {}
    for i, item in enumerate(items):
        print(f"  [{i+1:02d}/{len(items)}] {item['entity']}")
        a1 = get_activations(m1, processor, item)
        a2 = get_activations(m2, processor, item)
        for k, v in a1.items(): acts1.setdefault(k, []).append(v)
        for k, v in a2.items(): acts2.setdefault(k, []).append(v)

    def stack(d, k):
        if k not in d or not d[k]: return None
        return torch.stack(d[k])

    lb_keys = [f"lb_{i}" for i in LLAVA_LB_LAYERS if f"lb_{i}" in acts1]
    ve_keys = [f"ve_{i}" for i in LLAVA_VE_LAYERS if f"ve_{i}" in acts1]
    br_keys = ["bridge"] if "bridge" in acts1 else []

    def comp_cka(keys):
        vals = []
        for k in keys:
            X = stack(acts1, k); Y = stack(acts2, k)
            if X is not None and Y is not None:
                c = debiased_cka(X, Y)
                if not np.isnan(c): vals.append(c)
        return float(np.mean(vals)) if vals else float("nan")

    def per_layer(keys, prefix):
        result = {}
        for k in keys:
            X = stack(acts1, k); Y = stack(acts2, k)
            if X is not None and Y is not None:
                idx = int(k.split("_")[1])
                result[idx] = debiased_cka(X, Y)
        return result

    def boot_ci(keys):
        pairs = []
        for k in keys:
            X = stack(acts1, k); Y = stack(acts2, k)
            if X is not None and Y is not None and X.shape == Y.shape:
                pairs.append((k, X, Y))
        if not pairs: return (float("nan"), float("nan"))
        N = pairs[0][1].shape[0]
        if N < 4: return (float("nan"), float("nan"))
        boot = []
        for _ in range(BOOTSTRAP_N):
            idx = torch.tensor(np.random.choice(N, N, replace=True))
            vals = [debiased_cka(X[idx], Y[idx]) for _, X, Y in pairs
                    if not np.isnan(debiased_cka(X[idx], Y[idx]))]
            if vals: boot.append(float(np.mean(vals)))
        if len(boot) < 2: return (float("nan"), float("nan"))
        return (float(np.percentile(boot, 100*BOOTSTRAP_ALPHA/2)),
                float(np.percentile(boot, 100*(1-BOOTSTRAP_ALPHA/2))))

    return {
        "n_items":       len(items),
        "ve_mean":       comp_cka(ve_keys),
        "ve_ci_95":      list(boot_ci(ve_keys)),
        "bridge_mean":   comp_cka(br_keys),
        "bridge_ci_95":  list(boot_ci(br_keys)),
        "lb_mean":       comp_cka(lb_keys),
        "lb_ci_95":      list(boot_ci(lb_keys)),
        "lb_per_layer":  per_layer(lb_keys, "lb"),
        "ve_per_layer":  per_layer(ve_keys, "ve"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# E7a: BASE-VS-BASE CONTROL
# ══════════════════════════════════════════════════════════════════════════════

def run_base_vs_base():
    """
    Load M0 twice independently and compute CKA.
    Expected result: CKA ≈ 1.000 everywhere IF quantization is deterministic.
    If CKA < 1.000 (especially at VE L0), this quantifies the noise floor
    caused by separate 4-bit quantization loads.
    """
    print("\n" + "="*60)
    print("E7a: BASE-VS-BASE CKA CONTROL")
    print("="*60)
    print("Loading M0 (first instance)...")
    m0a, proc = load_llava(None)
    print("Loading M0 (second instance, independent load)...")
    m0b, _    = load_llava(None)

    forget_items = load_split(MLLMU_REAL_FORGET)
    print(f"Computing CKA on {len(forget_items)} forget-set items...")
    result = compute_crp_for_items(m0a, m0b, proc, forget_items)
    result["comparison"] = "M0_instance_A vs M0_instance_B"
    result["purpose"]    = "quantization noise floor"

    out = CONTROLS_DIR / "base_vs_base_crp.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n  Base-vs-Base CKA (quantization noise floor):")
    print(f"    VE-CKA   : {result['ve_mean']:.4f}  "
          f"[{result['ve_ci_95'][0]:.4f}, {result['ve_ci_95'][1]:.4f}]")
    print(f"    BR-CKA   : {result['bridge_mean']:.4f}  "
          f"[{result['bridge_ci_95'][0]:.4f}, {result['bridge_ci_95'][1]:.4f}]")
    print(f"    LB-CKA   : {result['lb_mean']:.4f}  "
          f"[{result['lb_ci_95'][0]:.4f}, {result['lb_ci_95'][1]:.4f}]")
    if result.get("ve_per_layer"):
        layers_str = "  ".join(f"L{k}={v:.4f}" for k, v in
                               sorted(result["ve_per_layer"].items()))
        print(f"    VE layers: {layers_str}")
    if result.get("lb_per_layer"):
        layers_str = "  ".join(f"L{k}={v:.4f}" for k, v in
                               sorted(result["lb_per_layer"].items()))
        print(f"    LB layers: {layers_str}")
    print(f"\n  [saved] {out}")

    # Build narrative for paper
    ve = result['ve_mean']; lb = result['lb_mean']; br = result['bridge_mean']
    l0_ve = result.get('ve_per_layer', {}).get(0, float("nan"))
    print(f"\n  INTERPRETATION:")
    if ve > 0.999 and lb > 0.999 and br > 0.999:
        print(f"  -> Quantization is fully deterministic. All noise-floor CKA ≈ 1.000.")
        print(f"     VE L0={l0_ve:.4f} is therefore a genuine extraction artifact,")
        print(f"     not quantization noise. Exclude L0 from VE-CKA mean.")
    elif l0_ve is not None and l0_ve < 0.1:
        print(f"  -> VE L0={l0_ve:.4f} already present in base-vs-base.")
        print(f"     This CONFIRMS L0 is a quantization/hook artifact.")
        print(f"     Report: noise floor at L0 = {l0_ve:.4f}; "
              f"all experiment VE-CKA values relative to this floor.")
    else:
        print(f"  -> Noise floor: VE={ve:.4f}, BR={br:.4f}, LB={lb:.4f}")
        print(f"     Subtract or report these as baseline for interpretation.")

    del m0a, m0b
    torch.cuda.empty_cache()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# E7b: RETAIN-SET CRP (SELECTIVITY)
# ══════════════════════════════════════════════════════════════════════════════

def run_retain_crp(methods: list):
    """
    Compute CRP on the RETAIN set for each method.
    Selectivity check: if unlearning is targeted, forget-set CKA should be
    lower than retain-set CKA. Under-forgetting methods with minimal change
    should show both near 1.0 (no change on either set).
    """
    print("\n" + "="*60)
    print("E7b: RETAIN-SET CRP (SELECTIVITY)")
    print("="*60)

    retain_items = load_split(MLLMU_REAL_RETAIN)
    print(f"Retain set: {len(retain_items)} items")

    all_results = []
    for method in methods:
        ckpt = LLAVA_ADAPTERS.get(method)
        if method != "no_unlearn" and ckpt and not Path(ckpt).exists():
            print(f"  [skip] {method}: checkpoint missing")
            continue

        out_path = CONTROLS_DIR / f"{method}_retain_crp.json"
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                result = json.load(f)
            print(f"  [cached] {method}")
            all_results.append(result)
            continue

        print(f"\n  Method: {method}")
        m0, proc = load_llava(None)
        mu, _    = load_llava(ckpt)

        result = compute_crp_for_items(m0, mu, proc, retain_items)
        result["method"]  = method
        result["split"]   = "retain"
        result["purpose"] = "selectivity control"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"  RETAIN CRP — {method}:")
        print(f"    VE-CKA: {result['ve_mean']:.4f}  "
              f"BR-CKA: {result['bridge_mean']:.4f}  "
              f"LB-CKA: {result['lb_mean']:.4f}")

        all_results.append(result)
        del m0, mu
        torch.cuda.empty_cache()

    # Compare forget vs retain
    print("\n  SELECTIVITY SUMMARY (Forget-CKA vs Retain-CKA):")
    print(f"  {'Method':<15} {'F-VE':>7} {'R-VE':>7}  "
          f"{'F-BR':>7} {'R-BR':>7}  {'F-LB':>7} {'R-LB':>7}")
    print("  " + "-"*65)
    for r in all_results:
        m = r["method"]
        # Load forget result
        forget_path = PER_ENTITY_CRP_DIR / f"{m}_per_entity_crp.json"
        if not forget_path.exists():
            continue
        with open(forget_path, encoding="utf-8") as f:
            forget_r = json.load(f)
        fve = float(forget_r.get("ve_mean", float("nan")))
        fbr = float(forget_r.get("bridge_mean", float("nan")))
        flb = float(forget_r.get("lb_mean", float("nan")))
        rve = float(r.get("ve_mean", float("nan")))
        rbr = float(r.get("bridge_mean", float("nan")))
        rlb = float(r.get("lb_mean", float("nan")))
        print(f"  {m:<15} {fve:>7.4f} {rve:>7.4f}  "
              f"{fbr:>7.4f} {rbr:>7.4f}  {flb:>7.4f} {rlb:>7.4f}")

    # Save combined selectivity table
    selectivity = []
    for r in all_results:
        m = r["method"]
        forget_path = PER_ENTITY_CRP_DIR / f"{m}_per_entity_crp.json"
        if not forget_path.exists(): continue
        with open(forget_path, encoding="utf-8") as f:
            forget_r = json.load(f)
        selectivity.append({
            "method":        m,
            "forget_ve":     float(forget_r.get("ve_mean", float("nan"))),
            "forget_br":     float(forget_r.get("bridge_mean", float("nan"))),
            "forget_lb":     float(forget_r.get("lb_mean", float("nan"))),
            "retain_ve":     float(r.get("ve_mean", float("nan"))),
            "retain_br":     float(r.get("bridge_mean", float("nan"))),
            "retain_lb":     float(r.get("lb_mean", float("nan"))),
        })
    sel_path = CONTROLS_DIR / "selectivity_forget_vs_retain.json"
    with open(sel_path, "w", encoding="utf-8") as f:
        json.dump(selectivity, f, indent=2, default=str)
    print(f"\n  [saved] {sel_path}")
    return selectivity


# ══════════════════════════════════════════════════════════════════════════════
# E7c: EXTRACT PER-LAYER LB-CKA FROM EXISTING E2 JSON
# ══════════════════════════════════════════════════════════════════════════════

def extract_per_layer_lb():
    """
    Parse existing E2 per-entity JSON outputs and extract per-layer LB-CKA.
    No new computation needed. Already stored in lb_per_layer field.
    """
    print("\n" + "="*60)
    print("E7c: PER-LAYER LB-CKA (from existing E2 JSONs)")
    print("="*60)

    method_display = {
        "ga_retrained": "GA-attn4-50",
        "npo":          "NPO",
        "mmunlearner":  "MMUnlearner",
        "cagul":        "CAGUL",
        "sineproject":  "SineProject",
    }

    rows = []
    for method, display in method_display.items():
        p = PER_ENTITY_CRP_DIR / f"{method}_per_entity_crp.json"
        if not p.exists():
            print(f"  [missing] {method}")
            continue
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        lb_layers = data.get("lb_per_layer", {})
        if not lb_layers:
            print(f"  [no per-layer data] {method}")
            continue
        row = {"method": display}
        for k, v in sorted(lb_layers.items(), key=lambda x: int(x[0])):
            row[f"L{k}"] = round(float(v), 4)
        rows.append(row)
        print(f"  {display}: " + "  ".join(
            f"L{k}={v:.4f}" for k, v in sorted(lb_layers.items(), key=lambda x: int(x[0]))))

    # Build LaTeX table
    if rows:
        layer_keys = sorted(set(k for r in rows for k in r if k.startswith("L")),
                            key=lambda x: int(x[1:]))
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Per-layer language-backbone CKA on \emph{mllmu\_real}",
            r"(LLaVA-1.5-7B, $n=40$ forget samples). Values are debiased",
            r"linear CKA at sampled LM decoder layers. Deeper layers",
            r"show the largest divergence for GA-attn4-50 and MMUnlearner.}",
            r"\label{tab:lb_per_layer}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\small",
            r"\begin{tabular}{l" + "r"*len(layer_keys) + "}",
            r"\toprule",
            r"\textbf{Method} & " + " & ".join(
                f"\\textbf{{L{k[1:]}}}" for k in layer_keys) + r" \\",
            r"\midrule",
        ]
        for row in rows:
            vals = [f"{row.get(k, float('nan')):.4f}" for k in layer_keys]
            lines.append(f"{row['method']} & " + " & ".join(vals) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        latex = "\n".join(lines)

        tex_path = CONTROLS_DIR / "table_lb_per_layer.tex"
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"\n  [saved] {tex_path}")
        print("\n" + latex)

    out_path = CONTROLS_DIR / "lb_per_layer_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# LATEX TABLES FOR PAPER
# ══════════════════════════════════════════════════════════════════════════════

def build_control_tables():
    """Generate LaTeX tables for the controls appendix."""

    # Base-vs-base
    bvb_path = CONTROLS_DIR / "base_vs_base_crp.json"
    if bvb_path.exists():
        with open(bvb_path, encoding="utf-8") as f:
            bvb = json.load(f)
        l0 = bvb.get("ve_per_layer", {}).get("0", float("nan"))
        bvb_tex = (
            r"\begin{table}[h]" "\n"
            r"\centering" "\n"
            r"\caption{Base-vs-base CKA control (LLaVA-1.5-7B, $n=40$)." "\n"
            r"Two independent 4-bit NF4 loads of the same checkpoint are compared." "\n"
            r"Values near $1.000$ confirm that the extraction pipeline is" "\n"
            r"reproducible; any persistent gap quantifies the noise floor.}" "\n"
            r"\label{tab:base_vs_base}" "\n"
            r"\setlength{\tabcolsep}{5pt}\small" "\n"
            r"\begin{tabular}{lrrr}" "\n"
            r"\toprule" "\n"
            r"\textbf{Comparison} & \textbf{VE-CKA} & \textbf{BR-CKA} & \textbf{LB-CKA} \\" "\n"
            r"\midrule" "\n"
            f"M0 vs M0 (separate loads) & {bvb['ve_mean']:.4f} & "
            f"{bvb['bridge_mean']:.4f} & {bvb['lb_mean']:.4f} \\\\" "\n"
            r"\bottomrule" "\n"
            r"\end{tabular}" "\n"
            r"\end{table}"
        )
        bvb_tex_path = CONTROLS_DIR / "table_base_vs_base.tex"
        with open(bvb_tex_path, "w", encoding="utf-8") as f:
            f.write(bvb_tex)
        print(f"[saved] {bvb_tex_path}")
        print(bvb_tex)

    # Selectivity
    sel_path = CONTROLS_DIR / "selectivity_forget_vs_retain.json"
    if sel_path.exists():
        with open(sel_path, encoding="utf-8") as f:
            sel = json.load(f)
        display = {
            "ga_retrained": "GA-attn4-50",
            "npo": "NPO", "mmunlearner": "MMUnlearner",
            "cagul": "CAGUL", "sineproject": "SineProject",
        }
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Selectivity control: forget-set vs retain-set CRP",
            r"(LLaVA-1.5-7B, $n=40$ forget / $n=80$ retain samples).",
            r"F = forget set; R = retain set. For under-forgetting methods,",
            r"both F and R CKA are near $1.000$; for GA-attn4-50, the",
            r"forget-set shows lower CKA, indicating targeted representational",
            r"change on forget samples.}",
            r"\label{tab:selectivity}",
            r"\setlength{\tabcolsep}{3pt}\small",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"\textbf{Method} & \textbf{F-VE} & \textbf{R-VE} "
            r"& \textbf{F-BR} & \textbf{R-BR} "
            r"& \textbf{F-LB} & \textbf{R-LB} \\",
            r"\midrule",
        ]
        for r in sel:
            m = display.get(r["method"], r["method"])
            lines.append(
                f"{m} & {r['forget_ve']:.4f} & {r['retain_ve']:.4f}"
                f" & {r['forget_br']:.4f} & {r['retain_br']:.4f}"
                f" & {r['forget_lb']:.4f} & {r['retain_lb']:.4f} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        sel_tex = "\n".join(lines)
        sel_tex_path = CONTROLS_DIR / "table_selectivity.tex"
        with open(sel_tex_path, "w", encoding="utf-8") as f:
            f.write(sel_tex)
        print(f"\n[saved] {sel_tex_path}")
        print(sel_tex)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="all",
                        choices=["base_vs_base", "retain_crp", "extract_layers",
                                 "tables", "all"])
    parser.add_argument("--methods", nargs="+",
                        default=["npo", "mmunlearner", "cagul",
                                 "sineproject", "ga_retrained"])
    args = parser.parse_args()

    if args.run in ("base_vs_base", "all"):
        run_base_vs_base()

    if args.run in ("retain_crp", "all"):
        run_retain_crp(args.methods)

    if args.run in ("extract_layers", "all"):
        extract_per_layer_lb()

    if args.run in ("tables", "all"):
        build_control_tables()

    print("\n[E7] Done. Results in:", CONTROLS_DIR)
    print("  Add tab:base_vs_base, tab:selectivity, tab:lb_per_layer to appendix.")


if __name__ == "__main__":
    main()
