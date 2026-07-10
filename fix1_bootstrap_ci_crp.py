"""
fix1_bootstrap_ci_crp.py  (complete replacement)
──────────────────────────────────────────────────
Produces the main mllmu_real CRP table.

Bootstrap CIs require raw activation matrices (shape [n_examples, d]) for
both M0 and Mu. The bootstrap unit is EXAMPLES, not layers.

If raw activations are not found:
  - Report aggregate point estimates only
  - Omit CI columns
  - Return WARN, never PASS for CI status

Never bootstraps lb_per_layer values as a proxy for sample-level CIs.

Usage:
    py fix1_bootstrap_ci_crp.py
    py fix1_bootstrap_ci_crp.py --n_boot 1000 --seed 42
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from exp_config import PER_ENTITY_CRP_DIR, RESULTS_DIR, ROOT

SEED   = 42
N_BOOT = 1000
ALPHA  = 0.05
CHECKS = []
START  = time.time()

def chk(label, verdict, detail):
    CHECKS.append({"check": label, "verdict": verdict, "detail": str(detail)})
    icon = {"PASS": "OK", "WARN": "!!", "FAIL": "XX"}.get(verdict, "??")
    print(f"  [{icon} {verdict}] {label}: {detail}")


# ── Verified aggregate point estimates ────────────────────────────────────────
# Source: validated E2 runs on mllmu_real.
# Used ONLY when raw activations are unavailable.
VERIFIED = {
    "npo":         {"ve": 0.9997, "br": 0.9986, "lb": 0.9931},
    "mmunlearner": {"ve": 0.9998, "br": 0.9965, "lb": 0.9336},
    "cagul":       {"ve": 0.9997, "br": 0.9973, "lb": 0.9927},
    "sineproject": {"ve": 0.9999, "br": 0.9986, "lb": 0.9980},
    "graddiff":    {"ve": 0.9728, "br": 0.8032, "lb": 0.8482},
    # GA excluded pending checkpoint provenance verification
}

METHOD_ORDER   = ["npo", "mmunlearner", "cagul", "sineproject", "graddiff"]
METHOD_DISPLAY = {
    "npo":         "NPO",
    "mmunlearner": "MMUnlearner",
    "cagul":       "CAGUL",
    "sineproject": "SineProject",
    "graddiff":    "GradDiff",
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: SEARCH FOR RAW ACTIVATION FILES
# ══════════════════════════════════════════════════════════════════════════════

def load_array(path: Path):
    """Attempt to load a numeric array from various formats."""
    suffix = path.suffix.lower()
    try:
        if suffix in (".pt", ".pth"):
            import torch
            obj = torch.load(str(path), map_location="cpu", weights_only=False)
            if isinstance(obj, dict):
                return obj
            return {"data": obj}
        elif suffix == ".npy":
            return {"data": np.load(str(path), allow_pickle=False)}
        elif suffix == ".npz":
            return dict(np.load(str(path), allow_pickle=False))
        elif suffix in (".pkl", ".pickle"):
            with open(path, "rb") as f:
                return pickle.load(f)
        elif suffix == ".json":
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        return {"_load_error": str(e)}
    return {}


def is_activation_matrix(obj) -> bool:
    """Check if obj looks like a 2D activation matrix [n_samples, d]."""
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.dim() == 2 and obj.shape[0] >= 4
    except ImportError:
        pass
    if isinstance(obj, np.ndarray):
        return obj.ndim == 2 and obj.shape[0] >= 4
    return False


def to_numpy(obj) -> np.ndarray:
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.float().cpu().numpy()
    except ImportError:
        pass
    if isinstance(obj, np.ndarray):
        return obj.astype(np.float32)
    return np.array(obj, dtype=np.float32)


def scan_activation_files(search_dirs: list) -> list:
    """
    Recursively find files that may contain raw activation matrices.
    Returns list of candidate dicts with metadata.
    """
    candidates = []
    suffixes   = {".pt", ".pth", ".npy", ".npz", ".pkl", ".pickle"}

    for base in search_dirs:
        base = Path(base)
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() not in suffixes:
                continue
            # Skip tiny files (< 1 KB) and very large files (> 2 GB)
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size < 1024 or size > 2 * 1024**3:
                continue

            # Infer method from path
            name_lower = p.stem.lower()
            method = None
            for m in METHOD_ORDER:
                if m in name_lower or m.replace("_","") in name_lower:
                    method = m
                    break

            # Infer component from path
            comp = None
            for c in ("ve", "ve_", "vision", "bridge", "br_", "projector",
                      "lb", "lm_", "language"):
                if c in name_lower:
                    comp = c[:2]
                    break

            # Infer original vs unlearned
            role = None
            for r_kw, r_val in [("m0","m0"),("base","m0"),("orig","m0"),
                                  ("mu","mu"),("unlearn","mu"),("adapted","mu")]:
                if r_kw in name_lower:
                    role = r_val
                    break

            candidates.append({
                "path":    str(p),
                "size_kb": round(size / 1024, 1),
                "method":  method,
                "comp":    comp,
                "role":    role,
                "suffix":  p.suffix.lower(),
            })

    return candidates


def inspect_activation_candidate(cand: dict) -> dict:
    """Load candidate and inspect array shapes."""
    p   = Path(cand["path"])
    obj = load_array(p)

    info = dict(cand)
    info["arrays_found"]  = []
    info["load_error"]    = obj.get("_load_error") if isinstance(obj, dict) else None

    if isinstance(obj, dict) and "_load_error" not in obj:
        for key, val in obj.items():
            try:
                arr = to_numpy(val)
                if arr.ndim >= 2:
                    info["arrays_found"].append({
                        "key":   key,
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                        "is_activation_matrix": is_activation_matrix(val),
                    })
            except Exception:
                pass
    elif not isinstance(obj, dict):
        try:
            arr = to_numpy(obj)
            if arr.ndim >= 2:
                info["arrays_found"].append({
                    "key":   "root",
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "is_activation_matrix": is_activation_matrix(obj),
                })
        except Exception:
            pass

    return info


def find_activation_pairs(candidates: list) -> dict:
    """
    Look for M0/Mu pairs per method per component.
    Returns {method: {comp: {"m0": path, "mu": path, "shape": ...}}}
    """
    pairs = {}
    for c in candidates:
        if not c.get("method") or not c.get("role"):
            continue
        m  = c["method"]
        co = c.get("comp", "unknown")
        r  = c["role"]
        pairs.setdefault(m, {}).setdefault(co, {})[r] = c["path"]

    # Report
    complete_pairs = {}
    for m, comps in pairs.items():
        for co, roles in comps.items():
            if "m0" in roles and "mu" in roles:
                complete_pairs.setdefault(m, {})[co] = {
                    "m0":  roles["m0"],
                    "mu":  roles["mu"],
                }
    return complete_pairs


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: DEBIASED CKA
# ══════════════════════════════════════════════════════════════════════════════

def debiased_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Debiased linear CKA on stacked activation matrices.
    X, Y: [n_samples, d]
    Bootstrap unit is SAMPLES (rows), not layers.
    """
    if X is None or Y is None:
        return float("nan")
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    n = X.shape[0]
    if n < 4:
        return float("nan")
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    K = X @ X.T
    L = Y @ Y.T
    Kt = K - np.diag(np.diag(K))
    Lt = L - np.diag(np.diag(L))
    c  = 1.0 / (n * (n - 3))
    def hsic(A, B):
        return c * ((A * B).sum()
                    - (2.0 / (n - 2)) * (A.sum(axis=1) * B.sum(axis=1)).sum()
                    + A.sum() * B.sum() / ((n - 1) * (n - 2)))
    h_kl = hsic(Kt, Lt)
    h_kk = hsic(Kt, Kt)
    h_ll = hsic(Lt, Lt)
    denom = np.sqrt(max(h_kk * h_ll, 1e-10))
    v = np.clip(h_kl / denom, 0.0, 1.0)
    return float(v) if np.isfinite(v) else float("nan")


def bootstrap_sample_level_ci(X: np.ndarray, Y: np.ndarray,
                               n_boot: int, alpha: float, rng) -> tuple:
    """
    Bootstrap CI by resampling EXAMPLES (rows) from paired activation matrices.
    X, Y: [n_examples, d] — must be matched row-for-row.
    Returns (lo, hi) where lo/hi are the alpha/2 and 1-alpha/2 percentiles.
    """
    n = X.shape[0]
    if n < 4:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)   # resample WITH replacement
        v   = debiased_cka(X[idx], Y[idx])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: PER-METHOD RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def process_method(method: str, activation_pairs: dict,
                   n_boot: int, rng) -> dict:
    """
    Attempt valid sample-level bootstrap for one method.
    Falls back to verified aggregate if raw activations are unavailable.
    """
    result = {
        "method":   method,
        "n_items":  0,
        "ci_type":  "unavailable",
        "ve_agg":   float("nan"), "ve_ci": (float("nan"), float("nan")), "ve_n": 0,
        "br_agg":   float("nan"), "br_ci": (float("nan"), float("nan")), "br_n": 0,
        "lb_agg":   float("nan"), "lb_ci": (float("nan"), float("nan")), "lb_n": 0,
        "source":   "none",
    }

    # --- Try raw activations ---
    m_pairs = activation_pairs.get(method, {})
    any_ci_computed = False

    for comp_label, comp_key in [("ve","ve"), ("br","br"), ("lb","lb"),
                                   ("bridge","br"), ("vision","ve"), ("language","lb")]:
        pair = m_pairs.get(comp_label)
        if not pair:
            continue
        try:
            X_data = load_array(Path(pair["m0"]))
            Y_data = load_array(Path(pair["mu"]))
            X = None; Y = None
            # Try to extract the activation matrix
            for data, target in [(X_data, "X"), (Y_data, "Y")]:
                if isinstance(data, dict):
                    for k, v in data.items():
                        if k == "_load_error": continue
                        try:
                            arr = to_numpy(v)
                            if arr.ndim == 2 and arr.shape[0] >= 4:
                                if target == "X": X = arr
                                else: Y = arr
                                break
                        except Exception:
                            pass
                else:
                    arr = to_numpy(data)
                    if arr.ndim == 2 and arr.shape[0] >= 4:
                        if target == "X": X = arr
                        else: Y = arr

            if X is None or Y is None or X.shape != Y.shape:
                continue
            n = X.shape[0]
            mean_cka = debiased_cka(X, Y)
            lo, hi   = bootstrap_sample_level_ci(X, Y, n_boot, ALPHA, rng)

            result["n_items"] = n
            result["source"]  = f"{pair['m0']} + {pair['mu']}"
            result["ci_type"] = "sample-level"
            if comp_key == "ve":
                result["ve_agg"] = mean_cka; result["ve_ci"] = (lo,hi); result["ve_n"] = n
            elif comp_key == "br":
                result["br_agg"] = mean_cka; result["br_ci"] = (lo,hi); result["br_n"] = n
            elif comp_key == "lb":
                result["lb_agg"] = mean_cka; result["lb_ci"] = (lo,hi); result["lb_n"] = n
            any_ci_computed = True
        except Exception as e:
            chk(f"{method} {comp_label} activation load", "WARN", str(e))

    # --- Fall back to verified aggregates ---
    fb = VERIFIED.get(method, {})
    for comp, key in [("ve","ve_agg"),("br","br_agg"),("lb","lb_agg")]:
        if np.isnan(result[key]) and comp in fb:
            result[key] = fb[comp]
            if not any_ci_computed:
                result["source"] = "verified_aggregate_fallback"

    # Set n_items default
    if result["n_items"] == 0:
        result["n_items"] = 40

    return result, any_ci_computed


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: LATEX TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_latex(rows: list, n_boot: int, has_sample_ci: bool) -> str:
    n = rows[0]["n_items"] if rows else 40

    if has_sample_ci:
        ci_note = (rf"Bootstrap 95\% CIs ({n_boot} resamples, bootstrap unit: "
                   rf"individual forget-set examples, $n={n}$).")
    else:
        ci_note = (r"Item-level bootstrap CIs could not be computed because "
                   r"raw activation matrices were not found. "
                   r"Values are aggregate point estimates.")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Component Residual Profile on \emph{mllmu\_real}",
        rf"(LLaVA-1.5-7B, $n={n}$ forget samples). " + ci_note + r"}",
        r"\label{tab:crp_main}",
        r"\setlength{\tabcolsep}{3pt}\small",
    ]

    if has_sample_ci:
        lines += [
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"\textbf{Method} & \textbf{VE-CKA [95\% CI]}"
            r" & \textbf{BR-CKA [95\% CI]} & \textbf{LB-CKA [95\% CI]} \\",
            r"\midrule",
        ]
        def fmt(m, ci):
            if np.isnan(m): return "---"
            if not np.isnan(ci[0]): return f"{m:.4f} [{ci[0]:.4f},{ci[1]:.4f}]"
            return f"{m:.4f}"
        for r in rows:
            d = METHOD_DISPLAY.get(r["method"], r["method"])
            lines.append(f"{d} & {fmt(r['ve_agg'],r['ve_ci'])}"
                         f" & {fmt(r['br_agg'],r['br_ci'])}"
                         f" & {fmt(r['lb_agg'],r['lb_ci'])} \\\\")
    else:
        lines += [
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"\textbf{Method} & \textbf{VE-CKA} & \textbf{BR-CKA} & \textbf{LB-CKA} \\",
            r"\midrule",
        ]
        for r in rows:
            d  = METHOD_DISPLAY.get(r["method"], r["method"])
            ve = f"{r['ve_agg']:.4f}" if not np.isnan(r["ve_agg"]) else "---"
            br = f"{r['br_agg']:.4f}" if not np.isnan(r["br_agg"]) else "---"
            lb = f"{r['lb_agg']:.4f}" if not np.isnan(r["lb_agg"]) else "---"
            lines.append(f"{d} & {ve} & {br} & {lb} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_boot",  type=int, default=N_BOOT)
    parser.add_argument("--seed",    type=int, default=SEED)
    parser.add_argument("--methods", nargs="+", default=METHOD_ORDER)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    search_dirs = [
        ROOT / "outputs",
        ROOT / "outputs" / "revision",
        ROOT / "outputs" / "crp_per_entity",
        RESULTS_DIR,
        PER_ENTITY_CRP_DIR,
    ]

    # ── Step 1: Activation inventory ─────────────────────────────────────────
    print("\n=== STEP 1: ACTIVATION FILE INVENTORY ===")
    candidates = scan_activation_files(search_dirs)
    print(f"  Found {len(candidates)} candidate activation files")

    inventory_lines = [
        "ACTIVATION FILE INVENTORY",
        f"Generated: {datetime.now().isoformat()}",
        f"Search dirs: {[str(d) for d in search_dirs]}",
        "=" * 70,
    ]
    for c in candidates:
        info = inspect_activation_candidate(c)
        inventory_lines.append(f"\n{info['path']}")
        inventory_lines.append(f"  method={info['method']}  comp={info['comp']}  "
                                f"role={info['role']}  size={info['size_kb']}KB")
        for arr in info.get("arrays_found", []):
            inventory_lines.append(f"  array key={arr['key']}  shape={arr['shape']}  "
                                    f"is_activation_matrix={arr['is_activation_matrix']}")
        if info.get("load_error"):
            inventory_lines.append(f"  LOAD ERROR: {info['load_error']}")

    inv_path = RESULTS_DIR / "activation_inventory.txt"
    with open(inv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(inventory_lines))
    print(f"  [saved] {inv_path}")

    # ── Step 2: Find paired activations ──────────────────────────────────────
    print("\n=== STEP 2: PAIRED ACTIVATION SEARCH ===")
    activation_pairs = find_activation_pairs(candidates)
    for m, comps in activation_pairs.items():
        for comp, pair in comps.items():
            print(f"  PAIR FOUND: {m} / {comp}")
            print(f"    M0: {pair['m0']}")
            print(f"    Mu: {pair['mu']}")
            chk(f"{m}/{comp} pair", "PASS", "M0+Mu found")

    if not activation_pairs:
        chk("Raw activation pairs", "WARN",
            "No M0+Mu activation pairs found. "
            "Sample-level bootstrap CIs cannot be computed. "
            "Table will show aggregate point estimates only.")

    # ── Step 3: Compute results ───────────────────────────────────────────────
    print(f"\n=== STEP 3: CRP RESULTS (seed={args.seed}, n_boot={args.n_boot}) ===")
    rows = []
    any_sample_ci = False

    for method in args.methods:
        print(f"\n  Method: {method}")
        row, had_ci = process_method(method, activation_pairs, args.n_boot, rng)
        rows.append(row)
        if had_ci:
            any_sample_ci = True

        ci_type = row["ci_type"]
        print(f"    CI type: {ci_type}")
        print(f"    VE: {row['ve_agg']:.4f}  CI={row['ve_ci']}  n={row['ve_n']}")
        print(f"    BR: {row['br_agg']:.4f}  CI={row['br_ci']}  n={row['br_n']}")
        print(f"    LB: {row['lb_agg']:.4f}  CI={row['lb_ci']}  n={row['lb_n']}")

        if had_ci:
            chk(f"{method} CI", "PASS",
                f"Sample-level bootstrap CI computed (n={row['n_items']})")
        else:
            chk(f"{method} CI", "WARN",
                "No raw activations found — using aggregate point estimates. "
                "Item-level bootstrap CIs unavailable.")

    # ── Step 4: Build outputs ─────────────────────────────────────────────────
    latex = build_latex(rows, args.n_boot, any_sample_ci)
    tex   = RESULTS_DIR / "table_crp_main.tex"
    with open(tex, "w", encoding="utf-8") as f: f.write(latex)
    chk("LaTeX table", "PASS", str(tex))

    # Validation JSON
    validation = {
        "bootstrap_unit":        "examples" if any_sample_ci else "unavailable",
        "n_samples":             rows[0]["n_items"] if rows else 40,
        "n_boot":                args.n_boot,
        "seed":                  args.seed,
        "sample_level_ci_computed": any_sample_ci,
        "ci_type":               "sample-level" if any_sample_ci else "unavailable",
        "paired_indices_verified": any_sample_ci,
        "note": ("" if any_sample_ci else
                 "Raw activation matrices not found. Bootstrap CIs require "
                 "paired [n_samples, d] matrices for M0 and Mu at each hook. "
                 "Re-run E2 with save_activations=True to enable valid CIs."),
        "methods": [{
            "method": r["method"],
            "n_items": r["n_items"],
            "ci_type": r["ci_type"],
            "source":  r["source"],
            "ve_mean": r["ve_agg"], "ve_ci_95": list(r["ve_ci"]), "ve_n": r["ve_n"],
            "br_mean": r["br_agg"], "br_ci_95": list(r["br_ci"]), "br_n": r["br_n"],
            "lb_mean": r["lb_agg"], "lb_ci_95": list(r["lb_ci"]), "lb_n": r["lb_n"],
        } for r in rows],
    }
    bv_path = RESULTS_DIR / "bootstrap_validation.json"
    with open(bv_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2, default=str)
    chk("Bootstrap validation JSON", "PASS", str(bv_path))

    # CSV
    csv_path = RESULTS_DIR / "crp_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = ["method","n_items","ci_type","ve_mean","ve_ci_lo","ve_ci_hi",
                  "br_mean","br_ci_lo","br_ci_hi","lb_mean","lb_ci_lo","lb_ci_hi"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "method": r["method"], "n_items": r["n_items"], "ci_type": r["ci_type"],
                "ve_mean": r["ve_agg"], "ve_ci_lo": r["ve_ci"][0], "ve_ci_hi": r["ve_ci"][1],
                "br_mean": r["br_agg"], "br_ci_lo": r["br_ci"][0], "br_ci_hi": r["br_ci"][1],
                "lb_mean": r["lb_agg"], "lb_ci_lo": r["lb_ci"][0], "lb_ci_hi": r["lb_ci"][1],
            })
    chk("CSV", "PASS", str(csv_path))

    # Run report
    elapsed = round(time.time() - START, 1)
    n_pass = sum(1 for c in CHECKS if c["verdict"]=="PASS")
    n_warn = sum(1 for c in CHECKS if c["verdict"]=="WARN")
    n_fail = sum(1 for c in CHECKS if c["verdict"]=="FAIL")
    report = {
        "run_time_sec": elapsed,
        "n_pass": n_pass, "n_warn": n_warn, "n_fail": n_fail,
        "checks": CHECKS,
        "validation": validation,
    }
    rj = RESULTS_DIR / "run_report.json"
    rt = RESULTS_DIR / "run_report.txt"
    with open(rj, "w", encoding="utf-8") as f: json.dump(report, f, indent=2, default=str)
    with open(rt, "w", encoding="utf-8") as f:
        f.write("FIX 1 RUN REPORT\n" + "="*60 + "\n\n")
        f.write(f"Time: {elapsed}s\n")
        f.write(f"Bootstrap unit: {validation['bootstrap_unit']}\n")
        f.write(f"N samples: {validation['n_samples']}\n")
        f.write(f"CI type: {validation['ci_type']}\n")
        f.write(f"Paired indices verified: {validation['paired_indices_verified']}\n\n")
        for c in CHECKS:
            icon = {"PASS":"OK","WARN":"!!","FAIL":"XX"}.get(c["verdict"],"??")
            f.write(f"[{icon} {c['verdict']:4s}] {c['check']}: {c['detail']}\n")
        f.write(f"\nSummary: {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL\n")
    chk("Run report", "PASS", str(rt))

    print(f"\n  Summary: {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL")
    if not any_sample_ci:
        print("\n  [WARN] Sample-level bootstrap CIs are NOT available.")
        print("         Table shows aggregate point estimates only.")
        print("         To enable valid CIs: re-run E2 with save_activations=True")
    print("\nLaTeX:\n" + latex)


if __name__ == "__main__":
    main()
