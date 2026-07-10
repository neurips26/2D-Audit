"""
eval_recovery.py
-----------------
Adversarial recovery evaluation for all unlearning methods.
Four attack types × 5 methods × N forget entities.

Attack types
------------
1. direct    - standard forget-set query
2. rephrase  - paraphrased query that avoids entity name
3. crop      - centre-cropped image (retains central face region)
4. perturb   - Gaussian-noise-perturbed image

Outputs
-------
outputs/eval_behavioural/
    recovery_results_{arch}.json       per-item, per-attack raw scores
    recovery_summary_{arch}.csv        method × attack rate table
    recovery_latex_{arch}.tex          LaTeX table
    recovery_correlation_{arch}.json   correlation with CKA/CNIS values

Usage
-----
    py eval_recovery.py
    py eval_recovery.py --methods ga npo
    py eval_recovery.py --resume
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from eval_config import (
    ARCH, DEVICE, LLAVA_BASE_MODEL, BLIP2_BASE_MODEL,
    CHECKPOINT_DIRS, RECOVERY_METHODS, OUT_DIR,
    MAX_NEW_TOKENS, TEMPERATURE,
    FORGET_DIR, EXISTING_CRP,
    CROP_SCALE, PERTURB_STD,
)
from eval_utils import (
    load_mllmu_split, load_llava_model, load_blip2_model,
    run_llava_inference, run_blip2_inference,
    score_response, attack_crop, attack_perturb,
    make_rephrase_question, save_json,
)


ATTACK_TYPES = ["direct", "rephrase", "crop", "perturb"]


# ------------------------------------------------------------------------
# SANITY CHECK
# ------------------------------------------------------------------------

def sanity_check():
    errs = []
    if not FORGET_DIR.exists():
        errs.append(f"FORGET_DIR not found: {FORGET_DIR}")
    for m in RECOVERY_METHODS:
        ckpt = CHECKPOINT_DIRS.get(m)
        if ckpt and not Path(ckpt).exists():
            errs.append(f"Checkpoint missing for '{m}': {ckpt}")
    if errs:
        print("\n[ERROR] Sanity check failed:")
        for e in errs: print(f"  - {e}")
        sys.exit(1)
    print("[OK] Sanity check passed.\n")


# ------------------------------------------------------------------------
# INFERENCE WRAPPER
# ------------------------------------------------------------------------

def build_infer_fn(method: str, arch: str):
    """Load the checkpoint and return a callable infer(question, image)->str."""
    ckpt = CHECKPOINT_DIRS[method]
    if arch == "llava":
        model, tokenizer, image_processor, _ = load_llava_model(
            LLAVA_BASE_MODEL, ckpt, DEVICE
        )
        def infer(question, image):
            return run_llava_inference(
                model, tokenizer, image_processor,
                question, image, MAX_NEW_TOKENS, TEMPERATURE
            )
    else:
        model, processor = load_blip2_model(BLIP2_BASE_MODEL, ckpt, DEVICE)
        def infer(question, image):
            return run_blip2_inference(
                model, processor, question, image, MAX_NEW_TOKENS, DEVICE
            )
    return infer


# ------------------------------------------------------------------------
# PER-ITEM RECOVERY
# ------------------------------------------------------------------------

def evaluate_item_recovery(item: dict, infer_fn) -> dict:
    """
    Run all four attack types on a single forget-set item.
    Returns dict keyed by attack type, values are score dicts.
    """
    base_image = Image.open(item["image"]).convert("RGB")
    results = {}

    # 1. Direct
    resp = infer_fn(item["question"], base_image)
    results["direct"] = score_response(resp, item)

    # 2. Rephrase
    rephrased_q = make_rephrase_question(item)
    resp = infer_fn(rephrased_q, base_image)
    results["rephrase"] = score_response(resp, item)
    results["rephrase"]["rephrase_question"] = rephrased_q

    # 3. Crop
    cropped_img = attack_crop(base_image, CROP_SCALE)
    resp = infer_fn(item["question"], cropped_img)
    results["crop"] = score_response(resp, item)

    # 4. Perturb
    perturbed_img = attack_perturb(base_image, PERTURB_STD)
    resp = infer_fn(item["question"], perturbed_img)
    results["perturb"] = score_response(resp, item)

    return results


# ------------------------------------------------------------------------
# METHOD EVALUATION
# ------------------------------------------------------------------------

def evaluate_method_recovery(method: str, arch: str) -> dict:
    """
    Run all four attacks for all forget-set items for one method.
    Returns structured results dict.
    """
    print(f"\n{'='*60}")
    print(f"  Recovery eval: {method.upper()}")
    print(f"{'='*60}")

    forget_items = load_mllmu_split(FORGET_DIR)
    infer_fn     = build_infer_fn(method, arch)

    per_item_results = []
    for i, item in enumerate(forget_items):
        print(f"  Item [{i+1}/{len(forget_items)}] entity={item['entity']}")
        item_results = evaluate_item_recovery(item, infer_fn)

        # Print one-line summary per item
        for attack in ATTACK_TYPES:
            s = item_results[attack]
            status = "OK" if s["correct"] else "X"
            print(f"    {attack:10s} {status}  resp='{s['response'][:60]}'")

        per_item_results.append({
            "entity": item["entity"],
            "image":  str(item["image"]),
            "results": item_results,
        })

    # -- Aggregate ----------------------------------------------------------
    attack_rates = {}
    for attack in ATTACK_TYPES:
        scores = [r["results"][attack]["correct"] for r in per_item_results]
        attack_rates[attack] = sum(scores) / len(scores) if scores else float("nan")

    mean_recovery = sum(attack_rates.values()) / len(attack_rates)

    print(f"\n[{method}] RECOVERY RATES:")
    for attack in ATTACK_TYPES:
        print(f"  {attack:12s}: {attack_rates[attack]:.4f}")
    print(f"  {'mean':12s}: {mean_recovery:.4f}")

    return {
        "method":       method,
        "attack_rates": attack_rates,
        "mean_recovery": mean_recovery,
        "n_items":      len(per_item_results),
        "per_item":     per_item_results,
    }


# ------------------------------------------------------------------------
# CORRELATION ANALYSIS
# ------------------------------------------------------------------------

def compute_correlations(summary: list[dict]) -> dict:
    """
    Compute Pearson and Spearman correlations between CKA/CNIS and mean recovery.
    Uses method-level aggregates (one point per method).
    """
    from scipy.stats import pearsonr, spearmanr

    methods   = [r["method"] for r in summary]
    recovery  = [r["mean_recovery"] for r in summary]

    crp_keys = ["ve", "br", "lb", "cnis"]
    crp_labels = {
        "ve":   "VE-CKA",
        "br":   "Bridge-CKA",
        "lb":   "LB-CKA",
        "cnis": "CNIS",
    }

    correlations = {}
    for key in crp_keys:
        crp_vals = []
        rec_vals = []
        for m, r in zip(methods, recovery):
            if m not in EXISTING_CRP:
                continue
            v = EXISTING_CRP[m].get(key)
            if v is None:
                continue  # MANU CNIS = None
            crp_vals.append(v)
            rec_vals.append(r)

        if len(crp_vals) < 3:
            correlations[key] = {"pearson_r": None, "pearson_p": None,
                                  "spearman_r": None, "spearman_p": None,
                                  "n": len(crp_vals)}
            continue

        import numpy as np
        pr, pp = pearsonr(crp_vals, rec_vals)
        sr, sp = spearmanr(crp_vals, rec_vals)
        correlations[key] = {
            "label":      crp_labels[key],
            "pearson_r":  float(pr),
            "pearson_p":  float(pp),
            "spearman_r": float(sr),
            "spearman_p": float(sp),
            "n":          len(crp_vals),
            "crp_vals":   crp_vals,
            "rec_vals":   rec_vals,
            "methods":    [m for m in methods if EXISTING_CRP.get(m, {}).get(key) is not None],
        }
        print(f"  {crp_labels[key]:15s}: "
              f"Pearson r={pr:.3f} p={pp:.3f}  "
              f"Spearman r={sr:.3f} p={sp:.3f}  (n={len(crp_vals)})")

    return correlations


def build_correlation_latex(correlations: dict) -> str:
    """Build correlation results LaTeX table."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Pearson and Spearman correlations between Component Residual Profile",
        r"metrics and mean adversarial recovery rate across five unlearning methods.",
        r"Higher positive correlation indicates that methods with more residual",
        r"representational similarity are more adversarially recoverable.}",
        r"\label{tab:correlation}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\small",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Pearson $r$} & $p$ & "
        r"\textbf{Spearman $\rho$} & $p$ & $n$ \\",
        r"\midrule",
    ]
    for key, c in correlations.items():
        label = c.get("label", key)
        if c.get("pearson_r") is None:
            lines.append(f"{label} & --- & --- & --- & --- & {c['n']} \\\\")
            continue
        pr = c["pearson_r"];  pp = c["pearson_p"]
        sr = c["spearman_r"]; sp = c["spearman_p"]
        p_str_p = "$< 0.001$" if pp < 0.001 else f"{pp:.3f}"
        p_str_s = "$< 0.001$" if sp < 0.001 else f"{sp:.3f}"
        lines.append(
            f"{label} & {pr:.3f} & {p_str_p} & {sr:.3f} & {p_str_s} & {c['n']} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------------
# LATEX RECOVERY TABLE
# ------------------------------------------------------------------------

def build_recovery_latex(summary: list[dict]) -> str:
    METHOD_ORDER = ["ga", "npo", "mmunlearner", "cagul"]
    DISPLAY = {
        "ga": "GA", "npo": "NPO", "mmunlearner": "MMUnlearner",
        "cagul": "CAGUL", "manu": "MANU",
    }
    rows_by_method = {r["method"]: r for r in summary}

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Adversarial recovery rates by attack type on \emph{mllmu\_real}",
        r"(LLaVA-1.5-7B). Recovery~=~fraction of forget-set entities the model",
        r"still identifies despite output-level suppression. Direct~=~standard",
        r"identity query; Rephrase~=~paraphrased without entity name;",
        r"Crop~=~centre-cropped image ($0.6\times$); Perturb~=~Gaussian noise",
        r"($\sigma{=}25$). Higher values indicate stronger residual accessibility of",
        r"forget-set entities.}",
        r"\label{tab:recovery}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Direct} & \textbf{Rephrase} & "
        r"\textbf{Crop} & \textbf{Perturb} & \textbf{Mean} \\",
        r"\midrule",
    ]

    for m in METHOD_ORDER:
        if m not in rows_by_method:
            continue
        r  = rows_by_method[m]
        ar = r["attack_rates"]
        mn = r["mean_recovery"]
        disp = DISPLAY.get(m, m)
        lines.append(
            f"{disp} & {ar['direct']:.3f} & {ar['rephrase']:.3f} & "
            f"{ar['crop']:.3f} & {ar['perturb']:.3f} & {mn:.3f} \\\\"
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
    parser.add_argument("--methods", nargs="+", default=RECOVERY_METHODS)
    parser.add_argument("--arch",   default=ARCH, choices=["llava", "blip2"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    sanity_check()

    results_path     = OUT_DIR / f"recovery_results_{args.arch}.json"
    summary_csv_path = OUT_DIR / f"recovery_summary_{args.arch}.csv"
    latex_path       = OUT_DIR / f"recovery_latex_{args.arch}.tex"
    corr_path        = OUT_DIR / f"recovery_correlation_{args.arch}.json"
    corr_latex_path  = OUT_DIR / f"correlation_latex_{args.arch}.tex"

    # -- Resume ----------------------------------------------------------------
    all_results: dict = {}
    if args.resume and results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            existing = json.load(f)
        all_results = {r["method"]: r for r in existing}
        print(f"[resume] Loaded {len(all_results)} existing results.")

    # -- Run -------------------------------------------------------------------
    for method in args.methods:
        if method in {"manu", "manu_lora"}:
            print("[skip] Invalid LLaVA MANU checkpoint; all LoRA-B tensors are zero.")
            continue
        if method not in CHECKPOINT_DIRS:
            print(f"[warn] Unknown method '{method}', skipping.")
            continue
        if method == "no_unlearn":
            print("[skip] no_unlearn has no checkpoint - skipping recovery eval.")
            continue
        if args.resume and method in all_results:
            print(f"[resume] Skipping '{method}' (already done).")
            continue
        result = evaluate_method_recovery(method, args.arch)
        all_results[method] = result
        save_json(list(all_results.values()), results_path)

    summary = list(all_results.values())

    # -- Summary CSV -----------------------------------------------------------
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "direct", "rephrase", "crop", "perturb",
                      "mean_recovery", "n_items"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary:
            writer.writerow({
                "method":        r["method"],
                "direct":        r["attack_rates"]["direct"],
                "rephrase":      r["attack_rates"]["rephrase"],
                "crop":          r["attack_rates"]["crop"],
                "perturb":       r["attack_rates"]["perturb"],
                "mean_recovery": r["mean_recovery"],
                "n_items":       r["n_items"],
            })
    print(f"\n[saved] {summary_csv_path}")

    # -- LaTeX tables ----------------------------------------------------------
    rec_latex = build_recovery_latex(summary)
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(rec_latex)
    print(f"[saved] {latex_path}")

    # -- Correlation analysis --------------------------------------------------
    print("\n[correlation] Computing CKA/CNIS vs. recovery correlations...")
    correlations = compute_correlations(summary)
    save_json(correlations, corr_path)

    corr_latex = build_correlation_latex(correlations)
    with open(corr_latex_path, "w", encoding="utf-8") as f:
        f.write(corr_latex)
    print(f"[saved] {corr_latex_path}")

    # -- Print tables ----------------------------------------------------------
    print("\n" + "="*60)
    print("RECOVERY TABLE (LaTeX):")
    print("="*60)
    print(rec_latex)

    print("\n" + "="*60)
    print("CORRELATION TABLE (LaTeX):")
    print("="*60)
    print(corr_latex)

    # -- Quick console summary -------------------------------------------------
    print("\n" + "="*60)
    print(f"{'Method':<15} {'Direct':>7} {'Rephrase':>9} {'Crop':>6} "
          f"{'Perturb':>8} {'Mean':>6}")
    print("-"*55)
    for r in summary:
        ar = r["attack_rates"]
        print(f"{r['method']:<15} {ar['direct']:>7.3f} {ar['rephrase']:>9.3f} "
              f"{ar['crop']:>6.3f} {ar['perturb']:>8.3f} "
              f"{r['mean_recovery']:>6.3f}")


if __name__ == "__main__":
    main()
