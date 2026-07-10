"""
eval_behavioural.py
-------------------
Inference-only behavioural evaluation for LLaVA unlearning checkpoints.

Outputs
-------
outputs/eval_behavioural/
    behavioural_results.json      per-method, per-item raw results
    behavioural_summary.csv       one row per method
    behavioural_latex.tex         ready-to-paste LaTeX table

Usage
-----
    py eval_behavioural.py                    # runs all methods
    py eval_behavioural.py --methods ga npo   # specific methods only
    py eval_behavioural.py --resume           # skip already-computed methods
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from PIL import Image

# -- make local imports work from any CWD -------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from eval_config import (
    ARCH, DEVICE, LLAVA_BASE_MODEL, BLIP2_BASE_MODEL,
    CHECKPOINT_DIRS, ALL_METHODS, OUT_DIR,
    MAX_NEW_TOKENS, TEMPERATURE,
    FORGET_DIR, RETAIN_DIR,
)
from eval_utils import (
    load_mllmu_split, load_llava_model, load_blip2_model,
    run_llava_inference, run_blip2_inference,
    score_response, aggregate_scores, save_json,
)


# ------------------------------------------------------------------------------
# SANITY CHECK
# ------------------------------------------------------------------------------

def sanity_check():
    errs = []
    if not FORGET_DIR.exists():
        errs.append(f"FORGET_DIR not found: {FORGET_DIR}")
    if not RETAIN_DIR.exists():
        errs.append(f"RETAIN_DIR not found: {RETAIN_DIR}")
    for method, ckpt in CHECKPOINT_DIRS.items():
        if method == "no_unlearn":
            continue
        if ckpt is None or not Path(ckpt).exists():
            errs.append(f"Checkpoint missing for '{method}': {ckpt}")
    if errs:
        print("\n[ERROR] Sanity check failed:")
        for e in errs:
            print(f"  - {e}")
        print("\nEdit eval_config.py to fix paths, then rerun.")
        sys.exit(1)
    print("[OK] Sanity check passed.\n")


# ------------------------------------------------------------------------------
# EVALUATION LOOP
# ------------------------------------------------------------------------------

def evaluate_method(method: str, arch: str = "llava") -> dict:
    """
    Run forget-set and retain-set evaluation for one method.
    Returns a result dict ready to be saved.
    """
    print(f"\n{'='*60}")
    print(f"  Evaluating: {method.upper()}")
    print(f"{'='*60}")

    # -- Load data -------------------------------------------------------------
    forget_items = load_mllmu_split(FORGET_DIR)
    retain_items = load_mllmu_split(RETAIN_DIR)

    # -- Load model ------------------------------------------------------------
    ckpt = CHECKPOINT_DIRS[method]
    if arch == "llava":
        model, tokenizer, image_processor, ctx_len = load_llava_model(
            LLAVA_BASE_MODEL, ckpt, DEVICE
        )
        def infer(question, image):
            return run_llava_inference(
                model, tokenizer, image_processor,
                question, image,
                MAX_NEW_TOKENS, TEMPERATURE
            )
    else:
        model, processor = load_blip2_model(BLIP2_BASE_MODEL, ckpt, DEVICE)
        def infer(question, image):
            return run_blip2_inference(model, processor, question, image,
                                       MAX_NEW_TOKENS, DEVICE)

    # -- Forget set ------------------------------------------------------------
    print(f"\n[{method}] Evaluating FORGET set ({len(forget_items)} items)...")
    forget_scores = []
    for i, item in enumerate(forget_items):
        image = Image.open(item["image"]).convert("RGB")
        response = infer(item["question"], image)
        score = score_response(response, item)
        forget_scores.append(score)
        if (i + 1) % 5 == 0 or (i + 1) == len(forget_items):
            correct_so_far = sum(s["correct"] for s in forget_scores)
            print(f"  [{i+1}/{len(forget_items)}] "
                  f"correct={correct_so_far}/{i+1}  "
                  f"last_resp='{response[:60]}...'")

    # -- Retain set ------------------------------------------------------------
    print(f"\n[{method}] Evaluating RETAIN set ({len(retain_items)} items)...")
    retain_scores = []
    for i, item in enumerate(retain_items):
        image = Image.open(item["image"]).convert("RGB")
        response = infer(item["question"], image)
        score = score_response(response, item)
        retain_scores.append(score)
        if (i + 1) % 10 == 0 or (i + 1) == len(retain_items):
            correct_so_far = sum(s["correct"] for s in retain_scores)
            print(f"  [{i+1}/{len(retain_items)}] "
                  f"correct={correct_so_far}/{i+1}  "
                  f"last_resp='{response[:60]}...'")

    # -- Aggregate -------------------------------------------------------------
    forget_agg = aggregate_scores(forget_scores)
    retain_agg = aggregate_scores(retain_scores)

    forget_acc  = forget_agg["correct_rate"]
    forget_rate = 1.0 - forget_acc          # fraction successfully suppressed
    retain_acc  = retain_agg["correct_rate"]

    print(f"\n[{method}] RESULTS:")
    print(f"  Forget Acc  = {forget_acc:.4f}  (fraction still recalled)")
    print(f"  Forget Rate = {forget_rate:.4f}  (fraction suppressed)")
    print(f"  Retain Acc  = {retain_acc:.4f}")

    return {
        "method":      method,
        "forget_acc":  forget_acc,
        "forget_rate": forget_rate,
        "retain_acc":  retain_acc,
        "forget_n":    forget_agg["n"],
        "retain_n":    retain_agg["n"],
        "forget_scores": forget_scores,
        "retain_scores": retain_scores,
    }


# ------------------------------------------------------------------------------
# LATEX TABLE
# ------------------------------------------------------------------------------

def build_latex_table(summary: list[dict]) -> str:
    """Build the behavioural results LaTeX table."""

    header = r"""\begin{table}[t]
\centering
\caption{Behavioural results on \emph{mllmu\_real} (LLaVA-1.5-7B).
Forget Acc~= fraction of forget-set queries the model still answers
correctly (lower is better for unlearning);
Forget Rate~= $1 -$ Forget Acc (higher is better);
Retain Acc~= fraction of retain-set queries answered correctly.
High forget rate with high retain accuracy is the ideal target;
no method achieves both.}
\label{tab:behavioral}
\setlength{\tabcolsep}{5pt}
\small
\begin{tabular}{lrrrr}
\toprule
\textbf{Method} & \textbf{Forget Acc}$\downarrow$
  & \textbf{Forget Rate}$\uparrow$
  & \textbf{Retain Acc}$\uparrow$
  & \textbf{Regime} \\
\midrule"""

    METHOD_ORDER = ["no_unlearn", "ga", "npo", "mmunlearner", "cagul"]
    REGIME = {
        "no_unlearn":   "Reference",
        "ga":           "Under-forgetting",
        "npo":          "Under-forgetting",
        "mmunlearner":  "Under-forgetting",
        "cagul":        "Under-forgetting",
        "manu":         "Under-forgetting",
    }
    DISPLAY = {
        "no_unlearn":  "No Unlearn",
        "ga":          "GA",
        "npo":         "NPO",
        "mmunlearner": "MMUnlearner",
        "cagul":       "CAGUL",
        "manu":        "MANU",
    }

    rows_by_method = {r["method"]: r for r in summary}
    lines = [header]
    for m in METHOD_ORDER:
        if m not in rows_by_method:
            continue
        r = rows_by_method[m]
        fa   = f"{r['forget_acc']:.4f}"
        fr   = f"{r['forget_rate']:.4f}"
        ra   = f"{r['retain_acc']:.4f}"
        reg  = REGIME.get(m, "---")
        disp = DISPLAY.get(m, m)
        lines.append(f"{disp} & {fa} & {fr} & {ra} & {reg} \\\\")

    lines.append(r"""\bottomrule
\end{tabular}
\end{table}""")
    return "\n".join(lines)


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        help="Methods to evaluate (default: all)")
    parser.add_argument("--arch",   default=ARCH, choices=["llava", "blip2"])
    parser.add_argument("--resume", action="store_true",
                        help="Skip methods whose JSON results already exist")
    args = parser.parse_args()

    sanity_check()

    results_path  = OUT_DIR / f"behavioural_results_{args.arch}.json"
    summary_path  = OUT_DIR / f"behavioural_summary_{args.arch}.csv"
    latex_path    = OUT_DIR / f"behavioural_latex_{args.arch}.tex"

    # -- Load existing results if resuming ------------------------------------
    all_results: dict[str, dict] = {}
    if args.resume and results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            existing = json.load(f)
        all_results = {r["method"]: r for r in existing}
        print(f"[resume] Loaded {len(all_results)} existing results.")

    # -- Run evaluation --------------------------------------------------------
    for method in args.methods:
        if method in {"manu", "manu_lora"}:
            print("[skip] Invalid LLaVA MANU checkpoint; all LoRA-B tensors are zero.")
            continue
        if method not in CHECKPOINT_DIRS:
            print(f"[warn] Unknown method '{method}', skipping.")
            continue
        if args.resume and method in all_results:
            print(f"[resume] Skipping '{method}' (already done).")
            continue
        result = evaluate_method(method, args.arch)
        all_results[method] = result

        # Save after each method in case of crash
        save_json(list(all_results.values()), results_path)

    # -- Summary CSV -----------------------------------------------------------
    summary = list(all_results.values())
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "method", "forget_acc", "forget_rate", "retain_acc",
            "forget_n", "retain_n"
        ])
        writer.writeheader()
        for r in summary:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    print(f"\n[saved] {summary_path}")

    # -- LaTeX table -----------------------------------------------------------
    latex = build_latex_table(summary)
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"[saved] {latex_path}")
    print("\n" + "="*60)
    print("LATEX TABLE:")
    print("="*60)
    print(latex)

    # -- Quick summary print ---------------------------------------------------
    print("\n" + "="*60)
    print(f"{'Method':<15} {'ForgetAcc':>10} {'ForgetRate':>11} {'RetainAcc':>10}")
    print("-"*50)
    for r in summary:
        print(f"{r['method']:<15} {r['forget_acc']:>10.4f} "
              f"{r['forget_rate']:>11.4f} {r['retain_acc']:>10.4f}")


if __name__ == "__main__":
    main()

