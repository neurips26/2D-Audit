"""
audit_bootstrap_cis.py
Audits paired example-level CRP bootstrap outputs.

Checks:
- all five methods exist
- bootstrap unit is examples
- n_samples == 40
- paired_indices_verified == true
- saved bootstrap arrays exist and contain 1000 finite replicates
- recomputed percentile intervals match JSON
- point estimate location relative to bootstrap distribution
- flags unusual cases where the point estimate lies outside percentile CI
- computes percentile, basic, and median-bias diagnostics without overwriting results

Usage:
    py .\audit_bootstrap_cis.py
"""

from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_config import RESULTS_DIR

ROOT = RESULTS_DIR / "paired_activation_bootstrap"
OUT = ROOT / "ci_audit"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = ["npo", "mmunlearner", "cagul", "sineproject", "graddiff"]
COMPONENTS = ["ve", "bridge", "lb"]
CHECKS = []

def chk(label, verdict, detail):
    CHECKS.append({"check": label, "verdict": verdict, "detail": str(detail)})
    icon = {"PASS":"OK","WARN":"!!","FAIL":"XX"}.get(verdict,"??")
    print(f"  [{icon} {verdict}] {label}: {detail}")

def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

rows = []
for method in METHODS:
    ci_path = ROOT / method / "bootstrap_ci.json"
    npz_path = ROOT / method / "bootstrap_samples.npz"
    if not ci_path.exists():
        chk(f"{method} CI JSON", "FAIL", ci_path)
        continue
    if not npz_path.exists():
        chk(f"{method} bootstrap samples", "FAIL", npz_path)
        continue

    data = json.loads(ci_path.read_text(encoding="utf-8"))
    arrs = np.load(npz_path)

    if data.get("bootstrap_unit") != "examples":
        chk(f"{method} bootstrap unit", "FAIL", data.get("bootstrap_unit"))
        continue
    if data.get("n_samples") != 40:
        chk(f"{method} sample count", "FAIL", data.get("n_samples"))
        continue
    if data.get("paired_indices_verified") is not True:
        chk(f"{method} pairing", "FAIL", data.get("paired_indices_verified"))
        continue

    chk(f"{method} provenance", "PASS", "examples, n=40, paired=true")

    for comp in COMPONENTS:
        key = comp
        if key not in arrs:
            chk(f"{method}/{comp} samples", "FAIL", "array missing")
            continue
        vals = np.asarray(arrs[key], dtype=np.float64)
        if vals.ndim != 1 or len(vals) != int(data["n_bootstrap"]):
            chk(f"{method}/{comp} samples", "FAIL",
                f"shape={vals.shape}, expected={data['n_bootstrap']}")
            continue
        if not np.isfinite(vals).all():
            chk(f"{method}/{comp} finite", "FAIL", "NaN/Inf found")
            continue

        point = float(data["point_estimates"][comp])
        saved_lo, saved_hi = map(float, data["ci_95"][comp])
        alpha = float(data["alpha"])
        pct_lo = float(np.percentile(vals, 100 * alpha / 2))
        pct_hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
        basic_lo = float(2 * point - pct_hi)
        basic_hi = float(2 * point - pct_lo)
        median = float(np.median(vals))
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1))
        bias = mean - point
        point_inside = pct_lo <= point <= pct_hi
        saved_match = abs(saved_lo-pct_lo) < 1e-10 and abs(saved_hi-pct_hi) < 1e-10

        if not saved_match:
            chk(f"{method}/{comp} saved CI", "FAIL",
                f"saved=[{saved_lo},{saved_hi}] recomputed=[{pct_lo},{pct_hi}]")
        elif not point_inside:
            chk(f"{method}/{comp} point-vs-percentile", "WARN",
                f"point={point:.4f} outside [{pct_lo:.4f},{pct_hi:.4f}], "
                f"bootstrap bias={bias:+.4f}")
        else:
            chk(f"{method}/{comp} CI audit", "PASS",
                f"point={point:.4f} inside [{pct_lo:.4f},{pct_hi:.4f}]")

        rows.append({
            "method": method,
            "component": comp,
            "point": point,
            "bootstrap_mean": mean,
            "bootstrap_median": median,
            "bootstrap_sd": sd,
            "bootstrap_bias_mean_minus_point": bias,
            "percentile_low": pct_lo,
            "percentile_high": pct_hi,
            "basic_low": basic_lo,
            "basic_high": basic_hi,
            "point_inside_percentile": point_inside,
            "saved_percentile_matches": saved_match,
            "n_bootstrap": len(vals),
        })

csv_path = OUT / "ci_audit.csv"
if rows:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

mm = next((r for r in rows if r["method"]=="mmunlearner" and r["component"]=="lb"), None)
conclusion = {}
if mm:
    conclusion = {
        "mmunlearner_lb_point": mm["point"],
        "percentile_ci": [mm["percentile_low"], mm["percentile_high"]],
        "basic_ci": [mm["basic_low"], mm["basic_high"]],
        "bootstrap_mean": mm["bootstrap_mean"],
        "bias": mm["bootstrap_bias_mean_minus_point"],
        "point_inside_percentile": mm["point_inside_percentile"],
        "interpretation": (
            "Percentile CI excludes the full-sample point estimate because the "
            "bootstrap distribution is upward-shifted. This is an estimator-bias "
            "diagnostic, not a file corruption signal, provided all provenance "
            "and recomputation checks pass."
        )
    }

npass = sum(c["verdict"]=="PASS" for c in CHECKS)
nwarn = sum(c["verdict"]=="WARN" for c in CHECKS)
nfail = sum(c["verdict"]=="FAIL" for c in CHECKS)
report = {
    "checks": CHECKS,
    "n_pass": npass, "n_warn": nwarn, "n_fail": nfail,
    "overall_verdict": "FAIL" if nfail else ("WARN" if nwarn else "PASS"),
    "mmunlearner_lb": conclusion,
    "csv": str(csv_path),
}
save_json(OUT / "ci_audit_report.json", report)

txt = [
    "CRP BOOTSTRAP CI AUDIT",
    "="*72,
    f"PASS={npass} WARN={nwarn} FAIL={nfail}",
    "",
    "MMUnlearner LB:",
    json.dumps(conclusion, indent=2),
]
(OUT / "ci_audit_report.txt").write_text("\n".join(txt), encoding="utf-8")

print(f"\nOverall: {report['overall_verdict']}")
print(f"Report: {OUT / 'ci_audit_report.json'}")
raise SystemExit(1 if nfail else 0)
