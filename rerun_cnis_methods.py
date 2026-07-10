from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "outputs" / "revision" / "cnis_rerun"
OUTDIR.mkdir(parents=True, exist_ok=True)
DEFAULT_METHODS = ["no_unlearn", "npo", "mmunlearner", "cagul"]


def extract_method(path: Path, method: str):
    data = json.loads(path.read_text(encoding="utf-8"))
    block = data.get("methods", {}).get(method, {})
    scores = block.get("cnis_scores", {})
    neighborhoods = block.get("neighborhood_results", {})
    values, excluded = [], []
    for entity in sorted(set(neighborhoods) | set(scores)):
        value = scores.get(entity)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
        else:
            excluded.append(entity)
    if len(values) >= 2:
        rng = np.random.default_rng(42)
        boots = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(5000)]
        ci = [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]
    else:
        ci = None
    return {"method": method, "total_entities": len(set(neighborhoods) | set(scores)), "included": len(values), "excluded": excluded, "mean": float(np.mean(values)) if values else None, "ci_95": ci, "per_entity": scores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--arch", default="llava", choices=["llava", "blip2"])
    parser.add_argument("--data_root", default=None)
    args = parser.parse_args()

    checks, summaries = [], []
    def check(name, passed, detail): checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    for method in args.methods:
        method_out = OUTDIR / method
        cmd = [sys.executable, str(ROOT / "run_audit.py"), "--arch", args.arch, "--methods", method, "--output_dir", str(method_out), "--no_unlearn", "--no_recovery"]
        if args.data_root: cmd += ["--data_root", args.data_root]
        print("RUN:", " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(ROOT), env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        check(f"{method} audit process", result.returncode == 0, f"exit={result.returncode}")
        raw = method_out / args.arch / f"{args.arch}_audit_results.json"
        check(f"{method} raw output", raw.exists(), raw)
        if raw.exists():
            summary = extract_method(raw, method)
            summaries.append(summary)
            check(f"{method} per-entity CNIS", summary["included"] >= 2 and summary["ci_95"] is not None, f"included={summary['included']}, total={summary['total_entities']}")

    overall = "PASS" if checks and all(x["passed"] for x in checks) else "FAIL"
    payload = {"overall": overall, "summaries": summaries, "checks": checks}
    out = OUTDIR / "cnis_method_counts_ci.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for x in checks: print(f"[{'PASS' if x['passed'] else 'FAIL'}] {x['name']}: {x['detail']}")
    print(f"OVERALL VERDICT: {overall}\nREPORT: {out}")
    return 0 if overall == "PASS" else 1

if __name__ == "__main__": raise SystemExit(main())
