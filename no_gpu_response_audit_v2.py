"""
no_gpu_response_audit_v2.py
===========================
CPU-only parser for the actual behavioural JSON schema:

[
  {
    "method": "...",
    "forget_acc": ...,
    "retain_acc": ...,
    "forget_scores": [{"entity": ..., "response": ..., "correct": ...}, ...],
    "retain_scores": [{"entity": ..., "response": ..., "correct": ...}, ...]
  },
  ...
]

Runs on both BLIP-2 and LLaVA where available.

Usage:
    py .\no_gpu_response_audit_v2.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
try:
    from exp_config import ROOT, RESULTS_DIR
except Exception:
    ROOT = Path(__file__).resolve().parent
    RESULTS_DIR = ROOT / "outputs" / "revision"

OUT = RESULTS_DIR / "no_gpu_response_audit_v2"
OUT.mkdir(parents=True, exist_ok=True)
CHECKS = []

FILES = {
    "blip2": [
        ROOT / "outputs/eval_behavioural/behavioural_results_blip2.json",
        ROOT / "outputs/SAFE_E2_20260618_170037/eval_behavioural/behavioural_results_blip2.json",
        ROOT / "outputs/eval_behavioural_E2_DONE/behavioural_results_blip2.json",
    ],
    "llava": [
        ROOT / "outputs/eval_behavioural/behavioural_results_llava.json",
        ROOT / "outputs/SAFE_E2_20260618_170037/eval_behavioural/behavioural_results_llava.json",
        ROOT / "outputs/eval_behavioural_E2_DONE/behavioural_results_llava.json",
        RESULTS_DIR / "mllmu_real_behavioural_all/behavioural_results.json",
        RESULTS_DIR / "mllmu_real_smoke_then_full/full/behavioural_results.json",
    ],
}

REFUSALS = (
    "i cannot", "i can't", "i do not know", "i don't know",
    "i am not sure", "i'm not sure", "unable to",
    "cannot identify", "sorry"
)


def chk(label: str, verdict: str, detail: Any):
    verdict = verdict.upper()
    CHECKS.append({"check": label, "verdict": verdict, "detail": str(detail)})
    icon = {"PASS": "OK", "WARN": "!!", "FAIL": "XX"}.get(verdict, "??")
    print(f"[{icon} {verdict}] {label}: {detail}")


def norm(x: Any) -> str:
    s = str(x or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"[^\w\s]", "", s)


def canon_method(x: Any) -> str:
    s = norm(x).replace(" ", "_")
    aliases = {
        "baseline": "base", "m0": "base", "no-unlearn": "no_unlearn",
        "mmun": "mmunlearner", "sineproj": "sineproject",
        "sine_project": "sineproject", "grad_diff": "graddiff",
    }
    return aliases.get(s, s)


def find_file(arch: str) -> Optional[Path]:
    for p in FILES[arch]:
        if p.exists():
            return p
    return None


def get_method(row: Dict[str, Any], idx: int) -> str:
    for k in ("method", "method_name", "name", "adapter", "model"):
        if row.get(k):
            return canon_method(row[k])
    return f"row_{idx}"


def get_response(item: Dict[str, Any]) -> str:
    for k in ("response", "generated", "prediction", "output"):
        if k in item:
            return str(item.get(k) or "")
    return ""


def get_correct(item: Dict[str, Any]) -> Optional[bool]:
    v = item.get("correct", item.get("is_correct"))
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        if v.lower() in {"true", "1", "yes", "correct"}:
            return True
        if v.lower() in {"false", "0", "no", "incorrect"}:
            return False
    return None


def get_entity(item: Dict[str, Any], idx: int) -> str:
    for k in ("entity", "entity_id", "entity_name", "id", "sample_id"):
        if item.get(k) is not None:
            return str(item[k])
    return f"idx_{idx}"


def item_key(item: Dict[str, Any], idx: int) -> str:
    return norm(get_entity(item, idx))


def acc(items: List[Dict[str, Any]]) -> Optional[float]:
    vals = [get_correct(x) for x in items]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def refusal_count(items: List[Dict[str, Any]]) -> int:
    n = 0
    for x in items:
        r = norm(get_response(x))
        if any(p in r for p in REFUSALS):
            n += 1
    return n


def jaccard(a: str, b: str) -> float:
    aa, bb = set(norm(a).split()), set(norm(b).split())
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def parse_records(data: Any) -> Dict[str, Dict[str, Any]]:
    out = {}
    if isinstance(data, list):
        for i, row in enumerate(data):
            if not isinstance(row, dict):
                continue
            method = get_method(row, i)
            out[method] = {
                "raw": row,
                "forget": [x for x in row.get("forget_scores", []) if isinstance(x, dict)],
                "retain": [x for x in row.get("retain_scores", []) if isinstance(x, dict)],
            }
        return out

    if isinstance(data, dict):
        # already keyed by methods
        for k, row in data.items():
            if not isinstance(row, dict):
                continue
            method = canon_method(k)
            out[method] = {
                "raw": row,
                "forget": [x for x in row.get("forget_scores", row.get("forget_results", [])) if isinstance(x, dict)],
                "retain": [x for x in row.get("retain_scores", row.get("retain_results", [])) if isinstance(x, dict)],
            }
    return out


def compare(base_items, method_items, method, split):
    b = {item_key(x, i): x for i, x in enumerate(base_items)}
    m = {item_key(x, i): x for i, x in enumerate(method_items)}
    common = sorted(set(b) & set(m))
    if not common:
        chk(f"{method} {split} alignment", "WARN", "0 common entities")
        return {"n_aligned": 0}, []

    exact = normalized = same_acc_diff = 0
    js = []
    examples = []
    for k in common:
        br, mr = get_response(b[k]), get_response(m[k])
        bc, mc = get_correct(b[k]), get_correct(m[k])
        exact += br == mr
        normalized += norm(br) == norm(mr)
        js.append(jaccard(br, mr))
        if bc is not None and mc is not None and bc == mc and norm(br) != norm(mr):
            same_acc_diff += 1
        if (norm(br) != norm(mr) or bc != mc) and len(examples) < 20:
            examples.append({
                "split": split,
                "entity": k,
                "base_response": br,
                "method": method,
                "method_response": mr,
                "base_correct": bc,
                "method_correct": mc,
            })

    n = len(common)
    overlap = normalized / n
    chk(
        f"{method} vs base [{split}]",
        "WARN" if overlap >= 0.95 else "PASS",
        f"{n} aligned; normalized overlap={overlap:.3f}; mean Jaccard={sum(js)/n:.3f}",
    )
    return {
        "n_aligned": n,
        "exact_overlap": exact / n,
        "normalized_overlap": overlap,
        "mean_jaccard": sum(js) / n,
        "same_correctness_different_response": same_acc_diff,
    }, examples


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def audit_arch(arch: str, sample_n: int):
    print("\n" + "=" * 76)
    print(f"{arch.upper()} RESPONSE AUDIT V2")
    print("=" * 76)

    path = find_file(arch)
    if path is None:
        chk(f"{arch} file", "FAIL", "No behavioural file found")
        return {"status": "missing"}

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    methods = parse_records(data)
    chk(f"{arch} file", "PASS", path)
    chk(f"{arch} methods", "PASS" if methods else "FAIL", sorted(methods))
    if not methods:
        return {"status": "unparsed", "source": str(path)}

    stats = {}
    all_rows = []
    for method, rec in methods.items():
        stats[method] = {}
        for split in ("forget", "retain"):
            items = rec[split]
            ia = acc(items)
            scalar_key = "forget_acc" if split == "forget" else "retain_acc"
            scalar = rec["raw"].get(scalar_key)
            try:
                scalar = float(scalar) if scalar is not None else None
            except Exception:
                scalar = None

            stats[method][split] = {
                "n": len(items),
                "per_item_accuracy": ia,
                "scalar_accuracy": scalar,
                "mean_response_length": (
                    sum(len(get_response(x)) for x in items) / len(items)
                    if items else None
                ),
                "empty": sum(not get_response(x).strip() for x in items),
                "refusals": refusal_count(items),
                "unique_responses": len({norm(get_response(x)) for x in items}),
            }

            chk(f"{arch} {method} {split} coverage",
                "PASS" if items else "WARN", len(items))
            if ia is not None and scalar is not None:
                diff = abs(ia - scalar)
                chk(
                    f"{arch} {method} {split} stored vs recomputed",
                    "PASS" if diff <= 1e-9 else "WARN",
                    f"stored={scalar:.4f}, recomputed={ia:.4f}, diff={diff:.6f}",
                )

            for i, x in enumerate(items):
                all_rows.append({
                    "arch": arch,
                    "method": method,
                    "split": split,
                    "idx": i,
                    "entity": get_entity(x, i),
                    "response": get_response(x),
                    "correct": get_correct(x),
                    "refusal": x.get("refusal"),
                    "response_length": len(get_response(x)),
                })

    base = next((m for m in methods if m in {"no_unlearn", "base"}), None)
    chk(f"{arch} base/no-unlearn", "PASS" if base else "WARN", base or "not found")

    comparisons = {}
    examples = []
    if base:
        for method in methods:
            if method == base:
                continue
            comparisons[method] = {}
            for split in ("forget", "retain"):
                c, ex = compare(
                    methods[base][split],
                    methods[method][split],
                    method,
                    split,
                )
                comparisons[method][split] = c
                examples.extend(ex)

    interpretation = None
    if arch == "blip2" and base:
        base_ret = stats[base]["retain"]["per_item_accuracy"]
        if base_ret is not None and base_ret < 0.40:
            interpretation = (
                f"The BLIP-2 no-unlearn baseline already has low retain accuracy "
                f"({base_ret:.4f}). Therefore low retain performance is pre-existing "
                f"under this evaluator rather than caused by the adapters. The "
                f"response-overlap statistics below determine whether identical "
                f"aggregate scores also correspond to identical textual outputs."
            )

        for split in ("forget", "retain"):
            sample_rows = []
            for i, x in enumerate(methods[base][split][:sample_n]):
                sample_rows.append({
                    "idx": i,
                    "entity": get_entity(x, i),
                    "response": get_response(x),
                    "correct": get_correct(x),
                    "refusal": x.get("refusal"),
                })
            write_csv(OUT / f"manual_blip2_base_{split}.csv", sample_rows)

    write_csv(OUT / f"{arch}_all_responses.csv", all_rows)

    return {
        "status": "ok",
        "source": str(path),
        "methods": sorted(methods),
        "base": base,
        "stats": stats,
        "comparisons": comparisons,
        "interpretation": interpretation,
        "examples": examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("llava", "blip2", "both"), default="both")
    ap.add_argument("--show_samples", type=int, default=20)
    args = ap.parse_args()

    archs = ("llava", "blip2") if args.arch == "both" else (args.arch,)
    results = {a: audit_arch(a, args.show_samples) for a in archs}

    with open(OUT / "disagreement_examples.txt", "w", encoding="utf-8") as f:
        for arch, res in results.items():
            f.write(f"\n{arch.upper()}\n{'='*76}\n")
            for x in res.get("examples", []):
                f.write(f"[{x['split']}] {x['method']} | {x['entity']}\n")
                f.write(f"Base correct={x['base_correct']}: {x['base_response']}\n")
                f.write(f"Method correct={x['method_correct']}: {x['method_response']}\n")
                f.write("-" * 76 + "\n")

    clean = {a: {k: v for k, v in r.items() if k != "examples"}
             for a, r in results.items()}
    with open(OUT / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump({"architectures": clean, "checks": CHECKS}, f, indent=2)

    p = sum(c["verdict"] == "PASS" for c in CHECKS)
    w = sum(c["verdict"] == "WARN" for c in CHECKS)
    x = sum(c["verdict"] == "FAIL" for c in CHECKS)
    overall = "FAIL" if x else ("WARN" if w else "PASS")

    with open(OUT / "audit_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"NO-GPU RESPONSE AUDIT V2\n{'='*76}\n")
        f.write(f"PASS={p} WARN={w} FAIL={x} OVERALL={overall}\n\n")
        for arch, res in clean.items():
            f.write(f"{arch.upper()}\n{'-'*76}\n")
            f.write(f"Source: {res.get('source')}\n")
            f.write(f"Base: {res.get('base')}\n")
            f.write(f"Methods: {res.get('methods')}\n")
            if res.get("interpretation"):
                f.write("\nInterpretation:\n")
                f.write(res["interpretation"] + "\n")
            f.write("\n")

    with open(OUT / "PASS_WARN_FAIL_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {"n_pass": p, "n_warn": w, "n_fail": x,
             "overall": overall, "checks": CHECKS},
            f, indent=2
        )

    print("\n" + "=" * 76)
    print("AUDIT V2 COMPLETE")
    print(f"PASS={p} WARN={w} FAIL={x} OVERALL={overall}")
    print(f"Output: {OUT}")
    print("Open:")
    print(OUT / "audit_summary.txt")
    print(OUT / "audit_summary.json")
    print(OUT / "disagreement_examples.txt")


if __name__ == "__main__":
    main()
