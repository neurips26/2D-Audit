from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
try:
    from exp_config import ROOT, RESULTS_DIR
except Exception:
    ROOT = Path(__file__).resolve().parent
    RESULTS_DIR = ROOT / "outputs" / "revision"

OUT = RESULTS_DIR / "no_gpu_response_audit"
OUT.mkdir(parents=True, exist_ok=True)
CHECKS = []

SEARCH = {
    "llava": [
        ROOT / "outputs/eval_behavioural/behavioural_results_llava.json",
        ROOT / "outputs/eval_behavioural_E2_DONE/behavioural_results_llava.json",
        ROOT / "outputs/SAFE_E2_20260618_170037/eval_behavioural/behavioural_results_llava.json",
        RESULTS_DIR / "mllmu_real_behavioural_all/behavioural_results.json",
        RESULTS_DIR / "mllmu_real_smoke_then_full/full/behavioural_results.json",
    ],
    "blip2": [
        ROOT / "outputs/eval_behavioural/behavioural_results_blip2.json",
        ROOT / "outputs/eval_behavioural_E2_DONE/behavioural_results_blip2.json",
        ROOT / "outputs/SAFE_E2_20260618_170037/eval_behavioural/behavioural_results_blip2.json",
        ROOT / "outputs/blip2_mllmu/blip2/blip2_audit_results.json",
        ROOT / "outputs/blip2/blip2_audit_results.json",
    ],
}
BASE_NAMES = {"base", "baseline", "no_unlearn", "no-unlearn", "m0"}
REFUSALS = ("i cannot", "i can't", "i don't know", "i do not know",
            "i'm not sure", "unable to", "cannot identify", "sorry")


def chk(name, verdict, detail):
    verdict = verdict.upper()
    CHECKS.append({"check": name, "verdict": verdict, "detail": str(detail)})
    icon = {"PASS": "OK", "WARN": "!!", "FAIL": "XX"}.get(verdict, "??")
    print(f"[{icon} {verdict}] {name}: {detail}")


def norm(x):
    s = str(x or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"[^\w\s]", "", s)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def get_first(d, keys, default=""):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d:
            return d.get(k)
    return default


def response(item):
    return str(get_first(item, ("response", "generated", "prediction", "output"), "") or "")


def answer(item):
    return str(get_first(item, ("answer", "gt", "ground_truth", "target", "label"), "") or "")


def question(item):
    return str(get_first(item, ("question", "prompt", "query", "instruction"), "") or "")


def entity(item):
    return str(get_first(item, ("entity_id", "entity", "entity_name", "sample_id", "uid", "id"), "") or "")


def correct(item):
    v = get_first(item, ("correct", "is_correct", "match", "scored_correct"), None)
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


def canon_method(x):
    s = norm(x).replace(" ", "_")
    aliases = {"baseline": "base", "m0": "base", "no-unlearn": "no_unlearn",
               "mmun": "mmunlearner", "sineproj": "sineproject",
               "sine_project": "sineproject", "grad_diff": "graddiff"}
    return aliases.get(s, s)


def find_file(arch):
    for p in SEARCH[arch]:
        if p.exists():
            return p
    return None


def detect_methods(data):
    if isinstance(data, dict):
        for k in ("methods", "results_by_method", "models", "evaluations"):
            if isinstance(data.get(k), dict):
                return {canon_method(m): v for m, v in data[k].items()}
        known = {}
        for k, v in data.items():
            ck = canon_method(k)
            if ck in BASE_NAMES or ck in {
                "ga", "ga-50", "npo", "mmunlearner", "cagul",
                "sineproject", "manu", "graddiff"
            }:
                if isinstance(v, (dict, list)):
                    known[ck] = v
        if known:
            return known
        if data.get("method") or data.get("method_name"):
            m = data.get("method") or data.get("method_name")
            return {canon_method(m): data}
    if isinstance(data, list):
        out = {}
        for row in data:
            if isinstance(row, dict) and (row.get("method") or row.get("method_name")):
                m = row.get("method") or row.get("method_name")
                out[canon_method(m)] = row
        return out
    return {}


def split_name(item, parent=None):
    raw = get_first(item, ("split", "subset", "set"), parent)
    s = norm(raw)
    if "forget" in s:
        return "forget"
    if "retain" in s or "locality" in s:
        return "retain"
    return None


def extract_items(mdata):
    out = {"forget": [], "retain": []}
    if isinstance(mdata, list):
        for x in mdata:
            if isinstance(x, dict):
                s = split_name(x)
                if s:
                    out[s].append(x)
        return out
    if not isinstance(mdata, dict):
        return out

    explicit = {
        "forget": ("forget_per_item", "forget_results", "forget_samples",
                   "forget_items", "forget_predictions", "forget_outputs"),
        "retain": ("retain_per_item", "retain_results", "retain_samples",
                   "retain_items", "retain_predictions", "retain_outputs",
                   "locality_results", "locality_samples"),
    }
    for split, keys in explicit.items():
        for k in keys:
            if isinstance(mdata.get(k), list):
                out[split] = [x for x in mdata[k] if isinstance(x, dict)]
                break
    if out["forget"] or out["retain"]:
        return out

    for k in ("per_item", "results", "samples", "predictions", "items"):
        if not isinstance(mdata.get(k), list):
            continue
        for x in mdata[k]:
            if isinstance(x, dict):
                s = split_name(x)
                if s:
                    out[s].append(x)
        if out["forget"] or out["retain"]:
            return out

    for split in ("forget", "retain"):
        v = mdata.get(split)
        if isinstance(v, list):
            out[split] = [x for x in v if isinstance(x, dict)]
        elif isinstance(v, dict):
            for k in ("per_item", "results", "samples", "items"):
                if isinstance(v.get(k), list):
                    out[split] = [x for x in v[k] if isinstance(x, dict)]
                    break
    return out


def key_for(item, split, idx):
    e, q, a = norm(entity(item)), norm(question(item)), norm(answer(item))
    if e and q:
        return f"{split}|e={e}|q={q}"
    if q and a:
        return f"{split}|q={q}|a={a}"
    if q:
        return f"{split}|q={q}"
    if e:
        return f"{split}|e={e}"
    return f"{split}|index={idx}"


def index_items(items, split):
    out = {}
    for i, x in enumerate(items):
        k = key_for(x, split, i)
        if k in out:
            k = f"{k}|dup={i}"
        out[k] = x
    return out


def scalar_acc(mdata, split):
    if not isinstance(mdata, dict):
        return None
    keys = ("forget_acc", "forget_accuracy", "f_acc", "fa") if split == "forget" else (
        "retain_acc", "retain_accuracy", "ret_acc", "ra", "locality_acc")
    for k in keys:
        if k in mdata:
            try:
                return float(mdata[k])
            except Exception:
                pass
    if isinstance(mdata.get("metrics"), dict):
        for k in keys:
            if k in mdata["metrics"]:
                try:
                    return float(mdata["metrics"][k])
                except Exception:
                    pass
    return None


def item_acc(items):
    vals = [correct(x) for x in items]
    vals = [v for v in vals if v is not None]
    return None if not vals else sum(vals) / len(vals)


def jaccard(a, b):
    aa, bb = set(norm(a).split()), set(norm(b).split())
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def compare(base, method, method_name):
    result = {}
    examples = []
    for split in ("forget", "retain"):
        bi, mi = index_items(base[split], split), index_items(method[split], split)
        common = sorted(set(bi) & set(mi))
        if not common:
            chk(f"{method_name} {split} alignment", "WARN", "0 aligned items")
            result[split] = {"n_aligned": 0}
            continue

        exact = normalized = same_acc_diff = 0
        js = []
        base_only = method_only = both_correct = both_wrong = 0
        for k in common:
            b, m = bi[k], mi[k]
            br, mr = response(b), response(m)
            bc, mc = correct(b), correct(m)
            exact += br == mr
            normalized += norm(br) == norm(mr)
            js.append(jaccard(br, mr))
            if bc is not None and mc is not None:
                if bc and mc:
                    both_correct += 1
                elif bc and not mc:
                    base_only += 1
                elif not bc and mc:
                    method_only += 1
                else:
                    both_wrong += 1
                if bc == mc and norm(br) != norm(mr):
                    same_acc_diff += 1
            if (norm(br) != norm(mr) or bc != mc) and len(examples) < 20:
                examples.append({
                    "split": split, "item_key": k, "entity": entity(b),
                    "question": question(b), "answer": answer(b),
                    "base_response": br, "method": method_name,
                    "method_response": mr, "base_correct": bc,
                    "method_correct": mc,
                })

        n = len(common)
        overlap = normalized / n
        chk(f"{method_name} vs base [{split}]",
            "WARN" if overlap >= 0.95 else "PASS",
            f"{n} aligned; normalized overlap={overlap:.3f}; mean Jaccard={sum(js)/n:.3f}")
        result[split] = {
            "n_aligned": n,
            "base_items": len(base[split]),
            "method_items": len(method[split]),
            "exact_overlap": exact / n,
            "normalized_overlap": overlap,
            "mean_jaccard": sum(js) / n,
            "same_correctness_different_response": same_acc_diff,
            "both_correct": both_correct,
            "base_only_correct": base_only,
            "method_only_correct": method_only,
            "both_wrong": both_wrong,
        }
    return result, examples


def provenance(data, path):
    out = {"source_file": str(path), "sha256": sha256(path),
           "prompt": None, "scorer": None, "scorer_version": None}
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                lk = str(k).lower()
                if out["prompt"] is None and lk in {"prompt", "prompt_format", "prompt_template", "template"} and isinstance(v, str):
                    out["prompt"] = v
                if out["scorer"] is None and lk in {"scorer", "evaluator", "metric", "scoring_function"} and isinstance(v, str):
                    out["scorer"] = v
                if out["scorer_version"] is None and lk in {"scorer_version", "evaluator_version", "metric_version"} and isinstance(v, str):
                    out["scorer_version"] = v
                walk(v)
        elif isinstance(x, list):
            for v in x[:50]:
                walk(v)
    walk(data)
    return out


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def audit_arch(arch, sample_n):
    print("\n" + "=" * 72)
    print(f"{arch.upper()} AUDIT")
    print("=" * 72)
    path = find_file(arch)
    if path is None:
        chk(f"{arch} result file", "FAIL", "No known file found")
        return {"status": "missing"}

    chk(f"{arch} result file", "PASS", path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    methods = detect_methods(data)
    if not methods:
        chk(f"{arch} method extraction", "FAIL", "Could not parse methods")
        return {"status": "unparsed", "source": str(path)}

    chk(f"{arch} methods", "PASS", sorted(methods))
    items = {m: extract_items(v) for m, v in methods.items()}
    stats = {}
    flat = []

    for m, split_data in items.items():
        stats[m] = {}
        for split in ("forget", "retain"):
            rows = split_data[split]
            responses = [response(x) for x in rows]
            ia, sa = item_acc(rows), scalar_acc(methods[m], split)
            stats[m][split] = {
                "n": len(rows),
                "per_item_accuracy": ia,
                "scalar_accuracy": sa,
                "mean_length": None if not rows else sum(len(r) for r in responses) / len(rows),
                "empty": sum(not r.strip() for r in responses),
                "refusals": sum(any(p in norm(r) for p in REFUSALS) for r in responses),
            }
            chk(f"{arch} {m} {split} coverage",
                "PASS" if rows else "WARN", len(rows))
            if ia is not None and sa is not None:
                diff = abs(ia - sa)
                chk(f"{arch} {m} {split} scalar vs per-item",
                    "PASS" if diff <= 1e-6 else "WARN",
                    f"scalar={sa:.4f}, per-item={ia:.4f}, diff={diff:.6f}")

            for i, x in enumerate(rows):
                flat.append({
                    "arch": arch, "method": m, "split": split, "idx": i,
                    "item_key": key_for(x, split, i), "entity": entity(x),
                    "question": question(x), "answer": answer(x),
                    "response": response(x), "correct": correct(x),
                    "response_length": len(response(x)),
                })

    base = next((m for m in methods if m in BASE_NAMES), None)
    chk(f"{arch} base/no-unlearn", "PASS" if base else "WARN", base or "not found")

    comparisons, examples = {}, []
    if base:
        for m in methods:
            if m == base:
                continue
            comparisons[m], ex = compare(items[base], items[m], m)
            examples.extend(ex)

    q2 = None
    if arch == "blip2":
        if not base:
            q2 = "No base/no-unlearn result was found, so pre-existing low retain accuracy cannot yet be established."
        else:
            base_ret = stats[base]["retain"]["per_item_accuracy"]
            if base_ret is None:
                base_ret = stats[base]["retain"]["scalar_accuracy"]
            if base_ret is None:
                q2 = "Base result exists, but retain accuracy could not be recovered."
            elif base_ret < 0.40:
                q2 = (
                    f"Base BLIP-2 retain accuracy is already low ({base_ret:.4f}). "
                    "Therefore the low retain score is pre-existing under the shared "
                    "prompt/scorer rather than introduced by unlearning. Use this as "
                    "a within-architecture non-discriminative-behaviour result, not "
                    "as evidence of successful forgetting."
                )
            else:
                q2 = (
                    f"Base BLIP-2 retain accuracy is {base_ret:.4f}; therefore the "
                    "low method scores cannot automatically be called pre-existing."
                )

        sample_method = base or next(iter(items))
        for split in ("forget", "retain"):
            rows = []
            for i, x in enumerate(items[sample_method][split][:sample_n]):
                rows.append({
                    "idx": i, "entity": entity(x), "question": question(x),
                    "answer": answer(x), "response": response(x),
                    "correct": correct(x),
                })
            write_csv(OUT / f"manual_samples_blip2_{split}.csv", rows)

    return {
        "status": "ok",
        "source": str(path),
        "provenance": provenance(data, path),
        "methods": sorted(methods),
        "base": base,
        "stats": stats,
        "comparisons": comparisons,
        "q2_interpretation": q2,
        "flat": flat,
        "examples": examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("llava", "blip2", "both"), default="both")
    ap.add_argument("--show_samples", type=int, default=20)
    args = ap.parse_args()

    archs = ("llava", "blip2") if args.arch == "both" else (args.arch,)
    results = {a: audit_arch(a, args.show_samples) for a in archs}

    write_csv(OUT / "per_item_responses.csv",
              [r for v in results.values() for r in v.get("flat", [])])

    with open(OUT / "disagreement_examples.txt", "w", encoding="utf-8") as f:
        for arch, v in results.items():
            f.write(f"\n{arch.upper()}\n{'='*72}\n")
            for x in v.get("examples", []):
                f.write(f"[{x['split']}] {x['method']} | {x['item_key']}\n")
                f.write(f"Question: {x['question']}\nAnswer: {x['answer']}\n")
                f.write(f"Base ({x['base_correct']}): {x['base_response']}\n")
                f.write(f"Method ({x['method_correct']}): {x['method_response']}\n")
                f.write("-" * 72 + "\n")

    clean = {}
    for a, v in results.items():
        clean[a] = {k: val for k, val in v.items() if k not in {"flat", "examples"}}
    with open(OUT / "audit_summary.json", "w", encoding="utf-8") as f:
        json.dump({"architectures": clean, "checks": CHECKS}, f, indent=2, default=str)

    p = sum(c["verdict"] == "PASS" for c in CHECKS)
    w = sum(c["verdict"] == "WARN" for c in CHECKS)
    x = sum(c["verdict"] == "FAIL" for c in CHECKS)
    overall = "FAIL" if x else ("WARN" if w else "PASS")

    with open(OUT / "PASS_WARN_FAIL_report.json", "w", encoding="utf-8") as f:
        json.dump({"n_pass": p, "n_warn": w, "n_fail": x,
                   "overall": overall, "checks": CHECKS}, f, indent=2)

    with open(OUT / "audit_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"NO-GPU RESPONSE AUDIT\n{'='*72}\n")
        f.write(f"PASS={p} WARN={w} FAIL={x} OVERALL={overall}\n\n")
        for a, v in clean.items():
            f.write(f"{a.upper()}\n{'-'*72}\n")
            f.write(f"Source: {v.get('source')}\n")
            f.write(f"Base: {v.get('base')}\n")
            f.write(f"Methods: {v.get('methods')}\n")
            if v.get("q2_interpretation"):
                f.write("\nReviewer Q2 interpretation:\n")
                f.write(v["q2_interpretation"] + "\n")
            f.write("\n")

    print("\n" + "=" * 72)
    print("AUDIT COMPLETE")
    print(f"PASS={p} WARN={w} FAIL={x} OVERALL={overall}")
    print(f"Output: {OUT}")
    print("Open first:")
    print(OUT / "audit_summary.txt")
    print(OUT / "disagreement_examples.txt")
    print(OUT / "manual_samples_blip2_forget.csv")
    print(OUT / "manual_samples_blip2_retain.csv")


if __name__ == "__main__":
    main()
