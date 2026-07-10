"""
find_saved_response_files.py
CPU-only discovery tool.

Purpose:
- Find JSON files that actually contain per-item responses/predictions.
- Distinguish aggregate summaries from response-level files.
- Report likely schema paths and counts.
- Prioritise BLIP-2 and LLaVA behavioural outputs.

Run:
    py .\find_saved_response_files.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
try:
    from exp_config import ROOT, RESULTS_DIR
except Exception:
    ROOT = Path(__file__).resolve().parent
    RESULTS_DIR = ROOT / "outputs" / "revision"

OUT = RESULTS_DIR / "response_file_discovery"
OUT.mkdir(parents=True, exist_ok=True)

RESPONSE_KEYS = {
    "response", "generated", "prediction", "output", "answer_pred",
    "model_response", "generated_text", "decoded", "completion"
}
QUESTION_KEYS = {"question", "prompt", "query", "instruction"}
ANSWER_KEYS = {"answer", "gt", "ground_truth", "target", "label"}
CORRECT_KEYS = {"correct", "is_correct", "match", "scored_correct"}
SPLIT_KEYS = {"split", "subset", "set"}


def inspect(obj: Any, path: str = "$", depth: int = 0, max_depth: int = 8):
    findings = []
    if depth > max_depth:
        return findings

    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj[:10]):
            sample_keys = set()
            for x in obj[:10]:
                sample_keys |= set(map(str, x.keys()))
            has_response = bool(sample_keys & RESPONSE_KEYS)
            has_question = bool(sample_keys & QUESTION_KEYS)
            has_answer = bool(sample_keys & ANSWER_KEYS)
            has_correct = bool(sample_keys & CORRECT_KEYS)
            if has_response or (has_question and has_answer):
                findings.append({
                    "json_path": path,
                    "list_length": len(obj),
                    "sample_keys": sorted(sample_keys),
                    "has_response": has_response,
                    "has_question": has_question,
                    "has_answer": has_answer,
                    "has_correct": has_correct,
                    "sample": obj[0] if obj else None,
                })
        for i, x in enumerate(obj[:20]):
            findings.extend(inspect(x, f"{path}[{i}]", depth + 1, max_depth))
    elif isinstance(obj, dict):
        keys = set(map(str, obj.keys()))
        if keys & RESPONSE_KEYS:
            findings.append({
                "json_path": path,
                "list_length": None,
                "sample_keys": sorted(keys),
                "has_response": True,
                "has_question": bool(keys & QUESTION_KEYS),
                "has_answer": bool(keys & ANSWER_KEYS),
                "has_correct": bool(keys & CORRECT_KEYS),
                "sample": obj,
            })
        for k, v in obj.items():
            findings.extend(inspect(v, f"{path}.{k}", depth + 1, max_depth))
    return findings


def main():
    roots = [
        ROOT / "outputs",
        RESULTS_DIR,
    ]
    seen = set()
    rows = []

    for base in roots:
        if not base.exists():
            continue
        for p in base.rglob("*.json"):
            rp = str(p.resolve()).lower()
            if rp in seen:
                continue
            seen.add(rp)

            name = p.name.lower()
            priority = any(t in str(p).lower() for t in (
                "blip2", "llava", "behav", "eval", "result", "unlok", "mllmu_real"
            ))
            if not priority:
                continue

            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            found = inspect(data)
            if not found:
                continue

            for fnd in found:
                rows.append({
                    "file": str(p),
                    "json_path": fnd["json_path"],
                    "list_length": fnd["list_length"],
                    "has_response": fnd["has_response"],
                    "has_question": fnd["has_question"],
                    "has_answer": fnd["has_answer"],
                    "has_correct": fnd["has_correct"],
                    "sample_keys": fnd["sample_keys"],
                    "sample": fnd["sample"],
                })

    # Rank likely per-item behavioural files first.
    rows.sort(
        key=lambda r: (
            not bool(r["list_length"]),
            not r["has_response"],
            not r["has_correct"],
            "blip2" not in r["file"].lower(),
            "llava" not in r["file"].lower(),
            r["file"],
            r["json_path"],
        )
    )

    report = {
        "root": str(ROOT),
        "results_dir": str(RESULTS_DIR),
        "n_candidates": len(rows),
        "candidates": rows,
    }

    with open(OUT / "response_file_candidates.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    with open(OUT / "response_file_candidates.txt", "w", encoding="utf-8") as f:
        f.write("SAVED RESPONSE FILE DISCOVERY\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Candidates found: {len(rows)}\n\n")
        for i, r in enumerate(rows[:100], 1):
            f.write(f"[{i}] {r['file']}\n")
            f.write(f"    JSON path: {r['json_path']}\n")
            f.write(f"    list_length: {r['list_length']}\n")
            f.write(
                f"    response={r['has_response']} question={r['has_question']} "
                f"answer={r['has_answer']} correct={r['has_correct']}\n"
            )
            f.write(f"    keys: {r['sample_keys']}\n")
            sample = r["sample"]
            if isinstance(sample, dict):
                compact = {k: sample.get(k) for k in sample if k in (
                    "method", "split", "question", "answer", "response",
                    "generated", "prediction", "correct", "is_correct"
                )}
                f.write(f"    sample: {compact}\n")
            f.write("\n")

    print("=" * 90)
    print("SAVED RESPONSE FILE DISCOVERY COMPLETE")
    print("=" * 90)
    print(f"Candidates found: {len(rows)}")
    print(f"Output: {OUT}")
    print("\nOpen:")
    print(OUT / "response_file_candidates.txt")
    print("\nTop candidates:")
    for i, r in enumerate(rows[:20], 1):
        print(
            f"{i:02d}. {r['file']} | {r['json_path']} | "
            f"n={r['list_length']} | response={r['has_response']} | "
            f"correct={r['has_correct']}"
        )


if __name__ == "__main__":
    main()
