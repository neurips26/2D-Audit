"""
repair_and_rescore_mllmu_bench.py
---------------------------------
Repairs MLLMU-Bench metadata and rescoring offline using the already-saved
1,194 generated responses. No GPU inference is rerun.

Dataset schema observed:
    question
    answer
    entity_id
    entity_name

The original UTF-8 loader used the wrong metadata conventions for this
benchmark. This script reconstructs each item from the source JSON, maps it to
the saved response by split + exact question, then calls the authoritative
eval_utils.score_response(response, item).

Outputs:
    outputs/revision/mllmu_bench_full_repaired/
        <method>_results.json
        <method>_per_item.csv
        mllmu_bench_summary.json
        mllmu_bench_summary.csv
        table_mllmu_bench.tex
        repair_report.json
        provenance_manifest.json

Usage:
    py .\repair_and_rescore_mllmu_bench.py
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_config import RESULTS_DIR, ROOT

SCRIPT_VERSION = "mllmu_bench_offline_repair_v1.0"
METHODS = ["base", "npo", "mmunlearner", "cagul", "sineproject", "graddiff"]
EXPECTED_FORGET = 80
EXPECTED_RETAIN = 119
EXPECTED_TOTAL = EXPECTED_FORGET + EXPECTED_RETAIN

SOURCE_DIR = RESULTS_DIR / "mllmu_bench_full"
OUT_DIR = RESULTS_DIR / "mllmu_bench_full_repaired"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKS: list[dict[str, Any]] = []


def chk(label: str, verdict: str, detail: Any) -> str:
    CHECKS.append({
        "check": label,
        "verdict": verdict,
        "detail": str(detail),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    icon = {"PASS": "OK", "WARN": "!!", "FAIL": "XX"}.get(verdict, "??")
    print(f"  [{icon} {verdict}] {label}: {detail}")
    return verdict


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalise_question(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def decode_json(path: Path) -> Any:
    raw = path.read_bytes()
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return json.loads(raw.decode(encoding))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError(f"Could not decode/parse {path}: {' | '.join(errors)}")


def find_dataset_root() -> Path:
    candidates = [
        Path(ROOT) / "data" / "mllmu_bench",
        Path(ROOT) / "data" / "MLLMU_Bench",
        Path(ROOT) / "data" / "MLLMU-Bench",
    ]
    for p in candidates:
        if (p / "forget").exists() and (p / "retain").exists():
            return p.resolve()
    raise FileNotFoundError("MLLMU-Bench root not found")


def collect_json_records(split_dir: Path, split_name: str) -> list[dict[str, Any]]:
    records = []

    for json_path in sorted(split_dir.rglob("*.json")):
        raw = decode_json(json_path)
        rows = raw if isinstance(raw, list) else [raw]

        for local_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RuntimeError(f"Non-dict record in {json_path}")

            question = row.get("question")
            answer = row.get("answer")
            entity_id = row.get("entity_id")
            entity_name = row.get("entity_name")

            if not question:
                raise RuntimeError(f"Missing question in {json_path}")
            if not answer or not str(answer).strip():
                raise RuntimeError(f"Missing/empty answer in {json_path}")
            if not entity_name or not str(entity_name).strip():
                raise RuntimeError(f"Missing/empty entity_name in {json_path}")

            aliases = [str(entity_name).strip()]
            if entity_id:
                eid = str(entity_id).strip()
                aliases.extend([
                    eid,
                    eid.replace("_", " "),
                    eid.replace("-", " "),
                ])

            # Preserve order, remove case-insensitive duplicates.
            dedup = []
            seen = set()
            for alias in aliases:
                key = alias.casefold()
                if alias and key not in seen:
                    seen.add(key)
                    dedup.append(alias)

            records.append({
                "split": split_name,
                "question": str(question),
                "answer": str(answer),
                "entity": str(entity_name),
                "entity_name": str(entity_name),
                "entity_id": str(entity_id or ""),
                "aliases": dedup,
                "_source_json": str(json_path.resolve()),
                "_source_local_index": local_index,
            })

    return records


def build_question_index(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        key = normalise_question(row["question"])
        index.setdefault(key, []).append(row)
    return index


def load_saved_method(method: str) -> dict[str, Any]:
    path = SOURCE_DIR / f"{method}_results.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_source_item(
    saved_row: dict[str, Any],
    question_index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    key = normalise_question(saved_row.get("question"))
    matches = question_index.get(key, [])

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise RuntimeError(
            f"No dataset record matches question: {saved_row.get('question')!r}"
        )

    # Disambiguate duplicate question text using split.
    split = saved_row.get("split")
    split_matches = [x for x in matches if x.get("split") == split]
    if len(split_matches) == 1:
        return split_matches[0]

    raise RuntimeError(
        f"Ambiguous dataset mapping for question {saved_row.get('question')!r}: "
        f"{len(matches)} matches"
    )


def build_latex(results: list[dict[str, Any]]) -> str:
    display = {
        "base": "M0",
        "npo": "NPO",
        "mmunlearner": "MMUnlearner",
        "cagul": "CAGUL",
        "sineproject": "SineProject",
        "graddiff": "GradDiff",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{LLaVA behavioural evaluation on MLLMU-Bench after "
        r"offline metadata repair and authoritative rescoring "
        r"($80$ forget and $119$ retain items).}",
        r"\label{tab:mllmu_bench_repaired}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Method} & \textbf{F-Acc.$\downarrow$} & "
        r"\textbf{F-Rate$\uparrow$} & \textbf{Ret-Acc.$\uparrow$} \\",
        r"\midrule",
    ]

    for result in results:
        lines.append(
            f"{display[result['method']]} & "
            f"{result['forget_acc']:.4f} & "
            f"{result['forget_rate']:.4f} & "
            f"{result['retain_acc']:.4f} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def main() -> int:
    try:
        import eval_utils
    except Exception as exc:
        print(f"[FATAL] Cannot import eval_utils: {exc}")
        return 1

    if not hasattr(eval_utils, "score_response"):
        print("[FATAL] eval_utils.score_response missing")
        return 1

    score_response = eval_utils.score_response
    evaluator_path = Path(eval_utils.__file__).resolve()

    chk(
        "authoritative scorer",
        "PASS",
        {
            "function": (
                f"{score_response.__module__}.{score_response.__qualname__}"
            ),
            "signature": str(inspect.signature(score_response)),
            "module": str(evaluator_path),
            "sha256": sha256_file(evaluator_path),
        },
    )

    dataset_root = find_dataset_root()
    forget_records = collect_json_records(dataset_root / "forget", "forget")
    retain_records = collect_json_records(dataset_root / "retain", "retain")

    if len(forget_records) != EXPECTED_FORGET:
        chk("forget source count", "FAIL", len(forget_records))
        return 1
    if len(retain_records) != EXPECTED_RETAIN:
        chk("retain source count", "FAIL", len(retain_records))
        return 1

    chk(
        "source schema",
        "PASS",
        f"forget={len(forget_records)}, retain={len(retain_records)}; "
        "question/answer/entity_id/entity_name validated",
    )

    all_source_records = forget_records + retain_records
    question_index = build_question_index(all_source_records)

    duplicate_question_keys = {
        key: rows for key, rows in question_index.items() if len(rows) > 1
    }
    if duplicate_question_keys:
        chk(
            "duplicate source questions",
            "WARN",
            f"{len(duplicate_question_keys)} duplicate question texts",
        )
    else:
        chk("duplicate source questions", "PASS", "none")

    results = []
    summary_rows = []
    original_vs_repaired = []

    for method in METHODS:
        saved = load_saved_method(method)
        saved_rows = saved.get("per_item", [])

        if len(saved_rows) != EXPECTED_TOTAL:
            chk(f"{method} saved row count", "FAIL", len(saved_rows))
            continue

        repaired_rows = []
        mapping_errors = []

        for saved_row in saved_rows:
            try:
                source_item = resolve_source_item(saved_row, question_index)
            except Exception as exc:
                mapping_errors.append(str(exc))
                continue

            response = str(saved_row.get("response", ""))
            score = score_response(response, source_item)

            repaired = dict(saved_row)
            repaired.update({
                "answer": source_item["answer"],
                "entity": source_item["entity"],
                "entity_name": source_item["entity_name"],
                "entity_id": source_item["entity_id"],
                "aliases": source_item["aliases"],
                "correct": bool(score["correct"]),
                "refusal": bool(score.get("refusal", False)),
                "source_json": source_item["_source_json"],
                "metadata_repaired": True,
            })
            repaired_rows.append(repaired)

            original_vs_repaired.append({
                "method": method,
                "split": repaired["split"],
                "item_id": repaired["item_id"],
                "question": repaired["question"],
                "entity_name": repaired["entity_name"],
                "saved_correct_before": bool(saved_row.get("correct")),
                "correct_after": bool(score["correct"]),
                "changed": bool(saved_row.get("correct")) != bool(score["correct"]),
                "response": response,
            })

        if mapping_errors:
            chk(
                f"{method} mapping",
                "FAIL",
                f"{len(mapping_errors)} errors; first={mapping_errors[0]}",
            )
            continue

        if len(repaired_rows) != EXPECTED_TOTAL:
            chk(
                f"{method} repaired row count",
                "FAIL",
                len(repaired_rows),
            )
            continue

        forget_rows = [x for x in repaired_rows if x["split"] == "forget"]
        retain_rows = [x for x in repaired_rows if x["split"] == "retain"]

        if len(forget_rows) != EXPECTED_FORGET or len(retain_rows) != EXPECTED_RETAIN:
            chk(
                f"{method} split counts",
                "FAIL",
                f"forget={len(forget_rows)}, retain={len(retain_rows)}",
            )
            continue

        forget_correct = sum(bool(x["correct"]) for x in forget_rows)
        retain_correct = sum(bool(x["correct"]) for x in retain_rows)

        forget_acc = forget_correct / EXPECTED_FORGET
        retain_acc = retain_correct / EXPECTED_RETAIN
        forget_rate = 1.0 - forget_acc

        repaired_result = {
            **{k: v for k, v in saved.items() if k != "per_item"},
            "script_version": SCRIPT_VERSION,
            "repair_type": "offline_metadata_repair_and_authoritative_rescore",
            "source_results_file": str(
                (SOURCE_DIR / f"{method}_results.json").resolve()
            ),
            "dataset_root": str(dataset_root),
            "evaluation_complete": True,
            "metadata_repaired": True,
            "forget_n": EXPECTED_FORGET,
            "retain_n": EXPECTED_RETAIN,
            "forget_correct": forget_correct,
            "retain_correct": retain_correct,
            "forget_acc": forget_acc,
            "forget_rate": forget_rate,
            "retain_acc": retain_acc,
            "per_item": repaired_rows,
        }

        result_path = OUT_DIR / f"{method}_results.json"
        csv_path = OUT_DIR / f"{method}_per_item.csv"

        save_json(result_path, repaired_result)

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(repaired_rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(repaired_rows)

        changed_count = sum(
            1
            for row in original_vs_repaired
            if row["method"] == method and row["changed"]
        )

        chk(
            f"{method} repaired",
            "PASS",
            f"F-Acc={forget_acc:.4f}, F-Rate={forget_rate:.4f}, "
            f"Ret-Acc={retain_acc:.4f}; changed={changed_count}/{EXPECTED_TOTAL}",
        )

        summary_rows.append({
            "method": method,
            "forget_n": EXPECTED_FORGET,
            "forget_correct": forget_correct,
            "forget_acc": forget_acc,
            "forget_rate": forget_rate,
            "retain_n": EXPECTED_RETAIN,
            "retain_correct": retain_correct,
            "retain_acc": retain_acc,
            "changed_scores": changed_count,
        })
        results.append(repaired_result)

    if len(results) != len(METHODS):
        chk("method coverage", "FAIL", f"{len(results)}/{len(METHODS)}")
        return 1

    summary_csv = OUT_DIR / "mllmu_bench_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    save_json(
        OUT_DIR / "mllmu_bench_summary.json",
        {
            "script_version": SCRIPT_VERSION,
            "repair_type": "offline_metadata_repair_and_authoritative_rescore",
            "source_directory": str(SOURCE_DIR.resolve()),
            "dataset_root": str(dataset_root),
            "scorer": {
                "function": (
                    f"{score_response.__module__}.{score_response.__qualname__}"
                ),
                "signature": str(inspect.signature(score_response)),
                "module": str(evaluator_path),
                "module_sha256": sha256_file(evaluator_path),
            },
            "methods": [
                {k: v for k, v in result.items() if k != "per_item"}
                for result in results
            ],
        },
    )

    (OUT_DIR / "table_mllmu_bench.tex").write_text(
        build_latex(results),
        encoding="utf-8",
    )

    with (OUT_DIR / "score_changes.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(original_vs_repaired[0].keys()),
        )
        writer.writeheader()
        writer.writerows(original_vs_repaired)

    n_pass = sum(c["verdict"] == "PASS" for c in CHECKS)
    n_warn = sum(c["verdict"] == "WARN" for c in CHECKS)
    n_fail = sum(c["verdict"] == "FAIL" for c in CHECKS)

    report = {
        "script_version": SCRIPT_VERSION,
        "overall_verdict": "FAIL" if n_fail else ("WARN" if n_warn else "PASS"),
        "checks": CHECKS,
        "n_pass": n_pass,
        "n_warn": n_warn,
        "n_fail": n_fail,
        "no_gpu_rerun": True,
        "source_responses_reused": EXPECTED_TOTAL * len(METHODS),
        "dataset_schema": [
            "question",
            "answer",
            "entity_id",
            "entity_name",
        ],
    }
    save_json(OUT_DIR / "repair_report.json", report)

    save_json(
        OUT_DIR / "provenance_manifest.json",
        {
            "script_version": SCRIPT_VERSION,
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_directory": str(SOURCE_DIR.resolve()),
            "output_directory": str(OUT_DIR.resolve()),
            "dataset_root": str(dataset_root),
            "scorer_module": str(evaluator_path),
            "scorer_module_sha256": sha256_file(evaluator_path),
            "no_gpu_rerun": True,
        },
    )

    print("\n" + "=" * 76)
    print("REPAIRED MLLMU-BENCH SUMMARY")
    print("=" * 76)
    print(
        f"{'Method':<16}"
        f"{'F-Acc':>10}"
        f"{'F-Rate':>10}"
        f"{'Ret-Acc':>10}"
        f"{'Changed':>10}"
    )
    for row in summary_rows:
        print(
            f"{row['method']:<16}"
            f"{row['forget_acc']:>10.4f}"
            f"{row['forget_rate']:>10.4f}"
            f"{row['retain_acc']:>10.4f}"
            f"{row['changed_scores']:>10}"
        )

    print(
        f"\nChecks: {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL "
        f"-> {OUT_DIR / 'repair_report.json'}"
    )

    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
