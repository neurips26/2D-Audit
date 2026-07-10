#!/usr/bin/env python3
"""
Collect matched cross-dataset unlearning results from the project tree.

Purpose
-------
Build one auditable table across:
- mllmu_real
- mllmu_bench
- unlok_vqa

The collector is deliberately conservative:
- it never invents missing values;
- it records source files for every row;
- it detects common metric aliases;
- it writes unmatched/ambiguous files for review;
- it produces PASS/WARN/FAIL reports.

Outputs
-------
outputs/revision/matched_cross_dataset/
    matched_results_long.csv
    matched_results_wide.csv
    matched_results.json
    source_inventory.csv
    unmatched_json_files.csv
    collection_report.json
    collection_report.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from exp_config import ROOT
except Exception:
    ROOT = SCRIPT_DIR

DEFAULT_SEARCH_ROOTS = [
    ROOT / "outputs" / "revision",
    ROOT / "outputs",
    ROOT / "results",
]

DEFAULT_OUTPUT = ROOT / "outputs" / "revision" / "matched_cross_dataset"

DATASET_ALIASES = {
    "mllmu_real": [
        "mllmu_real",
        "mllmu-real",
        "mllmureal",
        "real_mllmu",
    ],
    "mllmu_bench": [
        "mllmu_bench",
        "mllmu-bench",
        "mllmubench",
        "mllmu",
    ],
    "unlok_vqa": [
        "unlok_vqa",
        "unlok-vqa",
        "unlokvqa",
        "unlok",
    ],
}

METHOD_ALIASES = {
    "original": ["original", "orig", "base", "no_unlearn", "no-unlearn"],
    "retrain": ["retrain", "retr", "full_retrain", "full-retrain"],
    "ga": ["gradient_ascent", "gradient-ascent", "ga"],
    "npo": ["npo"],
    "mmunlearner": ["mmunlearner", "mm_unlearner", "mm-unlearner"],
    "manu": ["manu"],
    "cagul": ["cagul"],
    "sineproject": ["sineproject", "sine_project", "sine-project"],
    "graddiff": ["graddiff", "grad_diff", "grad-diff"],
    "random_label": ["random_label", "random-label", "randomlabel", "rl"],
    "mu_align": ["mu_align", "mu-align", "mualign"],
    "scrub": ["scrub"],
}

METRIC_ALIASES = {
    "forget_accuracy": [
        "forget_accuracy", "forget_acc", "forgetacc", "fa",
        "forget_score", "forget",
    ],
    "retain_accuracy": [
        "retain_accuracy", "retain_acc", "retainacc", "ra",
        "retain_score", "retain",
    ],
    "overall_accuracy": [
        "overall_accuracy", "overall_acc", "accuracy", "acc",
    ],
    "mia": [
        "mia", "membership_inference", "membership_inference_accuracy",
        "mia_accuracy", "mia_score",
    ],
    "ad": [
        "ad", "answer_distance", "answer_dist",
    ],
    "js": [
        "js", "js_divergence", "jensen_shannon",
    ],
    "cka_vision": [
        "ve_cka", "vision_cka", "cka_vision", "vision_encoder_cka",
    ],
    "cka_bridge": [
        "bridge_cka", "br_cka", "cka_bridge", "projector_cka",
    ],
    "cka_language": [
        "lb_cka", "language_cka", "cka_language", "decoder_cka",
    ],
    "cnis": [
        "cnis", "semantic_neighborhood", "semantic_neighbourhood",
    ],
}

SEED_KEYS = ["seed", "random_seed", "rng_seed"]
METHOD_KEYS = ["method", "method_name", "algorithm", "unlearning_method"]
DATASET_KEYS = ["dataset", "dataset_name", "benchmark", "data_name"]


def normalize_token(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def flatten_dict(
    obj: Any,
    prefix: str = "",
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if result is None:
        result = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            next_key = f"{prefix}.{key}" if prefix else str(key)
            flatten_dict(value, next_key, result)
    elif isinstance(obj, list):
        # Keep short scalar lists only; do not explode prediction arrays.
        if len(obj) <= 10 and all(
            isinstance(x, (str, int, float, bool, type(None)))
            for x in obj
        ):
            result[prefix] = obj
    else:
        result[prefix] = obj

    return result


def canonical_dataset(*values: Any) -> str | None:
    joined = " ".join(str(v or "") for v in values).casefold()

    # Test specific alias mllmu_real before generic mllmu.
    order = ["mllmu_real", "mllmu_bench", "unlok_vqa"]

    for canonical in order:
        for alias in DATASET_ALIASES[canonical]:
            if alias.casefold() in joined:
                return canonical

    return None


def canonical_method(*values: Any) -> str | None:
    joined = " ".join(str(v or "") for v in values).casefold()
    normalized = normalize_token(joined)

    # Longer aliases first to avoid GA matching arbitrary substrings.
    candidates = []
    for canonical, aliases in METHOD_ALIASES.items():
        for alias in aliases:
            candidates.append((len(alias), canonical, normalize_token(alias)))

    for _, canonical, alias in sorted(candidates, reverse=True):
        if re.search(rf"(^|_){re.escape(alias)}($|_)", normalized):
            return canonical

    return None


def find_value_by_keys(
    flattened: dict[str, Any],
    accepted_keys: Iterable[str],
) -> Any:
    accepted = {normalize_token(k) for k in accepted_keys}

    for full_key, value in flattened.items():
        leaf = normalize_token(full_key.split(".")[-1])
        if leaf in accepted:
            return value

    return None


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None

    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None

    text = str(value).strip().replace("%", "")
    try:
        result = float(text)
    except ValueError:
        return None

    if "%" in str(value):
        result /= 100.0

    return result if math.isfinite(result) else None


def extract_metrics(flattened: dict[str, Any]) -> dict[str, float]:
    normalized_map = {
        normalize_token(key): value
        for key, value in flattened.items()
    }
    found: dict[str, float] = {}

    for canonical, aliases in METRIC_ALIASES.items():
        candidates = []

        for full_key, value in normalized_map.items():
            leaf = full_key.split("_")[-1]
            for alias in aliases:
                alias_n = normalize_token(alias)

                # Exact leaf/full suffix preferred.
                score = 0
                if full_key == alias_n:
                    score = 100
                elif full_key.endswith("_" + alias_n):
                    score = 90
                elif alias_n in full_key:
                    score = 50

                if score:
                    number = numeric(value)
                    if number is not None:
                        candidates.append((score, len(full_key), number, full_key))

        if candidates:
            candidates.sort(key=lambda x: (-x[0], x[1], x[3]))
            found[canonical] = candidates[0][2]

    return found


def infer_seed(flattened: dict[str, Any], path: Path) -> int | None:
    value = find_value_by_keys(flattened, SEED_KEYS)
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    match = re.search(r"(?:seed|s)[_\-]?(\d{1,6})", str(path), re.I)
    return int(match.group(1)) if match else None


def iter_candidate_rows(data: Any) -> Iterable[dict[str, Any]]:
    if isinstance(data, dict):
        # Emit root object.
        yield data

        # Emit likely row collections.
        for key in (
            "results", "rows", "methods", "experiments",
            "summary", "metrics", "runs",
        ):
            value = data.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(value, dict):
                for child_key, item in value.items():
                    if isinstance(item, dict):
                        augmented = dict(item)
                        augmented.setdefault("_container_key", child_key)
                        yield augmented

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item


def collect(search_roots: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []

    seen_files: set[Path] = set()

    for search_root in search_roots:
        if not search_root.exists():
            continue

        for path in search_root.rglob("*.json"):
            resolved = path.resolve()

            if resolved in seen_files:
                continue
            seen_files.add(resolved)

            if "matched_cross_dataset" in normalize_token(str(path)):
                continue
            if path.name.casefold().endswith("outputs.json"):
                # Prediction dumps are not aggregate metric files.
                continue

            entry = {
                "source_file": str(resolved),
                "size_bytes": path.stat().st_size,
                "status": "",
                "dataset": "",
                "method": "",
                "seed": "",
                "metric_count": 0,
                "error": "",
            }

            try:
                data = read_json(path)
            except Exception as exc:
                entry["status"] = "json_read_error"
                entry["error"] = repr(exc)
                inventory.append(entry)
                continue

            file_rows = 0

            for candidate in iter_candidate_rows(data):
                flattened = flatten_dict(candidate)
                container_key = candidate.get("_container_key", "")

                dataset_value = find_value_by_keys(flattened, DATASET_KEYS)
                method_value = find_value_by_keys(flattened, METHOD_KEYS)

                dataset = canonical_dataset(
                    dataset_value,
                    path,
                    container_key,
                )
                method = canonical_method(
                    method_value,
                    path,
                    container_key,
                )
                metrics = extract_metrics(flattened)
                seed = infer_seed(flattened, path)

                if not dataset or not method or not metrics:
                    continue

                row = {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "source_file": str(resolved),
                    "source_container": str(container_key),
                }
                row.update(metrics)
                rows.append(row)
                file_rows += 1

                entry.update({
                    "status": "matched",
                    "dataset": dataset,
                    "method": method,
                    "seed": seed if seed is not None else "",
                    "metric_count": len(metrics),
                })

            if file_rows == 0:
                entry["status"] = "unmatched"

            inventory.append(entry)

    return rows, inventory


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Keep the richest row for each dataset/method/seed/source combination.
    """
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        key = (
            row["dataset"],
            row["method"],
            row.get("seed"),
            row["source_file"],
        )
        grouped[key].append(row)

    result = []

    for candidates in grouped.values():
        candidates.sort(
            key=lambda row: sum(
                1 for metric in METRIC_ALIASES
                if row.get(metric) is not None
            ),
            reverse=True,
        )
        result.append(candidates[0])

    return result


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Aggregate multiple seeds/sources by mean while preserving source count.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[(row["dataset"], row["method"])].append(row)

    aggregated = []

    for (dataset, method), items in sorted(grouped.items()):
        out = {
            "dataset": dataset,
            "method": method,
            "n_rows": len(items),
            "seeds": ",".join(
                str(x) for x in sorted({
                    item["seed"]
                    for item in items
                    if item.get("seed") is not None
                })
            ),
            "source_files": " | ".join(sorted({
                item["source_file"] for item in items
            })),
        }

        for metric in METRIC_ALIASES:
            values = [
                float(item[metric])
                for item in items
                if item.get(metric) is not None
            ]
            out[metric] = (
                sum(values) / len(values)
                if values
                else None
            )

        aggregated.append(out)

    return aggregated


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            writer.writerow({
                field: "" if row.get(field) is None else row.get(field)
                for field in fields
            })


def build_wide(aggregated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, dict[str, Any]] = defaultdict(dict)

    for row in aggregated:
        method = row["method"]
        dataset = row["dataset"]
        by_method[method]["method"] = method

        for metric in METRIC_ALIASES:
            by_method[method][f"{dataset}__{metric}"] = row.get(metric)

        by_method[method][f"{dataset}__n_rows"] = row["n_rows"]
        by_method[method][f"{dataset}__seeds"] = row["seeds"]

    return [by_method[key] for key in sorted(by_method)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="May be supplied more than once.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    search_roots = (
        [Path(x).resolve() for x in args.search_root]
        if args.search_root
        else [x.resolve() for x in DEFAULT_SEARCH_ROOTS]
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows, inventory = collect(search_roots)
    rows = deduplicate(rows)
    aggregated = aggregate(rows)
    wide = build_wide(aggregated)

    long_fields = [
        "dataset", "method", "n_rows", "seeds",
        *METRIC_ALIASES.keys(),
        "source_files",
    ]
    source_fields = [
        "source_file", "size_bytes", "status", "dataset",
        "method", "seed", "metric_count", "error",
    ]

    wide_fields = ["method"]
    for dataset in DATASET_ALIASES:
        for metric in METRIC_ALIASES:
            wide_fields.append(f"{dataset}__{metric}")
        wide_fields.extend([
            f"{dataset}__n_rows",
            f"{dataset}__seeds",
        ])

    write_csv(
        output / "matched_results_long.csv",
        aggregated,
        long_fields,
    )
    write_csv(
        output / "matched_results_wide.csv",
        wide,
        wide_fields,
    )
    write_csv(
        output / "source_inventory.csv",
        inventory,
        source_fields,
    )
    write_csv(
        output / "unmatched_json_files.csv",
        [x for x in inventory if x["status"] != "matched"],
        source_fields,
    )

    datasets_found = sorted({row["dataset"] for row in aggregated})
    methods_found = sorted({row["method"] for row in aggregated})

    coverage = {
        dataset: sorted({
            row["method"]
            for row in aggregated
            if row["dataset"] == dataset
        })
        for dataset in DATASET_ALIASES
    }

    matched_methods_all_three = sorted(
        set(coverage["mllmu_real"])
        & set(coverage["mllmu_bench"])
        & set(coverage["unlok_vqa"])
    )

    report = {
        "search_roots": [str(x) for x in search_roots],
        "output": str(output),
        "json_files_scanned": len(inventory),
        "matched_source_files": sum(
            1 for x in inventory if x["status"] == "matched"
        ),
        "raw_rows": len(rows),
        "aggregated_rows": len(aggregated),
        "datasets_found": datasets_found,
        "methods_found": methods_found,
        "coverage": coverage,
        "methods_present_on_all_three_datasets": matched_methods_all_three,
        "files": {
            "long_csv": str(output / "matched_results_long.csv"),
            "wide_csv": str(output / "matched_results_wide.csv"),
            "source_inventory": str(output / "source_inventory.csv"),
            "unmatched": str(output / "unmatched_json_files.csv"),
        },
    }

    if not aggregated:
        verdict = "FAIL"
        reason = "No aggregate result rows could be extracted."
        exit_code = 1
    elif len(datasets_found) < 2:
        verdict = "WARN"
        reason = "Results were found for fewer than two target datasets."
        exit_code = 2
    elif not matched_methods_all_three:
        verdict = "WARN"
        reason = (
            "Results were collected, but no method currently has matched "
            "entries on all three target datasets."
        )
        exit_code = 2
    else:
        verdict = "PASS"
        reason = (
            "At least one method has matched entries across all three datasets."
        )
        exit_code = 0

    report["overall_verdict"] = verdict
    report["reason"] = reason

    (output / "matched_results.json").write_text(
        json.dumps(aggregated, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "collection_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "MATCHED CROSS-DATASET RESULT COLLECTION",
        "=" * 92,
        f"JSON files scanned: {len(inventory)}",
        f"Matched source files: {report['matched_source_files']}",
        f"Aggregated rows: {len(aggregated)}",
        f"Datasets found: {', '.join(datasets_found) or 'none'}",
        f"Methods found: {', '.join(methods_found) or 'none'}",
        "",
        "COVERAGE",
        "-" * 92,
    ]

    for dataset, methods in coverage.items():
        lines.append(
            f"{dataset}: {', '.join(methods) if methods else 'none'}"
        )

    lines.extend([
        "",
        "METHODS PRESENT ON ALL THREE DATASETS",
        "-" * 92,
        ", ".join(matched_methods_all_three) or "none",
        "",
        f"LONG CSV: {output / 'matched_results_long.csv'}",
        f"WIDE CSV: {output / 'matched_results_wide.csv'}",
        f"INVENTORY: {output / 'source_inventory.csv'}",
        f"UNMATCHED: {output / 'unmatched_json_files.csv'}",
        f"REASON: {reason}",
        f"OVERALL VERDICT: {verdict}",
    ])

    (output / "collection_report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
