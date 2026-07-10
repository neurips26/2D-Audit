#!/usr/bin/env python3
"""
Audit local FIUBench data and the official FIUBench code repository before
running another learned-base training job.

The audit is deliberately read-only. It does not modify dataset files,
checkpoints, or model weights.

Outputs:
  outputs/revision/fiubench_official_audit/
      fiubench_official_audit.json
      fiubench_official_audit.txt
      malformed_qa_examples.json
      protocol_evidence.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from exp_config import ROOT
except Exception:
    ROOT = SCRIPT_DIR

DATA_ROOT = ROOT / "data" / "fiubench_official"
SNAPSHOT = DATA_ROOT / "official_snapshot"
DEFAULT_CODE_DIRS = [
    ROOT / "external" / "FIUBench_official",
    ROOT / "external" / "VLM_Unlearned_official",
    DATA_ROOT / "official_code",
]
OUT_DIR = ROOT / "outputs" / "revision" / "fiubench_official_audit"

DATA_FILENAMES = [
    "full.json",
    "full_data.json",
    "test_full.json",
    "idontknow.json",
    "split.json",
]

TRAINING_HINTS = (
    "finetune", "train", "learning_rate", "batch_size", "per_device_train",
    "gradient_accumulation", "num_train_epochs", "lora", "projector",
    "freeze", "vision_tower", "mm_projector", "model_name_or_path",
    "deepspeed", "warmup", "weight_decay"
)

CHECKPOINT_SUFFIXES = {
    ".safetensors", ".bin", ".pt", ".pth", ".ckpt"
}


def load_json_or_jsonl(path: Path) -> tuple[Any, str]:
    raw = path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(raw), "json"
    except json.JSONDecodeError:
        rows = []
        for line_no, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSONL at line {line_no}: {exc}"
                ) from exc
        if not rows:
            raise ValueError(f"{path}: no JSON or JSONL records")
        return rows, "jsonl"


def stable_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def flatten_candidate_records(data: Any) -> tuple[list[dict[str, Any]], str]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)], "top_level_list"

    if isinstance(data, dict):
        # Prefer common dataset containers.
        for key in ("data", "records", "items", "examples", "dataset"):
            value = data.get(key)
            if isinstance(value, list) and value and all(
                isinstance(x, dict) for x in value
            ):
                return list(value), f"dict[{key}]"

        # A mapping from identity IDs to identity dictionaries.
        dict_values = [v for v in data.values() if isinstance(v, dict)]
        if dict_values and len(dict_values) == len(data):
            rows = []
            for key, value in data.items():
                row = dict(value)
                row.setdefault("_source_key", str(key))
                rows.append(row)
            return rows, "identity_mapping"

    return [], "not_record_like"


def extract_qas(record: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("qa_list", "qas", "qa", "questions"):
        value = record.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    if "question" in record and "answer" in record:
        return [record]
    return []


def record_id(record: dict[str, Any]) -> str:
    for key in ("unique", "id", "identity_id", "person_id", "_source_key", "name"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def canonical_name(record: dict[str, Any]) -> str:
    for key in ("name", "full_name", "person_name"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize(text: Any) -> str:
    value = str(text or "").casefold()
    value = value.replace("’", "'")
    value = re.sub(r"\*+", "", value)
    value = re.sub(r"[^a-z0-9' ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def is_name_question(question: str) -> bool:
    q = normalize(question)
    return any(term in q for term in (
        "full name", "what is the name", "who is", "person's name",
        "person name", "identity"
    ))


def classify_malformed(
    record: dict[str, Any],
    qa: dict[str, Any],
) -> list[str]:
    question = str(qa.get("question", "")).strip()
    answer = str(qa.get("answer", "")).strip()
    qn = normalize(question)
    an = normalize(answer)
    name = normalize(canonical_name(record))
    reasons: list[str] = []

    if not question:
        reasons.append("missing_question")
    if not answer:
        reasons.append("missing_answer")
    if answer and len(an) < 3:
        reasons.append("answer_too_short")
    if len(answer) > 300:
        reasons.append("answer_too_long")
    if an in {
        "the person in the image",
        "person in the image",
        "the person shown",
        "person shown",
    }:
        reasons.append("generic_person_placeholder")
    if (
        "the person in the image" in an
        and name
        and name not in an
        and is_name_question(question)
    ):
        reasons.append("name_answer_missing_canonical_name")
    if is_name_question(question) and name and name not in an:
        reasons.append("name_answer_conflicts_with_metadata")
    if qn and an and qn == an:
        reasons.append("answer_equals_question")
    if re.search(
        r"(the )?person in the image'?s? full name is (the )?person in the image",
        an,
    ):
        reasons.append("self_referential_name_answer")
    return sorted(set(reasons))


def summarize_data_file(path: Path) -> dict[str, Any]:
    data, fmt = load_json_or_jsonl(path)
    records, structure = flatten_candidate_records(data)

    summary: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": stable_sha256(path),
        "format": fmt,
        "top_level_type": type(data).__name__,
        "record_structure": structure,
        "record_count": len(records),
    }

    if not records:
        if isinstance(data, dict):
            summary["top_level_keys"] = list(data.keys())[:100]
            summary["top_level_key_count"] = len(data)
            summary["split_node_sizes"] = {
                str(k): len(v)
                for k, v in data.items()
                if isinstance(v, (list, dict))
            }
        return summary

    ids = [record_id(r) for r in records]
    names = [canonical_name(r) for r in records]
    qa_counts = [len(extract_qas(r)) for r in records]

    malformed_counter: Counter[str] = Counter()
    malformed_examples = []
    total_qas = 0
    name_qas = 0

    for record in records:
        for qa in extract_qas(record):
            total_qas += 1
            if is_name_question(str(qa.get("question", ""))):
                name_qas += 1
            reasons = classify_malformed(record, qa)
            if reasons:
                malformed_counter.update(reasons)
                if len(malformed_examples) < 50:
                    malformed_examples.append({
                        "record_id": record_id(record),
                        "name": canonical_name(record),
                        "question": qa.get("question"),
                        "answer": qa.get("answer"),
                        "reasons": reasons,
                    })

    summary.update({
        "nonempty_id_count": sum(bool(x) for x in ids),
        "unique_id_count": len({x for x in ids if x}),
        "duplicate_id_count": sum(
            count - 1 for count in Counter(x for x in ids if x).values()
            if count > 1
        ),
        "nonempty_name_count": sum(bool(x) for x in names),
        "unique_name_count": len({normalize(x) for x in names if x}),
        "duplicate_name_count": sum(
            count - 1 for count in Counter(normalize(x) for x in names if x).values()
            if count > 1
        ),
        "qa_total": total_qas,
        "name_qa_total": name_qas,
        "qa_per_record_min": min(qa_counts) if qa_counts else 0,
        "qa_per_record_max": max(qa_counts) if qa_counts else 0,
        "qa_per_record_mean": (
            sum(qa_counts) / len(qa_counts) if qa_counts else 0.0
        ),
        "malformed_qa_total_flagged": sum(malformed_counter.values()),
        "malformed_reason_counts": dict(malformed_counter),
        "malformed_examples": malformed_examples,
        "is_exact_400_identity_candidate": (
            len(records) == 400
            and len({x for x in ids if x}) == 400
        ),
        "is_exact_8000_qa_candidate": total_qas == 8000,
    })
    return summary


def scan_code_repository(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "git_repository": (path / ".git").exists(),
        "training_files": [],
        "config_evidence": [],
        "checkpoint_files": [],
        "checkpoint_links": [],
        "readme_mentions": [],
    }
    if not path.exists():
        return result

    text_suffixes = {
        ".py", ".sh", ".bash", ".yaml", ".yml", ".json", ".toml",
        ".md", ".txt", ".cfg"
    }

    for file in path.rglob("*"):
        if not file.is_file():
            continue
        rel = file.relative_to(path).as_posix()
        lower = rel.casefold()

        if file.suffix.casefold() in CHECKPOINT_SUFFIXES:
            result["checkpoint_files"].append({
                "file": rel,
                "bytes": file.stat().st_size,
            })
            continue

        if file.suffix.casefold() not in text_suffixes:
            continue
        if file.stat().st_size > 5_000_000:
            continue

        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        if any(x in lower for x in ("train", "finetune", "forget")):
            result["training_files"].append(rel)

        lines = text.splitlines()
        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()
            normalized_line = stripped.casefold()

            if any(hint in normalized_line for hint in TRAINING_HINTS):
                if len(result["config_evidence"]) < 400:
                    result["config_evidence"].append({
                        "file": rel,
                        "line": line_no,
                        "text": stripped[:500],
                    })

            if re.search(
                r"https?://(?:huggingface\.co|drive\.google\.com|github\.com)/\S+",
                stripped,
                flags=re.I,
            ):
                if any(term in normalized_line for term in (
                    "checkpoint", "model", "weight", "pretrain", "finetune"
                )):
                    if len(result["checkpoint_links"]) < 100:
                        result["checkpoint_links"].append({
                            "file": rel,
                            "line": line_no,
                            "text": stripped[:1000],
                        })

            if file.name.casefold().startswith("readme") and any(
                term in normalized_line for term in (
                    "400", "8000", "fine-tun", "learn", "checkpoint",
                    "batch", "learning rate", "epoch", "llava", "llama"
                )
            ):
                if len(result["readme_mentions"]) < 300:
                    result["readme_mentions"].append({
                        "file": rel,
                        "line": line_no,
                        "text": stripped[:1000],
                    })

    result["training_files"] = sorted(set(result["training_files"]))
    result["checkpoint_file_count"] = len(result["checkpoint_files"])
    result["training_file_count"] = len(result["training_files"])
    result["official_checkpoint_candidate_found"] = (
        result["checkpoint_file_count"] > 0
        or len(result["checkpoint_links"]) > 0
    )
    result["recipe_evidence_found"] = len(result["config_evidence"]) > 0
    return result


def choose_code_dir(explicit: str) -> Path | None:
    if explicit:
        p = Path(explicit).resolve()
        return p
    for candidate in DEFAULT_CODE_DIRS:
        if candidate.exists():
            return candidate
    return None


def txt_bool(value: bool) -> str:
    return "PASS" if value else "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot",
        default=str(SNAPSHOT),
        help="Path containing FIUBench JSON/JSONL files.",
    )
    parser.add_argument(
        "--code_dir",
        default="",
        help="Cloned official code repository path.",
    )
    args = parser.parse_args()

    snapshot = Path(args.snapshot).resolve()
    code_dir = choose_code_dir(args.code_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data_summaries = {}
    load_errors = {}
    malformed_examples = {}

    for filename in DATA_FILENAMES:
        path = snapshot / filename
        if not path.exists():
            data_summaries[filename] = {
                "path": str(path),
                "exists": False,
            }
            continue
        try:
            summary = summarize_data_file(path)
            data_summaries[filename] = summary
            malformed_examples[filename] = summary.pop("malformed_examples", [])
        except Exception as exc:
            load_errors[filename] = repr(exc)
            data_summaries[filename] = {
                "path": str(path),
                "exists": True,
                "load_error": repr(exc),
            }

    code_summary = (
        scan_code_repository(code_dir)
        if code_dir is not None
        else {
            "path": None,
            "exists": False,
            "git_repository": False,
            "training_files": [],
            "config_evidence": [],
            "checkpoint_files": [],
            "checkpoint_links": [],
            "readme_mentions": [],
            "checkpoint_file_count": 0,
            "training_file_count": 0,
            "official_checkpoint_candidate_found": False,
            "recipe_evidence_found": False,
        }
    )

    exact_400_files = [
        name for name, summary in data_summaries.items()
        if summary.get("is_exact_400_identity_candidate")
    ]
    exact_8000_files = [
        name for name, summary in data_summaries.items()
        if summary.get("is_exact_8000_qa_candidate")
    ]
    compatible_400_8000_files = sorted(
        set(exact_400_files) & set(exact_8000_files)
    )

    full_data_count = data_summaries.get("full_data.json", {}).get("record_count")
    discrepancy_573_detected = full_data_count == 573

    split_summary = data_summaries.get("split.json", {})
    split_file_loaded = split_summary.get("exists", False) and not split_summary.get("load_error")

    official_400_found = len(exact_400_files) > 0
    official_8000_found = len(exact_8000_files) > 0
    official_recipe_found = bool(code_summary.get("recipe_evidence_found"))
    official_checkpoint_found = bool(
        code_summary.get("official_checkpoint_candidate_found")
    )

    # SAFE_TO_TRAIN requires an exact local candidate and some official recipe
    # evidence. It does not require an official checkpoint.
    safe_to_train = (
        len(compatible_400_8000_files) > 0
        and official_recipe_found
        and not load_errors
    )

    if safe_to_train:
        recommendation = (
            "REPRODUCE_OFFICIAL_STAGE_I: an exact 400-identity/8000-QA local "
            "candidate and official recipe evidence were found."
        )
        verdict = "PASS"
    elif official_400_found and official_recipe_found:
        recommendation = (
            "REVIEW_REQUIRED: a 400-identity candidate and recipe evidence were "
            "found, but the QA count/protocol does not exactly resolve to 8000."
        )
        verdict = "WARN"
    else:
        recommendation = (
            "DO_NOT_TRAIN_YET: official Stage-I dataset and/or training recipe "
            "has not been unambiguously resolved."
        )
        verdict = "WARN"

    report = {
        "audit_name": "FIUBench official Stage-I protocol audit",
        "snapshot": str(snapshot),
        "code_repository": str(code_dir) if code_dir else None,
        "official_public_claims_used_as_audit_targets": {
            "identity_count": 400,
            "qa_count": 8000,
            "note": (
                "These are audit targets from the official paper/repository. "
                "This script verifies whether local files match them."
            ),
        },
        "data_files": data_summaries,
        "load_errors": load_errors,
        "malformed_examples_file": str(
            OUT_DIR / "malformed_qa_examples.json"
        ),
        "code": {
            key: value for key, value in code_summary.items()
            if key not in {
                "config_evidence", "checkpoint_links", "readme_mentions"
            }
        },
        "protocol_evidence_file": str(
            OUT_DIR / "protocol_evidence.json"
        ),
        "gates": {
            "OFFICIAL_400_SET_FOUND": official_400_found,
            "OFFICIAL_8000_QA_SET_FOUND": official_8000_found,
            "EXACT_400_AND_8000_SAME_FILE": bool(
                compatible_400_8000_files
            ),
            "OFFICIAL_SPLIT_FILE_LOADED": split_file_loaded,
            "573_RECORD_DISCREPANCY_CONFIRMED": discrepancy_573_detected,
            "OFFICIAL_RECIPE_FOUND": official_recipe_found,
            "OFFICIAL_CHECKPOINT_FOUND": official_checkpoint_found,
            "SAFE_TO_TRAIN": safe_to_train,
        },
        "candidates": {
            "exact_400_identity_files": exact_400_files,
            "exact_8000_qa_files": exact_8000_files,
            "exact_400_and_8000_files": compatible_400_8000_files,
        },
        "recommendation": recommendation,
        "overall_verdict": verdict,
    }

    protocol_evidence = {
        "code_repository": str(code_dir) if code_dir else None,
        "training_files": code_summary.get("training_files", []),
        "config_evidence": code_summary.get("config_evidence", []),
        "checkpoint_files": code_summary.get("checkpoint_files", []),
        "checkpoint_links": code_summary.get("checkpoint_links", []),
        "readme_mentions": code_summary.get("readme_mentions", []),
    }

    json_path = OUT_DIR / "fiubench_official_audit.json"
    txt_path = OUT_DIR / "fiubench_official_audit.txt"
    malformed_path = OUT_DIR / "malformed_qa_examples.json"
    evidence_path = OUT_DIR / "protocol_evidence.json"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    malformed_path.write_text(
        json.dumps(malformed_examples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    evidence_path.write_text(
        json.dumps(protocol_evidence, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "FIUBENCH OFFICIAL STAGE-I PROTOCOL AUDIT",
        "=" * 92,
        f"Snapshot: {snapshot}",
        f"Official code: {code_dir if code_dir else 'NOT FOUND'}",
        "",
        "LOCAL DATA SUMMARY",
        "-" * 92,
    ]
    for filename in DATA_FILENAMES:
        s = data_summaries.get(filename, {})
        if not s.get("exists"):
            lines.append(f"{filename}: MISSING")
            continue
        if s.get("load_error"):
            lines.append(f"{filename}: LOAD ERROR — {s['load_error']}")
            continue
        lines.append(
            f"{filename}: records={s.get('record_count', 'N/A')}, "
            f"unique_ids={s.get('unique_id_count', 'N/A')}, "
            f"qa={s.get('qa_total', 'N/A')}, "
            f"malformed_flags={s.get('malformed_qa_total_flagged', 'N/A')}, "
            f"format={s.get('format')}/{s.get('record_structure')}"
        )

    lines.extend([
        "",
        "AUTOMATED GATES",
        "-" * 92,
    ])
    for key, value in report["gates"].items():
        lines.append(f"{key}: {txt_bool(bool(value))}")

    lines.extend([
        "",
        "CANDIDATES",
        "-" * 92,
        f"Exact 400-identity files: {exact_400_files or 'NONE'}",
        f"Exact 8000-QA files: {exact_8000_files or 'NONE'}",
        f"Exact 400+8000 same file: {compatible_400_8000_files or 'NONE'}",
        "",
        "DECISION",
        "-" * 92,
        recommendation,
        f"REPORT: {txt_path}",
        f"JSON: {json_path}",
        f"MALFORMED QA: {malformed_path}",
        f"PROTOCOL EVIDENCE: {evidence_path}",
        f"OVERALL VERDICT: {verdict}",
    ])
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
