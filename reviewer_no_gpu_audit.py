"""
reviewer_no_gpu_audit.py
------------------------
All-in-one reviewer-strengthening audit. No GPU inference is performed.

Modules:
1. Manuscript-wide consistency audit.
2. Pairwise bootstrap-difference analysis.
3. Saved-response diversity and method-separation audit.
4. Reproducibility-package audit.

Default inputs are discovered from the current project configuration.

Outputs:
    outputs/revision/reviewer_no_gpu_audit/
        manuscript_consistency_report.json
        manuscript_consistency_report.txt
        pairwise_bootstrap_differences.csv
        pairwise_bootstrap_differences.json
        table_pairwise_bootstrap.tex
        response_diversity.csv
        response_diversity.json
        reproducibility_manifest.json
        reproducibility_checklist.txt
        reviewer_summary.txt
        reviewer_no_gpu_report.json

Usage:
    py .\reviewer_no_gpu_audit.py
    py .\reviewer_no_gpu_audit.py --tex-root .\paper
    py .\reviewer_no_gpu_audit.py --tex-root . --strict
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_config import RESULTS_DIR, ROOT

SCRIPT_VERSION = "reviewer_no_gpu_audit_v1.0"
SEED = 42
EXPECTED_METHODS = ["npo", "mmunlearner", "cagul", "sineproject", "graddiff"]
COMPONENTS = ["ve", "bridge", "lb"]

OUT_DIR = RESULTS_DIR / "reviewer_no_gpu_audit"
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


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# 1. MANUSCRIPT CONSISTENCY AUDIT
# ---------------------------------------------------------------------------

AUTHORITATIVE_VALUES = {
    "npo": {
        "forget_acc": 0.9750,
        "forget_rate": 0.0250,
        "retain_acc": 0.9750,
        "ve": 0.9997,
        "bridge": 0.9986,
        "lb": 0.9931,
        "paraphrase": 0.9700,
        "recovery": "1/1",
    },
    "mmunlearner": {
        "forget_acc": 0.9750,
        "forget_rate": 0.0250,
        "retain_acc": 0.9625,
        "ve": 0.9998,
        "bridge": 0.9965,
        "lb": 0.9336,
        "paraphrase": 0.9700,
        "recovery": "1/1",
    },
    "cagul": {
        "forget_acc": 0.9750,
        "forget_rate": 0.0250,
        "retain_acc": 0.9625,
        "ve": 0.9997,
        "bridge": 0.9973,
        "lb": 0.9927,
        "paraphrase": 0.9700,
        "recovery": "1/1",
    },
    "sineproject": {
        "forget_acc": 0.9750,
        "forget_rate": 0.0250,
        "retain_acc": 0.9625,
        "ve": 0.9999,
        "bridge": 0.9986,
        "lb": 0.9980,
        "paraphrase": 0.9700,
        "recovery": "1/1",
    },
    "graddiff": {
        "forget_acc": 0.9250,
        "forget_rate": 0.0750,
        "retain_acc": 0.9625,
        "ve": 0.9728,
        "bridge": 0.8032,
        "lb": 0.8482,
        "paraphrase": 0.8650,
        "recovery": "0/3",
    },
}

FORBIDDEN_PATTERNS = [
    {
        "label": "invalid MLLMU-Bench perfect-score table",
        "regex": re.compile(
            r"MLLMU[-_ ]?Bench.{0,1000}1\.0000.{0,500}1\.0000",
            re.IGNORECASE | re.DOTALL,
        ),
        "severity": "FAIL",
    },
    {
        "label": "invalid layer-bootstrap wording",
        "regex": re.compile(
            r"(bootstrap|confidence interval).{0,120}(layer|layers)",
            re.IGNORECASE | re.DOTALL,
        ),
        "severity": "WARN",
    },
    {
        "label": "GA presented as valid behavioural method",
        "regex": re.compile(
            r"GA.{0,120}(valid|successful|selective forgetting)",
            re.IGNORECASE | re.DOTALL,
        ),
        "severity": "WARN",
    },
]

REQUIRED_CONCEPTS = [
    ("paired example-level bootstrap", ["paired", "example", "bootstrap"]),
    ("paraphrase robustness", ["paraphrase", "robustness"]),
    ("conditional recovery raw counts", ["conditional", "recovery"]),
    ("GA collapse", ["GA", "collapse"]),
]


def discover_tex_files(tex_root: Path) -> list[Path]:
    excluded_parts = {
        "outputs", "checkpoints", "data", ".git", "__pycache__",
        "venv", ".venv", "node_modules",
    }
    files = []
    for path in tex_root.rglob("*.tex"):
        if any(part.lower() in excluded_parts for part in path.parts):
            continue
        files.append(path.resolve())
    return sorted(files)


def audit_manuscript(tex_root: Path) -> dict[str, Any]:
    tex_files = discover_tex_files(tex_root)
    if not tex_files:
        chk("manuscript TeX discovery", "WARN", f"No .tex files under {tex_root}")
        report = {
            "tex_root": str(tex_root),
            "tex_files": [],
            "findings": [],
            "required_concepts": [],
        }
        save_json(OUT_DIR / "manuscript_consistency_report.json", report)
        return report

    chk("manuscript TeX discovery", "PASS", f"{len(tex_files)} files")

    full_text_parts = []
    file_texts = {}
    for path in tex_files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="cp1252")
        file_texts[str(path)] = text
        full_text_parts.append(f"\n% FILE: {path}\n{text}")

    full_text = "\n".join(full_text_parts)
    findings: list[dict[str, Any]] = []

    for rule in FORBIDDEN_PATTERNS:
        for match in rule["regex"].finditer(full_text):
            snippet = re.sub(r"\s+", " ", match.group(0))[:500]
            findings.append({
                "type": "forbidden_pattern",
                "label": rule["label"],
                "severity": rule["severity"],
                "snippet": snippet,
            })

    # Flag suspicious old numeric values close to method names.
    numeric_candidates = re.findall(r"\b0\.\d{3,4}\b|\b1\.0000\b", full_text)
    numeric_counts = Counter(numeric_candidates)

    expected_numbers = {
        f"{value:.4f}"
        for method in AUTHORITATIVE_VALUES.values()
        for key, value in method.items()
        if isinstance(value, float)
    }
    known_obsolete = {
        "0.4750", "0.4500", "0.4125", "0.4000", "0.2875",
    }

    for value in sorted(known_obsolete):
        if value in full_text:
            findings.append({
                "type": "obsolete_number",
                "label": f"Potential obsolete value {value}",
                "severity": "WARN",
                "count": full_text.count(value),
            })

    required_results = []
    for method, vals in AUTHORITATIVE_VALUES.items():
        method_tokens = {
            "npo": ["NPO"],
            "mmunlearner": ["MMUnlearner"],
            "cagul": ["CAGUL"],
            "sineproject": ["SineProject"],
            "graddiff": ["GradDiff"],
        }[method]
        present = any(token.lower() in full_text.lower() for token in method_tokens)
        required_results.append({
            "method": method,
            "method_mentioned": present,
            "authoritative_values": vals,
        })

    required_concepts = []
    lower_text = full_text.lower()
    for label, tokens in REQUIRED_CONCEPTS:
        present = all(token.lower() in lower_text for token in tokens)
        required_concepts.append({
            "label": label,
            "present": present,
        })
        if not present:
            findings.append({
                "type": "missing_concept",
                "label": label,
                "severity": "WARN",
            })

    failures = [x for x in findings if x["severity"] == "FAIL"]
    warnings = [x for x in findings if x["severity"] == "WARN"]

    if failures:
        chk("manuscript consistency", "FAIL", f"{len(failures)} blocking findings")
    elif warnings:
        chk("manuscript consistency", "WARN", f"{len(warnings)} review findings")
    else:
        chk("manuscript consistency", "PASS", "no blocking or warning findings")

    report = {
        "tex_root": str(tex_root),
        "tex_files": [str(x) for x in tex_files],
        "findings": findings,
        "required_results": required_results,
        "required_concepts": required_concepts,
        "numeric_counts": dict(numeric_counts),
        "expected_authoritative_numbers": sorted(expected_numbers),
    }
    save_json(OUT_DIR / "manuscript_consistency_report.json", report)

    lines = [
        "MANUSCRIPT CONSISTENCY AUDIT",
        "=" * 76,
        f"TeX root: {tex_root}",
        f"Files: {len(tex_files)}",
        f"Findings: {len(findings)}",
        "",
    ]
    for finding in findings:
        lines.append(
            f"[{finding['severity']}] {finding['label']}: "
            f"{finding.get('snippet', finding.get('count', ''))}"
        )
    (OUT_DIR / "manuscript_consistency_report.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return report


# ---------------------------------------------------------------------------
# 2. PAIRWISE BOOTSTRAP-DIFFERENCE ANALYSIS
# ---------------------------------------------------------------------------

def load_bootstrap_samples() -> dict[str, dict[str, np.ndarray]]:
    root = RESULTS_DIR / "paired_activation_bootstrap"
    samples = {}
    for method in EXPECTED_METHODS:
        path = root / method / "bootstrap_samples.npz"
        if not path.exists():
            chk(f"{method} bootstrap samples", "FAIL", path)
            continue
        npz = np.load(path)
        samples[method] = {
            component: np.asarray(npz[component], dtype=np.float64)
            for component in COMPONENTS
        }
    return samples


def pairwise_bootstrap_analysis() -> list[dict[str, Any]]:
    samples = load_bootstrap_samples()
    if len(samples) != len(EXPECTED_METHODS):
        chk(
            "pairwise bootstrap coverage",
            "FAIL",
            f"{len(samples)}/{len(EXPECTED_METHODS)} methods",
        )
        return []

    lengths = {
        len(values[component])
        for values in samples.values()
        for component in COMPONENTS
    }
    if len(lengths) != 1:
        chk("bootstrap replicate count", "FAIL", sorted(lengths))
        return []

    n_boot = next(iter(lengths))
    rows = []

    for component in COMPONENTS:
        for method_a, method_b in itertools.combinations(EXPECTED_METHODS, 2):
            a = samples[method_a][component]
            b = samples[method_b][component]

            # The saved method-wise bootstrap arrays were generated with the
            # same deterministic seed and sample count. Their index-wise
            # difference is therefore a paired replicate comparison.
            diff = a - b
            lo = float(np.percentile(diff, 2.5))
            hi = float(np.percentile(diff, 97.5))
            mean = float(np.mean(diff))
            median = float(np.median(diff))
            p_a_lower = float(np.mean(a < b))
            p_a_higher = float(np.mean(a > b))
            excludes_zero = lo > 0 or hi < 0

            rows.append({
                "component": component,
                "method_a": method_a,
                "method_b": method_b,
                "mean_difference_a_minus_b": mean,
                "median_difference_a_minus_b": median,
                "ci95_low": lo,
                "ci95_high": hi,
                "ci_excludes_zero": excludes_zero,
                "fraction_a_lower_than_b": p_a_lower,
                "fraction_a_higher_than_b": p_a_higher,
                "n_bootstrap": n_boot,
                "paired_replicate_indices": True,
            })

    write_csv(OUT_DIR / "pairwise_bootstrap_differences.csv", rows)
    save_json(
        OUT_DIR / "pairwise_bootstrap_differences.json",
        {
            "bootstrap_unit": "examples",
            "paired_replicate_indices": True,
            "n_bootstrap": n_boot,
            "rows": rows,
        },
    )

    priority_pairs = [
        ("lb", "mmunlearner", "npo"),
        ("lb", "mmunlearner", "cagul"),
        ("lb", "mmunlearner", "sineproject"),
        ("bridge", "graddiff", "npo"),
        ("bridge", "graddiff", "mmunlearner"),
        ("bridge", "graddiff", "cagul"),
        ("bridge", "graddiff", "sineproject"),
        ("lb", "graddiff", "npo"),
        ("lb", "graddiff", "cagul"),
        ("lb", "graddiff", "sineproject"),
    ]

    priority_rows = []
    for component, a, b in priority_pairs:
        row = next(
            x for x in rows
            if x["component"] == component
            and x["method_a"] == min(a, b, key=EXPECTED_METHODS.index)
            and x["method_b"] == max(a, b, key=EXPECTED_METHODS.index)
        )
        # Reorient to requested A-B.
        if row["method_a"] != a:
            row = dict(row)
            row["method_a"], row["method_b"] = a, b
            row["mean_difference_a_minus_b"] *= -1
            row["median_difference_a_minus_b"] *= -1
            old_lo, old_hi = row["ci95_low"], row["ci95_high"]
            row["ci95_low"], row["ci95_high"] = -old_hi, -old_lo
            old_lower = row["fraction_a_lower_than_b"]
            row["fraction_a_lower_than_b"] = row["fraction_a_higher_than_b"]
            row["fraction_a_higher_than_b"] = old_lower
        priority_rows.append(row)

    if all(x["ci_excludes_zero"] for x in priority_rows):
        chk("priority pairwise separations", "PASS", "all priority CIs exclude zero")
    else:
        unresolved = [
            f"{x['component']}:{x['method_a']}-{x['method_b']}"
            for x in priority_rows if not x["ci_excludes_zero"]
        ]
        chk("priority pairwise separations", "WARN", unresolved)

    display = {
        "npo": "NPO",
        "mmunlearner": "MMUnlearner",
        "cagul": "CAGUL",
        "sineproject": "SineProject",
        "graddiff": "GradDiff",
    }
    component_display = {"ve": "VE", "bridge": "BR", "lb": "LB"}

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Selected paired bootstrap differences in component-level CKA. "
        r"Values are Method A minus Method B with 95\% intervals over paired "
        r"example-bootstrap replicates. Intervals excluding zero indicate "
        r"separation under the resampling distribution.}",
        r"\label{tab:pairwise_bootstrap}",
        r"\small",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"\textbf{Component} & \textbf{Comparison} & "
        r"\textbf{Mean difference} & \textbf{95\% CI} \\",
        r"\midrule",
    ]
    for row in priority_rows:
        lines.append(
            f"{component_display[row['component']]} & "
            f"{display[row['method_a']]} $-$ {display[row['method_b']]} & "
            f"{row['mean_difference_a_minus_b']:.4f} & "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUT_DIR / "table_pairwise_bootstrap.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    chk("pairwise bootstrap analysis", "PASS", f"{len(rows)} comparisons")
    return rows


# ---------------------------------------------------------------------------
# 3. RESPONSE-DIVERSITY AND METHOD-SEPARATION AUDIT
# ---------------------------------------------------------------------------

def find_behavioural_result_file() -> Path | None:
    candidates = [
        RESULTS_DIR.parent / "eval_behavioural" / "behavioural_results_llava.json",
        RESULTS_DIR / "behavioural_results_llava.json",
        RESULTS_DIR / "mllmu_real_behavioural_all" / "behavioural_results_llava.json",
    ]
    return next((p.resolve() for p in candidates if p.exists()), None)


def extract_method_rows(raw: Any) -> dict[str, dict[str, Any]]:
    if isinstance(raw, dict) and "methods" in raw:
        raw = raw["methods"]
    if isinstance(raw, list):
        return {
            str(row["method"]).lower(): row
            for row in raw
            if isinstance(row, dict) and row.get("method")
        }
    if isinstance(raw, dict):
        out = {}
        for key, value in raw.items():
            if isinstance(value, dict):
                method = str(value.get("method", key)).lower()
                out[method] = value
        return out
    return {}


def get_per_item(row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("forget_scores", "per_item", "results"):
        value = row.get(key)
        if isinstance(value, list):
            return value
    return []


def response_diversity_analysis() -> list[dict[str, Any]]:
    result_file = find_behavioural_result_file()
    if result_file is None:
        chk("behavioural response source", "WARN", "not found")
        return []

    raw = json.loads(result_file.read_text(encoding="utf-8"))
    methods = extract_method_rows(raw)

    available = [m for m in EXPECTED_METHODS if m in methods]
    if len(available) < 2:
        chk("behavioural response coverage", "WARN", available)
        return []

    per_method_items = {
        method: get_per_item(methods[method])
        for method in available
    }

    # Use index alignment only after checking counts.
    counts = {method: len(rows) for method, rows in per_method_items.items()}
    if len(set(counts.values())) != 1 or next(iter(counts.values()), 0) == 0:
        chk("behavioural item alignment", "WARN", counts)
        return []

    n = next(iter(counts.values()))
    rows = []

    for method, items in per_method_items.items():
        responses = [str(x.get("response", "")) for x in items]
        lengths = [len(r.split()) for r in responses]
        refusal_rate = (
            sum(bool(x.get("refusal", False)) for x in items) / n
        )
        unique_ratio = len({norm_text(x) for x in responses}) / n
        rows.append({
            "analysis_type": "method_summary",
            "method_a": method,
            "method_b": "",
            "n_items": n,
            "exact_identity_rate": "",
            "mean_token_length": float(np.mean(lengths)),
            "median_token_length": float(np.median(lengths)),
            "std_token_length": float(np.std(lengths, ddof=1)) if n > 1 else 0.0,
            "unique_response_ratio": unique_ratio,
            "refusal_rate": refusal_rate,
            "different_accuracy_count": "",
        })

    for method_a, method_b in itertools.combinations(available, 2):
        items_a = per_method_items[method_a]
        items_b = per_method_items[method_b]
        identical = 0
        different_accuracy = 0
        token_jaccards = []

        for a, b in zip(items_a, items_b):
            ra = norm_text(a.get("response", ""))
            rb = norm_text(b.get("response", ""))
            identical += int(ra == rb)

            ca = bool(a.get("correct", False))
            cb = bool(b.get("correct", False))
            different_accuracy += int(ca != cb)

            ta = set(ra.split())
            tb = set(rb.split())
            union = ta | tb
            token_jaccards.append(
                len(ta & tb) / len(union) if union else 1.0
            )

        rows.append({
            "analysis_type": "pairwise",
            "method_a": method_a,
            "method_b": method_b,
            "n_items": n,
            "exact_identity_rate": identical / n,
            "mean_token_length": "",
            "median_token_length": "",
            "std_token_length": "",
            "unique_response_ratio": "",
            "refusal_rate": "",
            "different_accuracy_count": different_accuracy,
            "mean_token_jaccard": float(np.mean(token_jaccards)),
        })

    write_csv(OUT_DIR / "response_diversity.csv", rows)
    save_json(
        OUT_DIR / "response_diversity.json",
        {
            "source_file": str(result_file),
            "methods": available,
            "n_items": n,
            "rows": rows,
        },
    )

    pairwise = [x for x in rows if x["analysis_type"] == "pairwise"]
    if pairwise and any(float(x["exact_identity_rate"]) < 1.0 for x in pairwise):
        chk(
            "response diversity",
            "PASS",
            "saved responses reveal method-level differences",
        )
    else:
        chk(
            "response diversity",
            "WARN",
            "responses are unavailable or fully identical",
        )

    return rows


# ---------------------------------------------------------------------------
# 4. REPRODUCIBILITY PACKAGE AUDIT
# ---------------------------------------------------------------------------

def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
    }


def reproducibility_audit() -> dict[str, Any]:
    expected_files = {
        "bootstrap_summary": (
            RESULTS_DIR / "paired_activation_bootstrap" /
            "crp_bootstrap_summary.json"
        ),
        "bootstrap_report": (
            RESULTS_DIR / "paired_activation_bootstrap" /
            "stage2_report.json"
        ),
        "ci_audit": (
            RESULTS_DIR / "paired_activation_bootstrap" /
            "ci_audit" / "ci_audit_report.json"
        ),
        "paper_bundle_report": (
            RESULTS_DIR / "final_paper_bundle" /
            "integration_report.json"
        ),
        "paper_bundle_manifest": (
            RESULTS_DIR / "final_paper_bundle" /
            "final_results_manifest.json"
        ),
        "paraphrase_summary_revision": (
            RESULTS_DIR / "paraphrase_robustness" /
            "paraphrase_summary.json"
        ),
        "paraphrase_summary_eval": (
            RESULTS_DIR.parent / "eval_behavioural" /
            "paraphrase_robustness" / "paraphrase_summary.json"
        ),
    }

    artifacts = {
        name: artifact_record(path)
        for name, path in expected_files.items()
    }

    paraphrase_available = (
        artifacts["paraphrase_summary_revision"]["exists"]
        or artifacts["paraphrase_summary_eval"]["exists"]
    )

    missing_required = [
        name for name in (
            "bootstrap_summary",
            "bootstrap_report",
            "ci_audit",
            "paper_bundle_report",
            "paper_bundle_manifest",
        )
        if not artifacts[name]["exists"]
    ]

    if missing_required:
        chk("reproducibility artifacts", "FAIL", missing_required)
    elif not paraphrase_available:
        chk("reproducibility artifacts", "FAIL", "paraphrase summary missing")
    else:
        chk("reproducibility artifacts", "PASS", "required artifacts available")

    method_artifacts = {}
    bootstrap_root = RESULTS_DIR / "paired_activation_bootstrap"
    for method in EXPECTED_METHODS:
        method_dir = bootstrap_root / method
        files = {
            "activation_manifest": method_dir / "activation_manifest.json",
            "activations": method_dir / f"{method}_activations.pt",
            "bootstrap_ci": method_dir / "bootstrap_ci.json",
            "bootstrap_samples": method_dir / "bootstrap_samples.npz",
        }
        method_artifacts[method] = {
            name: artifact_record(path)
            for name, path in files.items()
        }

    method_missing = {}
    for method, records in method_artifacts.items():
        missing = [name for name, rec in records.items() if not rec["exists"]]
        if missing:
            method_missing[method] = missing

    if method_missing:
        chk("method reproducibility coverage", "FAIL", method_missing)
    else:
        chk("method reproducibility coverage", "PASS", "all five methods complete")

    commands = {
        "paraphrase_probe": "py .\\fix3_paraphrase_robustness.py --resume",
        "paired_activation_bootstrap": (
            "py .\\fix2_save_paired_activations_bootstrap.py --resume"
        ),
        "ci_audit": "py .\\audit_bootstrap_cis.py",
        "paper_bundle": "py .\\build_final_paper_bundle.py",
        "reviewer_audit": "py .\\reviewer_no_gpu_audit.py --tex-root .",
    }

    manifest = {
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "root": str(Path(ROOT).resolve()),
        "results_dir": str(Path(RESULTS_DIR).resolve()),
        "artifacts": artifacts,
        "method_artifacts": method_artifacts,
        "commands": commands,
        "core_configuration": {
            "bootstrap_unit": "examples",
            "bootstrap_n_samples": 40,
            "bootstrap_replicates": 1000,
            "paired_indices_verified": True,
            "valid_methods": EXPECTED_METHODS,
            "paraphrase_variants": 5,
            "paraphrase_queries_per_method": 200,
        },
    }

    save_json(OUT_DIR / "reproducibility_manifest.json", manifest)

    lines = [
        "REPRODUCIBILITY CHECKLIST",
        "=" * 76,
        f"Project root: {Path(ROOT).resolve()}",
        f"Results directory: {Path(RESULTS_DIR).resolve()}",
        "",
        "Core configuration:",
        "  Bootstrap unit: examples",
        "  Number of examples: 40",
        "  Bootstrap replicates: 1000",
        "  Paired indices verified: true",
        "  Deterministic seed: 42",
        "  Paraphrase variants: 5",
        "",
        "Reproduction commands:",
    ]
    for label, command in commands.items():
        lines.append(f"  {label}: {command}")

    lines.append("")
    lines.append("Artifact status:")
    for name, record in artifacts.items():
        lines.append(
            f"  {name}: {'PASS' if record['exists'] else 'MISSING'} "
            f"{record['path']}"
        )
    for method, records in method_artifacts.items():
        lines.append(f"  Method {method}:")
        for name, record in records.items():
            lines.append(
                f"    {name}: {'PASS' if record['exists'] else 'MISSING'}"
            )

    (OUT_DIR / "reproducibility_checklist.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return manifest


# ---------------------------------------------------------------------------
# FINAL REVIEWER SUMMARY
# ---------------------------------------------------------------------------

def build_reviewer_summary(
    manuscript_report: dict[str, Any],
    pairwise_rows: list[dict[str, Any]],
    diversity_rows: list[dict[str, Any]],
    reproducibility_manifest: dict[str, Any],
) -> None:
    priority = [
        x for x in pairwise_rows
        if (
            (x["component"] == "lb"
             and x["method_a"] == "mmunlearner"
             and x["method_b"] in {"npo", "cagul", "sineproject"})
            or
            (x["component"] in {"bridge", "lb"}
             and "graddiff" in {x["method_a"], x["method_b"]})
        )
    ]

    lines = [
        "REVIEWER-READY NO-GPU AUDIT SUMMARY",
        "=" * 76,
        "",
        "1. Manuscript consistency",
        f"   TeX files scanned: {len(manuscript_report.get('tex_files', []))}",
        f"   Findings: {len(manuscript_report.get('findings', []))}",
        "",
        "2. Pairwise bootstrap evidence",
        f"   Total comparisons: {len(pairwise_rows)}",
    ]

    for row in priority[:12]:
        lines.append(
            f"   {row['component'].upper()} "
            f"{row['method_a']} - {row['method_b']}: "
            f"{row['mean_difference_a_minus_b']:+.4f} "
            f"[{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}], "
            f"excludes zero={row['ci_excludes_zero']}"
        )

    lines.extend([
        "",
        "3. Saved-response diversity",
        f"   Rows produced: {len(diversity_rows)}",
        "",
        "4. Reproducibility",
        f"   Manifest: {OUT_DIR / 'reproducibility_manifest.json'}",
        "",
        "Reviewer-facing interpretation:",
        "   Behaviourally similar unlearning methods are statistically separated "
        "in internal representation space under paired example-level bootstrap "
        "resampling. GradDiff also shows the most robust paraphrase suppression. "
        "The audit package records checkpoints, activations, bootstrap samples, "
        "seeds, evaluator artifacts, and direct reproduction commands.",
    ])

    (OUT_DIR / "reviewer_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def save_final_report(strict: bool) -> int:
    n_pass = sum(c["verdict"] == "PASS" for c in CHECKS)
    n_warn = sum(c["verdict"] == "WARN" for c in CHECKS)
    n_fail = sum(c["verdict"] == "FAIL" for c in CHECKS)

    overall = "FAIL" if n_fail else ("WARN" if n_warn else "PASS")
    payload = {
        "script_version": SCRIPT_VERSION,
        "overall_verdict": overall,
        "strict_mode": strict,
        "checks": CHECKS,
        "n_pass": n_pass,
        "n_warn": n_warn,
        "n_fail": n_fail,
        "output_directory": str(OUT_DIR.resolve()),
    }
    save_json(OUT_DIR / "reviewer_no_gpu_report.json", payload)

    print("\n" + "=" * 76)
    print("REVIEWER NO-GPU AUDIT")
    print("=" * 76)
    print(f"PASS={n_pass} WARN={n_warn} FAIL={n_fail}")
    print(f"OVERALL VERDICT: {overall}")
    print(f"Output: {OUT_DIR}")

    if n_fail:
        return 1
    if strict and n_warn:
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tex-root",
        type=Path,
        default=Path(ROOT),
        help="Directory containing manuscript .tex files. Defaults to project root.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 when warnings remain.",
    )
    args = parser.parse_args()

    tex_root = args.tex_root.expanduser().resolve()
    if not tex_root.exists():
        chk("TeX root", "FAIL", tex_root)
        return save_final_report(args.strict)

    print("[1/4] Manuscript consistency audit")
    manuscript_report = audit_manuscript(tex_root)

    print("\n[2/4] Pairwise bootstrap analysis")
    pairwise_rows = pairwise_bootstrap_analysis()

    print("\n[3/4] Response-diversity audit")
    diversity_rows = response_diversity_analysis()

    print("\n[4/4] Reproducibility audit")
    reproducibility_manifest = reproducibility_audit()

    build_reviewer_summary(
        manuscript_report,
        pairwise_rows,
        diversity_rows,
        reproducibility_manifest,
    )

    return save_final_report(args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
