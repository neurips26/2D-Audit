#!/usr/bin/env python3
'''
AAAI reviewer-completion pipeline for the TwoDimAudit project.

Subcommands:
  preflight       Validate project inputs and saved activations.
  cpu             Run estimator-consistent bootstrap/BCa/LOEO/probe audits.
  sweep-crp       Compute CRP for discovered GradDiff and NPO sweep checkpoints.
  blip2-logprob   Score reference-answer likelihoods for existing BLIP-2 models.
  final           Consolidate all generated artifacts.

All new outputs are stored under:
  outputs/revision/reviewer_complete_pipeline/
'''

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

VERSION = "reviewer_complete_pipeline_v1.0"
SEED = 42
METHODS = ["npo", "mmunlearner", "cagul", "sineproject", "graddiff"]
DISPLAY = {
    "npo": "NPO",
    "mmunlearner": "MMUnlearner",
    "cagul": "CAGUL",
    "sineproject": "SineProject",
    "graddiff": "GradDiff",
}
PAIRWISE = [
    ("mmunlearner", "npo"),
    ("mmunlearner", "cagul"),
    ("mmunlearner", "sineproject"),
]
NORMAL = NormalDist()
CHECKS: list[dict[str, Any]] = []


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(label: str, verdict_value: str, detail: Any) -> None:
    CHECKS.append(
        {
            "check": label,
            "verdict": verdict_value,
            "detail": str(detail),
            "timestamp_utc": utc_now(),
        }
    )
    print(f"[{verdict_value}] {label}: {detail}")


def verdict() -> str:
    if any(row["verdict"] == "FAIL" for row in CHECKS):
        return "FAIL"
    if any(row["verdict"] == "WARN" for row in CHECKS):
        return "WARN"
    return "PASS"


def project_config():
    try:
        import exp_config
    except Exception as exc:
        raise RuntimeError(f"Could not import exp_config.py: {exc}") from exc

    results_dir = Path(
        getattr(exp_config, "RESULTS_DIR", ROOT / "outputs" / "revision")
    )
    if not results_dir.is_absolute():
        results_dir = (ROOT / results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    output = results_dir / "reviewer_complete_pipeline"
    output.mkdir(parents=True, exist_ok=True)
    return exp_config, results_dir, output


def resolve_path(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_torch_load(path: Path) -> Any:
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_text_report(path: Path, title: str) -> None:
    lines = [title, "=" * 88]
    lines.extend(
        f"[{row['verdict']}] {row['check']}: {row['detail']}" for row in CHECKS
    )
    lines.extend(["", f"OVERALL VERDICT: {verdict()}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Dataset and activation loading
# ---------------------------------------------------------------------------


def load_forget_items(exp_config: Any) -> list[dict[str, Any]]:
    split = resolve_path(getattr(exp_config, "MLLMU_REAL_FORGET", None))
    if split is None:
        try:
            import eval_config
            split = resolve_path(getattr(eval_config, "FORGET_DIR", None))
        except Exception:
            split = None
    if split is None or not split.exists():
        raise FileNotFoundError("Could not resolve the mllmu_real forget set.")

    try:
        import eval_utils
        items = list(eval_utils.load_mllmu_split(split))
        if items:
            return items
    except Exception:
        pass

    annotation = split / "annotations.json"
    if annotation.exists():
        raw = load_json(annotation)
        rows = raw if isinstance(raw, list) else [raw]
        result = []
        for row in rows:
            item = dict(row)
            image = Path(str(item.get("image", "")))
            if not image.is_absolute():
                image = split / image
            item["image"] = image.resolve()
            item["entity"] = str(item.get("entity", item.get("name", "unknown")))
            item["answer"] = str(item.get("answer", item.get("gt", "")))
            result.append(item)
        return result

    result: list[dict[str, Any]] = []
    for entity_dir in sorted(path for path in split.iterdir() if path.is_dir()):
        images = (
            list(entity_dir.glob("*.jpg"))
            + list(entity_dir.glob("*.jpeg"))
            + list(entity_dir.glob("*.png"))
        )
        jsons = list(entity_dir.glob("*.json"))
        if not images or not jsons:
            continue
        raw = load_json(jsons[0])
        rows = raw if isinstance(raw, list) else [raw]
        for row in rows:
            result.append(
                {
                    "entity": str(row.get("entity", row.get("name", entity_dir.name))),
                    "image": images[0].resolve(),
                    "question": str(row.get("question", "")),
                    "answer": str(row.get("answer", row.get("gt", ""))),
                    "aliases": row.get("aliases", []),
                }
            )
    return result


def find_activation_file(directory: Path, label: str) -> Path:
    candidates = [
        directory / f"{label}_activations.pt",
        directory / "paired_activations.pt",
        directory / "base_activations.pt",
        directory / "activations.pt",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    matches = sorted(directory.glob("*activations*.pt"))
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError(f"No activation file under {directory}")


def load_activation_bundle(directory: Path, label: str) -> dict[str, Any]:
    path = find_activation_file(directory, label)
    payload = safe_torch_load(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected activation payload: {path}")

    if isinstance(payload.get("activations"), dict):
        activations = payload["activations"]
        item_ids = list(payload.get("item_ids", []))
    else:
        activations = {
            key: value
            for key, value in payload.items()
            if hasattr(value, "shape") and getattr(value, "ndim", 0) >= 2
        }
        item_ids = list(payload.get("item_ids", [])) if isinstance(
            payload.get("item_ids"), list
        ) else []

    manifest = {}
    for candidate in (
        directory / "activation_manifest.json",
        directory / "base_manifest.json",
    ):
        if candidate.exists():
            manifest = load_json(candidate)
            break
    if not item_ids:
        item_ids = list(manifest.get("item_ids", []))
    if not activations:
        raise RuntimeError(f"No activation tensors found in {path}")

    return {
        "activations": activations,
        "item_ids": item_ids,
        "manifest": manifest,
        "file": path,
    }


def discover_activation_root(results_dir: Path) -> Path:
    candidates = [
        results_dir / "paired_activation_bootstrap",
        ROOT / "outputs" / "revision" / "paired_activation_bootstrap",
        ROOT / "results" / "paired_activation_bootstrap",
        ROOT / "outputs" / "paired_activation_bootstrap",
    ]
    for candidate in candidates:
        if (candidate / "base").exists():
            return candidate.resolve()

    matches = sorted(
        {path.parent.parent for path in ROOT.rglob("base_activations.pt")
         if path.parent.name == "base" and path.parent.parent.name == "paired_activation_bootstrap"},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0].resolve()

    raise FileNotFoundError(
        "Could not locate paired_activation_bootstrap. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def load_all_activations(results_dir: Path) -> dict[str, dict[str, Any]]:
    root = discover_activation_root(results_dir)
    bundles = {"base": load_activation_bundle(root / "base", "base")}
    for method in METHODS:
        bundles[method] = load_activation_bundle(root / method, method)

    n = int(next(iter(bundles["base"]["activations"].values())).shape[0])
    for label, bundle in bundles.items():
        for key, tensor in bundle["activations"].items():
            if int(tensor.shape[0]) != n:
                raise RuntimeError(
                    f"{label}/{key}: {tensor.shape[0]} rows; expected {n}"
                )
    return bundles


def component_keys(exp_config: Any, activations: dict[str, Any]):
    available = set(activations)
    ve_layers = list(getattr(exp_config, "LLAVA_VE_LAYERS", []))
    lb_layers = list(getattr(exp_config, "LLAVA_LB_LAYERS", []))

    ve = [f"ve_{index}" for index in ve_layers if f"ve_{index}" in available]
    lb = [f"lb_{index}" for index in lb_layers if f"lb_{index}" in available]
    if not ve:
        ve = sorted(
            [key for key in available if re.fullmatch(r"ve_\d+", key)],
            key=lambda value: int(value.split("_")[-1]),
        )
    if not lb:
        lb = sorted(
            [key for key in available if re.fullmatch(r"lb_\d+", key)],
            key=lambda value: int(value.split("_")[-1]),
        )
    bridge = [key for key in ("bridge", "bridge_0", "bridge_1") if key in available]
    if not bridge:
        bridge = sorted(key for key in available if key.startswith("bridge"))
    if not lb:
        raise RuntimeError("No language-backbone activation keys found.")
    return {"ve": ve, "bridge": bridge, "lb": lb}


# ---------------------------------------------------------------------------
# CRP-layer estimator and intervals
# ---------------------------------------------------------------------------


def debiased_linear_cka(x: Any, y: Any) -> float:
    import torch

    x = x.detach().float().cpu()
    y = y.detach().float().cpu()
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if y.ndim > 2:
        y = y.reshape(y.shape[0], -1)
    if x.shape[0] != y.shape[0] or x.shape[0] < 4:
        return float("nan")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        return float("nan")

    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    n = int(x.shape[0])
    k = x @ x.T
    l = y @ y.T
    k = k - torch.diag(torch.diag(k))
    l = l - torch.diag(torch.diag(l))
    ones = torch.ones((n, 1), dtype=k.dtype)

    def hsic(a, b):
        first = torch.trace(a @ b)
        second = ((ones.T @ a @ ones) * (ones.T @ b @ ones)) / ((n - 1) * (n - 2))
        third = (2.0 / (n - 2)) * (ones.T @ a @ b @ ones)
        return (first + second.squeeze() - third.squeeze()) / (n * (n - 3))

    numerator = hsic(k, l)
    denominator = torch.sqrt(torch.clamp(hsic(k, k) * hsic(l, l), min=0))
    if not torch.isfinite(denominator) or float(denominator) <= 0:
        return float("nan")
    return float((numerator / denominator).item())


def crp_component(
    base: dict[str, Any],
    method: dict[str, Any],
    keys: Sequence[str],
    indices: Sequence[int] | np.ndarray | None = None,
) -> float:
    import torch

    index_tensor = None
    if indices is not None:
        index_tensor = torch.as_tensor(np.asarray(indices), dtype=torch.long)
    values = []
    for key in keys:
        if key not in base or key not in method:
            continue
        x = base[key]
        y = method[key]
        if index_tensor is not None:
            x = x[index_tensor]
            y = y[index_tensor]
        value = debiased_linear_cka(x, y)
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def percentile_ci(values: np.ndarray, alpha: float):
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def basic_ci(theta: float, values: np.ndarray, alpha: float):
    low, high = percentile_ci(values, alpha)
    return 2 * theta - high, 2 * theta - low


def bca_ci(theta: float, bootstrap_values: np.ndarray, jackknife_values: np.ndarray, alpha: float):
    boot = np.asarray(bootstrap_values, dtype=np.float64)
    jack = np.asarray(jackknife_values, dtype=np.float64)
    boot = boot[np.isfinite(boot)]
    jack = jack[np.isfinite(jack)]
    if len(boot) < 100 or len(jack) < 4:
        return None

    proportion = (np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / len(boot)
    proportion = min(max(float(proportion), 1e-6), 1 - 1e-6)
    z0 = NORMAL.inv_cdf(proportion)
    jack_mean = float(np.mean(jack))
    differences = jack_mean - jack
    denominator = 6 * float(np.sum(differences**2)) ** 1.5
    acceleration = float(np.sum(differences**3) / denominator) if denominator > 0 else 0.0

    adjusted = []
    for probability in (alpha / 2, 1 - alpha / 2):
        z_alpha = NORMAL.inv_cdf(probability)
        denom = 1 - acceleration * (z0 + z_alpha)
        if abs(denom) < 1e-12:
            return None
        p = NORMAL.cdf(z0 + (z0 + z_alpha) / denom)
        adjusted.append(min(max(p, 0), 1))
    return float(np.quantile(boot, adjusted[0])), float(np.quantile(boot, adjusted[1]))


def example_replicates(n: int, count: int):
    rng = np.random.default_rng(SEED)
    return [rng.integers(0, n, size=n) for _ in range(count)]


def cluster_replicates(labels: Sequence[str], count: int):
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[str(label)].append(index)
    entities = sorted(groups)
    rng = np.random.default_rng(SEED + 1)
    replicates = []
    for _ in range(count):
        selected = rng.choice(entities, size=len(entities), replace=True)
        replicates.append(np.asarray([i for entity in selected for i in groups[str(entity)]], dtype=np.int64))
    return replicates, entities


def calculate_intervals(bundles, keys, labels, n_bootstrap: int, alpha: float):
    base = bundles["base"]["activations"]
    n = int(next(iter(base.values())).shape[0])
    ex_reps = example_replicates(n, n_bootstrap)
    cl_reps, entities = cluster_replicates(labels, n_bootstrap)
    all_indices = np.arange(n)
    ex_jack = [np.delete(all_indices, i) for i in range(n)]
    labels_array = np.asarray(labels)
    cl_jack = [np.where(labels_array != entity)[0] for entity in entities]

    rows = []
    raw = {
        "version": VERSION,
        "estimator": "CRP-layer: CKA per hook, then arithmetic mean",
        "n_examples": n,
        "n_entities": len(entities),
        "n_bootstrap": n_bootstrap,
        "methods": {},
    }

    for method_name in METHODS:
        method = bundles[method_name]["activations"]
        raw["methods"][method_name] = {}
        for component, hook_keys in keys.items():
            if not hook_keys:
                continue
            theta = crp_component(base, method, hook_keys)
            ex_values = np.asarray([crp_component(base, method, hook_keys, idx) for idx in ex_reps], dtype=np.float64)
            cl_values = np.asarray([crp_component(base, method, hook_keys, idx) for idx in cl_reps], dtype=np.float64)
            ex_values = ex_values[np.isfinite(ex_values)]
            cl_values = cl_values[np.isfinite(cl_values)]
            ex_jack_values = np.asarray([crp_component(base, method, hook_keys, idx) for idx in ex_jack], dtype=np.float64)
            cl_jack_values = np.asarray([crp_component(base, method, hook_keys, idx) for idx in cl_jack], dtype=np.float64)

            ex_pct = percentile_ci(ex_values, alpha)
            cl_pct = percentile_ci(cl_values, alpha)
            ex_basic = basic_ci(theta, ex_values, alpha)
            cl_basic = basic_ci(theta, cl_values, alpha)
            ex_bca = bca_ci(theta, ex_values, ex_jack_values, alpha)
            cl_bca = bca_ci(theta, cl_values, cl_jack_values, alpha)

            row = {
                "method": method_name,
                "display_name": DISPLAY[method_name],
                "component": component,
                "point_estimate": theta,
                "example_percentile_low": ex_pct[0],
                "example_percentile_high": ex_pct[1],
                "example_basic_low": ex_basic[0],
                "example_basic_high": ex_basic[1],
                "example_bca_low": ex_bca[0] if ex_bca else "",
                "example_bca_high": ex_bca[1] if ex_bca else "",
                "cluster_percentile_low": cl_pct[0],
                "cluster_percentile_high": cl_pct[1],
                "cluster_basic_low": cl_basic[0],
                "cluster_basic_high": cl_basic[1],
                "cluster_bca_low": cl_bca[0] if cl_bca else "",
                "cluster_bca_high": cl_bca[1] if cl_bca else "",
                "example_width": ex_pct[1] - ex_pct[0],
                "cluster_width": cl_pct[1] - cl_pct[0],
                "cluster_minus_example_width": (cl_pct[1] - cl_pct[0]) - (ex_pct[1] - ex_pct[0]),
                "example_bias": float(np.mean(ex_values) - theta),
                "cluster_bias": float(np.mean(cl_values) - theta),
                "point_inside_example_percentile": bool(ex_pct[0] <= theta <= ex_pct[1]),
                "point_inside_cluster_percentile": bool(cl_pct[0] <= theta <= cl_pct[1]),
            }
            rows.append(row)
            raw["methods"][method_name][component] = {
                "summary": row,
                "example_bootstrap_values": ex_values.tolist(),
                "cluster_bootstrap_values": cl_values.tolist(),
                "example_jackknife_values": ex_jack_values.tolist(),
                "cluster_jackknife_values": cl_jack_values.tolist(),
            }
    return rows, raw


# ---------------------------------------------------------------------------
# Influence and probe
# ---------------------------------------------------------------------------


def loeo_analysis(bundles, keys, labels):
    base = bundles["base"]["activations"]
    labels_array = np.asarray(labels)
    entities = sorted(set(labels))
    rows = []
    for method_name in METHODS:
        method = bundles[method_name]["activations"]
        full = {component: crp_component(base, method, hook_keys) for component, hook_keys in keys.items() if hook_keys}
        for entity in entities:
            indices = np.where(labels_array != entity)[0]
            for component, hook_keys in keys.items():
                if not hook_keys:
                    continue
                excluded = crp_component(base, method, hook_keys, indices)
                rows.append({
                    "method": method_name,
                    "display_name": DISPLAY[method_name],
                    "excluded_entity": entity,
                    "component": component,
                    "full_value": full[component],
                    "excluded_value": excluded,
                    "change": excluded - full[component],
                    "absolute_change": abs(excluded - full[component]),
                })
    return rows


def excluded_pairwise(bundles, lb_keys, labels, entity: str, n_bootstrap: int, alpha: float):
    labels_array = np.asarray(labels)
    retained = np.where(labels_array != entity)[0]
    base = bundles["base"]["activations"]
    rng = np.random.default_rng(SEED + 17)
    local_reps = [rng.integers(0, len(retained), size=len(retained)) for _ in range(n_bootstrap)]
    rows = []
    for method_a, method_b in PAIRWISE:
        a = bundles[method_a]["activations"]
        b = bundles[method_b]["activations"]
        full_a = crp_component(base, a, lb_keys)
        full_b = crp_component(base, b, lb_keys)
        excluded_a = crp_component(base, a, lb_keys, retained)
        excluded_b = crp_component(base, b, lb_keys, retained)
        differences = []
        for local_indices in local_reps:
            indices = retained[local_indices]
            va = crp_component(base, a, lb_keys, indices)
            vb = crp_component(base, b, lb_keys, indices)
            if math.isfinite(va) and math.isfinite(vb):
                differences.append(va - vb)
        differences = np.asarray(differences, dtype=np.float64)
        low, high = percentile_ci(differences, alpha)
        rows.append({
            "method_a": method_a,
            "method_b": method_b,
            "excluded_entity": entity,
            "full_difference_a_minus_b": full_a - full_b,
            "excluded_difference_a_minus_b": excluded_a - excluded_b,
            "excluded_ci_low": low,
            "excluded_ci_high": high,
            "ci_excludes_zero": bool(low > 0 or high < 0),
            "n_examples_after_exclusion": len(retained),
        })
    return rows


def ridge_scores(train_x, train_y, test_x, ridge: float):
    classes = sorted(set(train_y.tolist()))
    class_index = {label: i for i, label in enumerate(classes)}
    one_hot = np.zeros((len(train_y), len(classes)), dtype=np.float64)
    for row, label in enumerate(train_y):
        one_hot[row, class_index[label]] = 1
    mean = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1
    train = (train_x - mean) / scale
    test = (test_x - mean) / scale
    kernel = train @ train.T
    coefficients = np.linalg.solve(kernel + ridge * np.eye(kernel.shape[0]), one_hot)
    return test @ train.T @ coefficients, classes


def linear_probe(bundles, labels, lb_key: str, ridge: float):
    labels_array = np.asarray(labels)
    entities = sorted(set(labels))
    groups = {entity: np.where(labels_array == entity)[0].tolist() for entity in entities}
    invalid = {entity: rows for entity, rows in groups.items() if len(rows) != 2}
    if invalid:
        raise RuntimeError(f"Two-fold probe needs two examples per entity: {invalid}")

    rows = []
    for name in ["base"] + METHODS:
        tensor = bundles[name]["activations"][lb_key]
        features = tensor.detach().float().cpu().numpy()
        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        fold_accuracy = []
        for direction in (0, 1):
            train_idx = np.asarray([groups[e][direction] for e in entities])
            test_idx = np.asarray([groups[e][1 - direction] for e in entities])
            train_labels = np.asarray(entities)
            scores, classes = ridge_scores(features[train_idx], train_labels, features[test_idx], ridge)
            predictions = np.asarray([classes[i] for i in scores.argmax(axis=1)])
            fold_accuracy.append(float(np.mean(predictions == np.asarray(entities))))
        rows.append({
            "method": name,
            "display_name": "Original model" if name == "base" else DISPLAY[name],
            "activation_key": lb_key,
            "fold_1_accuracy": fold_accuracy[0],
            "fold_2_accuracy": fold_accuracy[1],
            "mean_accuracy": float(np.mean(fold_accuracy)),
            "chance_accuracy": 1 / len(entities),
            "n_entities": len(entities),
            "ridge": ridge,
        })
    return rows


# ---------------------------------------------------------------------------
# Config and manuscript audits
# ---------------------------------------------------------------------------


def flatten_json(value: Any, prefix: str = ""):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_json(child, name))
    elif isinstance(value, list):
        result[prefix] = value
    else:
        result[prefix] = value
    return result


def first_matching(flat, patterns):
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        for key, value in flat.items():
            if regex.search(key):
                return value
    return ""


def hyperparameter_inventory(exp_config):
    adapters = dict(getattr(exp_config, "LLAVA_ADAPTERS", {}))
    rows = []
    for method in METHODS:
        checkpoint = resolve_path(adapters.get(method))
        flat = {}
        sources = []
        if checkpoint and checkpoint.exists():
            for filename in (
                "training_meta.json", "adapter_config.json", "trainer_state.json",
                "training_args.json", "run_config.json", "config.json",
            ):
                path = checkpoint / filename
                if path.exists():
                    try:
                        for key, value in flatten_json(load_json(path)).items():
                            flat[f"{filename}:{key}"] = value
                        sources.append(str(path.resolve()))
                    except Exception:
                        pass
        row = {
            "method": method,
            "display_name": DISPLAY[method],
            "checkpoint": str(checkpoint) if checkpoint else "",
            "checkpoint_exists": bool(checkpoint and checkpoint.exists()),
            "steps": first_matching(flat, [r"(^|\.)steps$", r"global_step", r"max_steps"]),
            "learning_rate": first_matching(flat, [r"learning_rate", r"(^|\.)lr$"]),
            "batch_size": first_matching(flat, [r"batch_size", r"per_device_train_batch_size"]),
            "gradient_accumulation": first_matching(flat, [r"gradient_accumulation"]),
            "lora_rank": first_matching(flat, [r"lora_rank", r"(^|\.)r$"]),
            "lora_alpha": first_matching(flat, [r"lora_alpha"]),
            "lora_dropout": first_matching(flat, [r"lora_dropout"]),
            "target_modules": first_matching(flat, [r"target_modules"]),
            "beta": first_matching(flat, [r"(^|\.)beta$"]),
            "lambda_retain": first_matching(flat, [r"lambda_retain", r"retain_weight", r"retain_lambda"]),
            "weight_decay": first_matching(flat, [r"weight_decay"]),
            "seed": first_matching(flat, [r"(^|\.)seed$"]),
            "base_model": first_matching(flat, [r"base_model", r"base_model_name_or_path"]),
            "metadata_sources": " | ".join(sources),
        }
        required = ["steps", "learning_rate", "lora_rank", "lora_alpha", "target_modules", "seed"]
        missing = [field for field in required if row[field] in ("", None, [])]
        row["missing_required_fields"] = ", ".join(missing)
        row["config_status"] = "PASS" if not missing else "WARN"
        rows.append(row)
    return rows


def manuscript_audit(tex_root: Path):
    excluded = {"outputs", "checkpoints", ".git", "venv", ".venv", "__pycache__"}
    files = sorted(
        path for path in tex_root.rglob("*.tex")
        if not any(part.lower() in excluded for part in path.parts)
    )
    text = ""
    for path in files:
        try:
            text += "\n" + path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text += "\n" + path.read_text(encoding="cp1252")
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    references = re.findall(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", text)
    undefined = sorted(set(references) - labels)
    dangling = []
    for pattern in (
        r"Appendix\s*[;,.]", r"Table\s*[;,.]", r"Figure\s*[;,.]",
        r"Appendix~\\ref\{\}", r"Table~\\ref\{\}", r"Figure~\\ref\{\}",
    ):
        if re.search(pattern, text):
            dangling.append(pattern)
    lower = text.lower()
    return {
        "tex_root": str(tex_root.resolve()),
        "tex_files": [str(path.resolve()) for path in files],
        "undefined_references": undefined,
        "dangling_patterns": dangling,
        "availability_statement_present": any(
            phrase in lower for phrase in (
                "code availability", "data availability", "will be released",
                "artifact availability", "code and data",
            )
        ),
        "audit_scope_statement_present": "not regulatory" in lower or "not compliance" in lower,
        "legal_evidence_statement_present": "legal evidence" in lower,
    }


def ci_text(low, high):
    return "--" if low == "" or high == "" else f"[{float(low):.4f},{float(high):.4f}]"


def cluster_latex(rows):
    lb_rows = [row for row in rows if row["component"] == "lb"]
    lines = [
        r"\begin{table*}[!t]", r"\centering",
        r"\caption{Authoritative CRP-layer bootstrap sensitivity for LB-CKA. Both resampling schemes recompute CKA at every sampled language layer and then average the layerwise scores. BCa intervals account for material bootstrap bias.}",
        r"\label{tab:cluster_bootstrap_authoritative}",
        r"\setlength{\tabcolsep}{3pt}", r"\scriptsize",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"\textbf{Method} & \textbf{LB-CKA} & \textbf{Example BCa} & \textbf{Cluster BCa} & \textbf{$\Delta$ width} & \textbf{Cluster bias} \\",
        r"\midrule",
    ]
    for row in lb_rows:
        lines.append(
            f"{row['display_name']} & {row['point_estimate']:.4f} & "
            f"{ci_text(row['example_bca_low'], row['example_bca_high'])} & "
            f"{ci_text(row['cluster_bca_low'], row['cluster_bca_high'])} & "
            f"{row['cluster_minus_example_width']:+.4f} & {row['cluster_bias']:+.4f} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def probe_latex(rows):
    lines = [
        r"\begin{table}[!t]", r"\centering",
        r"\caption{Two-fold ridge linear-probe accuracy for forget-entity identity from the final sampled LLaVA language layer. Each fold trains on one example per entity and tests on the other.}",
        r"\label{tab:linear_probe}", r"\setlength{\tabcolsep}{5pt}", r"\small",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"\textbf{Checkpoint} & \textbf{Probe Acc.} & \textbf{Chance} \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(f"{row['display_name']} & {row['mean_accuracy']:.4f} & {row['chance_accuracy']:.4f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def run_preflight(_):
    exp_config, results_dir, output = project_config()
    for path in (
        ROOT / "fix2_save_paired_activations_bootstrap.py",
        ROOT / "train_graddiff_llava.py",
        ROOT / "adapter_guard.py",
    ):
        check(path.name, "PASS" if path.exists() else "WARN", path)

    npo_candidates = [
        ROOT / "stage3_npo_smoke_complete_fixed.py",
        ROOT / "stage3_npo_smoke_authoritative_fixed.py",
        ROOT / "stage3_npo_smoke_final_fixed.py",
        ROOT / "stage3_npo_smoke.py",
    ]
    found_npo = next((path for path in npo_candidates if path.exists()), None)
    check("NPO sweep script", "PASS" if found_npo else "WARN", found_npo)

    try:
        activation_root = discover_activation_root(results_dir)
        bundles = load_all_activations(results_dir)
        n = int(next(iter(bundles["base"]["activations"].values())).shape[0])
        check(
            "paired activation coverage",
            "PASS",
            f"root={activation_root}; {len(bundles)-1} methods; n={n}",
        )
    except Exception as exc:
        check("paired activation coverage", "FAIL", f"{type(exc).__name__}: {exc}")

    adapters = dict(getattr(exp_config, "LLAVA_ADAPTERS", {}))
    for method in METHODS:
        path = resolve_path(adapters.get(method))
        check(f"{method} checkpoint", "PASS" if path and path.exists() else "WARN", path)

    save_json(output / "preflight_report.json", {
        "version": VERSION,
        "project_root": str(ROOT),
        "results_dir": str(results_dir),
        "checks": CHECKS,
        "overall_verdict": verdict(),
    })
    save_text_report(output / "preflight_report.txt", "AAAI REVIEWER PIPELINE PREFLIGHT")
    print(f"REPORT JSON: {output / 'preflight_report.json'}")
    print(f"REPORT TXT:  {output / 'preflight_report.txt'}")
    print(f"OVERALL VERDICT: {verdict()}")
    return 1 if verdict() == "FAIL" else 0


def run_cpu(args):
    exp_config, results_dir, output = project_config()
    cpu_dir = output / "cpu_analysis"
    cpu_dir.mkdir(parents=True, exist_ok=True)

    bundles = load_all_activations(results_dir)
    keys = component_keys(exp_config, bundles["base"]["activations"])
    items = load_forget_items(exp_config)
    n = int(next(iter(bundles["base"]["activations"].values())).shape[0])
    if len(items) != n:
        raise RuntimeError(f"Dataset/activation mismatch: {len(items)} vs {n}")

    labels = [str(item.get("entity", item.get("name", f"entity_{i//2:02d}"))) for i, item in enumerate(items)]
    counts = Counter(labels)
    check("entity grouping", "PASS" if len(counts) == 20 and set(counts.values()) == {2} else "WARN", dict(counts))

    interval_rows, interval_raw = calculate_intervals(bundles, keys, labels, args.n_bootstrap, args.alpha)
    write_csv(cpu_dir / "authoritative_bootstrap_intervals.csv", interval_rows)
    save_json(cpu_dir / "authoritative_bootstrap_intervals.json", interval_raw)
    (cpu_dir / "table_cluster_bootstrap_authoritative.tex").write_text(cluster_latex(interval_rows), encoding="utf-8")

    uncovered = [row for row in interval_rows if not row["point_inside_example_percentile"] or not row["point_inside_cluster_percentile"]]
    check("percentile interval self-coverage", "WARN" if uncovered else "PASS", f"{len(uncovered)} affected rows; BCa/basic saved" if uncovered else "all covered")

    loeo_rows = loeo_analysis(bundles, keys, labels)
    write_csv(cpu_dir / "leave_one_entity_out.csv", loeo_rows)
    save_json(cpu_dir / "leave_one_entity_out.json", loeo_rows)
    mm_lb = [row for row in loeo_rows if row["method"] == "mmunlearner" and row["component"] == "lb"]
    influential = max(mm_lb, key=lambda row: row["absolute_change"])
    save_json(cpu_dir / "mmunlearner_most_influential_entity.json", influential)
    check("MMUnlearner influential entity", "PASS", f"{influential['excluded_entity']} ({influential['change']:+.6f})")

    pairwise_rows = excluded_pairwise(bundles, keys["lb"], labels, influential["excluded_entity"], args.n_bootstrap, args.alpha)
    write_csv(cpu_dir / "influential_entity_excluded_pairwise.csv", pairwise_rows)
    save_json(cpu_dir / "influential_entity_excluded_pairwise.json", pairwise_rows)
    separated = sum(row["ci_excludes_zero"] for row in pairwise_rows)
    check("pairwise separation after exclusion", "PASS" if separated == len(pairwise_rows) else "WARN", f"{separated}/{len(pairwise_rows)} intervals exclude zero")

    probe_rows = linear_probe(bundles, labels, keys["lb"][-1], args.probe_ridge)
    write_csv(cpu_dir / "linear_probe_l31.csv", probe_rows)
    save_json(cpu_dir / "linear_probe_l31.json", probe_rows)
    (cpu_dir / "table_linear_probe.tex").write_text(probe_latex(probe_rows), encoding="utf-8")
    check("linear probe", "PASS", f"{len(probe_rows)} checkpoints at {keys['lb'][-1]}")

    config_rows = hyperparameter_inventory(exp_config)
    write_csv(cpu_dir / "hyperparameter_inventory.csv", config_rows)
    save_json(cpu_dir / "hyperparameter_inventory.json", config_rows)
    incomplete = [row["method"] for row in config_rows if row["config_status"] != "PASS"]
    check("hyperparameter inventory", "WARN" if incomplete else "PASS", f"incomplete: {incomplete}" if incomplete else "complete")

    manuscript = manuscript_audit(Path(args.tex_root).resolve())
    save_json(cpu_dir / "manuscript_audit.json", manuscript)
    reference_problems = manuscript["undefined_references"] or manuscript["dangling_patterns"]
    check("manuscript references", "FAIL" if reference_problems else "PASS", reference_problems if reference_problems else "clean")
    check("availability statement", "PASS" if manuscript["availability_statement_present"] else "WARN", manuscript["availability_statement_present"])
    check("audit/legal scope statements", "PASS" if manuscript["audit_scope_statement_present"] and manuscript["legal_evidence_statement_present"] else "WARN", {
        "audit_scope": manuscript["audit_scope_statement_present"],
        "legal_evidence": manuscript["legal_evidence_statement_present"],
    })

    artifacts = {}
    for path in cpu_dir.iterdir():
        if path.is_file():
            artifacts[path.name] = {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    save_json(cpu_dir / "cpu_analysis_report.json", {
        "version": VERSION,
        "created_utc": utc_now(),
        "estimator": "CRP-layer",
        "component_keys": keys,
        "influential_entity": influential,
        "checks": CHECKS,
        "artifacts": artifacts,
        "overall_verdict": verdict(),
    })
    save_text_report(cpu_dir / "cpu_analysis_report.txt", "AAAI REVIEWER COMPLETE CPU ANALYSIS")
    print(f"Output: {cpu_dir}")
    print(f"OVERALL VERDICT: {verdict()}")
    return 1 if verdict() == "FAIL" else 0


# ---------------------------------------------------------------------------
# Sweep CRP stage
# ---------------------------------------------------------------------------


def recognised_checkpoint_files(path: Path) -> list[Path]:
    """Return checkpoint files recognised by the validated extraction code."""
    names = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
    )
    return [path / name for name in names if (path / name).is_file()]


def checkpoint_is_usable(path: Path) -> tuple[bool, str]:
    """Reject empty/interrupted checkpoint folders before loading a model."""
    files = recognised_checkpoint_files(path)
    if not files:
        return False, "no recognised checkpoint files"

    adapter_cfg = path / "adapter_config.json"
    if adapter_cfg.exists():
        weights = [
            path / "adapter_model.safetensors",
            path / "adapter_model.bin",
        ]
        if not any(weight.exists() and weight.stat().st_size > 0 for weight in weights):
            return False, "adapter_config.json exists but adapter weights are missing/empty"
    return True, ", ".join(file.name for file in files)


def discover_sweep_checkpoints():
    rows = []
    skipped = []

    grad_root = ROOT / "checkpoints" / "graddiff"
    if grad_root.exists():
        for path in sorted(grad_root.rglob("graddiff_llava_*steps")):
            if not path.is_dir():
                continue
            usable, detail = checkpoint_is_usable(path)
            row = {
                "method": "graddiff",
                "label": path.relative_to(grad_root).as_posix(),
                "checkpoint": path.resolve(),
                "checkpoint_detail": detail,
            }
            (rows if usable else skipped).append(row)

    npo_root = ROOT / "checkpoints" / "npo_sweep"
    if npo_root.exists():
        for path in sorted(p for p in npo_root.iterdir() if p.is_dir()):
            usable, detail = checkpoint_is_usable(path)
            row = {
                "method": "npo",
                "label": path.name,
                "checkpoint": path.resolve(),
                "checkpoint_detail": detail,
            }
            (rows if usable else skipped).append(row)

    unique = {str(row["checkpoint"]): row for row in rows}
    return list(unique.values()), skipped


def isolated_fix2(
    script: Path,
    isolated_results: Path,
    method: str,
    checkpoint: Path,
    log_path: Path,
):
    """
    Run the validated activation script in a child interpreter while capturing
    both stdout and stderr. Capturing prevents Windows PowerShell from treating
    child stderr as a terminating NativeCommandError.
    """
    wrapper = (
        "import runpy, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, r'{ROOT}')\n"
        f"sys.path.insert(0, r'{script.parent}')\n"
        "import exp_config\n"
        f"exp_config.RESULTS_DIR = Path(r'{isolated_results}')\n"
        "exp_config.LLAVA_ADAPTERS = dict(exp_config.LLAVA_ADAPTERS)\n"
        f"exp_config.LLAVA_ADAPTERS['{method}'] = Path(r'{checkpoint}')\n"
        f"sys.argv = [r'{script}', '--methods', '{method}', '--resume']\n"
        f"runpy.run_path(r'{script}', run_name='__main__')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", wrapper],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    combined = (
        "COMMAND CHECKPOINT: " + str(checkpoint) + "\n"
        + "RETURN CODE: " + str(completed.returncode) + "\n\n"
        + "STDOUT\n" + (completed.stdout or "")
        + "\nSTDERR\n" + (completed.stderr or "")
    )
    log_path.write_text(combined, encoding="utf-8")

    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print("[child stderr captured]")
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n")
    return int(completed.returncode), combined


def run_sweep_crp(args):
    _, _, output = project_config()
    fix2 = ROOT / "fix2_save_paired_activations_bootstrap.py"
    if not fix2.exists():
        check("fix2 script", "FAIL", fix2)
        print("OVERALL VERDICT: FAIL")
        return 1

    sweep_root = output / "sweep_crp"
    work_results = sweep_root / "work_results"
    point_root = sweep_root / "points"
    point_log_root = sweep_root / "point_logs"
    point_root.mkdir(parents=True, exist_ok=True)
    point_log_root.mkdir(parents=True, exist_ok=True)

    discovered, skipped = discover_sweep_checkpoints()
    checkpoints = [row for row in discovered if row["method"] in args.methods]
    skipped = [row for row in skipped if row["method"] in args.methods]

    for row in skipped:
        check(
            f"skip incomplete {row['method']} checkpoint",
            "WARN",
            f"{row['checkpoint']}: {row['checkpoint_detail']}",
        )

    if not checkpoints:
        check("usable sweep checkpoints", "FAIL", "none discovered")
        save_text_report(sweep_root / "sweep_crp_report.txt", "SWEEP CHECKPOINT CRP")
        print("OVERALL VERDICT: FAIL")
        return 1

    manifest = []
    completed_counts = Counter()
    failed_counts = Counter()

    for index, row in enumerate(checkpoints, start=1):
        safe_label = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            f"{row['method']}_{row['label']}",
        )
        target = point_root / safe_label
        point_log = point_log_root / f"{safe_label}.log"

        if args.resume and (target / "activation_manifest.json").exists():
            check(safe_label, "PASS", "valid cached point output found")
            completed_counts[row["method"]] += 1
            manifest.append({
                **row,
                "checkpoint": str(row["checkpoint"]),
                "output": str(target.resolve()),
                "point_log": str(point_log.resolve()),
                "status": "cached",
            })
            continue

        print("=" * 88)
        print(f"SWEEP CRP {index}/{len(checkpoints)}: {safe_label}")
        print("=" * 88)

        method_work = work_results / "paired_activation_bootstrap" / row["method"]
        if method_work.exists():
            shutil.rmtree(method_work)

        code, child_log = isolated_fix2(
            fix2,
            work_results,
            row["method"],
            row["checkpoint"],
            point_log,
        )

        expected_manifest = method_work / "activation_manifest.json"
        expected_activations = method_work / f"{row['method']}_activations.pt"
        if code != 0 or not expected_manifest.exists() or not expected_activations.exists():
            tail = "\n".join(child_log.splitlines()[-35:])
            check(
                safe_label,
                "WARN",
                f"point failed (exit={code}); full log={point_log}\n{tail}",
            )
            failed_counts[row["method"]] += 1
            manifest.append({
                **row,
                "checkpoint": str(row["checkpoint"]),
                "point_log": str(point_log.resolve()),
                "status": "failed",
                "exit_code": code,
            })
            continue

        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(method_work, target)
        check(safe_label, "PASS", target)
        completed_counts[row["method"]] += 1
        manifest.append({
            **row,
            "checkpoint": str(row["checkpoint"]),
            "output": str(target.resolve()),
            "point_log": str(point_log.resolve()),
            "status": "complete",
        })

    for method in args.methods:
        completed = completed_counts[method]
        failed = failed_counts[method]
        if completed >= 3:
            check(
                f"{method} trajectory coverage",
                "PASS",
                f"{completed} valid points; {failed} failed/incomplete points",
            )
        elif completed > 0:
            check(
                f"{method} trajectory coverage",
                "WARN",
                f"only {completed} valid point(s); need at least 3 for trajectory",
            )
        else:
            status = "WARN" if method == "npo" else "FAIL"
            check(
                f"{method} trajectory coverage",
                status,
                "no valid points completed",
            )

    save_json(sweep_root / "sweep_crp_manifest.json", {
        "version": VERSION,
        "requested_methods": args.methods,
        "completed_counts": dict(completed_counts),
        "failed_counts": dict(failed_counts),
        "skipped_incomplete_checkpoints": [
            {**row, "checkpoint": str(row["checkpoint"])} for row in skipped
        ],
        "points": manifest,
        "checks": CHECKS,
        "overall_verdict": verdict(),
    })
    save_text_report(sweep_root / "sweep_crp_report.txt", "SWEEP CHECKPOINT CRP")
    print(f"OVERALL VERDICT: {verdict()}")
    return 1 if verdict() == "FAIL" else 0


# ---------------------------------------------------------------------------
# BLIP-2 likelihood stage
# ---------------------------------------------------------------------------


def resolve_blip2(exp_config):
    base = getattr(exp_config, "BLIP2_BASE", None) or getattr(exp_config, "BLIP2_BASE_MODEL", None)
    adapters = dict(getattr(exp_config, "BLIP2_ADAPTERS", {}))
    if not base:
        try:
            import eval_config
            base = getattr(eval_config, "BLIP2_BASE_MODEL", None)
        except Exception:
            pass
    return base, adapters


def to_device(batch, device: str):
    if hasattr(batch, "to"):
        return batch.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def answer_likelihood(model, processor, image, question: str, answer: str, device: str):
    import torch
    prompt = processor(images=image, text=question, return_tensors="pt")
    full = processor(images=image, text=f"{question} {answer}".strip(), return_tensors="pt")
    prompt_length = int(prompt["input_ids"].shape[1])
    labels = full["input_ids"].clone()
    labels[:, :prompt_length] = -100
    answer_tokens = int((labels != -100).sum().item())
    if answer_tokens <= 0:
        raise RuntimeError("No answer tokens remained after masking.")
    batch = dict(full)
    batch["labels"] = labels
    batch = to_device(batch, device)
    with torch.inference_mode():
        output = model(**batch)
    loss = float(output.loss.detach().float().item())
    logits = output.logits.detach().float().cpu()
    label_ids = labels.cpu()
    positions = torch.where(label_ids[0] != -100)[0]
    first_position = int(positions[0])
    prediction_position = max(first_position - 1, 0)
    target_id = int(label_ids[0, first_position].item())
    vector = logits[0, prediction_position]
    target_value = vector[target_id]
    rank = int((vector > target_value).sum().item()) + 1
    top = torch.topk(vector, k=min(2, vector.numel()))
    best_id = int(top.indices[0].item())
    competitor = float(top.values[1].item()) if best_id == target_id and len(top.values) > 1 else float(top.values[0].item())
    return {
        "mean_answer_nll": loss,
        "total_answer_logprob": -loss * answer_tokens,
        "answer_token_count": answer_tokens,
        "first_answer_token_rank": rank,
        "first_answer_token_margin": float(target_value.item()) - competitor,
    }


def run_blip2_logprob(args):
    exp_config, _, output = project_config()
    out_dir = output / "blip2_logprob"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "blip2_logprob_results.json"
    if args.resume and result_path.exists():
        check("BLIP-2 likelihood", "PASS", "cached")
        print("OVERALL VERDICT: PASS")
        return 0

    base_name, adapters = resolve_blip2(exp_config)
    if not base_name:
        check("BLIP-2 base model", "FAIL", "not found in exp_config")
        print("OVERALL VERDICT: FAIL")
        return 1

    try:
        import eval_config
        import eval_utils
        import torch
        from PIL import Image
    except Exception as exc:
        check("BLIP-2 imports", "FAIL", exc)
        print("OVERALL VERDICT: FAIL")
        return 1

    forget_items = load_forget_items(exp_config)
    retain_dir = resolve_path(getattr(eval_config, "RETAIN_DIR", None))
    retain_items = list(eval_utils.load_mllmu_split(retain_dir)) if retain_dir and retain_dir.exists() else []
    items = [("forget", item) for item in forget_items] + [("retain", item) for item in retain_items]
    preferred = ["ga", "npo", "mmunlearner", "cagul", "manu"]
    methods = ["base"] + [method for method in preferred if method in adapters]
    summaries = []
    per_item_all = []

    for method in methods:
        checkpoint = None if method == "base" else resolve_path(adapters[method])
        if method != "base" and (checkpoint is None or not checkpoint.exists()):
            check(f"BLIP-2 {method}", "WARN", f"missing: {checkpoint}")
            continue
        model = processor = None
        try:
            model, processor = eval_utils.load_blip2_model(str(base_name), checkpoint, args.device)
            model.eval()
            method_rows = []
            for index, (split, item) in enumerate(items):
                image = Image.open(item["image"]).convert("RGB")
                row = answer_likelihood(model, processor, image, str(item["question"]), str(item.get("answer", "")), args.device)
                row.update({
                    "method": method,
                    "split": split,
                    "index": index,
                    "entity": str(item.get("entity", "")),
                    "question": str(item.get("question", "")),
                    "answer": str(item.get("answer", "")),
                })
                method_rows.append(row)
                print(f"[{method}] {index+1}/{len(items)}")
            per_item_all.extend(method_rows)
            for split in ("forget", "retain"):
                subset = [row for row in method_rows if row["split"] == split]
                if not subset:
                    continue
                summaries.append({
                    "method": method,
                    "split": split,
                    "n": len(subset),
                    "mean_answer_nll": float(np.mean([row["mean_answer_nll"] for row in subset])),
                    "mean_total_answer_logprob": float(np.mean([row["total_answer_logprob"] for row in subset])),
                    "mean_first_token_rank": float(np.mean([row["first_answer_token_rank"] for row in subset])),
                    "mean_first_token_margin": float(np.mean([row["first_answer_token_margin"] for row in subset])),
                })
            check(f"BLIP-2 {method}", "PASS", len(method_rows))
        except Exception as exc:
            check(f"BLIP-2 {method}", "FAIL", f"{type(exc).__name__}: {exc}")
        finally:
            if model is not None:
                del model
            if processor is not None:
                del processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_json(result_path, {
        "version": VERSION,
        "base_model": str(base_name),
        "summary": summaries,
        "per_item": per_item_all,
        "checks": CHECKS,
        "overall_verdict": verdict(),
    })
    write_csv(out_dir / "blip2_logprob_summary.csv", summaries)
    write_csv(out_dir / "blip2_logprob_per_item.csv", per_item_all)
    save_text_report(out_dir / "blip2_logprob_report.txt", "BLIP-2 REFERENCE-ANSWER LIKELIHOOD AUDIT")
    print(f"OVERALL VERDICT: {verdict()}")
    return 1 if verdict() == "FAIL" else 0


def run_final(_):
    _, _, output = project_config()
    expected = {
        "preflight": output / "preflight_report.json",
        "cpu": output / "cpu_analysis" / "cpu_analysis_report.json",
        "sweep_crp": output / "sweep_crp" / "sweep_crp_manifest.json",
        "blip2_logprob": output / "blip2_logprob" / "blip2_logprob_results.json",
    }
    artifacts = []
    for name, path in expected.items():
        exists = path.exists()
        check(name, "PASS" if exists else "WARN", path)
        artifacts.append({
            "artifact": name,
            "path": str(path.resolve()),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else None,
            "sha256": sha256(path) if exists else None,
        })
    save_json(output / "final_report.json", {
        "version": VERSION,
        "created_utc": utc_now(),
        "artifacts": artifacts,
        "checks": CHECKS,
        "overall_verdict": verdict(),
    })
    save_text_report(output / "final_report.txt", "AAAI REVIEWER COMPLETE PIPELINE")
    print(f"OVERALL VERDICT: {verdict()}")
    return 1 if verdict() == "FAIL" else 0


def build_parser():
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    cpu = commands.add_parser("cpu")
    cpu.add_argument("--n-bootstrap", type=int, default=2000)
    cpu.add_argument("--alpha", type=float, default=0.05)
    cpu.add_argument("--probe-ridge", type=float, default=1.0)
    cpu.add_argument("--tex-root", default=".")
    sweep = commands.add_parser("sweep-crp")
    sweep.add_argument("--resume", action="store_true")
    sweep.add_argument("--methods", nargs="+", choices=["graddiff", "npo"], default=["graddiff", "npo"])
    blip = commands.add_parser("blip2-logprob")
    blip.add_argument("--device", default="cuda")
    blip.add_argument("--resume", action="store_true")
    commands.add_parser("final")
    return root


def main():
    args = build_parser().parse_args()
    try:
        if args.command == "preflight":
            return run_preflight(args)
        if args.command == "cpu":
            return run_cpu(args)
        if args.command == "sweep-crp":
            return run_sweep_crp(args)
        if args.command == "blip2-logprob":
            return run_blip2_logprob(args)
        if args.command == "final":
            return run_final(args)
        raise RuntimeError(f"Unknown command: {args.command}")
    except KeyboardInterrupt:
        print("[FAIL] Interrupted by user.")
        print("OVERALL VERDICT: FAIL")
        return 130
    except Exception as exc:
        try:
            _, _, output = project_config()
            save_json(output / "fatal_error.json", {
                "version": VERSION,
                "command": getattr(args, "command", None),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "created_utc": utc_now(),
            })
        except Exception:
            pass
        print(f"[FAIL] {type(exc).__name__}: {exc}")
        print("OVERALL VERDICT: FAIL")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
