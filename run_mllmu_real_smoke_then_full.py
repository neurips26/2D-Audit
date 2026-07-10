#!/usr/bin/env python3
"""
One-command mllmu_real behavioural evaluation:

Phase 1: smoke test
- one forget item + one retain item for each of six checkpoints
- saves raw responses
- detects obvious empty/repetitive generation
- does not modify generation settings

Phase 2: full evaluation
- runs the existing eval_behavioural.evaluate_method() for all six methods
- resumes completed methods
- saves after every method
- continues after individual failures

No training and no generation patching.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import types
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "outputs" / "revision" / "mllmu_real_smoke_then_full"
SMOKE_DIR = OUT_DIR / "smoke"
FULL_DIR = OUT_DIR / "full"
PER_METHOD_DIR = FULL_DIR / "per_method"

METHODS = [
    "ga_attn4_50",
    "npo",
    "mmunlearner",
    "cagul",
    "sineproject",
    "graddiff",
]

DISPLAY = {
    "ga_attn4_50": "GA-attn4-50",
    "npo": "NPO",
    "mmunlearner": "MMUnlearner",
    "cagul": "CAGUL",
    "sineproject": "SineProject",
    "graddiff": "GradDiff",
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def inspect_adapter_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    sf = checkpoint_dir / "adapter_model.safetensors"
    bf = checkpoint_dir / "adapter_model.bin"

    if sf.exists():
        from safetensors.torch import load_file
        state = load_file(str(sf), device="cpu")
        source = sf
    elif bf.exists():
        import torch
        try:
            state = torch.load(str(bf), map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(str(bf), map_location="cpu")
        source = bf
    else:
        raise FileNotFoundError(
            f"No adapter_model.safetensors or adapter_model.bin in {checkpoint_dir}"
        )

    if not isinstance(state, dict):
        raise RuntimeError(f"Adapter state is not a dictionary: {source}")

    vals = []
    for name, tensor in state.items():
        if "lora_b" not in name.casefold():
            continue
        try:
            max_abs = float(tensor.detach().float().abs().max().item())
        except Exception:
            continue
        vals.append(max_abs)

    n_total = len(vals)
    n_nonzero = sum(math.isfinite(v) and v > 0 for v in vals)

    if n_total == 0:
        raise RuntimeError(f"No LoRA-B tensors found in {source}")
    if n_nonzero == 0:
        raise RuntimeError(f"All {n_total} LoRA-B tensors are zero in {source}")

    return {
        "checkpoint_dir": str(checkpoint_dir),
        "source_file": str(source),
        "n_lora_B": n_total,
        "n_nonzero_lora_B": n_nonzero,
        "max_abs_lora_B": max(vals),
        "active": True,
    }


def install_adapter_guard_compat() -> None:
    module = types.ModuleType("adapter_guard")

    def assert_adapter_is_active(checkpoint_dir: Any) -> dict[str, Any]:
        return inspect_adapter_checkpoint(Path(checkpoint_dir))

    module.assert_adapter_is_active = assert_adapter_is_active
    sys.modules["adapter_guard"] = module


def checkpoint_map() -> dict[str, Path]:
    import exp_config
    adapters = dict(exp_config.LLAVA_ADAPTERS)
    result = {
        "ga_attn4_50": adapters.get("ga_attn4_50"),
        "npo": adapters.get("npo"),
        "mmunlearner": adapters.get("mmunlearner"),
        "cagul": adapters.get("cagul"),
        "sineproject": adapters.get("sineproject"),
        "graddiff": adapters.get("graddiff"),
    }
    return {
        method: Path(path).resolve() if path is not None else None
        for method, path in result.items()
    }


def repetition_diagnostic(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    tokens = re.findall(r"[A-Za-z0-9']+", cleaned.casefold())

    if not tokens:
        return {
            "empty": True,
            "token_count": 0,
            "unique_token_count": 0,
            "dominant_token": "",
            "dominant_fraction": 0.0,
            "obvious_repetition": False,
        }

    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    dominant_token, dominant_count = max(
        counts.items(),
        key=lambda item: item[1],
    )
    dominant_fraction = dominant_count / len(tokens)

    obvious = (
        len(tokens) >= 8
        and (
            dominant_fraction >= 0.60
            or len(set(tokens)) <= 2
        )
    )

    return {
        "empty": False,
        "token_count": len(tokens),
        "unique_token_count": len(set(tokens)),
        "dominant_token": dominant_token,
        "dominant_fraction": round(dominant_fraction, 4),
        "obvious_repetition": obvious,
    }


def run_smoke(
    methods: list[str],
    checkpoints: dict[str, Path],
) -> tuple[list[dict[str, Any]], list[str]]:
    from PIL import Image
    import torch
    import eval_config
    from eval_utils import (
        load_mllmu_split,
        load_llava_model,
        run_llava_inference,
    )

    forget_items = load_mllmu_split(eval_config.FORGET_DIR)
    retain_items = load_mllmu_split(eval_config.RETAIN_DIR)

    if not forget_items:
        raise RuntimeError("mllmu_real forget split is empty.")
    if not retain_items:
        raise RuntimeError("mllmu_real retain split is empty.")

    samples = {
        "forget": forget_items[0],
        "retain": retain_items[0],
    }

    records = []
    failed_methods = []

    for index, method in enumerate(methods, start=1):
        print("\n" + "=" * 100)
        print(f"[{now()}] SMOKE [{index}/{len(methods)}] {method}")
        print(f"[checkpoint] {checkpoints[method]}")
        print("=" * 100)

        method_record = {
            "method": method,
            "display_name": DISPLAY[method],
            "checkpoint": str(checkpoints[method]),
            "started_at": now(),
            "samples": {},
            "status": "PASS",
            "error": None,
        }

        model = None

        try:
            model, processor, _, _ = load_llava_model(
                eval_config.LLAVA_BASE_MODEL,
                checkpoints[method],
                eval_config.DEVICE,
            )

            for split_name, item in samples.items():
                image = Image.open(item["image"]).convert("RGB")
                response = run_llava_inference(
                    model,
                    processor,
                    None,
                    item["question"],
                    image,
                    max_new_tokens=32,
                    temperature=0.0,
                )
                diag = repetition_diagnostic(response)

                method_record["samples"][split_name] = {
                    "question": item["question"],
                    "image": str(item["image"]),
                    "response": response,
                    "diagnostic": diag,
                }

                label = "REPETITIVE" if diag["obvious_repetition"] else "OK"
                if diag["empty"]:
                    label = "EMPTY"

                print(
                    f"[{method}] {split_name.upper()} [{label}] "
                    f"{response!r}"
                )

            if any(
                sample["diagnostic"]["empty"]
                for sample in method_record["samples"].values()
            ):
                method_record["status"] = "WARN"

            if any(
                sample["diagnostic"]["obvious_repetition"]
                for sample in method_record["samples"].values()
            ):
                method_record["status"] = "WARN"

        except Exception as exc:
            method_record["status"] = "FAIL"
            method_record["error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            failed_methods.append(method)
            print(
                f"[{now()}] SMOKE FAIL {method}: "
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            method_record["finished_at"] = now()
            records.append(method_record)
            save_json(SMOKE_DIR / f"{method}.json", method_record)

            if model is not None:
                del model
                torch.cuda.empty_cache()

    save_json(SMOKE_DIR / "smoke_results.json", records)

    return records, failed_methods


def load_existing_full() -> dict[str, dict[str, Any]]:
    existing = {}

    if not PER_METHOD_DIR.exists():
        return existing

    for path in PER_METHOD_DIR.glob("*.json"):
        if path.name.endswith("_FAILED.json"):
            continue

        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        method = result.get("method")
        if method in METHODS and result.get("status") == "PASS":
            existing[method] = result

    return existing


def normalize_full_result(
    method: str,
    checkpoint: Path,
    raw: dict[str, Any],
    started_at: str,
    finished_at: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    result = dict(raw)
    forget_acc = result.get("forget_acc")
    retain_acc = result.get("retain_acc")

    finite = (
        isinstance(forget_acc, (int, float))
        and isinstance(retain_acc, (int, float))
        and math.isfinite(float(forget_acc))
        and math.isfinite(float(retain_acc))
    )

    result.update({
        "method": method,
        "display_name": DISPLAY[method],
        "dataset": "mllmu_real",
        "architecture": "llava",
        "checkpoint": str(checkpoint),
        "seed": 42,
        "started_at": started_at,
        "finished_at": finished_at,
        "runtime_seconds": round(runtime_seconds, 2),
        "status": "PASS" if finite else "FAIL",
    })

    return result


def write_full_summary(results: dict[str, dict[str, Any]]) -> None:
    rows = [results[m] for m in METHODS if m in results]

    fields = [
        "method",
        "display_name",
        "forget_acc",
        "forget_rate",
        "retain_acc",
        "forget_n",
        "retain_n",
        "seed",
        "runtime_seconds",
        "checkpoint",
        "status",
    ]

    FULL_DIR.mkdir(parents=True, exist_ok=True)

    with (FULL_DIR / "behavioural_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    save_json(FULL_DIR / "behavioural_results.json", rows)


def run_full(
    methods: list[str],
    checkpoints: dict[str, Path],
    resume: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    import eval_behavioural

    eval_behavioural.CHECKPOINT_DIRS = checkpoints
    eval_behavioural.ALL_METHODS = list(methods)
    eval_behavioural.OUT_DIR = FULL_DIR

    results = load_existing_full()
    failures = []

    for index, method in enumerate(methods, start=1):
        if (
            resume
            and method in results
            and results[method].get("status") == "PASS"
        ):
            print(
                f"[{now()}] FULL [{index}/{len(methods)}] "
                f"RESUME SKIP {method}"
            )
            continue

        print("\n" + "=" * 100)
        print(f"[{now()}] FULL [{index}/{len(methods)}] START {method}")
        print(f"[checkpoint] {checkpoints[method]}")
        print("=" * 100)

        started_at = now()
        start_clock = time.perf_counter()

        try:
            raw = eval_behavioural.evaluate_method(method, "llava")
            finished_at = now()
            runtime = time.perf_counter() - start_clock

            result = normalize_full_result(
                method,
                checkpoints[method],
                raw,
                started_at,
                finished_at,
                runtime,
            )

            results[method] = result
            save_json(PER_METHOD_DIR / f"{method}.json", result)
            write_full_summary(results)

            print(
                f"[{finished_at}] FULL FINISH {method} | "
                f"ForgetAcc={result.get('forget_acc')} | "
                f"RetainAcc={result.get('retain_acc')} | "
                f"runtime={runtime:.2f}s | "
                f"status={result['status']}"
            )

        except Exception as exc:
            finished_at = now()
            runtime = time.perf_counter() - start_clock

            failure = {
                "method": method,
                "checkpoint": str(checkpoints[method]),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "started_at": started_at,
                "finished_at": finished_at,
                "runtime_seconds": round(runtime, 2),
            }
            failures.append(failure)
            save_json(
                PER_METHOD_DIR / f"{method}_FAILED.json",
                failure,
            )

            print(
                f"[{finished_at}] FULL FAIL {method}: "
                f"{type(exc).__name__}: {exc}"
            )

    write_full_summary(results)
    return results, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=METHODS,
    )
    parser.add_argument(
        "--stop-after-smoke",
        action="store_true",
        help="Run smoke only and do not start full evaluation.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    PER_METHOD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{now()}] MLLMU_REAL SMOKE-THEN-FULL START")
    print(f"[root] {ROOT}")
    print(f"[output] {OUT_DIR}")

    install_adapter_guard_compat()

    checkpoints = checkpoint_map()
    preflight_errors = []
    manifest = {
        "generated_at": now(),
        "dataset": "mllmu_real",
        "architecture": "llava",
        "methods": {},
    }

    for method in args.methods:
        checkpoint = checkpoints[method]

        try:
            activity = inspect_adapter_checkpoint(checkpoint)
            manifest["methods"][method] = {
                "checkpoint": str(checkpoint),
                "exists": True,
                "activity": activity,
            }
            print(
                f"[PASS] {method:<16} "
                f"{activity['n_nonzero_lora_B']}/"
                f"{activity['n_lora_B']} nonzero LoRA-B"
            )
        except Exception as exc:
            preflight_errors.append(
                f"{method}: {type(exc).__name__}: {exc}"
            )
            manifest["methods"][method] = {
                "checkpoint": str(checkpoint),
                "exists": bool(checkpoint and checkpoint.exists()),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(
                f"[FAIL] {method}: {type(exc).__name__}: {exc}"
            )

    save_json(OUT_DIR / "checkpoint_manifest.json", manifest)

    if preflight_errors:
        report = {
            "overall_verdict": "FAIL",
            "reason": "Checkpoint preflight failed.",
            "errors": preflight_errors,
        }
        save_json(OUT_DIR / "run_report.json", report)
        print("OVERALL VERDICT: FAIL")
        return 1

    # Import after installing compatibility module.
    import eval_behavioural  # noqa: F401

    smoke_records, smoke_failed = run_smoke(
        args.methods,
        checkpoints,
    )

    smoke_warn = [
        record["method"]
        for record in smoke_records
        if record["status"] == "WARN"
    ]

    print("\n" + "=" * 100)
    print("SMOKE SUMMARY")
    print("=" * 100)
    print(
        "PASS: "
        + ", ".join(
            record["method"]
            for record in smoke_records
            if record["status"] == "PASS"
        )
    )
    print(
        "WARN: "
        + (", ".join(smoke_warn) if smoke_warn else "none")
    )
    print(
        "FAIL: "
        + (", ".join(smoke_failed) if smoke_failed else "none")
    )

    if args.stop_after_smoke:
        verdict = "WARN" if smoke_warn else ("FAIL" if smoke_failed else "PASS")
        report = {
            "mode": "smoke_only",
            "smoke_warn_methods": smoke_warn,
            "smoke_failed_methods": smoke_failed,
            "overall_verdict": verdict,
            "reason": "Smoke-only run requested.",
        }
        save_json(OUT_DIR / "run_report.json", report)
        print(f"OVERALL VERDICT: {verdict}")
        return 0 if verdict == "PASS" else (2 if verdict == "WARN" else 1)

    if smoke_failed:
        report = {
            "mode": "smoke_then_full",
            "smoke_warn_methods": smoke_warn,
            "smoke_failed_methods": smoke_failed,
            "overall_verdict": "FAIL",
            "reason": (
                "At least one method failed to load or generate during smoke; "
                "full evaluation was not started."
            ),
        }
        save_json(OUT_DIR / "run_report.json", report)
        print(
            "Full evaluation not started because smoke had hard failures."
        )
        print("OVERALL VERDICT: FAIL")
        return 1

    # WARN due to repetitive output does not stop the full evaluation.
    # This preserves the true behavioural result without changing generation.
    results, failures = run_full(
        args.methods,
        checkpoints,
        args.resume,
    )

    completed = [
        method
        for method in args.methods
        if results.get(method, {}).get("status") == "PASS"
    ]
    missing = [
        method
        for method in args.methods
        if method not in completed
    ]

    if len(completed) == len(args.methods):
        verdict = "PASS"
        reason = (
            "Smoke completed and all full behavioural evaluations completed. "
            "Smoke repetition warnings, if any, are preserved as model behaviour."
        )
        exit_code = 0
    elif completed:
        verdict = "WARN"
        reason = (
            "Smoke completed, but one or more full evaluations failed."
        )
        exit_code = 2
    else:
        verdict = "FAIL"
        reason = "No full behavioural evaluation completed."
        exit_code = 1

    report = {
        "started_at": manifest["generated_at"],
        "finished_at": now(),
        "dataset": "mllmu_real",
        "architecture": "llava",
        "requested_methods": args.methods,
        "smoke_warn_methods": smoke_warn,
        "smoke_failed_methods": smoke_failed,
        "completed_methods": completed,
        "missing_or_failed_methods": missing,
        "full_failures": failures,
        "smoke_results": str(SMOKE_DIR / "smoke_results.json"),
        "summary_csv": str(FULL_DIR / "behavioural_summary.csv"),
        "results_json": str(FULL_DIR / "behavioural_results.json"),
        "overall_verdict": verdict,
        "reason": reason,
    }
    save_json(OUT_DIR / "run_report.json", report)

    lines = [
        "MLLMU_REAL SMOKE-THEN-FULL REPORT",
        "=" * 100,
        f"Smoke warnings: {', '.join(smoke_warn) if smoke_warn else 'none'}",
        f"Smoke failures: {', '.join(smoke_failed) if smoke_failed else 'none'}",
        f"Full completed: {', '.join(completed) if completed else 'none'}",
        f"Full missing/failed: {', '.join(missing) if missing else 'none'}",
        "",
        "FULL RESULTS",
        "-" * 100,
    ]

    for method in args.methods:
        result = results.get(method)
        if result and result.get("status") == "PASS":
            lines.append(
                f"{method:<16} "
                f"ForgetAcc={float(result['forget_acc']):.4f} "
                f"RetainAcc={float(result['retain_acc']):.4f} "
                f"forget_n={result.get('forget_n', '')} "
                f"retain_n={result.get('retain_n', '')} "
                f"runtime={result.get('runtime_seconds', '')}s"
            )
        else:
            lines.append(f"{method:<16} FAILED OR MISSING")

    lines.extend([
        "",
        f"SMOKE JSON: {SMOKE_DIR / 'smoke_results.json'}",
        f"SUMMARY CSV: {FULL_DIR / 'behavioural_summary.csv'}",
        f"RESULTS JSON: {FULL_DIR / 'behavioural_results.json'}",
        f"REASON: {reason}",
        f"OVERALL VERDICT: {verdict}",
    ])

    (OUT_DIR / "run_report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "\n".join(lines))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
