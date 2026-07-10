"""
Robust GradDiff training for LLaVA-1.5-7B.

Objective
---------
    L = -L_forget + lambda_retain * L_retain

Key safeguards
--------------
* Answer-only supervision: prompt/image tokens are masked with -100.
* Deterministic seeds.
* Gradient clipping and finite-loss checks.
* Bounded retries (no infinite loop on a bad sample).
* Unique run/checkpoint names encode LR, lambda, seed, and step.
* Checkpoints at requested schedule points.
* Full or sampled behavioural validation with PASS/WARN/FAIL report files.
* LoRA health checks and machine-readable training manifest.

Examples
--------
Smoke test:
    py -u train_graddiff_llava_fixed.py --steps 2 --save_steps 2 --seed 42

Main run:
    py -u train_graddiff_llava_fixed.py --steps 50 --lambda_retain 1.0 --lr 1e-4 --seed 42

Check a checkpoint on the full available splits:
    py -u train_graddiff_llava_fixed.py --check_ckpt "<checkpoint path>" --eval_forget_n 0 --eval_retain_n 0

Verify every saved adapter:
    py -u train_graddiff_llava_fixed.py --verify_only
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_config import (  # type: ignore
    DEVICE,
    LLAVA_BASE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LR,
    MLLMU_REAL_FORGET,
    MLLMU_REAL_RETAIN,
    ROOT,
)

DEFAULT_SAVE_STEPS = [5, 10, 20, 50]
DEFAULT_LAMBDA = 1.0
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_MAX_RETRIES = 20
GRADDIFF_ROOT = ROOT / "checkpoints" / "graddiff"
REPORT_ROOT = ROOT / "outputs" / "revision" / "graddiff"


@dataclass(frozen=True)
class RunSpec:
    lr: float
    lambda_retain: float
    seed: int
    max_steps: int
    max_grad_norm: float

    @property
    def run_id(self) -> str:
        return (
            f"lr{float_tag(self.lr)}_"
            f"lambda{float_tag(self.lambda_retain)}_"
            f"seed{self.seed}"
        )


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def float_tag(value: float) -> str:
    """Filesystem-safe, stable representation for hyperparameters."""
    if value == 0:
        return "0"
    text = f"{value:.8g}".lower().replace("+", "")
    return text.replace(".", "p").replace("-", "m")


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Exact determinism is not guaranteed for every quantized CUDA kernel, but
    # these settings remove avoidable stochasticity.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def cleanup_cuda(*objects: Any) -> None:
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sanitize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def answer_matches(response: str, item: dict[str, Any]) -> bool:
    response_norm = normalize_answer(response)
    candidates = [item.get("answer", ""), *item.get("aliases", [])]
    for candidate in candidates:
        candidate_norm = normalize_answer(str(candidate))
        if candidate_norm and candidate_norm in response_norm:
            return True
    return False


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def _coerce_item(item: dict[str, Any], split_dir: Path) -> dict[str, Any] | None:
    if "image" not in item or "question" not in item:
        return None
    answer = item.get("answer", item.get("gt", ""))
    if not sanitize_text(answer):
        return None
    image_path = Path(str(item["image"]))
    if not image_path.is_absolute():
        image_path = split_dir / image_path
    if not image_path.exists():
        print(f"  [WARN] Missing image: {image_path}")
        return None
    result = dict(item)
    result["image"] = image_path
    result["question"] = sanitize_text(item["question"])
    result["answer"] = sanitize_text(answer)
    result.setdefault("aliases", [])
    return result


def load_items(split_dir: Path) -> list[dict[str, Any]]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    annotation_file = split_dir / "annotations.json"
    loaded: list[dict[str, Any]] = []

    if annotation_file.exists():
        raw = json.loads(annotation_file.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"Expected a list in {annotation_file}")
        for item in raw:
            if isinstance(item, dict):
                parsed = _coerce_item(item, split_dir)
                if parsed is not None:
                    loaded.append(parsed)
        return loaded

    for entity_dir in sorted(split_dir.iterdir()):
        if not entity_dir.is_dir():
            continue
        images = sorted(entity_dir.glob("*.jpg")) + sorted(entity_dir.glob("*.jpeg")) + sorted(entity_dir.glob("*.png"))
        json_files = sorted(entity_dir.glob("*.json"))
        if not images or not json_files:
            continue
        raw = json.loads(json_files[0].read_text(encoding="utf-8"))
        questions = raw if isinstance(raw, list) else [raw]
        for question in questions:
            if not isinstance(question, dict):
                continue
            parsed = _coerce_item(
                {
                    "entity": entity_dir.name,
                    "image": images[0],
                    "question": question.get("question", ""),
                    "answer": question.get("answer", question.get("gt", "")),
                    "aliases": question.get("aliases", []),
                },
                split_dir,
            )
            if parsed is not None:
                loaded.append(parsed)
    return loaded


class CyclingSampler:
    """Deterministically reshuffles items at each epoch."""

    def __init__(self, items: list[dict[str, Any]], seed: int):
        if not items:
            raise ValueError("CyclingSampler received an empty item list")
        self.items = items
        self.rng = random.Random(seed)
        self.order = list(range(len(items)))
        self.position = len(self.order)

    def next(self) -> dict[str, Any]:
        if self.position >= len(self.order):
            self.rng.shuffle(self.order)
            self.position = 0
        index = self.order[self.position]
        self.position += 1
        return self.items[index]


# -----------------------------------------------------------------------------
# Model construction
# -----------------------------------------------------------------------------

def get_bnb_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def build_train_model():
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    print("[GradDiff] Loading base model...")
    processor = AutoProcessor.from_pretrained(LLAVA_BASE)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_BASE,
        quantization_config=get_bnb_config(),
        device_map=DEVICE,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.print_trainable_parameters()
    return model, processor


def check_lora_health(model, label: str) -> dict[str, int]:
    counts = {
        "total": 0,
        "nonzero": 0,
        "vision_total": 0,
        "vision_nonzero": 0,
        "language_total": 0,
        "language_nonzero": 0,
    }
    for name, parameter in model.named_parameters():
        if "lora_B" not in name:
            continue
        counts["total"] += 1
        nonzero = bool(torch.isfinite(parameter).all() and parameter.detach().abs().max().item() > 1e-8)
        counts["nonzero"] += int(nonzero)
        if "vision_tower" in name or "vision_model" in name:
            counts["vision_total"] += 1
            counts["vision_nonzero"] += int(nonzero)
        elif "language_model" in name:
            counts["language_total"] += 1
            counts["language_nonzero"] += int(nonzero)

    print(
        f"  [health:{label}] lora_B={counts['nonzero']}/{counts['total']} | "
        f"vision={counts['vision_nonzero']}/{counts['vision_total']} | "
        f"language={counts['language_nonzero']}/{counts['language_total']}"
    )
    return counts


# -----------------------------------------------------------------------------
# Answer-only supervision
# -----------------------------------------------------------------------------

def _find_last_subsequence(sequence: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(sequence):
        return None
    for start in range(len(sequence) - len(pattern), -1, -1):
        if sequence[start : start + len(pattern)] == pattern:
            return start
    return None


def build_answer_only_batch(processor, item: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Create a multimodal batch whose labels supervise only assistant-answer tokens."""
    image = Image.open(item["image"]).convert("RGB")
    prefix = f"USER: <image>\n{item['question']} ASSISTANT:"
    answer = sanitize_text(item["answer"])
    full_text = f"{prefix} {answer}"

    batch = processor(text=full_text, images=image, return_tensors="pt")
    labels = batch["input_ids"].clone()
    token_ids = labels[0].tolist()

    tokenizer = processor.tokenizer
    answer_candidates: list[list[int]] = []
    for candidate in (f" {answer}", answer):
        ids = tokenizer(candidate, add_special_tokens=False).input_ids
        if ids and ids not in answer_candidates:
            answer_candidates.append(ids)

    answer_start: int | None = None
    for answer_ids in answer_candidates:
        answer_start = _find_last_subsequence(token_ids, answer_ids)
        if answer_start is not None:
            break

    if answer_start is None:
        # Fallback: process the prefix through the same multimodal processor.
        # Clamp to leave at least one supervised token.
        prefix_batch = processor(text=prefix, images=image, return_tensors="pt")
        answer_start = min(int(prefix_batch["input_ids"].shape[1]), labels.shape[1] - 1)

    labels[:, :answer_start] = -100
    if "attention_mask" in batch:
        labels[batch["attention_mask"] == 0] = -100

    supervised = int((labels != -100).sum().item())
    if supervised <= 0:
        raise RuntimeError("Answer-only masking produced zero supervised tokens")

    batch["labels"] = labels
    return {key: value.to(DEVICE) if torch.is_tensor(value) else value for key, value in batch.items()}


def forward_answer_loss(model, processor, item: dict[str, Any]) -> torch.Tensor:
    batch = build_answer_only_batch(processor, item)
    output = model(**batch)
    loss = output.loss
    if loss is None or not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite answer loss: {loss}")
    return loss


# -----------------------------------------------------------------------------
# Checkpoint/report helpers
# -----------------------------------------------------------------------------

def checkpoint_dir(run_dir: Path, step: int) -> Path:
    return run_dir / f"graddiff_llava_{step}steps"


def save_checkpoint(model, processor, run_dir: Path, step: int, spec: RunSpec, health: dict[str, int]) -> Path:
    destination = checkpoint_dir(run_dir, step)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(destination))
    processor.save_pretrained(str(destination))
    metadata = {
        "method": "graddiff",
        "objective": "-forget_answer_ce + lambda_retain * retain_answer_ce",
        "answer_only_labels": True,
        "step": step,
        "run": asdict(spec),
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "base_model": LLAVA_BASE,
        "health": health,
    }
    atomic_json_dump(metadata, destination / "training_meta.json")
    return destination


def write_training_report(run_dir: Path, payload: dict[str, Any]) -> None:
    atomic_json_dump(payload, run_dir / "training_report.json")
    status = payload.get("overall_verdict", "UNKNOWN")
    lines = [
        "GRADDIFF TRAINING REPORT",
        f"Run: {payload.get('run_id')}",
        f"Completed steps: {payload.get('completed_steps')}/{payload.get('requested_steps')}",
        f"Successful updates: {payload.get('successful_updates')}",
        f"Failed attempts: {payload.get('failed_attempts')}",
        f"Final lora_B: {payload.get('final_health', {}).get('nonzero', 0)}/{payload.get('final_health', {}).get('total', 0)}",
        f"OVERALL VERDICT: {status}",
    ]
    (run_dir / "training_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train(
    max_steps: int,
    lambda_retain: float,
    learning_rate: float,
    seed: int,
    save_steps: Iterable[int],
    max_grad_norm: float,
    max_retries: int,
    output_root: Path,
) -> Path:
    if max_steps <= 0:
        raise ValueError("--steps must be positive")
    if learning_rate <= 0:
        raise ValueError("--lr must be positive")
    if lambda_retain < 0:
        raise ValueError("--lambda_retain must be non-negative")

    set_deterministic_seed(seed)
    spec = RunSpec(
        lr=learning_rate,
        lambda_retain=lambda_retain,
        seed=seed,
        max_steps=max_steps,
        max_grad_norm=max_grad_norm,
    )
    run_dir = output_root / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print(f"GRADDIFF RUN: {spec.run_id}")
    print(f"steps={max_steps} lr={learning_rate:g} lambda={lambda_retain:g} seed={seed}")
    print("objective=-forget_answer_CE + lambda*retain_answer_CE")
    print("=" * 88)

    model, processor = build_train_model()
    initial_health = check_lora_health(model, "before_training")

    forget_items = load_items(MLLMU_REAL_FORGET)
    retain_items = load_items(MLLMU_REAL_RETAIN)
    if not forget_items or not retain_items:
        cleanup_cuda(model)
        raise RuntimeError(
            f"Dataset loading failed: forget={len(forget_items)}, retain={len(retain_items)}"
        )
    print(f"[PASS] Loaded forget={len(forget_items)} retain={len(retain_items)}")

    forget_sampler = CyclingSampler(forget_items, seed + 101)
    retain_sampler = CyclingSampler(retain_items, seed + 202)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        cleanup_cuda(model)
        raise RuntimeError("No trainable parameters")

    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    model.train()

    requested_save_steps = sorted({int(step) for step in save_steps if 0 < int(step) <= max_steps} | {max_steps})
    history: list[dict[str, Any]] = []
    completed_steps = 0
    failed_attempts = 0
    consecutive_failures = 0
    gradient_check_passed = False
    start_time = time.time()

    try:
        while completed_steps < max_steps:
            forget_item = forget_sampler.next()
            retain_item = retain_sampler.next()
            optimizer.zero_grad(set_to_none=True)

            try:
                # Backpropagate the two terms separately to avoid retaining both
                # forward graphs in GPU memory at the same time.
                forget_loss = forward_answer_loss(model, processor, forget_item)
                (-forget_loss).backward()
                forget_value = float(forget_loss.detach().item())
                del forget_loss

                retain_loss = forward_answer_loss(model, processor, retain_item)
                (lambda_retain * retain_loss).backward()
                retain_value = float(retain_loss.detach().item())
                del retain_loss

                if not gradient_check_passed:
                    gradient_check_passed = any(
                        parameter.grad is not None
                        and torch.isfinite(parameter.grad).all()
                        and parameter.grad.detach().abs().max().item() > 1e-10
                        for name, parameter in model.named_parameters()
                        if "lora_B" in name and parameter.requires_grad
                    )
                    if not gradient_check_passed:
                        raise RuntimeError("Gradients do not reach trainable lora_B tensors")
                    print("[PASS] Gradients reach lora_B")

                gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm).item())
                if not math.isfinite(gradient_norm):
                    raise FloatingPointError(f"Non-finite gradient norm: {gradient_norm}")

                optimizer.step()
                completed_steps += 1
                consecutive_failures = 0

                total_value = -forget_value + lambda_retain * retain_value
                record = {
                    "step": completed_steps,
                    "total_objective": total_value,
                    "forget_answer_ce": forget_value,
                    "retain_answer_ce": retain_value,
                    "gradient_norm_before_clip": gradient_norm,
                    "forget_entity": forget_item.get("entity"),
                    "retain_entity": retain_item.get("entity"),
                }
                history.append(record)

                if completed_steps == 1 or completed_steps % 5 == 0:
                    print(
                        f"step={completed_steps:3d} total={total_value:+.5f} "
                        f"forget_ce={forget_value:.5f} retain_ce={retain_value:.5f} "
                        f"grad_norm={gradient_norm:.4f}"
                    )

                if completed_steps in requested_save_steps:
                    health = check_lora_health(model, f"step_{completed_steps}")
                    saved = save_checkpoint(model, processor, run_dir, completed_steps, spec, health)
                    print(f"[PASS] Saved checkpoint: {saved}")
                    atomic_json_dump(history, run_dir / "training_history.json")

            except (RuntimeError, ValueError, FloatingPointError, OSError) as exc:
                optimizer.zero_grad(set_to_none=True)
                failed_attempts += 1
                consecutive_failures += 1
                print(
                    f"[WARN] Failed attempt at completed_step={completed_steps}: "
                    f"{type(exc).__name__}: {exc}"
                )
                if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if consecutive_failures >= max_retries:
                    raise RuntimeError(
                        f"Aborting after {consecutive_failures} consecutive failed attempts"
                    ) from exc

        final_health = check_lora_health(model, "after_training")
        overall = "PASS" if final_health["nonzero"] > 0 and completed_steps == max_steps else "FAIL"
        report = {
            "run_id": spec.run_id,
            "run": asdict(spec),
            "requested_steps": max_steps,
            "completed_steps": completed_steps,
            "successful_updates": len(history),
            "failed_attempts": failed_attempts,
            "gradient_check_passed": gradient_check_passed,
            "initial_health": initial_health,
            "final_health": final_health,
            "elapsed_seconds": round(time.time() - start_time, 2),
            "checkpoints": [str(checkpoint_dir(run_dir, step)) for step in requested_save_steps],
            "overall_verdict": overall,
        }
        write_training_report(run_dir, report)
        atomic_json_dump(history, run_dir / "training_history.json")
        print(f"OVERALL VERDICT: {overall}")
        print(f"REPORT: {run_dir / 'training_report.txt'}")
        print(f"JSON: {run_dir / 'training_report.json'}")
        return checkpoint_dir(run_dir, max_steps)
    finally:
        cleanup_cuda(model)


# -----------------------------------------------------------------------------
# Behavioural evaluation
# -----------------------------------------------------------------------------

def load_eval_model(checkpoint: Path | None):
    from peft import PeftModel
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(LLAVA_BASE)
    base = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_BASE,
        quantization_config=get_bnb_config(),
        device_map=DEVICE,
    )
    model = base if checkpoint is None else PeftModel.from_pretrained(base, str(checkpoint))
    model.eval()
    model.config.use_cache = True
    return model, base, processor


def generate_answer(model, processor, item: dict[str, Any], max_new_tokens: int) -> str:
    image = Image.open(item["image"]).convert("RGB")
    prompt = f"USER: <image>\n{item['question']} ASSISTANT:"
    batch = processor(text=prompt, images=image, return_tensors="pt")
    batch = {key: value.to(DEVICE) if torch.is_tensor(value) else value for key, value in batch.items()}
    with torch.inference_mode():
        output = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    decoded = processor.decode(output[0], skip_special_tokens=True)
    if "ASSISTANT:" in decoded:
        decoded = decoded.rsplit("ASSISTANT:", 1)[-1]
    return decoded.strip()


def select_eval_items(items: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    return items if n <= 0 else items[: min(n, len(items))]


def evaluate_split(model, processor, items: list[dict[str, Any]], max_new_tokens: int, label: str) -> dict[str, Any]:
    correct = 0
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        response = generate_answer(model, processor, item, max_new_tokens)
        matched = answer_matches(response, item)
        correct += int(matched)
        rows.append(
            {
                "index": index,
                "entity": item.get("entity"),
                "answer": item.get("answer"),
                "response": response,
                "correct": matched,
            }
        )
        if index == 1 or index % 10 == 0 or index == len(items):
            print(f"  [{label}] {index}/{len(items)} correct={correct}")
    accuracy = correct / len(items) if items else float("nan")
    return {"n": len(items), "correct": correct, "accuracy": accuracy, "rows": rows}


def _prediction_key(row: dict[str, Any], position: int) -> str:
    entity = sanitize_text(row.get("entity"))
    answer = sanitize_text(row.get("answer"))
    question = sanitize_text(row.get("question"))
    if entity or question:
        return f"{entity}::{question}::{answer}"
    return f"position::{position}"


def _load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("predictions", "items", "results", "rows", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise ValueError(f"Unsupported prediction JSON schema: {path}")


def _paired_flip_analysis(
    base_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    base_map = {
        _prediction_key(row, i): row for i, row in enumerate(base_rows)
    }
    method_map = {
        _prediction_key(row, i): row for i, row in enumerate(method_rows)
    }

    common = sorted(set(base_map) & set(method_map))
    if not common and len(base_rows) == len(method_rows):
        pairs = list(zip(base_rows, method_rows))
    else:
        pairs = [(base_map[key], method_map[key]) for key in common]

    counts = {
        "successful_forgetting": 0,
        "unexpected_recovery": 0,
        "still_remembered": 0,
        "unchanged_incorrect": 0,
        "utility_damage": 0,
        "utility_improvement": 0,
        "utility_preserved": 0,
        "unchanged_failure": 0,
    }
    details: list[dict[str, Any]] = []

    for index, (base_row, method_row) in enumerate(pairs, start=1):
        base_correct = bool(base_row.get("correct", False))
        method_correct = bool(method_row.get("correct", False))

        if split == "forget":
            if base_correct and not method_correct:
                category = "successful_forgetting"
            elif not base_correct and method_correct:
                category = "unexpected_recovery"
            elif base_correct and method_correct:
                category = "still_remembered"
            else:
                category = "unchanged_incorrect"
        else:
            if base_correct and not method_correct:
                category = "utility_damage"
            elif not base_correct and method_correct:
                category = "utility_improvement"
            elif base_correct and method_correct:
                category = "utility_preserved"
            else:
                category = "unchanged_failure"

        counts[category] += 1
        details.append(
            {
                "index": index,
                "entity": method_row.get("entity", base_row.get("entity")),
                "answer": method_row.get("answer", base_row.get("answer")),
                "base_response": base_row.get("response"),
                "method_response": method_row.get("response"),
                "base_correct": base_correct,
                "method_correct": method_correct,
                "category": category,
            }
        )

    return {
        "n_aligned": len(pairs),
        "counts": counts,
        "details": details,
    }


def _classify_relative(
    forget_accuracy: float,
    retain_accuracy: float,
    base_forget_accuracy: float,
    base_retain_accuracy: float,
    retain_tolerance: float,
    collapse_retain_drop: float,
) -> tuple[str, str]:
    delta_forget = forget_accuracy - base_forget_accuracy
    delta_retain = retain_accuracy - base_retain_accuracy

    if delta_retain <= -abs(collapse_retain_drop):
        return "COLLAPSE_OR_SEVERE_UTILITY_DAMAGE", "WARN"

    if delta_forget < 0 and delta_retain >= -abs(retain_tolerance):
        return "DIRECTIONAL_SELECTIVE_FORGETTING", "PASS"

    if abs(delta_forget) < 1e-12 and delta_retain >= -abs(retain_tolerance):
        return "UNDER_FORGETTING", "WARN"

    if delta_forget < 0 and delta_retain < -abs(retain_tolerance):
        return "FORGETTING_WITH_UTILITY_DAMAGE", "WARN"

    if delta_forget > 0:
        return "UNEXPECTED_RECOVERY_OR_RELEARNING", "WARN"

    return "NO_CLEAR_SELECTIVE_EFFECT", "WARN"


def behavioral_check(
    checkpoint: Path | None,
    eval_forget_n: int,
    eval_retain_n: int,
    max_new_tokens: int,
    selective_forget_threshold: float,
    selective_retain_threshold: float,
    retain_tolerance: float = 0.05,
    collapse_retain_drop: float = 0.20,
) -> dict[str, Any]:
    if checkpoint is not None and not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    forget_items = select_eval_items(load_items(MLLMU_REAL_FORGET), eval_forget_n)
    retain_items = select_eval_items(load_items(MLLMU_REAL_RETAIN), eval_retain_n)
    model_label = "BASE_MODEL" if checkpoint is None else str(checkpoint)
    print(f"[Eval] checkpoint={model_label}")
    print(f"[Eval] forget={len(forget_items)} retain={len(retain_items)}")

    model, base, processor = load_eval_model(checkpoint)
    try:
        forget = evaluate_split(model, processor, forget_items, max_new_tokens, "forget")
        retain = evaluate_split(model, processor, retain_items, max_new_tokens, "retain")
    finally:
        cleanup_cuda(model, base)

    forget_accuracy = float(forget["accuracy"])
    retain_accuracy = float(retain["accuracy"])
    forget_rate = 1.0 - forget_accuracy

    report_dir = (
        REPORT_ROOT / "base_model"
        if checkpoint is None
        else REPORT_ROOT / checkpoint.parent.name / checkpoint.name
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    base_summary_path = REPORT_ROOT / "base_model" / "behavioral_summary.json"
    base_forget_predictions_path = REPORT_ROOT / "base_model" / "forget_predictions.json"
    base_retain_predictions_path = REPORT_ROOT / "base_model" / "retain_predictions.json"

    paired_forget = None
    paired_retain = None

    if checkpoint is None:
        classification = "BASELINE"
        verdict = "PASS"
        base_forget_accuracy = forget_accuracy
        base_retain_accuracy = retain_accuracy
        delta_forget = 0.0
        delta_retain = 0.0
    elif base_summary_path.exists():
        base_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))
        base_forget_accuracy = float(
            base_summary.get("forget_accuracy", base_summary.get("forget_acc"))
        )
        base_retain_accuracy = float(
            base_summary.get("retain_accuracy", base_summary.get("retain_acc"))
        )
        delta_forget = forget_accuracy - base_forget_accuracy
        delta_retain = retain_accuracy - base_retain_accuracy
        classification, verdict = _classify_relative(
            forget_accuracy=forget_accuracy,
            retain_accuracy=retain_accuracy,
            base_forget_accuracy=base_forget_accuracy,
            base_retain_accuracy=base_retain_accuracy,
            retain_tolerance=retain_tolerance,
            collapse_retain_drop=collapse_retain_drop,
        )

        if base_forget_predictions_path.exists() and base_retain_predictions_path.exists():
            paired_forget = _paired_flip_analysis(
                _load_prediction_rows(base_forget_predictions_path),
                forget["rows"],
                "forget",
            )
            paired_retain = _paired_flip_analysis(
                _load_prediction_rows(base_retain_predictions_path),
                retain["rows"],
                "retain",
            )
    else:
        base_forget_accuracy = float("nan")
        base_retain_accuracy = float("nan")
        delta_forget = float("nan")
        delta_retain = float("nan")
        classification = "BASELINE_REQUIRED"
        verdict = "WARN"

    result = {
        "checkpoint": model_label,
        "forget": {key: value for key, value in forget.items() if key != "rows"},
        "retain": {key: value for key, value in retain.items() if key != "rows"},
        "forget_accuracy": forget_accuracy,
        "forget_acc": forget_accuracy,
        "forget_rate": forget_rate,
        "retain_accuracy": retain_accuracy,
        "retain_acc": retain_accuracy,
        "base_forget_accuracy": base_forget_accuracy,
        "base_retain_accuracy": base_retain_accuracy,
        "delta_forget_accuracy": delta_forget,
        "delta_retain_accuracy": delta_retain,
        "relative_criteria": {
            "retain_tolerance": retain_tolerance,
            "collapse_retain_drop": collapse_retain_drop,
        },
        "legacy_absolute_thresholds": {
            "forget_accuracy_max": selective_forget_threshold,
            "retain_accuracy_min": selective_retain_threshold,
            "used_for_classification": False,
        },
        "paired_forget": (
            {key: value for key, value in paired_forget.items() if key != "details"}
            if paired_forget is not None else None
        ),
        "paired_retain": (
            {key: value for key, value in paired_retain.items() if key != "details"}
            if paired_retain is not None else None
        ),
        "classification": classification,
        "overall_verdict": verdict,
    }

    atomic_json_dump(result, report_dir / "behavioral_summary.json")
    atomic_json_dump(forget["rows"], report_dir / "forget_predictions.json")
    atomic_json_dump(retain["rows"], report_dir / "retain_predictions.json")

    if paired_forget is not None:
        atomic_json_dump(paired_forget["details"], report_dir / "forget_flip_details.json")
    if paired_retain is not None:
        atomic_json_dump(paired_retain["details"], report_dir / "retain_flip_details.json")

    lines = [
        "GRADDIFF BEHAVIOURAL CHECK",
        f"Checkpoint: {model_label}",
        f"ForgetAcc: {forget_accuracy:.4f}",
        f"ForgetRate: {forget_rate:.4f}",
        f"RetainAcc: {retain_accuracy:.4f}",
        f"Base ForgetAcc: {base_forget_accuracy:.4f}",
        f"Base RetainAcc: {base_retain_accuracy:.4f}",
        f"Delta ForgetAcc: {delta_forget:+.4f}",
        f"Delta RetainAcc: {delta_retain:+.4f}",
        f"Classification: {classification}",
    ]

    if paired_forget is not None:
        fc = paired_forget["counts"]
        lines.extend(
            [
                f"Successful forgetting flips: {fc['successful_forgetting']}",
                f"Unexpected recovery flips: {fc['unexpected_recovery']}",
            ]
        )
    if paired_retain is not None:
        rc = paired_retain["counts"]
        lines.extend(
            [
                f"Retain utility damage: {rc['utility_damage']}",
                f"Retain utility improvement: {rc['utility_improvement']}",
            ]
        )

    lines.append(f"OVERALL VERDICT: {verdict}")
    text_report = "\n".join(lines) + "\n"
    (report_dir / "behavioral_summary.txt").write_text(text_report, encoding="utf-8")

    print("=" * 88)
    print(text_report.strip())
    print(f"REPORT: {report_dir / 'behavioral_summary.txt'}")
    print(f"JSON: {report_dir / 'behavioral_summary.json'}")
    return result


# -----------------------------------------------------------------------------
# Adapter verification
# -----------------------------------------------------------------------------

def verify_adapter_file(adapter_file: Path) -> dict[str, Any]:
    from safetensors import safe_open

    total = 0
    nonzero = 0
    finite = True
    max_abs = 0.0
    with safe_open(str(adapter_file), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if "lora_B" not in key:
                continue
            tensor = handle.get_tensor(key)
            total += 1
            finite = finite and bool(torch.isfinite(tensor).all())
            current_max = float(tensor.abs().max().item())
            max_abs = max(max_abs, current_max)
            nonzero += int(current_max > 1e-8)
    verdict = "PASS" if total > 0 and nonzero > 0 and finite else "FAIL"
    return {
        "adapter_file": str(adapter_file),
        "lora_B_total": total,
        "lora_B_nonzero": nonzero,
        "finite": finite,
        "max_abs": max_abs,
        "verdict": verdict,
    }


def verify_checkpoints(output_root: Path) -> dict[str, Any]:
    adapter_files = sorted(output_root.rglob("adapter_model.safetensors")) if output_root.exists() else []
    results = [verify_adapter_file(path) for path in adapter_files]
    for row in results:
        print(
            f"[{row['verdict']}] {row['adapter_file']} | "
            f"lora_B={row['lora_B_nonzero']}/{row['lora_B_total']} "
            f"max_abs={row['max_abs']:.6g}"
        )
    overall = "PASS" if results and all(row["verdict"] == "PASS" for row in results) else "FAIL"
    report = {"checkpoints": results, "overall_verdict": overall}
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(report, REPORT_ROOT / "checkpoint_health.json")
    lines = ["GRADDIFF CHECKPOINT HEALTH"] + [
        f"[{row['verdict']}] {row['adapter_file']} lora_B={row['lora_B_nonzero']}/{row['lora_B_total']}"
        for row in results
    ] + [f"OVERALL VERDICT: {overall}"]
    (REPORT_ROOT / "checkpoint_health.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"OVERALL VERDICT: {overall}")
    return report


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust answer-only GradDiff training")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lambda_retain", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_steps", nargs="+", type=int, default=DEFAULT_SAVE_STEPS)
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--output_root", type=Path, default=GRADDIFF_ROOT)
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--check_ckpt", type=Path)
    parser.add_argument("--check_base", action="store_true", help="Evaluate the untouched base model with the identical evaluator")
    parser.add_argument("--eval_forget_n", type=int, default=0, help="0 means the full available split")
    parser.add_argument("--eval_retain_n", type=int, default=0, help="0 means the full available split")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--selective_forget_threshold", type=float, default=0.50)
    parser.add_argument("--selective_retain_threshold", type=float, default=0.80)
    parser.add_argument("--retain_tolerance", type=float, default=0.05, help="Allowed RetainAcc drop versus base")
    parser.add_argument("--collapse_retain_drop", type=float, default=0.20, help="RetainAcc drop versus base treated as severe collapse")
    parser.add_argument("--skip_final_eval", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_base and args.check_ckpt is not None:
        print("[FATAL] Use only one of --check_base or --check_ckpt")
        return 2
    try:
        if args.verify_only:
            report = verify_checkpoints(args.output_root)
            return 0 if report["overall_verdict"] == "PASS" else 1

        if args.check_base:
            behavioral_check(
                checkpoint=None,
                eval_forget_n=args.eval_forget_n,
                eval_retain_n=args.eval_retain_n,
                max_new_tokens=args.max_new_tokens,
                selective_forget_threshold=args.selective_forget_threshold,
                selective_retain_threshold=args.selective_retain_threshold,
                retain_tolerance=args.retain_tolerance,
                collapse_retain_drop=args.collapse_retain_drop,
            )
            return 0

        if args.check_ckpt is not None:
            behavioral_check(
                checkpoint=args.check_ckpt,
                eval_forget_n=args.eval_forget_n,
                eval_retain_n=args.eval_retain_n,
                max_new_tokens=args.max_new_tokens,
                selective_forget_threshold=args.selective_forget_threshold,
                selective_retain_threshold=args.selective_retain_threshold,
                retain_tolerance=args.retain_tolerance,
                collapse_retain_drop=args.collapse_retain_drop,
            )
            return 0

        final_checkpoint = train(
            max_steps=args.steps,
            lambda_retain=args.lambda_retain,
            learning_rate=args.lr,
            seed=args.seed,
            save_steps=args.save_steps,
            max_grad_norm=args.max_grad_norm,
            max_retries=args.max_retries,
            output_root=args.output_root,
        )

        if not args.skip_final_eval:
            behavioral_check(
                checkpoint=final_checkpoint,
                eval_forget_n=args.eval_forget_n,
                eval_retain_n=args.eval_retain_n,
                max_new_tokens=args.max_new_tokens,
                selective_forget_threshold=args.selective_forget_threshold,
                selective_retain_threshold=args.selective_retain_threshold,
                retain_tolerance=args.retain_tolerance,
                collapse_retain_drop=args.collapse_retain_drop,
            )

        print("\n[NEXT] Add the exact selected checkpoint path to exp_config.py only after evaluation.")
        print(f"[NEXT] Final checkpoint: {final_checkpoint}")
        return 0
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
