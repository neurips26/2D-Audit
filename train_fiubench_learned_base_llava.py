#!/usr/bin/env python3
"""
Train and validate a FIUBench learned/base LLaVA checkpoint.

Scientific role
---------------
FIUBench contains fictitious identities. The original LLaVA model does not know
them, so the correct protocol is:

  1. Learn the FIUBench identities.
  2. Use this learned checkpoint as M0 for FIUBench.
  3. Derive FIUBench-specific unlearned checkpoints from M0.
  4. Compare each unlearned checkpoint against M0 for behavior and CRP.

This script trains a LoRA adapter on the official full_data.json JSONL records
and evaluates the resulting learned checkpoint on the official forget10 and
retain10 subsets prepared earlier.

Key safeguards
--------------
- Reads ordinary JSON or JSONL/NDJSON.
- Uses official `unique` IDs and image filenames.
- Trains on answer tokens only; prompt/image tokens are masked.
- Deterministic seed and deterministic per-epoch order.
- Gradient accumulation and clipping.
- Bounded retries.
- LoRA-B gradient and checkpoint health checks.
- Resumable training from the saved adapter.
- Identity-balanced evaluation (N questions per identity).
- Machine-readable TXT/JSON reports.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_config import (
    DEVICE,
    LLAVA_BASE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    MAX_NEW_TOKENS,
    ROOT,
)

DEFAULT_SEED = 42
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 1
DEFAULT_GRAD_ACCUM = 4
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_MAX_RETRIES = 30
DEFAULT_EVAL_QAS_PER_IDENTITY = 1

FIU_ROOT = ROOT / "data" / "fiubench_official"
SNAPSHOT = FIU_ROOT / "official_snapshot"
FULL_DATA = SNAPSHOT / "full_data.json"
PREPARED = FIU_ROOT / "prepared_forget10"
FORGET_ANN = PREPARED / "forget" / "annotations.json"
RETAIN_ANN = PREPARED / "retain" / "annotations.json"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "fiubench_learned_base"
RESULT_ROOT = ROOT / "outputs" / "revision" / "fiubench_learned_base"


def atomic_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def atomic_text(lines: Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows: list[Any] = []
        for line_number, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Could not parse {path} as JSONL at line {line_number}: {exc}"
                ) from exc
        if not rows:
            raise ValueError(f"No JSON/JSONL records found in {path}")
        return rows


def flatten_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [dict(x) for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("data", "items", "records", "examples", "dataset"):
            value = data.get(key)
            if isinstance(value, list):
                return [dict(x) for x in value if isinstance(x, dict)]
        rows: list[dict[str, Any]] = []
        for key, value in data.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("_source_key", key)
                rows.append(row)
        if rows:
            return rows
    raise ValueError("Unsupported FIUBench record structure.")


def normalize_text(text: Any) -> str:
    value = str(text or "").casefold()
    value = re.sub(r"[^a-z0-9$+.\- ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def clean_answer(value: Any) -> str:
    text = str(value or "").strip()
    # FIUBench contains variants such as "The **a: ...**".
    text = re.sub(r"^\s*the\s+", "", text, flags=re.I)
    text = re.sub(r"^\s*\*{0,2}\s*a\s*:\s*", "", text, flags=re.I)
    text = text.replace("**", "").strip()
    return text


def find_image_root() -> Path:
    candidates = [
        SNAPSHOT / "SFHQ",
        FIU_ROOT / "sfhq_part1_download",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "SFHQ image directory not found. Expected either "
        f"{candidates[0]} or {candidates[1]}"
    )


def build_image_index(image_root: Path) -> dict[str, Path]:
    print(f"[data] Indexing images under: {image_root}")
    index: dict[str, Path] = {}
    count = 0
    for path in image_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.casefold() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue
        count += 1
        index.setdefault(path.name.casefold(), path.resolve())
        index.setdefault(path.stem.casefold(), path.resolve())
    print(f"[PASS] Indexed {count} image files")
    return index


def resolve_record_image(row: dict[str, Any], image_index: dict[str, Path]) -> Path | None:
    raw = str(row.get("image_path", row.get("image", ""))).replace("\\", "/").strip()
    if raw:
        path = Path(raw)
        for key in (path.name.casefold(), path.stem.casefold()):
            if key in image_index:
                return image_index[key]

    unique = str(row.get("unique", "")).strip()
    if unique:
        for stem in (f"sfhq_pt1_{unique}", unique):
            if stem.casefold() in image_index:
                return image_index[stem.casefold()]
    return None


def choose_qas(
    qa_list: list[dict[str, Any]],
    max_qas_per_identity: int,
) -> list[dict[str, Any]]:
    valid = [
        qa for qa in qa_list
        if isinstance(qa, dict)
        and str(qa.get("question", "")).strip()
        and str(qa.get("answer", "")).strip()
    ]
    if max_qas_per_identity <= 0 or len(valid) <= max_qas_per_identity:
        return valid

    # Preserve broad coverage by selecting evenly across the identity's QA list.
    indices = np.linspace(
        0,
        len(valid) - 1,
        num=max_qas_per_identity,
        dtype=int,
    )
    return [valid[int(i)] for i in indices]


def load_training_items(
    max_qas_per_identity: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not FULL_DATA.exists():
        raise FileNotFoundError(f"Missing official training metadata: {FULL_DATA}")

    records = flatten_records(load_json_or_jsonl(FULL_DATA))
    image_root = find_image_root()
    image_index = build_image_index(image_root)

    items: list[dict[str, Any]] = []
    missing_images: list[str] = []
    identities_without_qa: list[str] = []

    for row in records:
        unique = str(row.get("unique", "")).strip()
        entity = str(row.get("name", unique or "unknown")).strip()
        image = resolve_record_image(row, image_index)
        if image is None:
            missing_images.append(unique or entity)
            continue

        qa_list = row.get("qa_list", [])
        if not isinstance(qa_list, list):
            identities_without_qa.append(unique or entity)
            continue

        selected_qas = choose_qas(qa_list, max_qas_per_identity)
        if not selected_qas:
            identities_without_qa.append(unique or entity)
            continue

        for qa_index, qa in enumerate(selected_qas):
            keywords = qa.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = [keywords]

            items.append(
                {
                    "unique": unique,
                    "entity": entity,
                    "image": image,
                    "question": str(qa["question"]).strip(),
                    "answer": clean_answer(qa["answer"]),
                    "keywords": [str(x).strip() for x in keywords if str(x).strip()],
                    "qa_index": qa_index,
                }
            )

    report = {
        "source_records": len(records),
        "training_items": len(items),
        "unique_identities": len({x["unique"] for x in items}),
        "missing_image_identities": missing_images,
        "identities_without_qa": identities_without_qa,
        "max_qas_per_identity": max_qas_per_identity,
    }
    return items, report


def get_bnb_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def build_new_model():
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    print("[model] Loading original LLaVA base...")
    processor = AutoProcessor.from_pretrained(LLAVA_BASE, use_fast=False)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_BASE,
        quantization_config=get_bnb_config(),
        device_map=DEVICE,
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=False,
    )

    lora = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, processor


def load_adapter_for_training(checkpoint: Path):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from peft import PeftModel, prepare_model_for_kbit_training

    print(f"[model] Resuming FIUBench adapter: {checkpoint}")
    processor = AutoProcessor.from_pretrained(checkpoint, use_fast=False)
    base = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_BASE,
        quantization_config=get_bnb_config(),
        device_map=DEVICE,
    )
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=False,
    )
    model = PeftModel.from_pretrained(
        base,
        str(checkpoint),
        is_trainable=True,
    )
    model.print_trainable_parameters()
    return model, processor


def load_adapter_for_eval(checkpoint: Path):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(checkpoint, use_fast=False)
    base = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_BASE,
        quantization_config=get_bnb_config(),
        device_map=DEVICE,
    )
    model = PeftModel.from_pretrained(base, str(checkpoint))
    model.eval()
    return model, processor


def build_answer_only_batch(
    processor,
    image: Image.Image,
    question: str,
    answer: str,
) -> dict[str, torch.Tensor]:
    prefix = f"USER: <image>\n{question} ASSISTANT:"
    full = f"{prefix} {answer}"

    full_inputs = processor(
        text=full,
        images=image,
        return_tensors="pt",
    ).to(DEVICE)

    prefix_inputs = processor(
        text=prefix,
        images=image,
        return_tensors="pt",
    ).to(DEVICE)

    labels = full_inputs["input_ids"].clone()
    prefix_len = int(prefix_inputs["input_ids"].shape[1])
    labels[:, :prefix_len] = -100

    if not torch.any(labels != -100):
        raise RuntimeError("Answer-only masking removed all supervised tokens.")

    full_inputs["labels"] = labels
    return full_inputs


def lora_health(model) -> dict[str, Any]:
    total = nonzero = 0
    maximum = 0.0

    for name, param in model.named_parameters():
        if "lora_B" not in name:
            continue
        total += 1
        value = float(param.detach().abs().max().item())
        maximum = max(maximum, value)
        nonzero += int(value > 1e-8)

    return {
        "total_lora_B": total,
        "nonzero_lora_B": nonzero,
        "max_abs_lora_B": maximum,
        "trained": total > 0 and nonzero > 0,
    }


def checkpoint_health(checkpoint: Path) -> dict[str, Any]:
    from safetensors import safe_open

    model_file = checkpoint / "adapter_model.safetensors"
    if not model_file.exists():
        return {
            "checkpoint": str(checkpoint),
            "exists": False,
            "total_lora_B": 0,
            "nonzero_lora_B": 0,
            "trained": False,
        }

    total = nonzero = 0
    maximum = 0.0

    with safe_open(str(model_file), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if "lora_B" not in key:
                continue
            total += 1
            value = float(handle.get_tensor(key).abs().max().item())
            maximum = max(maximum, value)
            nonzero += int(value > 1e-8)

    return {
        "checkpoint": str(checkpoint),
        "exists": True,
        "total_lora_B": total,
        "nonzero_lora_B": nonzero,
        "max_abs_lora_B": maximum,
        "trained": total > 0 and nonzero > 0,
    }


def save_checkpoint(
    model,
    processor,
    checkpoint: Path,
    metadata: dict[str, Any],
) -> None:
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint))
    processor.save_pretrained(str(checkpoint))
    atomic_json(metadata, checkpoint / "training_meta.json")


def train(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    set_all_seeds(args.seed)

    items, data_report = load_training_items(args.max_qas_per_identity)
    if not items:
        raise RuntimeError("No FIUBench training items were constructed.")
    if data_report["missing_image_identities"]:
        raise RuntimeError(
            f"Missing images for {len(data_report['missing_image_identities'])} identities."
        )

    run_tag = (
        f"epochs{args.epochs}_q{args.max_qas_per_identity}_"
        f"lr{str(args.lr).replace('.', 'p').replace('-', 'm')}_seed{args.seed}"
    )
    run_dir = CHECKPOINT_ROOT / run_tag
    final_checkpoint = run_dir / "final"
    report_dir = RESULT_ROOT / run_tag
    report_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_from:
        model, processor = load_adapter_for_training(Path(args.resume_from))
    else:
        model, processor = build_new_model()

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    model.train()

    total_micro_steps = len(items) * args.epochs
    expected_optimizer_steps = math.ceil(total_micro_steps / args.grad_accum)
    print("=" * 96)
    print("FIUBENCH LEARNED-BASE TRAINING")
    print(f"Identities: {data_report['unique_identities']}")
    print(f"Training QA items: {len(items)}")
    print(f"Epochs: {args.epochs}")
    print(f"Gradient accumulation: {args.grad_accum}")
    print(f"Expected optimizer steps: {expected_optimizer_steps}")
    print(f"Output: {run_dir}")
    print("=" * 96)

    log: list[dict[str, Any]] = []
    optimizer_step = 0
    micro_step = 0
    retries = 0
    grad_verified = False
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        order = list(range(len(items)))
        random.Random(args.seed + epoch).shuffle(order)

        for position, item_index in enumerate(order):
            item = items[item_index]

            try:
                image = Image.open(item["image"]).convert("RGB")
                batch = build_answer_only_batch(
                    processor,
                    image,
                    item["question"],
                    item["answer"],
                )

                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss: {float(loss.item())}")

                scaled_loss = loss / args.grad_accum
                scaled_loss.backward()
                micro_step += 1
                retries = 0

                should_step = (
                    micro_step % args.grad_accum == 0
                    or position == len(order) - 1
                )

                grad_norm = None
                if should_step:
                    if not grad_verified:
                        has_gradient = any(
                            p.grad is not None
                            and torch.isfinite(p.grad).all()
                            and float(p.grad.detach().abs().max().item()) > 1e-10
                            for name, p in model.named_parameters()
                            if "lora_B" in name and p.requires_grad
                        )
                        if not has_gradient:
                            raise RuntimeError("No valid gradient reached LoRA-B.")
                        print("[PASS] LoRA-B gradient flow verified")
                        grad_verified = True

                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            trainable,
                            args.max_grad_norm,
                        ).item()
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

                    if (
                        optimizer_step == 1
                        or optimizer_step % args.log_every == 0
                        or optimizer_step == expected_optimizer_steps
                    ):
                        elapsed = time.time() - start_time
                        rate = optimizer_step / max(elapsed, 1e-9)
                        remaining = max(expected_optimizer_steps - optimizer_step, 0)
                        eta_seconds = remaining / max(rate, 1e-9)
                        print(
                            f"epoch={epoch + 1}/{args.epochs} "
                            f"opt_step={optimizer_step}/{expected_optimizer_steps} "
                            f"loss={float(loss.item()):.5f} "
                            f"grad_norm={grad_norm:.4f} "
                            f"eta_min={eta_seconds / 60:.1f}"
                        )

                    if (
                        args.save_every > 0
                        and optimizer_step % args.save_every == 0
                    ):
                        checkpoint = run_dir / f"step_{optimizer_step}"
                        metadata = {
                            "method": "fiubench_learned_base",
                            "base_model": LLAVA_BASE,
                            "optimizer_step": optimizer_step,
                            "micro_step": micro_step,
                            "epoch": epoch + 1,
                            "seed": args.seed,
                            "learning_rate": args.lr,
                            "gradient_accumulation": args.grad_accum,
                            "max_qas_per_identity": args.max_qas_per_identity,
                            "training_items": len(items),
                            "identities": data_report["unique_identities"],
                        }
                        save_checkpoint(model, processor, checkpoint, metadata)
                        print(f"[PASS] Saved intermediate checkpoint: {checkpoint}")

                log.append(
                    {
                        "epoch": epoch + 1,
                        "micro_step": micro_step,
                        "optimizer_step": optimizer_step,
                        "loss": float(loss.item()),
                        "grad_norm": grad_norm,
                        "unique": item["unique"],
                        "entity": item["entity"],
                        "qa_index": item["qa_index"],
                    }
                )

                del batch, loss, scaled_loss, image

            except Exception as exc:
                retries += 1
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(
                    f"[WARN] epoch={epoch + 1} item={position + 1}/{len(order)} "
                    f"retry={retries}/{args.max_retries}: {exc}"
                )
                if retries >= args.max_retries:
                    raise RuntimeError(
                        "Maximum consecutive training retries reached."
                    ) from exc

    final_health = lora_health(model)
    metadata = {
        "method": "fiubench_learned_base",
        "role": "M0_for_fiubench_unlearning",
        "base_model": LLAVA_BASE,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "gradient_accumulation": args.grad_accum,
        "max_grad_norm": args.max_grad_norm,
        "max_qas_per_identity": args.max_qas_per_identity,
        "optimizer_steps": optimizer_step,
        "micro_steps": micro_step,
        "training_items": len(items),
        "training_identities": data_report["unique_identities"],
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "answer_only_masking": True,
        "health": final_health,
    }
    save_checkpoint(model, processor, final_checkpoint, metadata)

    atomic_json(log, report_dir / "training_log.json")
    training_report = {
        "dataset": "FIUBench",
        "data": data_report,
        "training": metadata,
        "final_checkpoint": str(final_checkpoint),
        "elapsed_seconds": time.time() - start_time,
        "overall_verdict": "PASS" if final_health["trained"] else "FAIL",
    }
    atomic_json(training_report, report_dir / "training_report.json")
    atomic_text(
        [
            "FIUBENCH LEARNED-BASE TRAINING REPORT",
            "=" * 88,
            f"Training identities: {data_report['unique_identities']}",
            f"Training QA items: {len(items)}",
            f"Epochs: {args.epochs}",
            f"Optimizer steps: {optimizer_step}",
            f"LoRA-B nonzero: {final_health['nonzero_lora_B']}/{final_health['total_lora_B']}",
            f"Final checkpoint: {final_checkpoint}",
            f"OVERALL VERDICT: {training_report['overall_verdict']}",
        ],
        report_dir / "training_report.txt",
    )

    print(f"[PASS] Final checkpoint saved: {final_checkpoint}")
    print(f"REPORT: {report_dir / 'training_report.txt'}")
    print(f"JSON: {report_dir / 'training_report.json'}")
    print(f"OVERALL VERDICT: {training_report['overall_verdict']}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_checkpoint, training_report


def load_prepared_annotations(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prepared annotation file missing: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    valid = []
    for row in rows:
        image = Path(row["image"])
        if image.exists():
            item = dict(row)
            item["image"] = image
            valid.append(item)
    return valid


def identity_balanced_subset(
    items: list[dict[str, Any]],
    qas_per_identity: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item.get("entity", "unknown"))].append(item)

    selected: list[dict[str, Any]] = []
    for entity in sorted(grouped):
        group = grouped[entity]
        if qas_per_identity <= 0 or len(group) <= qas_per_identity:
            selected.extend(group)
            continue

        # Prefer full-name questions for q=1; otherwise spread across QA types.
        if qas_per_identity == 1:
            name_candidates = [
                x for x in group
                if "name" in normalize_text(x.get("question", ""))
            ]
            selected.append(name_candidates[0] if name_candidates else group[0])
        else:
            indices = np.linspace(
                0,
                len(group) - 1,
                num=qas_per_identity,
                dtype=int,
            )
            selected.extend(group[int(i)] for i in indices)
    return selected


def decode_new_tokens(
    processor,
    generated: torch.Tensor,
    prompt_length: int,
) -> str:
    tokens = generated[0, prompt_length:]
    return processor.decode(tokens, skip_special_tokens=True).strip()


def expected_terms(item: dict[str, Any]) -> list[str]:
    terms: list[str] = []

    keywords = item.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = [keywords]
    terms.extend(str(x) for x in keywords if str(x).strip())

    # Entity is a useful alias for identity-name questions.
    entity = str(item.get("entity", "")).strip()
    if entity:
        terms.append(entity)

    # Full answer is a fallback, but keywords are preferred.
    answer = str(item.get("answer", "")).strip()
    if answer:
        terms.append(answer)

    unique: list[str] = []
    seen = set()
    for term in terms:
        normalized = normalize_text(term)
        if normalized and normalized not in seen:
            unique.append(normalized)
            seen.add(normalized)
    return unique


def score_response(response: str, item: dict[str, Any]) -> bool:
    normalized_response = normalize_text(response)
    if not normalized_response:
        return False

    for term in expected_terms(item):
        # Avoid accepting extremely generic one-character terms.
        if len(term) >= 2 and term in normalized_response:
            return True
    return False


def evaluate_split(
    model,
    processor,
    items: list[dict[str, Any]],
    split_name: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    correct = 0

    for index, item in enumerate(items, start=1):
        image = Image.open(item["image"]).convert("RGB")
        prompt = f"USER: <image>\n{item['question']} ASSISTANT:"
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        prompt_length = int(inputs["input_ids"].shape[1])
        response = decode_new_tokens(processor, generated, prompt_length)
        is_correct = score_response(response, item)
        correct += int(is_correct)

        outputs.append(
            {
                "split": split_name,
                "entity": item.get("entity"),
                "question": item.get("question"),
                "answer": item.get("answer"),
                "keywords": item.get("keywords", []),
                "response": response,
                "correct": is_correct,
            }
        )

        if index == 1 or index % 10 == 0 or index == len(items):
            print(
                f"[eval:{split_name}] {index}/{len(items)} "
                f"correct={correct} accuracy={correct / index:.4f}"
            )

        del inputs, generated, image

    accuracy = correct / len(items) if items else float("nan")
    return {
        "split": split_name,
        "items": len(items),
        "correct": correct,
        "accuracy": accuracy,
        "outputs": outputs,
    }


def evaluate(
    checkpoint: Path,
    qas_per_identity: int,
    max_new_tokens: int,
    min_accuracy: float,
) -> dict[str, Any]:
    health = checkpoint_health(checkpoint)
    if not health["trained"]:
        raise RuntimeError(f"Checkpoint failed LoRA health verification: {health}")

    forget_all = load_prepared_annotations(FORGET_ANN)
    retain_all = load_prepared_annotations(RETAIN_ANN)
    forget = identity_balanced_subset(forget_all, qas_per_identity)
    retain = identity_balanced_subset(retain_all, qas_per_identity)

    print("=" * 96)
    print("FIUBENCH LEARNED-BASE EVALUATION")
    print(f"Checkpoint: {checkpoint}")
    print(f"Forget identities/items: {len({x['entity'] for x in forget})}/{len(forget)}")
    print(f"Retain identities/items: {len({x['entity'] for x in retain})}/{len(retain)}")
    print(f"Minimum required accuracy per split: {min_accuracy:.4f}")
    print("=" * 96)

    model, processor = load_adapter_for_eval(checkpoint)
    forget_result = evaluate_split(
        model,
        processor,
        forget,
        "forget10",
        max_new_tokens,
    )
    retain_result = evaluate_split(
        model,
        processor,
        retain,
        "retain10",
        max_new_tokens,
    )

    overall_accuracy = (
        forget_result["correct"] + retain_result["correct"]
    ) / (
        forget_result["items"] + retain_result["items"]
    )

    gate_pass = (
        forget_result["accuracy"] >= min_accuracy
        and retain_result["accuracy"] >= min_accuracy
    )

    report = {
        "dataset": "FIUBench",
        "role": "learned_base_gate",
        "checkpoint": str(checkpoint),
        "checkpoint_health": health,
        "qas_per_identity": qas_per_identity,
        "max_new_tokens": max_new_tokens,
        "minimum_split_accuracy": min_accuracy,
        "forget10": {
            key: value for key, value in forget_result.items()
            if key != "outputs"
        },
        "retain10": {
            key: value for key, value in retain_result.items()
            if key != "outputs"
        },
        "overall_accuracy": overall_accuracy,
        "scientific_gate_pass": gate_pass,
        "overall_verdict": "PASS" if gate_pass else "WARN",
    }

    run_name = checkpoint.parent.name if checkpoint.name == "final" else checkpoint.name
    report_dir = RESULT_ROOT / run_name
    atomic_json(report, report_dir / "evaluation_summary.json")
    atomic_json(
        forget_result["outputs"],
        report_dir / "forget10_outputs.json",
    )
    atomic_json(
        retain_result["outputs"],
        report_dir / "retain10_outputs.json",
    )
    atomic_text(
        [
            "FIUBENCH LEARNED-BASE EVALUATION SUMMARY",
            "=" * 88,
            f"Checkpoint: {checkpoint}",
            f"Questions per identity: {qas_per_identity}",
            f"Forget10 accuracy: {forget_result['accuracy']:.4f} "
            f"({forget_result['correct']}/{forget_result['items']})",
            f"Retain10 accuracy: {retain_result['accuracy']:.4f} "
            f"({retain_result['correct']}/{retain_result['items']})",
            f"Overall accuracy: {overall_accuracy:.4f}",
            f"Required split accuracy: {min_accuracy:.4f}",
            f"SCIENTIFIC GATE: {'PASS' if gate_pass else 'NOT YET PASSED'}",
            f"OVERALL VERDICT: {report['overall_verdict']}",
        ],
        report_dir / "evaluation_summary.txt",
    )

    print("")
    print(f"Forget10 accuracy: {forget_result['accuracy']:.4f}")
    print(f"Retain10 accuracy: {retain_result['accuracy']:.4f}")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print(f"SCIENTIFIC GATE: {'PASS' if gate_pass else 'NOT YET PASSED'}")
    print(f"REPORT: {report_dir / 'evaluation_summary.txt'}")
    print(f"JSON: {report_dir / 'evaluation_summary.json'}")
    print(f"OVERALL VERDICT: {report['overall_verdict']}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("train", "eval", "train_eval", "verify"),
        default="train_eval",
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--max_qas_per_identity", type=int, default=20)
    parser.add_argument("--grad_accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULT_MAX_GRAD_NORM)
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument(
        "--eval_qas_per_identity",
        type=int,
        default=DEFAULT_EVAL_QAS_PER_IDENTITY,
        help="0 evaluates every prepared question.",
    )
    parser.add_argument("--eval_max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument(
        "--min_accuracy",
        type=float,
        default=0.60,
        help="Minimum required accuracy on each official split.",
    )
    args = parser.parse_args()

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.mode == "verify":
        if not args.checkpoint:
            print("[FAIL] --checkpoint is required for verify mode.")
            return 1
        result = checkpoint_health(Path(args.checkpoint))
        print(json.dumps(result, indent=2))
        print(f"OVERALL VERDICT: {'PASS' if result['trained'] else 'FAIL'}")
        return 0 if result["trained"] else 1

    checkpoint: Path | None = None

    if args.mode in {"train", "train_eval"}:
        checkpoint, training_report = train(args)
        if training_report["overall_verdict"] != "PASS":
            return 1

    if args.mode in {"eval", "train_eval"}:
        if checkpoint is None:
            if not args.checkpoint:
                print("[FAIL] --checkpoint is required for eval mode.")
                return 1
            checkpoint = Path(args.checkpoint)

        report = evaluate(
            checkpoint=checkpoint,
            qas_per_identity=args.eval_qas_per_identity,
            max_new_tokens=args.eval_max_new_tokens,
            min_accuracy=args.min_accuracy,
        )

        # WARN means training/evaluation completed but the learned-base gate is
        # not strong enough yet. Return 2 so PowerShell can distinguish it.
        return 0 if report["overall_verdict"] == "PASS" else 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
