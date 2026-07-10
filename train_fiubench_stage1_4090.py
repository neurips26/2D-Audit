#!/usr/bin/env python3
"""
24-GB-safe FIUBench Stage-I learned-base training.

Protocol alignment:
- Uses the prepared official400 manifest only.
- Seed 233, two epochs, LR 1e-5, weight decay 0.01.
- Frozen vision tower.
- Trainable multimodal projector.
- 4-bit QLoRA for the language model because a 24 GB RTX 4090 cannot reproduce
  the official full 11B language-model fine-tuning configuration.

This is a resource-constrained reproduction, not an exact official full-FT run.

Outputs include:
- adapter weights
- separately saved projector weights
- training reports
- identity evaluation
- mismatched-image and blank-image controls
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
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file, load_file

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from exp_config import DEVICE, LLAVA_BASE, ROOT

DATA_DIR = ROOT / "data" / "fiubench_official" / "prepared_official400"
REPORT_PATH = DATA_DIR / "preparation_report.json"
TRAIN_PATH = DATA_DIR / "full_train.jsonl"
FORGET_PATH = DATA_DIR / "forget10.jsonl"
RETAIN_PATH = DATA_DIR / "retain15.jsonl"
CKPT_ROOT = ROOT / "checkpoints" / "fiubench_stage1_4090"
OUT_ROOT = ROOT / "outputs" / "revision" / "fiubench_stage1_4090"


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {line_no}: {exc}") from exc
    return rows


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(text: Any) -> str:
    value = str(text or "").casefold()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def is_name_question(question: str) -> bool:
    q = normalize(question)
    return any(x in q for x in (
        "full name", "what is the name", "who is", "person name", "person s name"
    ))


def flatten_training_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for row in records:
        for qa in row.get("qa_list", []):
            q = str(qa.get("question", "")).strip()
            a = str(qa.get("answer", "")).strip()
            if not q or not a:
                continue
            items.append({
                "unique": row["unique"],
                "name": row.get("name", ""),
                "image": row["image"],
                "question": q,
                "answer": a,
            })
    return items


def bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def find_projector_modules(model) -> list[tuple[str, torch.nn.Parameter]]:
    found = []
    keywords = ("multi_modal_projector", "mm_projector", "language_projection")
    for name, param in model.named_parameters():
        if any(k in name for k in keywords):
            found.append((name, param))
    return found


def build_model(rank: int, alpha: int, dropout: float):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    processor = AutoProcessor.from_pretrained(LLAVA_BASE, use_fast=False)
    base = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_BASE,
        quantization_config=bnb_config(),
        device_map=DEVICE,
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(base, config)

    projector = find_projector_modules(model)
    if not projector:
        raise RuntimeError("No multimodal projector parameters were found.")
    for _, param in projector:
        param.requires_grad = True

    # Explicitly keep the vision tower frozen.
    for name, param in model.named_parameters():
        if "vision_tower" in name or "vision_model" in name:
            param.requires_grad = False

    return model, processor, [name for name, _ in projector]


def answer_only_batch(processor, image: Image.Image, question: str, answer: str):
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
        raise RuntimeError("Answer masking removed all supervised tokens.")
    full_inputs["labels"] = labels
    return full_inputs


def save_projector(model, path: Path) -> dict[str, Any]:
    state = {}
    for name, param in model.named_parameters():
        if any(k in name for k in (
            "multi_modal_projector", "mm_projector", "language_projection"
        )):
            state[name] = param.detach().cpu().contiguous()
    if not state:
        raise RuntimeError("No projector state found to save.")
    save_file(state, str(path))
    return {
        "file": str(path),
        "tensor_count": len(state),
        "nonzero_tensors": sum(
            int(float(t.abs().max().item()) > 0.0) for t in state.values()
        ),
    }


def load_projector(model, path: Path) -> None:
    state = load_file(str(path))
    params = dict(model.named_parameters())
    missing = []
    for name, tensor in state.items():
        if name not in params:
            missing.append(name)
            continue
        params[name].data.copy_(tensor.to(params[name].device, dtype=params[name].dtype))
    if missing:
        raise RuntimeError(f"Projector tensors not found in model: {missing[:5]}")


def checkpoint_health(model) -> dict[str, Any]:
    lora_total = lora_nonzero = 0
    projector_total = projector_nonzero = 0
    for name, param in model.named_parameters():
        max_abs = float(param.detach().abs().max().item())
        if "lora_B" in name:
            lora_total += 1
            lora_nonzero += int(max_abs > 1e-8)
        if any(k in name for k in (
            "multi_modal_projector", "mm_projector", "language_projection"
        )):
            projector_total += 1
            projector_nonzero += int(max_abs > 1e-8)
    return {
        "lora_B_total": lora_total,
        "lora_B_nonzero": lora_nonzero,
        "projector_tensor_total": projector_total,
        "projector_tensor_nonzero": projector_nonzero,
        "trained": (
            lora_total > 0
            and lora_nonzero > 0
            and projector_total > 0
            and projector_nonzero > 0
        ),
    }


def identity_subset(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in records:
        name_qas = [
            qa for qa in row.get("qa_list", [])
            if is_name_question(str(qa.get("question", "")))
        ]
        qa = name_qas[0] if name_qas else (
            row.get("qa_list", [{}])[0] if row.get("qa_list") else {}
        )
        selected.append({
            "unique": row["unique"],
            "name": row.get("name", ""),
            "image": row["image"],
            "question": qa.get(
                "question",
                "What is the full name of the person in this image?"
            ),
            "answer": qa.get("answer", row.get("name", "")),
        })
    return selected


def exact_name_score(response: str, name: str) -> bool:
    rn, nn = normalize(response), normalize(name)
    return bool(nn) and nn in rn


def generate_response(model, processor, item, image_path: str, max_new_tokens: int):
    image = Image.open(image_path).convert("RGB")
    prompt = f"USER: <image>\n{item['question']} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    prompt_len = int(inputs["input_ids"].shape[1])
    response = processor.decode(
        output[0, prompt_len:],
        skip_special_tokens=True,
    ).strip()
    return response


def evaluate_control(
    model,
    processor,
    items: list[dict[str, Any]],
    mode: str,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    image_paths = [x["image"] for x in items]
    outputs = []
    correct = 0

    for i, item in enumerate(items):
        if mode == "correct":
            image_path = item["image"]
        elif mode == "mismatched":
            choices = [
                p for p in image_paths if p != item["image"]
            ]
            image_path = rng.choice(choices)
        elif mode == "blank":
            source = Image.open(item["image"]).convert("RGB")
            blank_path = OUT_ROOT / "_blank_control.png"
            if not blank_path.exists():
                Image.new("RGB", source.size, (0, 0, 0)).save(blank_path)
            image_path = str(blank_path)
        else:
            raise ValueError(mode)

        response = generate_response(
            model, processor, item, image_path, max_new_tokens
        )
        ok = exact_name_score(response, item["name"])
        correct += int(ok)
        outputs.append({
            "mode": mode,
            "unique": item["unique"],
            "name": item["name"],
            "question": item["question"],
            "image_used": image_path,
            "response": response,
            "correct": ok,
        })
        if (i + 1) % 10 == 0 or i + 1 == len(items):
            print(
                f"[eval:{mode}] {i+1}/{len(items)} "
                f"accuracy={correct/(i+1):.4f}"
            )

    return {
        "mode": mode,
        "items": len(items),
        "correct": correct,
        "accuracy": correct / len(items) if items else float("nan"),
        "outputs": outputs,
    }


def verify_data_gate() -> dict[str, Any]:
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing preparation report: {REPORT_PATH}")
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    required = (
        report.get("overall_verdict") == "PASS"
        and report.get("official400_count") == 400
        and report.get("forget10_count") == 40
        and report.get("retain15_count") == 60
        and not report.get("missing_images")
    )
    if not required:
        raise RuntimeError(
            "Official400 data gate failed. Run "
            ".\\run_prepare_fiubench_official400.ps1 first."
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=12)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--eval_max_new_tokens", type=int, default=32)
    parser.add_argument("--min_name_accuracy", type=float, default=0.60)
    parser.add_argument("--max_mismatch_accuracy", type=float, default=0.30)
    args = parser.parse_args()

    set_seed(args.seed)
    data_report = verify_data_gate()

    records = read_jsonl(TRAIN_PATH)
    forget_records = read_jsonl(FORGET_PATH)
    retain_records = read_jsonl(RETAIN_PATH)
    items = flatten_training_items(records)
    if not items:
        raise RuntimeError("No training items.")

    run_tag = (
        f"ep{args.epochs}_lr{str(args.lr).replace('.', 'p').replace('-', 'm')}"
        f"_r{args.lora_rank}_seed{args.seed}"
    )
    ckpt = CKPT_ROOT / run_tag / "final"
    out = OUT_ROOT / run_tag
    ckpt.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    model, processor, projector_names = build_model(
        args.lora_rank,
        args.lora_alpha,
        args.lora_dropout,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_micro = len(items) * args.epochs
    total_updates = math.ceil(total_micro / args.grad_accum)
    print("=" * 96)
    print("FIUBENCH STAGE-I 4090 RESOURCE-CONSTRAINED REPRODUCTION")
    print(f"Base: {LLAVA_BASE}")
    print(f"Identities: {len(records)}")
    print(f"Training QA: {len(items)}")
    print(f"Epochs: {args.epochs}")
    print(f"Effective batch: {args.batch_size * args.grad_accum}")
    print(f"Expected optimizer updates: {total_updates}")
    print(f"Projector tensors trainable: {len(projector_names)}")
    print("=" * 96)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro = update = 0
    logs = []
    start = time.time()

    for epoch in range(args.epochs):
        order = list(range(len(items)))
        random.Random(args.seed + epoch).shuffle(order)

        for pos, idx in enumerate(order):
            item = items[idx]
            image = Image.open(item["image"]).convert("RGB")
            batch = answer_only_batch(
                processor, image, item["question"], item["answer"]
            )
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at micro-step {micro+1}")

            (loss / args.grad_accum).backward()
            micro += 1
            should_step = (
                micro % args.grad_accum == 0
                or pos == len(order) - 1
            )
            grad_norm = None
            if should_step:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    trainable, args.max_grad_norm
                ).item())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1

                if update == 1 or update % args.log_every == 0:
                    elapsed = time.time() - start
                    eta = (
                        (total_updates - update)
                        / max(update / max(elapsed, 1e-9), 1e-9)
                    )
                    print(
                        f"epoch={epoch+1}/{args.epochs} "
                        f"update={update}/{total_updates} "
                        f"loss={float(loss.item()):.5f} "
                        f"grad_norm={grad_norm:.4f} "
                        f"eta_min={eta/60:.1f}"
                    )

            logs.append({
                "epoch": epoch + 1,
                "micro_step": micro,
                "optimizer_step": update,
                "loss": float(loss.item()),
                "grad_norm": grad_norm,
                "unique": item["unique"],
            })
            del batch, loss, image

    health = checkpoint_health(model)
    model.save_pretrained(str(ckpt))
    processor.save_pretrained(str(ckpt))
    projector_info = save_projector(
        model, ckpt / "multimodal_projector.safetensors"
    )
    metadata = {
        "protocol": "resource_constrained_fiubench_stage1",
        "not_exact_official_full_ft": True,
        "reason": "24GB RTX 4090",
        "base_model": LLAVA_BASE,
        "official_alignment": {
            "identities": 400,
            "forget10": 40,
            "retain15": 60,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": 0.0,
            "seed": args.seed,
            "vision_frozen": True,
            "projector_trainable": True,
        },
        "resource_adaptation": {
            "language_training": "4bit_QLoRA",
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "micro_batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
        },
        "training_items": len(items),
        "optimizer_updates": update,
        "health": health,
        "projector": projector_info,
        "data_report": data_report,
    }
    atomic_json(ckpt / "stage1_metadata.json", metadata)
    atomic_json(out / "training_log.json", logs)
    atomic_json(out / "training_report.json", metadata)

    print("[PASS] Checkpoint saved.")
    print(f"CHECKPOINT: {ckpt}")

    model.eval()
    forget_items = identity_subset(forget_records)
    retain_items = identity_subset(retain_records)

    results = {}
    for split_name, split_items in (
        ("forget10", forget_items),
        ("retain15", retain_items),
    ):
        results[f"{split_name}_correct"] = evaluate_control(
            model, processor, split_items, "correct",
            args.eval_max_new_tokens, args.seed
        )
        results[f"{split_name}_mismatched"] = evaluate_control(
            model, processor, split_items, "mismatched",
            args.eval_max_new_tokens, args.seed + 1
        )
        results[f"{split_name}_blank"] = evaluate_control(
            model, processor, split_items, "blank",
            args.eval_max_new_tokens, args.seed + 2
        )

    forget_acc = results["forget10_correct"]["accuracy"]
    retain_acc = results["retain15_correct"]["accuracy"]
    mismatch_max = max(
        results["forget10_mismatched"]["accuracy"],
        results["retain15_mismatched"]["accuracy"],
    )
    blank_max = max(
        results["forget10_blank"]["accuracy"],
        results["retain15_blank"]["accuracy"],
    )

    scientific_pass = (
        forget_acc >= args.min_name_accuracy
        and retain_acc >= args.min_name_accuracy
        and mismatch_max <= args.max_mismatch_accuracy
        and blank_max <= args.max_mismatch_accuracy
        and health["trained"]
    )

    summary = {
        "checkpoint": str(ckpt),
        "forget10_name_accuracy": forget_acc,
        "retain15_name_accuracy": retain_acc,
        "max_mismatched_accuracy": mismatch_max,
        "max_blank_accuracy": blank_max,
        "minimum_name_accuracy": args.min_name_accuracy,
        "maximum_control_accuracy": args.max_mismatch_accuracy,
        "checkpoint_health": health,
        "scientific_gate_pass": scientific_pass,
        "overall_verdict": "PASS" if scientific_pass else "WARN",
    }

    atomic_json(out / "evaluation_summary.json", summary)
    for key, value in results.items():
        atomic_json(out / f"{key}_outputs.json", value["outputs"])

    text = "\n".join([
        "FIUBENCH STAGE-I 4090 EVALUATION",
        "=" * 88,
        f"Forget10 correct-image name accuracy: {forget_acc:.4f}",
        f"Retain15 correct-image name accuracy: {retain_acc:.4f}",
        f"Maximum mismatched-image accuracy: {mismatch_max:.4f}",
        f"Maximum blank-image accuracy: {blank_max:.4f}",
        f"Minimum required name accuracy: {args.min_name_accuracy:.4f}",
        f"Maximum allowed control accuracy: {args.max_mismatch_accuracy:.4f}",
        f"SCIENTIFIC GATE: {'PASS' if scientific_pass else 'NOT YET PASSED'}",
        f"CHECKPOINT: {ckpt}",
        f"JSON: {out / 'evaluation_summary.json'}",
        f"OVERALL VERDICT: {summary['overall_verdict']}",
    ]) + "\n"
    (out / "evaluation_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0 if scientific_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
