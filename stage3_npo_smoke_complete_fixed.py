r"""
Validated NPO strength sweep for the LLaVA mllmu_real audit.

This file is bundled because the local project reported that no NPO sweep
script was present. It follows the previously audited newline-prompt scorer:

    USER: <image>\n{question}\nASSISTANT:

and compares every candidate against the same cached base-model evaluation.

Usage
-----
py -u .\stage3_npo_smoke_complete_fixed.py --sweep --resume
py -u .\stage3_npo_smoke_complete_fixed.py --eval_only --sweep --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp_config import (
    LLAVA_BASE,
    LLAVA_ADAPTERS,
    DEVICE,
    MLLMU_REAL_FORGET,
    MLLMU_REAL_RETAIN,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    ROOT,
    RESULTS_DIR,
    MAX_NEW_TOKENS,
)

VERSION = "stage3_npo_smoke_complete_fixed_v3"
SEED = 42
RETAIN_TOLERANCE = 0.05
COLLAPSE_TOKEN_RATIO = 0.30
COLLAPSE_ITEM_FRACTION = 0.50

NPO_DIR = Path(ROOT) / "checkpoints" / "npo_sweep"
OUT_DIR = Path(RESULTS_DIR) / "npo_sweep"
NPO_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SWEEP_CONFIGS = [
    {"beta": 0.1, "lr": 1e-4, "steps": 50, "label": "b0.1_s50"},
    {"beta": 0.5, "lr": 1e-4, "steps": 50, "label": "b0.5_s50"},
    {"beta": 1.0, "lr": 1e-4, "steps": 50, "label": "b1.0_s50"},
    {"beta": 0.1, "lr": 1e-4, "steps": 100, "label": "b0.1_s100"},
    {"beta": 0.5, "lr": 1e-4, "steps": 100, "label": "b0.5_s100"},
]

CHECKS: list[dict[str, Any]] = []


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def chk(label: str, verdict: str, detail: Any) -> None:
    CHECKS.append({
        "check": label,
        "verdict": verdict,
        "detail": str(detail),
        "timestamp_utc": now(),
    })
    print(f"[{verdict}] {label}: {detail}")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def overall_verdict() -> str:
    if any(row["verdict"] == "FAIL" for row in CHECKS):
        return "FAIL"
    if any(row["verdict"] == "WARN" for row in CHECKS):
        return "WARN"
    return "PASS"


def save_report(tag: str = "") -> None:
    payload = {
        "version": VERSION,
        "seed": SEED,
        "checks": CHECKS,
        "overall_verdict": overall_verdict(),
        "created_utc": now(),
    }
    save_json(OUT_DIR / f"stage3_report{tag}.json", payload)
    lines = ["NPO STRENGTH SWEEP", "=" * 80]
    lines.extend(
        f"[{row['verdict']}] {row['check']}: {row['detail']}" for row in CHECKS
    )
    lines.extend(["", f"OVERALL VERDICT: {overall_verdict()}"])
    (OUT_DIR / f"stage3_report{tag}.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def load_items(split_dir: Path) -> list[dict[str, Any]]:
    split_dir = Path(split_dir)
    annotation = split_dir / "annotations.json"
    items: list[dict[str, Any]] = []

    if annotation.exists():
        raw = json.loads(annotation.read_text(encoding="utf-8"))
        rows = raw if isinstance(raw, list) else []
        for index, original in enumerate(rows):
            item = dict(original)
            image = Path(str(item.get("image", "")))
            if not image.is_absolute():
                image = split_dir / image
            if not image.exists():
                continue
            item["image"] = image.resolve()
            item["entity"] = str(item.get("entity", item.get("name", index)))
            item["answer"] = str(item.get("answer", item.get("gt", "")))
            item["aliases"] = list(item.get("aliases", []))
            item["item_id"] = f"{item['entity']}_{index:03d}"
            items.append(item)
        return items

    uid = 0
    for entity_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        images = (
            list(entity_dir.glob("*.jpg"))
            + list(entity_dir.glob("*.jpeg"))
            + list(entity_dir.glob("*.png"))
        )
        jsons = list(entity_dir.glob("*.json"))
        if not images or not jsons:
            continue
        raw = json.loads(jsons[0].read_text(encoding="utf-8"))
        rows = raw if isinstance(raw, list) else [raw]
        for row in rows:
            items.append({
                "entity": str(row.get("entity", entity_dir.name)),
                "item_id": f"{entity_dir.name}_{uid:03d}",
                "image": images[0].resolve(),
                "question": str(row.get("question", "")),
                "answer": str(row.get("answer", row.get("gt", ""))),
                "aliases": list(row.get("aliases", [])),
            })
            uid += 1
    return items


def get_bnb():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def original_npo_adapter_config() -> dict[str, Any]:
    path = Path(LLAVA_ADAPTERS.get("npo", "")) / "adapter_config.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def build_trainable_model():
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    processor = AutoProcessor.from_pretrained(LLAVA_BASE, use_fast=False)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_BASE,
        quantization_config=get_bnb(),
        device_map=DEVICE,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=False,
    )

    old = original_npo_adapter_config()
    target_modules = old.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    config = LoraConfig(
        r=int(old.get("r", LORA_RANK)),
        lora_alpha=int(old.get("lora_alpha", LORA_ALPHA)),
        target_modules=target_modules,
        lora_dropout=float(old.get("lora_dropout", LORA_DROPOUT)),
        bias=str(old.get("bias", "none")),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model, processor, config


def load_base_model():
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(LLAVA_BASE, use_fast=False)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_BASE,
        quantization_config=get_bnb(),
        device_map=DEVICE,
    )
    model.eval()
    return model, processor


def load_adapter(path: Path):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(LLAVA_BASE, use_fast=False)
    base = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_BASE,
        quantization_config=get_bnb(),
        device_map=DEVICE,
    )
    model = PeftModel.from_pretrained(base, str(path), is_trainable=False)
    model.eval()
    return model, processor


def score_item(response: str, item: dict[str, Any]) -> bool:
    response = response.lower().strip()
    candidates = [item.get("answer", "")] + list(item.get("aliases", []))
    return any(
        str(candidate).lower().strip() in response
        for candidate in candidates
        if str(candidate).strip()
    )


def evaluate_model(model: Any, processor: Any, items: list[dict[str, Any]]):
    model.eval()
    per_item = []
    correct = 0
    collapsed = 0

    for index, item in enumerate(items):
        image = Image.open(item["image"]).convert("RGB")
        prompt = f"USER: <image>\n{item['question']}\nASSISTANT:"
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(DEVICE)
        prompt_length = int(inputs["input_ids"].shape[1])

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )

        generated = output[0][prompt_length:]
        response = processor.decode(generated, skip_special_tokens=True).strip()
        unique_ratio = len(set(generated.tolist())) / max(len(generated), 1)
        is_collapsed = (
            len(generated) > 4
            and unique_ratio < COLLAPSE_TOKEN_RATIO
        )
        is_correct = score_item(response, item)
        correct += int(is_correct)
        collapsed += int(is_collapsed)

        per_item.append({
            "index": index,
            "item_id": item["item_id"],
            "entity": item["entity"],
            "answer": item.get("answer", ""),
            "response": response[:300],
            "correct": is_correct,
            "collapsed": is_collapsed,
            "unique_token_ratio": unique_ratio,
        })
        print(f"  eval {index + 1:03d}/{len(items):03d}")

    n = len(items)
    collapse_fraction = collapsed / n if n else math.nan
    return {
        "n": n,
        "n_correct": correct,
        "accuracy": correct / n if n else math.nan,
        "n_collapsed": collapsed,
        "collapse_fraction": collapse_fraction,
        "collapsed": collapse_fraction > COLLAPSE_ITEM_FRACTION,
        "per_item": per_item,
    }


def base_scores(
    forget_items: list[dict[str, Any]],
    retain_items: list[dict[str, Any]],
):
    cache = OUT_DIR / "base_model_scores.json"
    scorer_version = "v3_newline_prompt"
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if (
            payload.get("scorer_version") == scorer_version
            and payload.get("forget", {}).get("n") == len(forget_items)
            and payload.get("retain", {}).get("n") == len(retain_items)
        ):
            print("[cached] base model behavioural scores")
            return payload

    print("Evaluating base model once...")
    model, processor = load_base_model()
    payload = {
        "scorer_version": scorer_version,
        "forget": evaluate_model(model, processor, forget_items),
        "retain": evaluate_model(model, processor, retain_items),
    }
    save_json(cache, payload)
    del model
    torch.cuda.empty_cache()
    return payload


def sequence_log_probability(model: Any, processor: Any, item: dict[str, Any]):
    image = Image.open(item["image"]).convert("RGB")
    text = (
        f"USER: <image>\n{item['question']}\n"
        f"ASSISTANT: {item['answer']}"
    )
    inputs = processor(text=text, images=image, return_tensors="pt").to(DEVICE)
    labels = inputs["input_ids"].clone()
    output = model(**inputs, labels=labels, use_cache=False)
    return -output.loss


def train_npo(
    beta: float,
    learning_rate: float,
    steps: int,
    label: str,
    forget_items: list[dict[str, Any]],
    resume: bool,
):
    checkpoint = NPO_DIR / label
    if resume and (
        (checkpoint / "adapter_config.json").exists()
        and (
            (checkpoint / "adapter_model.safetensors").exists()
            or (checkpoint / "adapter_model.bin").exists()
        )
    ):
        print(f"[cached] {checkpoint}")
        return checkpoint

    torch.manual_seed(SEED)
    model, processor, lora_config = build_trainable_model()

    model.eval()
    reference = {}
    print("Computing reference forget log-probabilities...")
    with torch.inference_mode():
        for index, item in enumerate(forget_items):
            reference[item["item_id"]] = float(
                sequence_log_probability(model, processor, item).item()
            )
            print(f"  reference {index + 1:03d}/{len(forget_items):03d}")

    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=0.0,
    )
    gradient_verified = False

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        item = forget_items[step % len(forget_items)]
        current = sequence_log_probability(model, processor, item)
        reference_value = torch.tensor(
            reference[item["item_id"]],
            device=current.device,
            dtype=current.dtype,
        )
        loss = -F.logsigmoid(-beta * (current - reference_value))
        loss.backward()

        if not gradient_verified:
            has_gradient = any(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                and parameter.grad.abs().max().item() > 1e-10
                for name, parameter in model.named_parameters()
                if "lora_B" in name and parameter.requires_grad
            )
            if not has_gradient:
                raise RuntimeError("No non-zero finite gradients at LoRA-B")
            gradient_verified = True
            print("[PASS] non-zero LoRA-B gradient")

        optimizer.step()
        if (step + 1) % max(steps // 5, 1) == 0 or step == 0:
            print(
                f"  train {step + 1:03d}/{steps:03d} "
                f"loss={float(loss.item()):.6f}"
            )

    nonzero_lora_b = sum(
        1
        for name, parameter in model.named_parameters()
        if "lora_B" in name and parameter.detach().abs().max().item() > 1e-8
    )
    if nonzero_lora_b == 0:
        raise RuntimeError("All LoRA-B tensors remained zero after training")

    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint))
    processor.save_pretrained(str(checkpoint))
    save_json(checkpoint / "training_meta.json", {
        "version": VERSION,
        "method": "NPO",
        "beta": beta,
        "learning_rate": learning_rate,
        "steps": steps,
        "seed": SEED,
        "batch_size": 1,
        "gradient_accumulation": 1,
        "lora_rank": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "target_modules": sorted(lora_config.target_modules),
        "nonzero_lora_B_tensors": nonzero_lora_b,
        "created_utc": now(),
    })
    del model
    torch.cuda.empty_cache()
    return checkpoint


def evaluate_checkpoint(
    checkpoint: Path,
    label: str,
    forget_items: list[dict[str, Any]],
    retain_items: list[dict[str, Any]],
    base: dict[str, Any],
):
    print(f"Evaluating {label}...")
    model, processor = load_adapter(checkpoint)
    forget = evaluate_model(model, processor, forget_items)
    retain = evaluate_model(model, processor, retain_items)
    del model
    torch.cuda.empty_cache()

    base_forget = float(base["forget"]["accuracy"])
    base_retain = float(base["retain"]["accuracy"])
    delta_forget = float(forget["accuracy"] - base_forget)
    delta_retain = float(retain["accuracy"] - base_retain)

    suppressed = [
        candidate["item_id"]
        for original, candidate in zip(
            base["forget"]["per_item"], forget["per_item"]
        )
        if original["correct"] and not candidate["correct"]
    ]

    if forget["collapsed"] or retain["collapsed"]:
        result_verdict = "FAIL"
        reason = "generation collapse detected"
    elif delta_retain < -RETAIN_TOLERANCE:
        result_verdict = "FAIL"
        reason = (
            f"retain accuracy changed by {delta_retain:+.4f}, "
            f"below tolerance {-RETAIN_TOLERANCE:.4f}"
        )
    elif delta_forget < -0.01:
        result_verdict = "PASS"
        reason = "forget accuracy improved while retain utility was preserved"
    else:
        result_verdict = "WARN"
        reason = "no meaningful forget-accuracy improvement"

    return {
        "version": VERSION,
        "label": label,
        "checkpoint": str(checkpoint.resolve()),
        "base_forget_accuracy": base_forget,
        "base_retain_accuracy": base_retain,
        "forget": forget,
        "retain": retain,
        "forget_accuracy": float(forget["accuracy"]),
        "forget_rate": 1 - float(forget["accuracy"]),
        "retain_accuracy": float(retain["accuracy"]),
        "delta_forget_accuracy": delta_forget,
        "delta_retain_accuracy": delta_retain,
        "n_suppressed": len(suppressed),
        "suppressed_item_ids": suppressed,
        "verdict": result_verdict,
        "reason": reason,
    }


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize current and legacy cached evaluation JSON files.

    Older cached files may keep accuracy under result["forget"]["accuracy"]
    and result["retain"]["accuracy"] without duplicating the values at the
    top level. The sweep summary must accept both schemas.
    """
    normalized = dict(result)

    forget = normalized.get("forget")
    retain = normalized.get("retain")
    forget = forget if isinstance(forget, dict) else {}
    retain = retain if isinstance(retain, dict) else {}

    base_forget = normalized.get("base_forget_accuracy")
    base_retain = normalized.get("base_retain_accuracy")

    forget_accuracy = normalized.get("forget_accuracy", forget.get("accuracy"))
    retain_accuracy = normalized.get("retain_accuracy", retain.get("accuracy"))

    if forget_accuracy is not None:
        forget_accuracy = float(forget_accuracy)
        normalized["forget_accuracy"] = forget_accuracy
        normalized.setdefault("forget_rate", 1.0 - forget_accuracy)

    if retain_accuracy is not None:
        retain_accuracy = float(retain_accuracy)
        normalized["retain_accuracy"] = retain_accuracy

    if (
        normalized.get("delta_forget_accuracy") is None
        and forget_accuracy is not None
        and base_forget is not None
    ):
        normalized["delta_forget_accuracy"] = (
            forget_accuracy - float(base_forget)
        )

    if (
        normalized.get("delta_retain_accuracy") is None
        and retain_accuracy is not None
        and base_retain is not None
    ):
        normalized["delta_retain_accuracy"] = (
            retain_accuracy - float(base_retain)
        )

    if normalized.get("n_suppressed") is None:
        suppressed = normalized.get("suppressed_item_ids", [])
        normalized["n_suppressed"] = (
            len(suppressed) if isinstance(suppressed, list) else 0
        )

    normalized.setdefault("label", "unknown")
    normalized.setdefault("checkpoint", "")
    normalized.setdefault("verdict", "WARN")
    normalized.setdefault("reason", "legacy/incomplete cached result")

    missing = [
        key for key in ("forget_accuracy", "forget_rate", "retain_accuracy")
        if normalized.get(key) is None
    ]
    normalized["summary_complete"] = not missing
    normalized["summary_missing_fields"] = missing
    return normalized


def write_summary(results: list[dict[str, Any]]) -> None:
    normalized_results = [normalize_result(result) for result in results]

    save_json(OUT_DIR / "npo_sweep_summary.json", {
        "version": VERSION,
        "results": normalized_results,
        "checks": CHECKS,
        "overall_verdict": overall_verdict(),
    })

    rows = []
    for result in normalized_results:
        rows.append({
            "label": result.get("label", "unknown"),
            "checkpoint": result.get("checkpoint", ""),
            "forget_accuracy": result.get("forget_accuracy"),
            "forget_rate": result.get("forget_rate"),
            "retain_accuracy": result.get("retain_accuracy"),
            "delta_forget_accuracy": result.get("delta_forget_accuracy"),
            "delta_retain_accuracy": result.get("delta_retain_accuracy"),
            "n_suppressed": result.get("n_suppressed", 0),
            "verdict": result.get("verdict", "WARN"),
            "reason": result.get("reason", ""),
            "summary_complete": result.get("summary_complete", False),
            "summary_missing_fields": ",".join(
                result.get("summary_missing_fields", [])
            ),
        })

    if rows:
        with (OUT_DIR / "npo_sweep_summary.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print("\nNPO SWEEP SUMMARY")
    print("=" * 100)
    print(
        f"{'Config':<20} {'F-Acc':>8} {'F-Rate':>8} "
        f"{'Ret-Acc':>9} {'Suppr':>7} {'Verdict':>8}"
    )

    for row in rows:
        f_acc = (
            f"{float(row['forget_accuracy']):.4f}"
            if row["forget_accuracy"] is not None else "NA"
        )
        f_rate = (
            f"{float(row['forget_rate']):.4f}"
            if row["forget_rate"] is not None else "NA"
        )
        r_acc = (
            f"{float(row['retain_accuracy']):.4f}"
            if row["retain_accuracy"] is not None else "NA"
        )
        print(
            f"{row['label']:<20} {f_acc:>8} {f_rate:>8} "
            f"{r_acc:>9} {int(row['n_suppressed']):>7d} "
            f"{row['verdict']:>8}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--smoke_steps", type=int, default=15)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    forget_items = load_items(Path(MLLMU_REAL_FORGET))
    retain_items = load_items(Path(MLLMU_REAL_RETAIN))
    if not forget_items or not retain_items:
        chk(
            "dataset loading",
            "FAIL",
            f"forget={len(forget_items)}, retain={len(retain_items)}",
        )
        save_report("_data_fail")
        return 1
    chk("dataset loading", "PASS", f"forget={len(forget_items)}, retain={len(retain_items)}")

    base = base_scores(forget_items, retain_items)
    smoke = {
        "beta": args.beta,
        "lr": args.lr,
        "steps": args.smoke_steps,
        "label": f"smoke_b{args.beta}_s{args.smoke_steps}",
    }
    configurations = [smoke] + (SWEEP_CONFIGS if args.sweep else [])
    results = []

    for index, config in enumerate(configurations):
        label = config["label"]
        print("\n" + "=" * 90)
        print(f"NPO {'SMOKE' if index == 0 else 'SWEEP'}: {label}")
        print("=" * 90)

        checkpoint = NPO_DIR / label
        result_path = OUT_DIR / f"{label}_eval.json"

        if args.resume and result_path.exists() and checkpoint.exists():
            try:
                cached = load_items  # keep linter quiet; no side effect
                result = normalize_result(
                    json.loads(result_path.read_text(encoding="utf-8"))
                )
                if result.get("checkpoint") == str(checkpoint.resolve()):
                    print(f"[cached] evaluation {label}")
                    results.append(result)
                    chk(label, result.get("verdict", "WARN"), result.get("reason", "cached"))
                    continue
            except Exception:
                pass

        if not args.eval_only:
            try:
                checkpoint = train_npo(
                    config["beta"],
                    config["lr"],
                    config["steps"],
                    label,
                    forget_items,
                    args.resume,
                )
            except Exception as exc:
                chk(f"{label} training", "FAIL", f"{type(exc).__name__}: {exc}")
                if index == 0:
                    save_report("_smoke_fail")
                    return 1
                continue
        elif not (
            (checkpoint / "adapter_config.json").exists()
            and (
                (checkpoint / "adapter_model.safetensors").exists()
                or (checkpoint / "adapter_model.bin").exists()
            )
        ):
            chk(f"{label} checkpoint", "WARN", f"missing: {checkpoint}")
            continue

        try:
            result = evaluate_checkpoint(
                checkpoint,
                label,
                forget_items,
                retain_items,
                base,
            )
        except Exception as exc:
            chk(f"{label} evaluation", "FAIL", f"{type(exc).__name__}: {exc}")
            if index == 0:
                save_report("_smoke_eval_fail")
                return 1
            continue

        save_json(result_path, result)
        results.append(result)
        chk(label, result["verdict"], result["reason"])

        if index == 0 and result["verdict"] == "FAIL":
            write_summary(results)
            save_report("_smoke_fail")
            return 1

    write_summary(results)
    save_report()
    print(f"OVERALL VERDICT: {overall_verdict()}")
    return 1 if overall_verdict() == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
