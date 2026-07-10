"""
retrain_ga_llava.py
--------------------
Retrains LLaVA GA (Gradient Ascent) adapter correctly.
Fixes the zero lora_B problem by ensuring:
  1. prepare_model_for_kbit_training() before get_peft_model()
  2. Optimizer created AFTER get_peft_model(), on requires_grad params only
  3. Gradient ascent via (-loss).backward() - NOT loss.item() detach
  4. Gradient check after first step - aborts if lora_B still zero
  5. save_pretrained() called only AFTER training, not before

Usage:
    py retrain_ga_llava.py
    py retrain_ga_llava.py --steps 50 --verify_only
    py retrain_ga_llava.py --steps 100
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from exp_config import (
    LLAVA_BASE, LLAVA_ADAPTERS, DEVICE,
    MLLMU_REAL_FORGET,
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LR,
    SCHEDULE_DIR,
)

GA_OUTPUT = SCHEDULE_DIR / "llava_ga_retrained_attn4_50steps"


# ------------------------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------------------------

def check_adapter_health(model, label: str) -> dict:
    """
    Check whether lora_B weights are nonzero (trained) or zero (dead).
    Returns dict with counts and raises if all-zero (after training).
    """
    stats = {"total_A": 0, "nonzero_A": 0,
             "total_B": 0, "nonzero_B": 0,
             "dead": False}

    for name, param in model.named_parameters():
        if "lora_A" in name:
            stats["total_A"] += 1
            if param.abs().max().item() > 1e-8:
                stats["nonzero_A"] += 1
        elif "lora_B" in name:
            stats["total_B"] += 1
            if param.abs().max().item() > 1e-8:
                stats["nonzero_B"] += 1

    stats["dead"] = stats["total_B"] > 0 and stats["nonzero_B"] == 0
    print(f"  [health:{label}] "
          f"lora_A {stats['nonzero_A']}/{stats['total_A']}  "
          f"lora_B {stats['nonzero_B']}/{stats['total_B']}  "
          f"{'DEAD - all-B-zero' if stats['dead'] else 'OK'}")
    return stats


def check_gradient_flows(model, loss):
    """
    After first backward, confirm that at least one lora_B has a gradient.
    Returns True if gradients reach lora_B, False otherwise.
    """
    for name, param in model.named_parameters():
        if "lora_B" in name and param.grad is not None:
            if param.grad.abs().max().item() > 1e-10:
                return True
    return False


# ------------------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------------------

def load_forget_items() -> list:
    ann = MLLMU_REAL_FORGET / "annotations.json"
    if ann.exists():
        with open(ann, encoding="utf-8") as f:
            items = json.load(f)
        result = []
        for item in items:
            p = Path(item["image"])
            if p.exists():
                item["image"] = p
                result.append(item)
        return result
    # subdirectory layout
    items = []
    for entity_dir in sorted(MLLMU_REAL_FORGET.iterdir()):
        if not entity_dir.is_dir(): continue
        imgs  = list(entity_dir.glob("*.jpg")) + list(entity_dir.glob("*.png"))
        jsons = list(entity_dir.glob("*.json"))
        if not imgs or not jsons: continue
        with open(jsons[0], encoding="utf-8") as f:
            qa = json.load(f)
        for q in (qa if isinstance(qa, list) else [qa]):
            items.append({
                "entity":   entity_dir.name,
                "image":    imgs[0],
                "question": q["question"],
                "answer":   q.get("answer", q.get("gt", "")),
            })
    return items


# ------------------------------------------------------------------------------
# MODEL SETUP
# ------------------------------------------------------------------------------

def get_bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def build_trainable_model():
    """
    Correct order:
      1. Load base model in 4-bit
      2. prepare_model_for_kbit_training  -> enables grad on non-quant layers
      3. get_peft_model with LoRA config  -> adds trainable lora_A, lora_B
      4. Build optimizer on requires_grad params
    """
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    print("[train] Loading base model...")
    processor = AutoProcessor.from_pretrained(LLAVA_BASE)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_BASE,
        quantization_config=get_bnb_config(),
        device_map=DEVICE,
    )

    # Step 2: MUST be before get_peft_model
    print("[train] prepare_model_for_kbit_training...")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=False,  # False avoids compat issues in 4-bit
    )

    # Step 3: Apply LoRA - language backbone only (matching original setup)
    lora_config = LoraConfig(
        r              = LORA_RANK,
        lora_alpha     = LORA_ALPHA,
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        task_type      = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Verify lora_B starts at zero (initialization state - expected)
    check_adapter_health(model, "before_training")

    # Step 4: Optimizer AFTER get_peft_model, on requires_grad params ONLY
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[train] Trainable tensors: {len(trainable)}")
    assert len(trainable) > 0, "No trainable parameters found!"

    optimizer = torch.optim.AdamW(trainable, lr=LR)

    return model, processor, optimizer


# ------------------------------------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------------------------------------

def compute_loss(model, processor, item: dict):
    """
    Standard cross-entropy loss on the forget item.
    GA will MAXIMIZE this loss by doing (-loss).backward().
    """
    image  = Image.open(item["image"]).convert("RGB")
    prompt = f"USER: <image>\n{item['question']} ASSISTANT: {item['answer']}"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    labels = inputs["input_ids"].clone()
    outputs = model(**inputs, labels=labels)
    return outputs.loss   # IMPORTANT: return the tensor, NOT .item()


def train_ga(n_steps: int, output_dir: Path):
    print(f"\n[train] GA Gradient Ascent - {n_steps} steps -> {output_dir}")

    if output_dir.exists() and (output_dir / "adapter_config.json").exists():
        # Check if existing adapter is trained
        from safetensors import safe_open
        st_path = output_dir / "adapter_model.safetensors"
        if st_path.exists():
            nonzero_b = 0
            total_b   = 0
            with safe_open(str(st_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if "lora_B" in key:
                        total_b += 1
                        t = f.get_tensor(key)
                        if t.abs().max().item() > 1e-8:
                            nonzero_b += 1
            if total_b > 0 and nonzero_b > 0:
                print(f"  [skip] Existing adapter is trained "
                      f"(lora_B {nonzero_b}/{total_b} nonzero). Skipping.")
                return
            else:
                print(f"  [warn] Existing adapter is dead (lora_B all zero). Retraining.")

    model, processor, optimizer = build_trainable_model()
    forget_items = load_forget_items()
    print(f"[train] Forget set: {len(forget_items)} items")

    model.train()
    step             = 0
    gradient_checked = False

    while step < n_steps:
        for item in forget_items:
            if step >= n_steps:
                break

            loss = compute_loss(model, processor, item)

            # -- GRADIENT ASCENT: maximise CE loss ----------------------------
            # Do NOT call loss.item() before backward - that detaches the graph
            ga_loss = -loss
            ga_loss.backward()

            # -- Verify gradient flows to lora_B on first step ----------------
            if not gradient_checked:
                if not check_gradient_flows(model, loss):
                    print("\n[FATAL] Gradients do NOT reach lora_B after first backward!")
                    print("  Possible causes:")
                    print("  - prepare_model_for_kbit_training not called")
                    print("  - LoRA target_modules mismatch")
                    print("  - Loss detached from graph")
                    sys.exit(1)
                else:
                    print("  [grad-check] Gradients reach lora_B OK")
                gradient_checked = True

            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % 10 == 0 or step == n_steps:
                print(f"  Step {step:3d}/{n_steps}  loss={loss.item():.4f}")

    # -- Health check BEFORE saving --------------------------------------------
    stats = check_adapter_health(model, "after_training")
    if stats["dead"]:
        print("\n[FATAL] Training completed but all lora_B weights are still zero!")
        print("  The adapter would be saved in a dead state. Aborting save.")
        sys.exit(1)

    # -- Save ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))

    meta = {
        "method":      "ga",
        "steps":       n_steps,
        "base_model":  LLAVA_BASE,
        "lora_rank":   LORA_RANK,
        "lora_alpha":  LORA_ALPHA,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "training":    "gradient_ascent",
        "loss_sign":   "negated_CE",
    }
    with open(output_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[train] Saved to: {output_dir}")
    print(f"  lora_B trained: {stats['nonzero_B']}/{stats['total_B']}")

    del model
    torch.cuda.empty_cache()


# ------------------------------------------------------------------------------
# VERIFY EXISTING ADAPTERS
# ------------------------------------------------------------------------------

def verify_all_adapters():
    """Check lora_B health for all configured adapters without loading models."""
    try:
        from safetensors import safe_open
    except ImportError:
        print("pip install safetensors")
        return

    print("\n=== ADAPTER HEALTH CHECK (safetensors) ===\n")
    for method, ckpt in LLAVA_ADAPTERS.items():
        if ckpt is None:
            print(f"  {method:<20} -> base model (no adapter)")
            continue
        ckpt = Path(ckpt)
        st   = ckpt / "adapter_model.safetensors"
        if not ckpt.exists():
            print(f"  {method:<20} -> MISSING checkpoint dir")
            continue
        if not st.exists():
            print(f"  {method:<20} -> no safetensors file")
            continue

        total_b = nonzero_b = 0
        with safe_open(str(st), framework="pt", device="cpu") as f:
            for key in f.keys():
                if "lora_B" in key:
                    total_b += 1
                    if f.get_tensor(key).abs().max().item() > 1e-8:
                        nonzero_b += 1

        status = "OK TRAINED" if nonzero_b > 0 else "X DEAD (all-B-zero)"
        print(f"  {method:<20} -> lora_B {nonzero_b:3d}/{total_b:<3d}  {status}")


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",       type=int, default=50)
    parser.add_argument("--output",      type=str, default=str(GA_OUTPUT),
                        help="Output directory for the retrained adapter")
    parser.add_argument("--verify_only", action="store_true",
                        help="Only check adapter health, do not train")
    args = parser.parse_args()

    if args.verify_only:
        verify_all_adapters()
        return

    verify_all_adapters()
    print()

    output_dir = Path(args.output)
    train_ga(args.steps, output_dir)

    print(f"\n[done] Update exp_config.py:")
    print(f'    "ga": Path(r"{output_dir}"),')


if __name__ == "__main__":
    main()


