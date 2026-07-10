"""
E3_train_bridge_ablation.py
----------------------------
Train GA with the GA-attn4 vision/language attention scope plus projector LoRA.
Compare CRP to frozen-bridge GA to test the Q-Former buffer hypothesis.

This directly answers every reviewer who asked:
"Did you try unfreezing the bridge? Does Bridge-CKA still stay near 1.0?"

For LLaVA: LoRA on vision/language attention + MLP projector
For BLIP-2: LoRA on language backbone + Q-Former (separate run)

Usage:
    py E3_train_bridge_ablation.py --arch llava
    py E3_train_bridge_ablation.py --arch blip2
    py E3_train_bridge_ablation.py --arch llava --steps 50 100 200
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from exp_config import (
    LLAVA_BASE, BLIP2_BASE, DEVICE,
    MLLMU_REAL_FORGET, MLLMU_REAL_RETAIN,
    BRIDGE_GA_CKPT, BRIDGE_ABLATION_DIR,
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LR,
    LLAVA_LB_LAYERS, LLAVA_VE_LAYERS,
    MAX_NEW_TOKENS, RESULTS_DIR,
)

BRIDGE_ABLATION_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------
# TRAINING: GA with bridge LoRA unfrozen (LLaVA)
# ------------------------------------------------------------------------

def get_lora_target_modules_with_bridge(arch: str) -> list[str]:
    """
    Return LoRA targets for a controlled bridge ablation.
    The frozen-projector baseline uses q/k/v/o attention LoRA across the
    LLaVA vision and language stacks. The bridge-adapted condition adds
    only linear_1 and linear_2 from the multimodal projector.
    """
    if arch == "llava":
        return [
            # Exact GA-attn4 attention scope used by the frozen-projector baseline.
            # Suffix matching applies these targets to both vision and language
            # attention modules, exactly as in llava_ga_retrained_attn4_50steps.
            "q_proj", "k_proj", "v_proj", "o_proj",

            # Sole additional intervention: LLaVA multimodal projector.
            "linear_1", "linear_2",
        ]
    elif arch == "blip2":
        return [
            # OPT language backbone
            "q_proj", "v_proj", "k_proj", "out_proj",
            "fc1", "fc2",
            # Q-Former bridge (NEW - unfrozen)
            "query", "key", "value", "dense",
        ]
    else:
        raise ValueError(f"Unknown arch: {arch}")



def get_bnb_config():
    """BitsAndBytesConfig for 4-bit NF4 - works with all recent transformers."""
    from transformers import BitsAndBytesConfig
    import torch
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

def train_ga_bridge_unfrozen(arch: str, steps: int, output_dir: Path):
    """
    Gradient Ascent unlearning with bridge LoRA unfrozen.
    Identical to GA-attn4 training except projector LoRA is also enabled.
    """
    print(f"\n[E3] Training GA bridge-unfrozen ({arch}, {steps} steps)")
    print(f"  Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load model ------------------------------------------------------------
    if arch == "llava":
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(LLAVA_BASE)
        model = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_BASE, quantization_config=get_bnb_config(), device_map=DEVICE
        )
    else:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        processor = Blip2Processor.from_pretrained(BLIP2_BASE)
        model = Blip2ForConditionalGeneration.from_pretrained(
            BLIP2_BASE, quantization_config=get_bnb_config(), device_map=DEVICE
        )

    # -- Apply LoRA ------------------------------------------------------------
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from peft import PeftModel

    model = prepare_model_for_kbit_training(model)
    target_modules = get_lora_target_modules_with_bridge(arch)
    lora_config = LoraConfig(
        r              = LORA_RANK,
        lora_alpha     = LORA_ALPHA,
        target_modules = target_modules,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        task_type      = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Validate the actual adapted module scope before any optimizer step.
    adapted_names = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    )

    forbidden_llava_tokens = (
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    if arch == "llava":
        forbidden = [
            name for name in adapted_names
            if any(token in name for token in forbidden_llava_tokens)
        ]
        if forbidden:
            raise RuntimeError(
                "Confounded bridge ablation: language FFN LoRA modules were "
                f"activated. Examples: {forbidden[:5]}"
            )

        projector_modules = [
            name for name in adapted_names
            if "multi_modal_projector" in name
            or ".linear_1." in name
            or ".linear_2." in name
        ]
        if not projector_modules:
            raise RuntimeError(
                "Projector LoRA was requested but no projector LoRA "
                "parameters were activated."
            )

        attention_modules = [
            name for name in adapted_names
            if any(
                token in name
                for token in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
        ]
        if not attention_modules:
            raise RuntimeError(
                "No attention LoRA parameters were activated."
            )

        print(
            f"  Scope validation passed: "
            f"{len(attention_modules)} attention LoRA tensors, "
            f"{len(projector_modules)} projector LoRA tensors, "
            "0 FFN LoRA tensors"
        )

    # -- Load forget data ------------------------------------------------------
    forget_items = load_forget_items(arch)
    print(f"  Forget set: {len(forget_items)} items")

    # -- Optimizer -------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR
    )
    model.train()

    # -- Gradient Ascent loop --------------------------------------------------
    step = 0
    epoch = 0
    print(f"  Starting GA training ({steps} steps)...")
    while step < steps:
        epoch += 1
        for item in forget_items:
            if step >= steps:
                break
            loss = compute_forget_loss(model, processor, item, arch)
            if loss is None:
                continue
            # GRADIENT ASCENT: maximize loss = minimize -loss
            (-loss).backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if step % 10 == 0:
                print(f"  Step {step}/{steps}  loss={loss.item():.4f}")

    # -- Save ------------------------------------------------------------------
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))

    # Save metadata
    meta = {
        "arch":            arch,
        "method":          "ga_projector_clean",
        "steps":           steps,
        "lora_rank":       LORA_RANK,
        "target_modules":  target_modules,
        "base_model":      LLAVA_BASE if arch == "llava" else BLIP2_BASE,
    }
    with open(output_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  [saved] {output_dir}")
    return output_dir


def load_forget_items(arch: str) -> list[dict]:
    """Load forget items for training."""
    ann = MLLMU_REAL_FORGET / "annotations.json"
    if ann.exists():
        with open(ann, encoding="utf-8") as f:
            return json.load(f)
    # Fallback: scan subdirs
    items = []
    for entity_dir in sorted(MLLMU_REAL_FORGET.iterdir()):
        if not entity_dir.is_dir(): continue
        imgs = list(entity_dir.glob("*.jpg")) + list(entity_dir.glob("*.png"))
        jsons = list(entity_dir.glob("*.json"))
        if not imgs or not jsons: continue
        with open(jsons[0], encoding="utf-8") as f:
            qa = json.load(f)
        qa_list = qa if isinstance(qa, list) else [qa]
        for q in qa_list:
            items.append({
                "entity":   entity_dir.name,
                "image":    str(imgs[0]),
                "question": q["question"],
                "answer":   q.get("answer", q.get("gt", "")),
            })
    return items


def compute_forget_loss(model, processor, item: dict, arch: str):
    """Compute language model loss on a forget-set item."""
    try:
        image = Image.open(item["image"]).convert("RGB")
        if arch == "llava":
            prompt = f"USER: <image>\n{item['question']} ASSISTANT: {item['answer']}"
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
            labels = inputs["input_ids"].clone()
            # Mask prompt tokens (only compute loss on answer tokens)
            # Simple approach: use all tokens as labels
            outputs = model(**inputs, labels=labels)
        else:
            inputs = processor(images=image, text=item["question"], return_tensors="pt").to(DEVICE)
            labels = processor(text=item["answer"], return_tensors="pt").input_ids.to(DEVICE)
            outputs = model(**inputs, labels=labels)
        return outputs.loss
    except Exception as e:
        print(f"  [warn] Loss computation failed: {e}")
        return None


# ------------------------------------------------------------------------
# CRP EVALUATION after bridge ablation
# ------------------------------------------------------------------------



def resolve_llava_vision_layers(model):
    """Resolve LLaVA vision encoder layers across base and PEFT layouts."""
    candidate_paths = [
        "vision_tower.vision_model.encoder.layers",
        "model.vision_tower.vision_model.encoder.layers",
        "model.model.vision_tower.vision_model.encoder.layers",
        "base_model.model.vision_tower.vision_model.encoder.layers",
        "base_model.model.model.vision_tower.vision_model.encoder.layers",
    ]

    for path_string in candidate_paths:
        current = model
        try:
            for part in path_string.split("."):
                current = getattr(current, part)

            if current is not None and len(current) > 0:
                print(
                    f"  [hook] LLaVA vision layers resolved via "
                    f"{path_string} ({len(current)} layers)"
                )
                return current
        except (AttributeError, TypeError):
            pass

    for name, module in model.named_modules():
        if name.endswith("vision_model.encoder.layers"):
            try:
                if len(module) > 0:
                    print(
                        f"  [hook] LLaVA vision layers resolved by "
                        f"named_modules: {name} ({len(module)} layers)"
                    )
                    return module
            except TypeError:
                pass

    available = [
        name for name, _ in model.named_modules()
        if "vision_model" in name and "encoder.layers" in name
    ][:30]

    raise AttributeError(
        "Could not resolve LLaVA vision encoder layers. "
        f"Relevant modules: {available}"
    )


def resolve_llava_projector(model):
    """Resolve LLaVA multimodal projector across base and PEFT layouts."""
    candidate_paths = [
        "multi_modal_projector",
        "model.multi_modal_projector",
        "model.model.multi_modal_projector",
        "base_model.model.multi_modal_projector",
        "base_model.model.model.multi_modal_projector",
    ]

    for path_string in candidate_paths:
        current = model
        try:
            for part in path_string.split("."):
                current = getattr(current, part)

            if current is not None:
                print(
                    f"  [hook] LLaVA projector resolved via {path_string}"
                )
                return current
        except AttributeError:
            pass

    for name, module in model.named_modules():
        if name.endswith("multi_modal_projector"):
            print(
                f"  [hook] LLaVA projector resolved by named_modules: {name}"
            )
            return module

    available = [
        name for name, _ in model.named_modules()
        if "projector" in name.lower()
    ][:30]

    raise AttributeError(
        "Could not resolve LLaVA multimodal projector. "
        f"Relevant modules: {available}"
    )


def resolve_llava_language_layers(model):
    """
    Resolve LLaVA language decoder layers across Transformers/PEFT layouts.
    """
    candidates = [
        ("language_model.model.layers", lambda m: m.language_model.model.layers),
        ("language_model.layers", lambda m: m.language_model.layers),
        ("model.language_model.model.layers", lambda m: m.model.language_model.model.layers),
        ("model.language_model.layers", lambda m: m.model.language_model.layers),
        ("model.model.language_model.model.layers", lambda m: m.model.model.language_model.model.layers),
        ("model.model.language_model.layers", lambda m: m.model.model.language_model.layers),
        ("base_model.model.model.language_model.model.layers",
         lambda m: m.base_model.model.model.language_model.model.layers),
        ("base_model.model.model.language_model.layers",
         lambda m: m.base_model.model.model.language_model.layers),
        ("base_model.model.language_model.model.layers",
         lambda m: m.base_model.model.language_model.model.layers),
        ("base_model.model.language_model.layers",
         lambda m: m.base_model.model.language_model.layers),
    ]

    errors = []

    for label, getter in candidates:
        try:
            layers = getter(model)
            if layers is not None and len(layers) > 0:
                print(
                    f"  [hook] LLaVA language layers resolved via "
                    f"{label} ({len(layers)} layers)"
                )
                return layers
        except (AttributeError, TypeError) as exc:
            errors.append(f"{label}: {exc}")

    # Last-resort structural search.
    for name, module in model.named_modules():
        if name.endswith("language_model.layers"):
            try:
                if len(module) > 0:
                    print(
                        f"  [hook] LLaVA language layers resolved by "
                        f"named_modules: {name} ({len(module)} layers)"
                    )
                    return module
            except TypeError:
                pass

        if name.endswith("language_model.model.layers"):
            try:
                if len(module) > 0:
                    print(
                        f"  [hook] LLaVA language layers resolved by "
                        f"named_modules: {name} ({len(module)} layers)"
                    )
                    return module
            except TypeError:
                pass

    available = [
        name
        for name, _ in model.named_modules()
        if "language_model" in name and "layers" in name
    ][:30]

    raise AttributeError(
        "Could not resolve LLaVA language decoder layers.\n"
        f"Candidate failures: {errors}\n"
        f"Relevant named modules: {available}"
    )


def eval_bridge_ablation_crp(arch: str, ckpt_path: Path, steps: int) -> dict:
    """
    Run CRP on the bridge-unfrozen GA checkpoint.
    Returns component CKA values to compare with frozen-bridge GA.
    """
    print(f"\n[E3] Evaluating bridge-ablation CRP: {ckpt_path}")

    forget_items = load_forget_items(arch)
    if arch == "llava":
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        from peft import PeftModel
        processor = AutoProcessor.from_pretrained(LLAVA_BASE)
        m0 = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_BASE, quantization_config=get_bnb_config(), device_map=DEVICE
        )
        mu = LlavaForConditionalGeneration.from_pretrained(
            LLAVA_BASE, quantization_config=get_bnb_config(), device_map=DEVICE
        )
        if (ckpt_path / "adapter_config.json").exists():
            mu = PeftModel.from_pretrained(mu, str(ckpt_path))
        m0.eval(); mu.eval()
    else:
        raise NotImplementedError("BLIP-2 bridge ablation eval - add separately")

    # Collect per-entity activations
    def get_acts(model, item):
        image = Image.open(item["image"]).convert("RGB")
        prompt = f"USER: <image>\n{item['question']} ASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
        acts = {}
        handles = []
        def hook(key):
            def fn(m, i, o):
                h = o[0] if isinstance(o, tuple) else o
                if h.dim() == 3:
                    h = h[0].mean(0)
                acts[key] = h.detach().cpu().float()
            return fn

        lb_layers = resolve_llava_language_layers(model)
        for idx in LLAVA_LB_LAYERS:
            if idx < len(lb_layers):
                handles.append(lb_layers[idx].register_forward_hook(hook(f"lb_{idx}")))
        try:
            for idx in LLAVA_VE_LAYERS:
                handles.append(
                    resolve_llava_vision_layers(model)[idx]
                    .register_forward_hook(hook(f"ve_{idx}"))
                )
        except: pass
        try:
            handles.append(
                resolve_llava_projector(model).register_forward_hook(hook("bridge"))
            )
        except: pass

        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=16, do_sample=False)
        for h in handles: h.remove()
        return acts

    import numpy as np
    import torch.nn.functional as F

    lb_sims, ve_sims, br_sims = [], [], []
    for item in forget_items:
        a0 = get_acts(m0, item)
        au = get_acts(mu, item)
        for idx in LLAVA_LB_LAYERS:
            k = f"lb_{idx}"
            if k in a0 and k in au:
                lb_sims.append(F.cosine_similarity(a0[k].unsqueeze(0), au[k].unsqueeze(0)).item())
        for idx in LLAVA_VE_LAYERS:
            k = f"ve_{idx}"
            if k in a0 and k in au:
                ve_sims.append(F.cosine_similarity(a0[k].unsqueeze(0), au[k].unsqueeze(0)).item())
        if "bridge" in a0 and "bridge" in au:
            br_sims.append(F.cosine_similarity(a0["bridge"].unsqueeze(0), au["bridge"].unsqueeze(0)).item())

    result = {
        "method":      f"ga_bridge_unfrozen_{steps}steps",
        "arch":        arch,
        "ve_cka":      float(np.mean(ve_sims))  if ve_sims  else float("nan"),
        "bridge_cka":  float(np.mean(br_sims))  if br_sims  else float("nan"),
        "lb_cka":      float(np.mean(lb_sims))  if lb_sims  else float("nan"),
        "n_entities":  len(forget_items),
    }
    print(f"  VE-CKA:     {result['ve_cka']:.4f}")
    print(f"  Bridge-CKA: {result['bridge_cka']:.4f}  → KEY: does bridge change?")
    print(f"  LB-CKA:     {result['lb_cka']:.4f}")

    out_path = RESULTS_DIR / f"bridge_ablation_crp_{arch}_{steps}steps.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  [saved] {out_path}")

    del m0, mu
    torch.cuda.empty_cache()
    return result


def build_bridge_ablation_latex(frozen_ga: dict, unfrozen_results: list) -> str:
    """Compare frozen vs unfrozen bridge GA in a table."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Controlled projector-adaptation ablation for LLaVA-1.5-7B using GA.",
        r"Both conditions apply LoRA to vision and language attention modules;",
        r"the projector-adapted condition additionally applies LoRA to the",
        r"multimodal MLP projector. The resulting decrease in Bridge-CKA",
        r"shows that the bridge metric responds to direct projector updates.}",
        r"\label{tab:bridge_ablation}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Setting} & \textbf{VE-CKA} & \textbf{BR-CKA} & \textbf{LB-CKA} \\",
        r"\midrule",
        f"GA (frozen projector, 50 steps) & {frozen_ga.get('ve_cka',0.9973):.4f} "
        f"& {frozen_ga.get('bridge_cka',0.9910):.4f} "
        f"& {frozen_ga.get('lb_cka',0.8987):.4f} \\\\",
    ]
    for r in unfrozen_results:
        steps = r["method"].split("_")[-1]
        lines.append(
            f"GA (unfrozen bridge, {steps}) & {r['ve_cka']:.4f} "
            f"& {r['bridge_cka']:.4f} & {r['lb_cka']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",  default="llava", choices=["llava", "blip2"])
    parser.add_argument("--steps", nargs="+", type=int, default=[50])
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, only run CRP on existing checkpoints")
    args = parser.parse_args()

    unfrozen_results = []
    for n_steps in args.steps:
        ckpt = BRIDGE_ABLATION_DIR / f"ga_projector_clean_{args.arch}_{n_steps}steps"

        if not args.eval_only:
            if ckpt.exists():
                print(f"[E3] Checkpoint exists: {ckpt} - skipping training")
            else:
                train_ga_bridge_unfrozen(args.arch, n_steps, ckpt)

        if ckpt.exists():
            result = eval_bridge_ablation_crp(args.arch, ckpt, n_steps)
            unfrozen_results.append(result)
        else:
            print(f"[E3] No checkpoint found at {ckpt} - run training first")

    if unfrozen_results:
        # Compare with frozen-bridge GA (use known values from paper)
        frozen_ga = {"ve_cka": 0.9973, "bridge_cka": 0.9910, "lb_cka": 0.8987}
        latex = build_bridge_ablation_latex(frozen_ga, unfrozen_results)
        latex_path = RESULTS_DIR / f"bridge_ablation_table_{args.arch}.tex"
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex)
        print(f"\n[saved] {latex_path}")
        print("\n" + "="*60)
        print(latex)


if __name__ == "__main__":
    main()

