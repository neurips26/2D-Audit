"""
train_npo_tuned.py
───────────────────
Tuned NPO training targeting selective forgetting on mllmu_real.

NPO is theoretically better suited for selective forgetting than GradDiff
because it uses a reference model to bound how far the forget distribution
can shift, preventing the collapse that hits unconstrained GA.

NPO loss = -log sigma(-beta * (log p_mu(y|x) - log p_ref(y|x)))

This pushes forget-set outputs away from the reference model while
naturally preventing over-forgetting because the sigma function saturates.

Sweep: beta in {0.1, 0.5, 1.0} at 50 and 100 steps.
Expected: ForgetAcc drops below 0.5 with RetainAcc staying above 0.8
at the right beta/step combination.

Usage:
    py train_npo_tuned.py --beta 0.5 --steps 100
    py train_npo_tuned.py --sweep          # runs all beta/step combos
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from exp_config import (
    LLAVA_BASE, DEVICE,
    MLLMU_REAL_FORGET, MLLMU_REAL_RETAIN,
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT,
    ROOT, RESULTS_DIR,
)

NPO_DIR = ROOT / "checkpoints" / "npo_tuned"
NPO_DIR.mkdir(parents=True, exist_ok=True)

SWEEP_CONFIGS = [
    {"beta": 0.1, "lr": 1e-4, "steps": 50},
    {"beta": 0.5, "lr": 1e-4, "steps": 50},
    {"beta": 1.0, "lr": 1e-4, "steps": 50},
    {"beta": 0.5, "lr": 1e-4, "steps": 100},
    {"beta": 0.1, "lr": 1e-4, "steps": 100},
]


def get_bnb():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)


def load_items(split_dir: Path) -> list:
    ann = split_dir / "annotations.json"
    if ann.exists():
        with open(ann, encoding="utf-8") as f:
            items = json.load(f)
        result = []
        for item in items:
            p = Path(item["image"])
            if not p.is_absolute(): p = split_dir / p
            if p.exists():
                item["image"] = p
                result.append(item)
        return result
    items = []
    for d in sorted(split_dir.iterdir()):
        if not d.is_dir(): continue
        imgs  = list(d.glob("*.jpg")) + list(d.glob("*.png"))
        jsons = list(d.glob("*.json"))
        if not imgs or not jsons: continue
        with open(jsons[0], encoding="utf-8") as f:
            qa = json.load(f)
        for q in (qa if isinstance(qa, list) else [qa])[:1]:
            items.append({"entity": d.name, "image": imgs[0],
                          "question": q["question"],
                          "answer":   q.get("answer", "")})
    return items


def build_model():
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    proc  = AutoProcessor.from_pretrained(LLAVA_BASE)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_BASE, quantization_config=get_bnb(), device_map=DEVICE)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    cfg   = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=LORA_DROPOUT, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model, proc


def compute_log_prob(model, proc, item: dict) -> torch.Tensor:
    """Compute log p(answer | question, image) under current model."""
    image  = Image.open(item["image"]).convert("RGB")
    prompt = f"USER: <image>\n{item['question']} ASSISTANT: {item['answer']}"
    inputs = proc(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    labels = inputs["input_ids"].clone()
    out    = model(**inputs, labels=labels)
    # CE loss = -mean log prob; we want log prob = -CE * n_tokens
    return -out.loss


def npo_loss(log_prob_mu: torch.Tensor,
             log_prob_ref: float,
             beta: float) -> torch.Tensor:
    """
    NPO loss for one forget example.
    Pushes mu output away from ref output proportionally to beta.
    """
    diff = beta * (log_prob_mu - log_prob_ref)
    return -F.logsigmoid(-diff)


def check_lora_health(model, label: str) -> bool:
    nonzero = 0; total = 0
    for n, p in model.named_parameters():
        if "lora_B" in n:
            total += 1
            if p.abs().max().item() > 1e-8: nonzero += 1
    print(f"  [health:{label}] lora_B {nonzero}/{total} nonzero")
    return nonzero > 0


def quick_eval(ckpt_path: Path, n_forget: int = 20, n_retain: int = 20) -> dict:
    """Quick ForgetAcc / RetainAcc check."""
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    proc  = AutoProcessor.from_pretrained(LLAVA_BASE)
    base  = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_BASE, quantization_config=get_bnb(), device_map=DEVICE)
    model = PeftModel.from_pretrained(base, str(ckpt_path))
    model.eval()

    def score(split_dir, n):
        items   = load_items(split_dir)[:n]
        correct = 0
        for item in items:
            image  = Image.open(item["image"]).convert("RGB")
            prompt = f"USER: <image>\n{item['question']} ASSISTANT:"
            inputs = proc(text=prompt, images=image, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            resp = proc.decode(out[0], skip_special_tokens=True)
            if "ASSISTANT:" in resp: resp = resp.split("ASSISTANT:")[-1].strip()
            if item["answer"].lower() in resp.lower(): correct += 1
        return correct / len(items) if items else float("nan")

    fa = score(MLLMU_REAL_FORGET, n_forget)
    ra = score(MLLMU_REAL_RETAIN, n_retain)
    del model, base; torch.cuda.empty_cache()
    return {"forget_acc": fa, "retain_acc": ra, "forget_rate": 1.0-fa}


def train_npo(beta: float, lr: float, steps: int, out_dir: Path) -> dict:
    """Train one NPO configuration and return behavioural results."""
    ckpt_name = f"npo_beta{beta}_lr{lr}_steps{steps}"
    ckpt_path = out_dir / ckpt_name

    if (ckpt_path / "adapter_config.json").exists():
        print(f"  [cached] {ckpt_name}")
    else:
        print(f"\n  Training: beta={beta}  lr={lr}  steps={steps}")
        model, proc  = build_model()
        forget_items = load_items(MLLMU_REAL_FORGET)

        # Compute reference log probs ONCE from M0
        print("  Computing reference log probs...")
        ref_log_probs = {}
        model.eval()
        with torch.no_grad():
            for item in forget_items:
                ref_log_probs[item["entity"]] = compute_log_prob(
                    model, proc, item).item()
        model.train()

        trainable  = [p for p in model.parameters() if p.requires_grad]
        optimizer  = torch.optim.AdamW(trainable, lr=lr)
        log        = []
        step       = 0
        cycle      = 0
        grad_check = False

        while step < steps:
            item        = forget_items[cycle % len(forget_items)]
            cycle      += 1
            ref_lp      = ref_log_probs[item["entity"]]

            log_prob_mu = compute_log_prob(model, proc, item)
            loss        = npo_loss(log_prob_mu, ref_lp, beta)
            loss.backward()

            if not grad_check:
                has_grad = any(
                    p.grad is not None and p.grad.abs().max().item() > 1e-10
                    for n, p in model.named_parameters()
                    if "lora_B" in n and p.requires_grad)
                if not has_grad:
                    print("  [FATAL] No gradients reach lora_B"); sys.exit(1)
                print("  [grad-check] lora_B receives gradients")
                grad_check = True

            optimizer.step(); optimizer.zero_grad()
            step += 1
            log.append({"step": step, "loss": round(loss.item(), 4),
                        "log_prob_mu": round(log_prob_mu.item(), 4),
                        "ref_lp": round(ref_lp, 4)})

            if step % 10 == 0:
                print(f"  Step {step:3d}  loss={loss.item():.4f}  "
                      f"log_p_mu={log_prob_mu.item():.4f}  ref={ref_lp:.4f}")

        check_lora_health(model, f"final")
        ckpt_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_path))
        proc.save_pretrained(str(ckpt_path))
        meta = {"method":"npo_tuned","beta":beta,"lr":lr,"steps":steps,
                "base_model":LLAVA_BASE}
        with open(ckpt_path/"training_meta.json","w") as f:
            json.dump(meta, f, indent=2)
        with open(ckpt_path/"training_log.json","w") as f:
            json.dump(log, f, indent=2)
        del model; torch.cuda.empty_cache()
        print(f"  [saved] {ckpt_path}")

    # Evaluate
    print(f"  Evaluating {ckpt_name}...")
    results = quick_eval(ckpt_path)
    results.update({"beta": beta, "lr": lr, "steps": steps,
                    "checkpoint": str(ckpt_path)})

    # Add to exp_config reminder
    fa  = results["forget_acc"]
    ra  = results["retain_acc"]
    fr  = results["forget_rate"]
    print(f"  ForgetAcc={fa:.4f}  ForgetRate={fr:.4f}  RetainAcc={ra:.4f}")

    if fa < 0.5 and ra > 0.8:
        print("  *** SELECTIVE FORGETTING ACHIEVED — add to exp_config.py ***")
        print(f'  "npo_tuned_{ckpt_name.replace("/","_")}": '
              f'ROOT / "checkpoints" / "npo_tuned" / "{ckpt_name}"')
    elif fa < 0.5:
        print("  *** ForgetAcc < 0.5 but RetainAcc also dropped — try higher beta ***")
    else:
        print("  Under-forgetting — try higher beta or more steps")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta",  type=float, default=0.5)
    parser.add_argument("--lr",    type=float, default=1e-4)
    parser.add_argument("--steps", type=int,   default=100)
    parser.add_argument("--sweep", action="store_true",
                        help="Run all beta/step configurations")
    args = parser.parse_args()

    all_results = []
    configs = SWEEP_CONFIGS if args.sweep else [
        {"beta": args.beta, "lr": args.lr, "steps": args.steps}]

    for cfg in configs:
        r = train_npo(cfg["beta"], cfg["lr"], cfg["steps"], NPO_DIR)
        all_results.append(r)

    # Summary
    print("\n" + "="*60)
    print("NPO TUNING SWEEP SUMMARY")
    print("="*60)
    print(f"  {'Config':<30} {'F-Acc':>8} {'F-Rate':>8} {'Ret-Acc':>8}")
    print("  " + "-"*55)
    for r in all_results:
        label = f"beta={r['beta']} steps={r['steps']}"
        print(f"  {label:<30} {r['forget_acc']:>8.4f} "
              f"{r['forget_rate']:>8.4f} {r['retain_acc']:>8.4f}")

    best = min(all_results, key=lambda r: r["forget_acc"])
    print(f"\n  Best ForgetAcc: {best['forget_acc']:.4f} "
          f"at beta={best['beta']} steps={best['steps']}")

    out = NPO_DIR / "npo_tuning_summary.json"
    with open(out, "w") as f: json.dump(all_results, f, indent=2, default=str)
    print(f"  [saved] {out}")

    print("\n  After finding best config:")
    print("  1. Add checkpoint to exp_config.py as 'npo_tuned'")
    print("  2. Run E2_save_activations.py --methods npo_tuned")
    print("  3. Run E2_run_audit_per_entity.py --methods npo_tuned")


if __name__ == "__main__":
    main()
