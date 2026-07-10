import sys
import types
import math
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

def inspect_adapter_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    sf = checkpoint_dir / "adapter_model.safetensors"

    from safetensors.torch import load_file
    state = load_file(str(sf), device="cpu")

    vals = []
    for name, tensor in state.items():
        if "lora_b" in name.lower():
            m = float(tensor.float().abs().max().item())
            vals.append(m)

    nonzero = sum(math.isfinite(v) and v > 0 for v in vals)

    if not vals or nonzero == 0:
        raise RuntimeError(f"Inactive adapter: {checkpoint_dir}")

    return {
        "n_lora_B": len(vals),
        "n_nonzero_lora_B": nonzero,
        "max_abs_lora_B": max(vals),
        "active": True,
    }

guard = types.ModuleType("adapter_guard")
guard.assert_adapter_is_active = inspect_adapter_checkpoint
sys.modules["adapter_guard"] = guard

from PIL import Image
import exp_config
import eval_config
from eval_utils import load_mllmu_split, load_llava_model, run_llava_inference

items = load_mllmu_split(eval_config.FORGET_DIR)
item = items[0]
image = Image.open(item["image"]).convert("RGB")

tests = {
    "BASE": None,
    "GA-attn4-50": exp_config.LLAVA_ADAPTERS["ga_attn4_50"],
    "NPO": exp_config.LLAVA_ADAPTERS["npo"],
}

print("=" * 100)
print("QUESTION:", item["question"])
print("EXPECTED:", item.get("answer", item.get("answers", "")))
print("IMAGE:", item["image"])
print("=" * 100)

for name, checkpoint in tests.items():
    print(f"\n[{name}] loading: {checkpoint}")

    model, processor, _, _ = load_llava_model(
        eval_config.LLAVA_BASE_MODEL,
        checkpoint,
        eval_config.DEVICE,
    )

    response = run_llava_inference(
        model,
        processor,
        None,
        item["question"],
        image,
        max_new_tokens=32,
        temperature=0.0,
    )

    print(f"[{name}] RESPONSE: {response!r}")

    del model
    import torch
    torch.cuda.empty_cache()

print("\nOVERALL VERDICT: PASS - diagnostic completed")
