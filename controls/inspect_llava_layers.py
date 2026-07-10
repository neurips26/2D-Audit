import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("\n=== TOP LEVEL ATTRIBUTES CONTAINING model/language/text ===")
for name in dir(model):
    if any(x in name.lower() for x in ["model", "language", "text", "llm"]):
        print(name)

print("\n=== NAMED MODULES CONTAINING layers ===")
count = 0
for name, module in model.named_modules():
    if "layers" in name.lower() or "language" in name.lower() or "text" in name.lower():
        print(name, "->", type(module).__name__)
        count += 1
        if count >= 120:
            break

print("\n=== POSSIBLE DECODER LAYER MODULES ===")
for name, module in model.named_modules():
    if any(pattern in name for pattern in [
        "language_model.model.layers.16",
        "language_model.layers.16",
        "model.layers.16",
        "text_model.layers.16",
        "model.language_model.layers.16"
    ]):
        print(name, "->", type(module).__name__)
