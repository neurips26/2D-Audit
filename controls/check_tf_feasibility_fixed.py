import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)

print("Preparing quantization config...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model in 4-bit using quantization_config...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("Checking language model layers...")
try:
    n_layers = len(model.language_model.model.layers)
    print(f"Language layers found: {n_layers}")
except Exception as e:
    print("Could not access model.language_model.model.layers")
    print(e)
    raise SystemExit(1)

print("Checking hook capture on a language layer...")
box = {}

def hook_fn(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    box["shape"] = tuple(hidden.shape)

handle = model.language_model.model.layers[16].register_forward_hook(hook_fn)

try:
    vocab = model.config.text_config.vocab_size
    input_ids = torch.randint(10, min(vocab, 1000), (1, 12)).to(model.device)
    attention_mask = torch.ones_like(input_ids).to(model.device)
    labels = input_ids.clone()

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    handle.remove()

    print("Teacher-forced text forward appears supported.")
    print("Hook captured hidden shape:", box.get("shape"))
    print("FEASIBLE=True")

except Exception as e:
    handle.remove()
    print("Teacher-forced forward failed.")
    print(repr(e))
    print("FEASIBLE=False")
