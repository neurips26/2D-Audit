import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading model in 4-bit...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    load_in_4bit=True,
    device_map="auto"
)

print("Checking language model layers...")
try:
    n_layers = len(model.language_model.model.layers)
    print(f"Language layers found: {n_layers}")
except Exception as e:
    print("Could not access model.language_model.model.layers")
    print(e)
    raise SystemExit(1)

print("Checking forward with labels/output_hidden_states...")
try:
    vocab = model.config.text_config.vocab_size
    input_ids = torch.randint(0, min(vocab, 1000), (1, 12)).to(model.device)
    attention_mask = torch.ones_like(input_ids).to(model.device)
    labels = input_ids.clone()

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )

    print("Teacher-forced forward appears supported.")
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        print(f"Top-level hidden_states length: {len(out.hidden_states)}")
    else:
        print("No top-level hidden_states returned, but hooks may still work.")
    print("FEASIBLE=True")

except Exception as e:
    print("Teacher-forced forward failed.")
    print(repr(e))
    print("FEASIBLE=False")
