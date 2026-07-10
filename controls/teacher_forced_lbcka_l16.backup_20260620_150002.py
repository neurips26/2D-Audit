from adapter_guard import assert_adapter_is_active
import json
import gc
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

try:
    from probes.cka_rsa import linear_cka
    USE_PROJECT_CKA = True
except Exception:
    USE_PROJECT_CKA = False

BASE_MODEL = "llava-hf/llava-1.5-7b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HOOK_LAYER = 16

OUTDIR = Path("outputs/controls")
OUTDIR.mkdir(parents=True, exist_ok=True)

ADAPTERS = {
    "ga": r"checkpoints\llava_ga_adapter",
    "npo": r"checkpoints\llava_npo_adapter",
    "mmunlearner": r"checkpoints\llava_mmunlearner_adapter",
    "cagul": r"checkpoints\llava_cagul_adapter",
    # Keep MANU optional. It is useful for diagnostics but do not overclaim.
}

GEN_TOKEN_MAIN_LB = {
    "ga": 0.8987,
    "npo": 0.8972,
    "mmunlearner": 0.8760,
    "cagul": 0.9035,
    "manu": 0.5212,
}

def fallback_linear_cka(X, Y):
    X = X.double()
    Y = Y.double()
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    gram_xy = torch.linalg.norm(Y.T @ X, ord="fro") ** 2
    gram_xx = torch.linalg.norm(X.T @ X, ord="fro")
    gram_yy = torch.linalg.norm(Y.T @ Y, ord="fro")
    denom = gram_xx * gram_yy
    return float((gram_xy / denom).item()) if denom > 1e-12 else float("nan")

def compute_cka(X, Y):
    if USE_PROJECT_CKA:
        return float(linear_cka(X, Y))
    return fallback_linear_cka(X, Y)

def find_examples(max_n=5):
    roots = [
        Path("data/mllmu_real/forget"),
        Path("data/mllmu_bench/forget"),
        Path("data/clear/forget"),
    ]

    examples = []
    for root in roots:
        if not root.exists():
            continue

        for ent_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            img = ent_dir / "image.jpg"
            qa = ent_dir / "qa_pairs.json"
            if not img.exists() or not qa.exists():
                continue

            try:
                qas = json.loads(qa.read_text(encoding="utf-8"))
            except Exception:
                continue

            if isinstance(qas, dict):
                qas = qas.get("qa_pairs", qas.get("questions", qas.get("data", [])))

            if not isinstance(qas, list) or not qas:
                continue

            first = qas[0]
            if not isinstance(first, dict):
                continue

            question = (
                first.get("question")
                or first.get("q")
                or first.get("prompt")
                or "Who is this person?"
            )
            answer = (
                first.get("answer")
                or first.get("a")
                or first.get("target")
                or first.get("label")
                or ent_dir.name.replace("_", " ")
            )

            examples.append({
                "entity": ent_dir.name,
                "image": str(img),
                "question": str(question),
                "answer": str(answer),
                "source": str(root),
            })

            if len(examples) >= max_n:
                return examples

    return examples

def load_base_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model

def load_adapter_model(adapter_path):
    base = load_base_model()
    assert_adapter_is_active(adapter_path)
    model = PeftModel.from_pretrained(
        base,
        str(adapter_path),
        is_trainable=False,
    )
    model.eval()
    return model

def get_layer(model):
    # Works for current installed transformers version:
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers[HOOK_LAYER]

    # Works for PEFT wrapper:
    if hasattr(model, "base_model"):
        inner = model.base_model.model
        if hasattr(inner, "model") and hasattr(inner.model, "language_model"):
            return inner.model.language_model.layers[HOOK_LAYER]

    raise RuntimeError("Could not find language layer path for this model.")

def extract_tf_activation(model, processor, item):
    image = Image.open(item["image"]).convert("RGB")

    # Full teacher-forced sequence: same prompt + same answer tokens for M0 and Mu.
    prompt = f"USER: <image>\n{item['question']}\nASSISTANT: {item['answer']}"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    # Move tensors to model's first device
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    labels = inputs["input_ids"].clone()

    # Pool only final answer-token span approximately.
    answer_ids = processor.tokenizer(
        item["answer"],
        add_special_tokens=False
    ).input_ids
    target_len = max(1, min(len(answer_ids), inputs["input_ids"].shape[1]))

    box = {}

    def hook_fn(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        # hidden shape: [batch, seq, dim]
        box["h"] = hidden[0, -target_len:, :].mean(dim=0).detach().cpu().float()

    layer = get_layer(model)
    handle = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(**inputs, labels=labels)

    handle.remove()

    if "h" not in box:
        raise RuntimeError("Hook did not capture hidden state")

    return box["h"]

print("=== TEACHER-FORCED LB-CKA CONTROL ===")
print("Finding examples...")
examples = find_examples(max_n=5)

if len(examples) < 3:
    raise SystemExit(f"ERROR: Found only {len(examples)} examples. Need at least 3, preferably 5.")

print(f"Using {len(examples)} examples:")
for e in examples:
    print(" ", e["entity"], "|", e["answer"], "|", e["source"])

print("\nLoading processor...")
processor = AutoProcessor.from_pretrained(BASE_MODEL, use_fast=False)

print("Loading original/base model...")
m0 = load_base_model()

base_acts = []
usable_examples = []

print("\nExtracting base activations...")
for item in examples:
    try:
        h = extract_tf_activation(m0, processor, item)
        base_acts.append(h)
        usable_examples.append(item)
        print("  OK base:", item["entity"])
    except Exception as e:
        print("  FAIL base:", item["entity"], "->", repr(e))

if len(base_acts) < 3:
    raise SystemExit("ERROR: Fewer than 3 usable base activations.")

results = {}

for method, adapter_path in ADAPTERS.items():
    if not Path(adapter_path).exists():
        print(f"\nSkipping {method}: adapter path missing: {adapter_path}")
        results[method] = {"status": "missing_adapter", "adapter": adapter_path}
        continue

    print(f"\n=== Method: {method} ===")
    try:
        mu = load_adapter_model(adapter_path)
    except Exception as e:
        print("  FAIL loading adapter:", repr(e))
        results[method] = {"status": "load_failed", "error": repr(e)}
        continue

    mu_acts = []
    kept_base = []
    kept_entities = []

    for h0, item in zip(base_acts, usable_examples):
        try:
            hu = extract_tf_activation(mu, processor, item)
            kept_base.append(h0)
            mu_acts.append(hu)
            kept_entities.append(item["entity"])
            print("  OK:", item["entity"])
        except Exception as e:
            print("  FAIL:", item["entity"], "->", repr(e))

    if len(mu_acts) < 3:
        results[method] = {
            "status": "insufficient",
            "n": len(mu_acts),
            "entities": kept_entities,
        }
    else:
        X = torch.stack(kept_base)
        Y = torch.stack(mu_acts)
        cka = compute_cka(X, Y)

        results[method] = {
            "status": "ok",
            "n": len(mu_acts),
            "entities": kept_entities,
            "tf_lb_cka_l16": cka,
            "gen_token_main_lb_cka": GEN_TOKEN_MAIN_LB.get(method),
        }

        print(f"  TF LB-CKA L{HOOK_LAYER}: {cka:.4f}")
        print(f"  Main generated-token LB-CKA: {GEN_TOKEN_MAIN_LB.get(method)}")

    del mu
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

out = {
    "control": "teacher_forced_lb_cka",
    "hook_layer": HOOK_LAYER,
    "examples_requested": len(examples),
    "examples_used_base": len(base_acts),
    "examples": usable_examples,
    "used_project_linear_cka": USE_PROJECT_CKA,
    "results": results,
}

outpath = OUTDIR / "teacher_forced_lbcka_l16_results.json"
outpath.write_text(json.dumps(out, indent=2), encoding="utf-8")

print("\n=== SUMMARY ===")
print(f"{'Method':<14} {'n':>3} {'TF-L16':>10} {'Gen-main':>10} {'status':>12}")
for m, r in results.items():
    print(
        f"{m:<14} "
        f"{str(r.get('n','-')):>3} "
        f"{str(round(r.get('tf_lb_cka_l16', float('nan')), 4)) if r.get('status')=='ok' else '-':>10} "
        f"{str(r.get('gen_token_main_lb_cka','-')):>10} "
        f"{r.get('status','?'):>12}"
    )

print("\nSaved:", outpath)
