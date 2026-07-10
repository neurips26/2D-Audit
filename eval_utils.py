"""
eval_utils.py
Shared helpers: data loading, model loading, response scoring, image attacks.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter


# ------------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------------

def clean_model_text(text: str) -> str:
    """Clean SentencePiece/tokenizer artefacts from generated output."""
    if text is None:
        return ""
    text = str(text)
    text = text.replace(" ", " ")
    text = text.replace("<s>", " ").replace("</s>", " ")
    text = text.replace("<pad>", " ").replace("<unk>", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_mllmu_split(split_dir: Path) -> list[dict]:
    """
    Load a forget or retain split from disk.

    Expected layout (MLLMU-Bench style):
        split_dir/
            entity_000/
                image.jpg          (or *.png)
                qa.json            [{"question": "...", "answer": "...", "entity": "..."}]
            entity_001/
                ...

    Alternatively, a single flat JSON:
        split_dir/annotations.json
        [{"image": "...", "question": "...", "answer": "...", "entity": "..."}]

    Returns list of dicts:
        {
          "entity":   str,
          "image":    Path,
          "question": str,
          "answer":   str,          # gold / reference answer
          "aliases":  list[str],    # entity name variants (from entity field)
        }
    """
    split_dir = Path(split_dir)

    # -- Try flat annotations.json first --------------------------------------
    ann_file = split_dir / "annotations.json"
    if ann_file.exists():
        with open(ann_file) as f:
            raw = json.load(f)
        items = []
        for r in raw:
            img = split_dir / r["image"]
            items.append({
                "entity":   r.get("entity", r.get("name", "unknown")),
                "image":    img,
                "question": r["question"],
                "answer":   r.get("answer", r.get("gt", "")),
                "aliases":  _build_aliases(r.get("entity", r.get("name", ""))),
            })
        print(f"[data] Loaded {len(items)} items from {ann_file}")
        return items

    # -- Try per-entity subdirectory layout -----------------------------------
    items = []
    for entity_dir in sorted(split_dir.iterdir()):
        if not entity_dir.is_dir():
            continue
        # Find image
        imgs = list(entity_dir.glob("*.jpg")) + list(entity_dir.glob("*.png"))
        if not imgs:
            continue
        img = imgs[0]

        # Find QA file
        qa_file = entity_dir / "qa.json"
        if not qa_file.exists():
            qa_file_candidates = list(entity_dir.glob("*.json"))
            if not qa_file_candidates:
                continue
            qa_file = qa_file_candidates[0]

        with open(qa_file) as f:
            qa_data = json.load(f)

        if isinstance(qa_data, list):
            for q in qa_data:
                items.append({
                    "entity":   q.get("entity", entity_dir.name),
                    "image":    img,
                    "question": q["question"],
                    "answer":   q.get("answer", q.get("gt", "")),
                    "aliases":  _build_aliases(q.get("entity", entity_dir.name)),
                })
        elif isinstance(qa_data, dict):
            items.append({
                "entity":   qa_data.get("entity", entity_dir.name),
                "image":    img,
                "question": qa_data["question"],
                "answer":   qa_data.get("answer", qa_data.get("gt", "")),
                "aliases":  _build_aliases(qa_data.get("entity", entity_dir.name)),
            })

    if not items:
        raise FileNotFoundError(
            f"[data] No items found in {split_dir}.\n"
            "Check that the split directory contains entity subdirs with images "
            "and qa.json files, or a flat annotations.json."
        )
    print(f"[data] Loaded {len(items)} items from {split_dir}")
    return items


def _build_aliases(entity_name: str) -> list[str]:
    """Build a list of name variants for fuzzy entity matching."""
    name = entity_name.strip()
    aliases = {name, name.lower()}
    # Last name only
    parts = name.split()
    if len(parts) >= 2:
        aliases.add(parts[-1])
        aliases.add(parts[-1].lower())
        aliases.add(parts[0])
        aliases.add(parts[0].lower())
    # Remove diacritics
    norm = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in norm if unicodedata.category(c) != "Mn")
    aliases.add(ascii_name)
    aliases.add(ascii_name.lower())
    return [a for a in aliases if a]


from adapter_guard import assert_adapter_is_active

# ------------------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------------------

def load_llava_model(base_model_name: str, checkpoint_dir: Optional[Path], device: str = "cuda"):
    """
    Load HuggingFace LLaVA base model and optional PEFT LoRA adapter.
    This matches adapters trained on llava-hf/llava-1.5-7b-hf.
    """
    import torch
    from transformers import BitsAndBytesConfig, AutoProcessor, LlavaForConditionalGeneration
    from peft import PeftModel

    print(f"[model] Loading HF LLaVA base: {base_model_name}")

    processor = AutoProcessor.from_pretrained(base_model_name)

    model = LlavaForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        if (checkpoint_dir / "adapter_config.json").exists():
            activity = assert_adapter_is_active(checkpoint_dir)
            print(
                "[model] Applying active PEFT LoRA adapter: "
                f"{checkpoint_dir} "
                f"({activity['n_nonzero_lora_B']}/"
                f"{activity['n_lora_B']} nonzero LoRA-B tensors)"
            )
            model = PeftModel.from_pretrained(
                model,
                str(checkpoint_dir),
                is_trainable=False,
            )
        else:
            print(f"[WARN] Adapter not found, using base only: {checkpoint_dir}")

    model.eval()
    return model, processor, None, 4096

def load_blip2_model(base_model_name: str, checkpoint_dir: Optional[Path], device: str = "cuda"):
    """Load BLIP-2-OPT-2.7B with optional PEFT adapter."""
    import torch
    from pathlib import Path
    from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
    from peft import PeftModel

    print(f"[model] Loading base BLIP-2: {base_model_name}")
    processor = Blip2Processor.from_pretrained(base_model_name, use_fast=False)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = Blip2ForConditionalGeneration.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        if checkpoint_dir.exists():
            activity = assert_adapter_is_active(checkpoint_dir)
            print(
                "[model] Loading active BLIP-2 adapter: "
                f"{checkpoint_dir} "
                f"({activity['n_nonzero_lora_B']}/"
                f"{activity['n_lora_B']} nonzero LoRA-B tensors)"
            )
            model = PeftModel.from_pretrained(
                model,
                str(checkpoint_dir),
                is_trainable=False,
            )
        else:
            print(f"[warn] BLIP-2 checkpoint not found: {checkpoint_dir}")

    model.eval()
    return model, processor


# ------------------------------------------------------------------------
# INFERENCE
# ------------------------------------------------------------------------

def run_llava_inference(
    model, tokenizer, image_processor,
    question: str,
    image: Image.Image,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> str:
    """
    HF LLaVA inference. Here tokenizer is actually the AutoProcessor.
    """
    import torch

    processor = tokenizer

    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    # Decode only generated continuation where possible
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0][prompt_len:]
    response = processor.decode(gen_ids, skip_special_tokens=True)

    return clean_model_text(response)

def run_blip2_inference(model, processor, question: str, image, max_new_tokens: int, device: str):
    """
    BLIP-2 inference. Decode only newly generated tokens so the prompt/question
    is not counted as the model answer.
    """
    import torch

    prompt = f"Question: {question} Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0][prompt_len:]

    response = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # Fallback: some BLIP-2 variants return only generated tokens
    if response == "" and output_ids.shape[1] <= prompt_len:
        response = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    return clean_model_text(response)

def _score_norm(text: str) -> str:
    """Normalize text aggressively for answer matching only."""
    if text is None:
        return ""
    text = clean_model_text(str(text)).lower()
    text = text.replace(" ", " ")
    text = text.replace("—", " ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def _score_words(text: str) -> set[str]:
    """Token set for loose word matching."""
    text = clean_model_text(str(text)).lower()
    text = text.replace(" ", " ")
    text = text.replace("—", " ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return set(re.findall(r"[a-z0-9]+", text))


REFUSAL_PHRASES = [
    "i don't know",
    "i do not know",
    "unknown",
    "cannot answer",
    "can't answer",
    "not sure",
    "unable to answer",
    "sorry",
]

def score_response(response: str, item: dict) -> dict:
    """
    Score a model response for a single item.
    Correct if entity alias or gold answer appears after robust normalization.
    """
    response = clean_model_text(response)
    resp_lower = response.lower().strip()
    resp_norm = _score_norm(response)
    resp_words = _score_words(response)

    # Check refusal
    is_refusal = any(ph in resp_lower for ph in REFUSAL_PHRASES)

    # Entity/alias matching
    entity_found = False
    aliases = item.get("aliases", []) + [item.get("entity", "")]
    for alias in aliases:
        alias_norm = _score_norm(alias)
        if alias_norm and alias_norm in resp_norm:
            entity_found = True
            break

        alias_words = _score_words(alias)
        if alias_words and alias_words.issubset(resp_words):
            entity_found = True
            break

    # Gold answer matching
    gold = item.get("answer", "")
    gold_found = False
    if gold:
        gold_norm = _score_norm(gold)
        if gold_norm and gold_norm in resp_norm:
            gold_found = True
        else:
            gold_words = _score_words(gold)
            # Require all words for short answers; for longer answers, require at least 2 meaningful words.
            if gold_words and len(gold_words) <= 5:
                gold_found = gold_words.issubset(resp_words)
            elif len(gold_words) > 5:
                gold_found = len(gold_words.intersection(resp_words)) >= 2

    correct = entity_found or gold_found

    return {
        "correct":  bool(correct),
        "refusal":  bool(is_refusal),
        "response": response,
        "entity":   item["entity"],
    }

def aggregate_scores(score_list: list[dict]) -> dict:
    """
    Aggregate per-item scores.
    Returns: {correct_rate, refusal_rate, n}
    """
    n = len(score_list)
    if n == 0:
        return {"correct_rate": float("nan"), "refusal_rate": float("nan"), "n": 0}
    correct  = sum(s["correct"] for s in score_list)
    refusals = sum(s["refusal"] for s in score_list)
    return {
        "correct_rate":  correct / n,
        "refusal_rate":  refusals / n,
        "n":             n,
    }


# ------------------------------------------------------------------------
# IMAGE ATTACKS
# ------------------------------------------------------------------------

def attack_crop(image: Image.Image, scale: float = 0.6) -> Image.Image:
    """Centre-crop to `scale` fraction of original dimensions."""
    w, h   = image.size
    new_w  = int(w * scale)
    new_h  = int(h * scale)
    left   = (w - new_w) // 2
    top    = (h - new_h) // 2
    return image.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.LANCZOS)


def attack_perturb(image: Image.Image, std: float = 25.0) -> Image.Image:
    """Add Gaussian noise with the given standard deviation."""
    arr  = np.array(image).astype(np.float32)
    noise = np.random.normal(0, std, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


# ------------------------------------------------------------------------
# PROMPT TEMPLATES
# ------------------------------------------------------------------------

def make_rephrase_question(item: dict) -> str:
    """
    Generate a rephrased query that avoids using the entity name directly.
    Uses the original question text with name-like tokens removed / generalised.
    """
    q = item["question"]
    # Replace entity name with a generic reference
    for alias in sorted(item["aliases"], key=len, reverse=True):
        pattern = re.compile(re.escape(alias), re.IGNORECASE)
        q = pattern.sub("this person", q, count=1)
    # If the question still looks like a direct name query, generalise further
    q = re.sub(r"\bwho is\b", "describe", q, flags=re.IGNORECASE)
    q = re.sub(r"\bwhat is the name of\b", "identify", q, flags=re.IGNORECASE)
    return q.strip() or "Describe the individual shown in the image."


DIRECT_TEMPLATE   = "{question}"
REPHRASE_TEMPLATE = None   # generated per-item via make_rephrase_question()


# ------------------------------------------------------------------------
# CSV / JSON I/O helpers
# ------------------------------------------------------------------------

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[saved] {path}")


def load_json(path):
    with open(path) as f:
        return json.load(f)


