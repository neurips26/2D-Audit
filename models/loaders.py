"""
Model loaders for LLaVA-1.5-7B and BLIP-2-OPT-2.7B.
Handles 4-bit quantisation and LoRA adapter management.
"""

import torch
import logging
from typing import Optional, Dict, Any

from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
)

from adapter_guard import assert_adapter_is_active

from config import (
    LLAVA_MODEL_ID,
    BLIP2_MODEL_ID,
    UnlearningConfig,
)

logger = logging.getLogger(__name__)


# -- BitsAndBytes 4-bit config -------------------------------------------------
def _bnb_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# -- LLaVA-1.5 -----------------------------------------------------------------
def load_llava(
    model_id: str = LLAVA_MODEL_ID,
    load_in_4bit: bool = True,
    device_map: str = "auto",
    adapter_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load LLaVA-1.5-7B (optionally quantised to 4-bit).
    Returns dict with keys: model, processor.
    """
    logger.info(f"Loading LLaVA from {model_id}  (4bit={load_in_4bit})")

    quant_cfg = _bnb_4bit_config() if load_in_4bit else None

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_cfg,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    processor = LlavaProcessor.from_pretrained(model_id)

    if adapter_path is not None:
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        activity = assert_adapter_is_active(adapter_path)
        logger.info(
            "Adapter activity verified: "
            f"{activity['n_nonzero_lora_B']}/"
            f"{activity['n_lora_B']} LoRA-B tensors are nonzero"
        )
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
        )

        # Do not merge adapters into a quantized base model.
        # The project merge-fidelity diagnostic showed that 4-bit
        # merge_and_unload does not preserve model outputs faithfully.

    model.eval()
    logger.info("LLaVA loaded OK")
    return {"model": model, "processor": processor, "arch": "llava"}


def load_llava_with_lora(
    base_model_id: str = LLAVA_MODEL_ID,
    cfg: Optional[UnlearningConfig] = None,
    load_in_4bit: bool = True,
) -> Dict[str, Any]:
    """Load LLaVA + attach fresh LoRA adapters for unlearning."""
    if cfg is None:
        cfg = UnlearningConfig()

    bundle = load_llava(base_model_id, load_in_4bit=load_in_4bit)
    model = bundle["model"]

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    bundle["model"] = model
    return bundle


# -- BLIP-2 OPT-2.7B ----------------------------------------------------------
def load_blip2(
    model_id: str = BLIP2_MODEL_ID,
    load_in_4bit: bool = True,
    device_map: str = "auto",
    adapter_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load BLIP-2-OPT-2.7B (optionally quantised to 4-bit).
    Returns dict with keys: model, processor.
    """
    logger.info(f"Loading BLIP-2 from {model_id}  (4bit={load_in_4bit})")

    quant_cfg = _bnb_4bit_config() if load_in_4bit else None

    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_cfg,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    processor = Blip2Processor.from_pretrained(model_id)

    if adapter_path is not None:
        logger.info(f"Loading LoRA adapter from {adapter_path}")
        activity = assert_adapter_is_active(adapter_path)
        logger.info(
            "Adapter activity verified: "
            f"{activity['n_nonzero_lora_B']}/"
            f"{activity['n_lora_B']} LoRA-B tensors are nonzero"
        )
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
        )

        # Do not merge adapters into a quantized base model.
        # The project merge-fidelity diagnostic showed that 4-bit
        # merge_and_unload does not preserve model outputs faithfully.

    model.eval()
    logger.info("BLIP-2 loaded OK")
    return {"model": model, "processor": processor, "arch": "blip2"}


def load_blip2_with_lora(
    base_model_id: str = BLIP2_MODEL_ID,
    cfg: Optional[UnlearningConfig] = None,
    load_in_4bit: bool = True,
) -> Dict[str, Any]:
    """Load BLIP-2 + attach fresh LoRA adapters for unlearning."""
    if cfg is None:
        cfg = UnlearningConfig()

    bundle = load_blip2(base_model_id, load_in_4bit=load_in_4bit)
    model = bundle["model"]

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    bundle["model"] = model
    return bundle


# -- Convenience loader --------------------------------------------------------
def load_model(
    arch: str,
    load_in_4bit: bool = True,
    adapter_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load either 'llava' or 'blip2' by name."""
    if arch == "llava":
        return load_llava(load_in_4bit=load_in_4bit, adapter_path=adapter_path)
    elif arch == "blip2":
        return load_blip2(load_in_4bit=load_in_4bit, adapter_path=adapter_path)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def save_adapter(model, path: str) -> None:
    """Save LoRA adapter weights."""
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path)
        logger.info(f"Adapter saved to {path}")


def free_model(bundle: Dict[str, Any]) -> None:
    """Release GPU memory."""
    import gc
    del bundle["model"]
    gc.collect()
    torch.cuda.empty_cache()

