"""
Runtime/file-system guard for PEFT/LoRA adapters.

Supported calls:
    assert_adapter_is_active(model)
    assert_adapter_is_active(model=model)
    assert_adapter_is_active(checkpoint_dir)
    assert_adapter_is_active(checkpoint_dir=checkpoint_dir)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch


def _find_model(args: Iterable[Any], kwargs: Dict[str, Any]) -> Optional[torch.nn.Module]:
    for key in ("model", "peft_model", "adapter_model", "wrapped_model", "module"):
        value = kwargs.get(key)
        if isinstance(value, torch.nn.Module):
            return value
    for value in args:
        if isinstance(value, torch.nn.Module):
            return value
    return None


def _find_checkpoint_dir(args: Iterable[Any], kwargs: Dict[str, Any]) -> Optional[Path]:
    for key in ("checkpoint_dir", "checkpoint", "adapter_dir", "adapter_path", "path"):
        value = kwargs.get(key)
        if isinstance(value, (str, Path)):
            return Path(value)
    for value in args:
        if isinstance(value, (str, Path)):
            return Path(value)
    return None


def _active_adapter_names(model: torch.nn.Module) -> list[str]:
    names: list[str] = []
    for attr in ("active_adapters", "active_adapter"):
        if not hasattr(model, attr):
            continue
        value = getattr(model, attr)
        try:
            value = value() if callable(value) else value
        except Exception:
            continue
        if value is None:
            continue
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, (list, tuple, set)):
            names.extend(str(x) for x in value)
    return list(dict.fromkeys(names))


def _inspect_loaded_model(
    model: torch.nn.Module,
    *,
    require_nonzero_b: bool,
    atol: float,
    expected_adapter: Optional[str],
) -> Dict[str, Any]:
    lora_a_names: list[str] = []
    lora_b_names: list[str] = []
    nonzero_lora_b: list[str] = []
    trainable_lora: list[str] = []

    for name, param in model.named_parameters():
        lname = name.lower()
        if "lora_a" in lname:
            lora_a_names.append(name)
        if "lora_b" in lname:
            lora_b_names.append(name)
            tensor = param.detach()
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"Non-finite values found in `{name}`.")
            if tensor.numel() and tensor.abs().max().item() > atol:
                nonzero_lora_b.append(name)
        if "lora_" in lname and param.requires_grad:
            trainable_lora.append(name)

    if not lora_a_names and not lora_b_names:
        raise RuntimeError("No LoRA parameters were found in the loaded model.")
    if not lora_b_names:
        raise RuntimeError("LoRA parameters were found, but no LoRA-B tensors were present.")
    if require_nonzero_b and not nonzero_lora_b:
        raise RuntimeError(
            f"All {len(lora_b_names)} LoRA-B tensors are zero or below "
            f"the activity tolerance ({atol:g})."
        )

    active_adapters = _active_adapter_names(model)
    if expected_adapter and active_adapters and expected_adapter not in active_adapters:
        raise RuntimeError(
            f"Expected adapter `{expected_adapter}` is not active. "
            f"Active adapter(s): {active_adapters}"
        )

    has_peft_config = hasattr(model, "peft_config")
    if not has_peft_config:
        for nested_attr in ("base_model", "model"):
            nested = getattr(model, nested_attr, None)
            if nested is not None and hasattr(nested, "peft_config"):
                has_peft_config = True
                break

    return {
        "adapter_active": True,
        "inspection_mode": "loaded_model",
        "has_peft_config": has_peft_config,
        "active_adapters": active_adapters,
        "lora_a_tensors": len(lora_a_names),
        "lora_b_tensors": len(lora_b_names),
        "n_lora_B": len(lora_b_names),
        "nonzero_lora_b_tensors": len(nonzero_lora_b),
        "n_nonzero_lora_B": len(nonzero_lora_b),
        "trainable_lora_tensors": len(trainable_lora),
        "activity_tolerance": atol,
    }


def _load_adapter_state_dict(checkpoint_dir: Path):
    safetensors_path = checkpoint_dir / "adapter_model.safetensors"
    bin_path = checkpoint_dir / "adapter_model.bin"

    if safetensors_path.exists():
        from safetensors.torch import load_file
        return load_file(str(safetensors_path), device="cpu"), safetensors_path

    if bin_path.exists():
        try:
            obj = torch.load(str(bin_path), map_location="cpu", weights_only=True)
        except TypeError:
            obj = torch.load(str(bin_path), map_location="cpu")
        if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
            obj = obj["state_dict"]
        if not isinstance(obj, dict):
            raise RuntimeError(f"Unsupported adapter checkpoint structure in {bin_path}.")
        return obj, bin_path

    raise RuntimeError(
        f"No adapter weights found in {checkpoint_dir}. Expected "
        "`adapter_model.safetensors` or `adapter_model.bin`."
    )


def _inspect_checkpoint(
    checkpoint_dir: Path,
    *,
    require_nonzero_b: bool,
    atol: float,
) -> Dict[str, Any]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"Adapter checkpoint directory does not exist: {checkpoint_dir}")

    config_path = checkpoint_dir / "adapter_config.json"
    if not config_path.exists():
        raise RuntimeError(f"Missing adapter_config.json in {checkpoint_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    state_dict, weights_path = _load_adapter_state_dict(checkpoint_dir)

    lora_a_names: list[str] = []
    lora_b_names: list[str] = []
    nonzero_lora_b: list[str] = []
    nonfinite_names: list[str] = []

    for name, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        lname = str(name).lower()
        if "lora_a" in lname:
            lora_a_names.append(str(name))
        if "lora_b" in lname:
            lora_b_names.append(str(name))
            tensor = value.detach().float()
            if not torch.isfinite(tensor).all():
                nonfinite_names.append(str(name))
            elif tensor.numel() and tensor.abs().max().item() > atol:
                nonzero_lora_b.append(str(name))

    if nonfinite_names:
        raise RuntimeError(
            "Non-finite values found in adapter tensors: "
            + ", ".join(nonfinite_names[:5])
        )
    if not lora_a_names and not lora_b_names:
        raise RuntimeError(f"No LoRA tensors were found in {weights_path}.")
    if not lora_b_names:
        raise RuntimeError(f"No LoRA-B tensors were found in {weights_path}.")
    if require_nonzero_b and not nonzero_lora_b:
        raise RuntimeError(
            f"All {len(lora_b_names)} LoRA-B tensors in {weights_path.name} "
            f"are zero or below the activity tolerance ({atol:g})."
        )

    return {
        "adapter_active": True,
        "inspection_mode": "checkpoint_files",
        "checkpoint_dir": str(checkpoint_dir),
        "adapter_config": str(config_path),
        "weights_file": str(weights_path),
        "peft_type": config.get("peft_type"),
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "target_modules": config.get("target_modules"),
        "lora_a_tensors": len(lora_a_names),
        "lora_b_tensors": len(lora_b_names),
        "n_lora_B": len(lora_b_names),
        "nonzero_lora_b_tensors": len(nonzero_lora_b),
        "n_nonzero_lora_B": len(nonzero_lora_b),
        "activity_tolerance": atol,
    }


def assert_adapter_is_active(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    require_nonzero_b = bool(kwargs.get("require_nonzero_b", True))
    atol = float(kwargs.get("atol", 1e-8))
    expected_adapter = kwargs.get("expected_adapter")

    model = _find_model(args, kwargs)
    if model is not None:
        return _inspect_loaded_model(
            model,
            require_nonzero_b=require_nonzero_b,
            atol=atol,
            expected_adapter=expected_adapter,
        )

    checkpoint_dir = _find_checkpoint_dir(args, kwargs)
    if checkpoint_dir is not None:
        return _inspect_checkpoint(
            checkpoint_dir,
            require_nonzero_b=require_nonzero_b,
            atol=atol,
        )

    raise RuntimeError(
        "assert_adapter_is_active() expected either a loaded torch model "
        "or an adapter checkpoint directory."
    )
