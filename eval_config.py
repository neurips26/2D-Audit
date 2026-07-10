"""
eval_config.py
Inference-only behavioural and recovery evaluation config.

Edit this file first:
1. ROOT
2. CHECKPOINT_DIRS
3. If needed, model IDs
"""

import os
from pathlib import Path


# ------------------------------------------------------------------------
# Root
# ------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent


# ------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------

DATA_ROOT = ROOT / "data" / "mllmu_real"
FORGET_DIR = DATA_ROOT / "forget"
RETAIN_DIR = DATA_ROOT / "retain"


# ------------------------------------------------------------------------
# Architecture / model
# ------------------------------------------------------------------------

ARCH = "llava"  # run "llava" first; use "blip2" later
DEVICE = "cuda"

LLAVA_BASE_MODEL = "llava-hf/llava-1.5-7b-hf"
BLIP2_BASE_MODEL = "Salesforce/blip2-opt-2.7b"

LOAD_IN_4BIT = False
SEED = 42


# ------------------------------------------------------------------------
# Checkpoints
# ------------------------------------------------------------------------
# no_unlearn = None means base model only.
# LoRA methods should point to adapter directory.
# MANU may be full-model checkpoint; loader should auto-detect.

CHECKPOINT_DIRS = {
    "no_unlearn": None,

    "ga": ROOT / "checkpoints" / "llava_ga_adapter",
    "npo": ROOT / "checkpoints" / "llava_npo_adapter",
    "mmunlearner": ROOT / "checkpoints" / "llava_mmunlearner_adapter",
    "cagul": ROOT / "checkpoints" / "llava_cagul_adapter",
    "manu": ROOT / "checkpoints" / "llava_manu_adapter",
}

ALL_METHODS = [
    "no_unlearn",
    "ga",
    "npo",
    "mmunlearner",
    "cagul",
    "manu",
]

RECOVERY_METHODS = [
    "ga",
    "npo",
    "mmunlearner",
    "cagul",
    "manu",
]

METHOD_DISPLAY_NAMES = {
    "no_unlearn": "No-unl.",
    "ga": "GA",
    "npo": "NPO",
    "mmunlearner": "MMUn.",
    "cagul": "CAGUL",
    "manu": "MANU",
}


# ------------------------------------------------------------------------
# Existing CRP / CNIS values from paper
# ------------------------------------------------------------------------
# Used for merge/correlation only. Not recomputed here.

EXISTING_CRP = {
    "no_unlearn": {
        "ve": 1.0000,
        "br": 1.0000,
        "lb": 1.0000,
        "cnis": 0.870,
        "retain": None,
    },
    "ga": {
        "ve": 0.9973,
        "br": 0.9910,
        "lb": 0.8987,
        "cnis": 0.869,
        "retain": 1.000,
    },
    "npo": {
        "ve": 0.9970,
        "br": 0.9861,
        "lb": 0.8972,
        "cnis": 0.872,
        "retain": 1.000,
    },
    "mmunlearner": {
        "ve": 0.9972,
        "br": 0.9863,
        "lb": 0.8760,
        "cnis": 0.869,
        "retain": 1.000,
    },
    "cagul": {
        "ve": 0.9972,
        "br": 0.9876,
        "lb": 0.9035,
        "cnis": 0.858,
        "retain": 1.000,
    },
    "manu": {
        "ve": 0.4645,
        "br": 0.2670,
        "lb": 0.5212,
        "cnis": None,
        "retain": 0.000,
    },
}


# ------------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------------

MAX_NEW_TOKENS = 64
TEMPERATURE = 0.0
DO_SAMPLE = False

# Keep this 1 for reliability on RTX 4090.
BATCH_SIZE = 1

# Use None for full run. Use small numbers for debugging.
MAX_FORGET_ITEMS = None
MAX_RETAIN_ITEMS = None


# ------------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------------
# "entity_or_answer" is safest:
# correct if response contains entity/alias OR short gold answer.

MATCH_MODE = "entity_or_answer"

REFUSAL_PHRASES = [
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "not enough information",
    "i cannot identify",
    "i can't identify",
    "unknown",
    "not sure who",
]

MAX_GOLD_WORDS_FOR_SIMPLE_MATCH = 5
LOWERCASE_MATCH = True
STRIP_PUNCTUATION = True


# ------------------------------------------------------------------------
# Recovery attacks
# ------------------------------------------------------------------------

RECOVERY_ATTACKS = [
    "direct",
    "rephrase",
    "crop",
    "perturb",
]

# Centre crop fraction. 0.65 is reasonable: partial but not destroyed.
CROP_SCALE = 0.65

# Gaussian noise std in 0-255 pixel units.
# 25 is too aggressive. Use 8 or 10 for fair perturbation.
PERTURB_STD = 8.0

MIN_IMAGE_SIZE = 64


# ------------------------------------------------------------------------
# Output paths
# ------------------------------------------------------------------------

OUT_DIR = ROOT / "outputs" / "eval_behavioural"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEHAVIOURAL_JSON = OUT_DIR / f"behavioral_results_{ARCH}.json"
BEHAVIOURAL_CSV = OUT_DIR / f"behavioral_results_{ARCH}.csv"

RECOVERY_JSON = OUT_DIR / f"recovery_results_{ARCH}.json"
RECOVERY_CSV = OUT_DIR / f"recovery_results_{ARCH}.csv"

CORRELATION_JSON = OUT_DIR / f"correlation_results_{ARCH}.json"
CORRELATION_CSV = OUT_DIR / f"correlation_results_{ARCH}.csv"

TABLE_BEHAVIOURAL_TEX = OUT_DIR / f"TABLE_behavioral_{ARCH}.tex"
TABLE_RECOVERY_TEX = OUT_DIR / f"TABLE_recovery_{ARCH}.tex"
TABLE_CORRELATION_TEX = OUT_DIR / f"TABLE_correlation_{ARCH}.tex"

PAPER_INSERT_TEX = OUT_DIR / f"paper_insert_{ARCH}.tex"


# ------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------

def get_base_model_name():
    if ARCH.lower() == "llava":
        return LLAVA_BASE_MODEL
    if ARCH.lower() == "blip2":
        return BLIP2_BASE_MODEL
    raise ValueError(f"Unsupported ARCH: {ARCH}")


def is_lora_checkpoint(path):
    if path is None:
        return False
    path = Path(path)
    return (path / "adapter_config.json").exists()


def is_full_model_checkpoint(path):
    if path is None:
        return False

    path = Path(path)

    markers = [
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
    ]

    if any((path / marker).exists() for marker in markers):
        return True

    if list(path.glob("model-*.safetensors")):
        return True

    if list(path.glob("pytorch_model-*.bin")):
        return True

    return False


def checkpoint_type(path):
    if path is None:
        return "base"

    if is_lora_checkpoint(path):
        return "lora"

    if is_full_model_checkpoint(path):
        return "full"

    return "unknown"


def validate_config(strict=False):
    ok = True

    print(f"[CONFIG] ROOT        = {ROOT}")
    print(f"[CONFIG] DATA_ROOT   = {DATA_ROOT}")
    print(f"[CONFIG] FORGET_DIR  = {FORGET_DIR}")
    print(f"[CONFIG] RETAIN_DIR  = {RETAIN_DIR}")
    print(f"[CONFIG] ARCH        = {ARCH}")
    print(f"[CONFIG] BASE_MODEL  = {get_base_model_name()}")
    print(f"[CONFIG] OUT_DIR     = {OUT_DIR}")

    if not ROOT.exists():
        print(f"[X] ROOT missing: {ROOT}")
        ok = False

    if not FORGET_DIR.exists():
        print(f"[X] FORGET_DIR missing: {FORGET_DIR}")
        ok = False

    if not RETAIN_DIR.exists():
        print(f"[X] RETAIN_DIR missing: {RETAIN_DIR}")
        ok = False

    for method in ALL_METHODS:
        path = CHECKPOINT_DIRS.get(method)

        if method == "no_unlearn":
            print("[OK] no_unlearn: base model")
            continue

        if path is None:
            print(f"[X] {method}: checkpoint path is None")
            ok = False
            continue

        path = Path(path)

        if not path.exists():
            print(f"[X] {method}: checkpoint missing: {path}")
            ok = False
            continue

        ctype = checkpoint_type(path)

        if ctype == "unknown":
            print(f"[X] {method}: checkpoint found but type unknown: {path}")
            ok = False
        else:
            print(f"[OK] {method}: {ctype} checkpoint detected at {path}")

    if ok:
        print("[RESULT] CONFIG OK")
    else:
        print("[RESULT] CONFIG HAS ERRORS")

    if strict and not ok:
        raise RuntimeError("Config validation failed.")

    return ok


if __name__ == "__main__":
    validate_config(strict=False)
