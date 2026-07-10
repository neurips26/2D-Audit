"""
exp_config.py
-------------
Master config for all revision experiments.

Fixed:
- No hardcoded C:\\Users\\Abdullah path.
- ROOT defaults to the folder where this exp_config.py file exists.
- You can still override ROOT using EVAL_ROOT if needed.
"""

from pathlib import Path
import os

# -Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â
# ROOT
# -Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â
# Default ROOT = current project folder containing exp_config.py
# Example:
# C:\Users\34998855\AppData\Roaming\JetBrains\PyCharm2025.1\extensions\com.intellij.database\2DUnl
#
# Optional override:
#   $env:EVAL_ROOT="D:\some_other_project"
# -Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â-Â

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("EVAL_ROOT", THIS_DIR)).resolve()

# -- Models --------------------------------------------------------------------
LLAVA_BASE = "llava-hf/llava-1.5-7b-hf"
BLIP2_BASE = "Salesforce/blip2-opt-2.7b"
DEVICE = "cuda"

# -- Existing LLaVA checkpoints ------------------------------------------------
LLAVA_ADAPTERS = {
    "no_unlearn": None,

    # DEAD/ZERO-B in current scan. Keep path for diagnostics, but do not use for CRP.
    "ga": ROOT / "checkpoints" / "schedule_sensitivity" / "llava_ga_attn4_20steps",

    # Trained LLaVA adapters, LoRA_B confirmed nonzero.
    "npo": ROOT / "checkpoints" / "llava_npo_adapter",
    "mmunlearner": ROOT / "checkpoints" / "llava_mmunlearner_adapter",
    "cagul": ROOT / "checkpoints" / "llava_cagul_adapter",
    "sineproject": ROOT / "checkpoints" / "llava_sineproject_adapter",

    # MANU LoRA is DEAD/ZERO-B. Real MANU should be full-weight if available.
    "manu_lora": ROOT / "checkpoints" / "llava_manu_adapter",
    "manu_full": ROOT / "checkpoints" / "llava_manu_full",

    # Retrained GA target. retrain_ga_llava.py should save here.
    "ga_retrained": ROOT / "checkpoints" / "schedule_sensitivity" / "llava_ga_attn4_20steps",
    "graddiff": ROOT / "checkpoints" / "graddiff" / "lr0p0001_lambda1_seed42" / "graddiff_llava_50steps",
    "graddiff_5": ROOT / "checkpoints" / "graddiff" / "lr0p0001_lambda1_seed42" / "graddiff_llava_5steps",
    "graddiff_10": ROOT / "checkpoints" / "graddiff" / "lr0p0001_lambda1_seed42" / "graddiff_llava_10steps",
    "graddiff_20": ROOT / "checkpoints" / "graddiff" / "lr0p0001_lambda1_seed42" / "graddiff_llava_20steps",
    "ga_attn4_50": ROOT / "checkpoints" / "schedule_sensitivity" / "llava_ga_retrained_attn4_50steps",
}

# -- BLIP-2 checkpoints --------------------------------------------------------
# All BLIP-2 adapters were confirmed trained in your LoRA_B scan.
BLIP2_ADAPTERS = {
    "no_unlearn": None,
    "ga": ROOT / "checkpoints" / "blip2_ga_adapter",
    "npo": ROOT / "checkpoints" / "blip2_npo_adapter",
    "mmunlearner": ROOT / "checkpoints" / "blip2_mmunlearner_adapter",
    "cagul": ROOT / "checkpoints" / "blip2_cagul_adapter",
    "manu": ROOT / "checkpoints" / "blip2_manu_adapter",
    "sineproject": ROOT / "checkpoints" / "blip2_sineproject_adapter",
}

# -- Valid methods for CRP -----------------------------------------------------
# Do not include current llava_ga_adapter or llava_manu_adapter because LoRA_B is zero.
LLAVA_METHODS_FOR_CRP = ["npo", "mmunlearner", "cagul", "sineproject", "ga_retrained", "graddiff"]

BLIP2_METHODS_FOR_CRP = [
    "ga",
    "npo",
    "mmunlearner",
    "cagul",
    "manu",
    "sineproject",
]

# -- Existing MLLMU data -------------------------------------------------------
MLLMU_REAL_FORGET = ROOT / "data" / "mllmu_real" / "forget"
MLLMU_REAL_RETAIN = ROOT / "data" / "mllmu_real" / "retain"

# -- E1: UnLOK-VQA paths -------------------------------------------------------
UNLOK_DIR = ROOT / "data" / "unlok_vqa"
UNLOK_JSON = UNLOK_DIR / "unlok_vqa.json"
UNLOK_IMAGES_DIR = UNLOK_DIR / "images"
UNLOK_FORGET_DIR = UNLOK_DIR / "forget"
UNLOK_RETAIN_DIR = UNLOK_DIR / "retain"

# -- E2: Per-entity CRP output dir ---------------------------------------------
PER_ENTITY_CRP_DIR = ROOT / "outputs" / "crp_per_entity"

# -- E3: Bridge ablation checkpoints -------------------------------------------
BRIDGE_ABLATION_DIR = ROOT / "checkpoints" / "bridge_ablation"
BRIDGE_GA_CKPT = BRIDGE_ABLATION_DIR / "ga_bridge_unfrozen"

# -- E5: Schedule sensitivity checkpoints --------------------------------------
SCHEDULE_DIR = ROOT / "checkpoints" / "schedule_sensitivity"
SCHEDULE_STEPS = [50, 100, 200]

# -- Output dirs ---------------------------------------------------------------
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "revision"

# -- Create required dirs safely -----------------------------------------------
DIRS_TO_CREATE = [
    OUTPUTS_DIR,
    RESULTS_DIR,
    PER_ENTITY_CRP_DIR,
    SCHEDULE_DIR,
    BRIDGE_ABLATION_DIR,
    UNLOK_DIR,
    UNLOK_IMAGES_DIR,
    UNLOK_FORGET_DIR,
    UNLOK_RETAIN_DIR,
]

for d in DIRS_TO_CREATE:
    d.mkdir(parents=True, exist_ok=True)

# -- CRP hook config -----------------------------------------------------------
LLAVA_VE_LAYERS = [6, 12, 18, 23]
LLAVA_LB_LAYERS = [0, 8, 16, 24, 31]
LLAVA_BRIDGE_LAYERS = [0, 1]

BLIP2_VE_LAYERS = [0, 6, 12, 18, 23]
BLIP2_LB_LAYERS = [0, 8, 16, 24, 31]
BLIP2_BRIDGE_LAYERS = [0, 1]

# Hook paths in model
LLAVA_HOOK_PATHS = {
    "ve": "model.vision_tower.vision_model.encoder.layers",
    "bridge": "model.multi_modal_projector",
    "lb": "model.language_model.layers",
}

BLIP2_HOOK_PATHS = {
    "ve": "vision_model.encoder.layers",
    "bridge": "qformer.encoder.layer",
    "lb": "language_model.model.decoder.layers",
}

# -- UnLOK-VQA split sizes -----------------------------------------------------
UNLOK_FORGET_N = 400
UNLOK_RETAIN_N = 105

# -- Training hyperparameters --------------------------------------------------
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LR = 1e-4
BATCH_SIZE = 4

# -- Inference -----------------------------------------------------------------
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.0

# -- Bootstrap CI --------------------------------------------------------------
BOOTSTRAP_N = 1000
BOOTSTRAP_ALPHA = 0.05

print(f"[config] ROOT = {ROOT}")
print(f"[config] RESULTS_DIR = {RESULTS_DIR}")







