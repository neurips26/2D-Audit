
"""
AAAI Audit Configuration
Two-Dimensional Residual Knowledge Audit of Unlearned MLLMs
"""

import os
from dataclasses import dataclass, field
from typing import List


DATA_ROOT       = os.path.join("data")
MLLMU_ROOT      = os.path.join(DATA_ROOT, "mllmu_bench")
OUTPUT_ROOT     = os.path.join("outputs")
CHECKPOINT_ROOT = os.path.join("checkpoints")

for d in [DATA_ROOT, MLLMU_ROOT, OUTPUT_ROOT, CHECKPOINT_ROOT]:
    os.makedirs(d, exist_ok=True)


LLAVA_MODEL_ID  = "llava-hf/llava-1.5-7b-hf"
BLIP2_MODEL_ID  = "Salesforce/blip2-opt-2.7b"


LLAVA_HOOK_SPEC = {
    "vision_encoder": {
        "pattern": "model.vision_tower.vision_model.encoder.layers",
        "sample_layers": [0, 6, 12, 18, 23],
    },
    "bridge": {
        "pattern": "model.multi_modal_projector",
        "sample_layers": ["linear_1", "linear_2"],
    },
    "language_backbone": {
        "pattern": "model.language_model.layers",
        "sample_layers": [0, 8, 16, 24, 31],
    },
}


BLIP2_HOOK_SPEC = {
    "vision_encoder": {
        "pattern": "vision_model.encoder.layers",
        "sample_layers": [0, 8, 16, 24, 31, 38],
    },
    "bridge": {
        "pattern": "qformer.encoder.layer",
        "sample_layers": [0, 3, 6, 9, 11],
    },
    "language_backbone": {
        "pattern": "language_model.model.decoder.layers",
        "sample_layers": [0, 6, 12, 18, 23],
    },
}


UNLEARNING_METHODS = [
    "ga",
    "npo",
    "mmunlearner",
    "manu",
    "cagul",
    "sineproject",
]


@dataclass
class UnlearningConfig:
    method: str = "ga"
    forget_lr: float = 1e-5
    retain_lr: float = 1e-5
    num_steps: int = 1
    batch_size: int = 2
    beta: float = 0.1
    alpha: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    prune_ratio: float = 0.1
    proj_lambda: float = 0.01
    attn_threshold: float = 0.1


@dataclass
class ProbeConfig:
    n_samples: int = 64
    batch_size: int = 4
    use_linear_cka: bool = True
    rsa_metric: str = "cosine"
    n_bootstrap: int = 1000


@dataclass
class RecoveryConfig:
    n_rephrase: int = 10
    attack_types: List[str] = field(
        default_factory=lambda: ["rephrase", "visual_crop", "visual_perturb"]
    )
    recovery_threshold: float = 0.3


@dataclass
class NeighborhoodConfig:
    max_hops: int = 1
    relation_types: List[str] = field(default_factory=lambda: [
        "P108", "P50", "P57", "P161", "P170",
        "P175", "P54", "P1344", "P69", "P463",
    ])
    max_neighbors_per_relation: int = 5
    cache_path: str = os.path.join(DATA_ROOT, "wikidata_cache.json")


@dataclass
class ExperimentConfig:
    architectures: List[str] = field(default_factory=lambda: ["llava", "blip2"])
    methods: List[str] = field(default_factory=lambda: UNLEARNING_METHODS)
    datasets: List[str] = field(
        default_factory=lambda: ["mllmu_bench", "fiubench", "clear", "unlok_vqa"]
    )
    seed: int = 42
    device: str = "cuda"
    load_in_4bit: bool = True
    output_dir: str = OUTPUT_ROOT
    unlearning: UnlearningConfig = field(default_factory=UnlearningConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    neighborhood: NeighborhoodConfig = field(default_factory=NeighborhoodConfig)


DEFAULT_CFG = ExperimentConfig()
