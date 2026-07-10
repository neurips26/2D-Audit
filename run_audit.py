"""
run_audit.py - Main pipeline for the Two-Dimensional Residual Knowledge Audit

Usage:
    py run_audit.py --arch llava --methods ga npo --debug
    py run_audit.py --arch blip2 --methods all
    py run_audit.py --arch llava --methods all --no_unlearn  (skip unlearning, load cached)

Pipeline:
    1. Load original model
    2. For each unlearning method:
       a. Apply unlearning -> save adapter
       b. Extract representations (original + unlearned) via hooks
       c. Compute CKA + RSA per component
       d. Run recovery test
       e. Evaluate neighborhood integrity (CNIS)
    3. Correlate CKA with recovery scores (key result)
    4. Save all tables + plots
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from pathlib import Path
from typing import Dict, List, Optional, Any

# -- Local imports -------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    ExperimentConfig,
    UnlearningConfig,
    LLAVA_HOOK_SPEC,
    BLIP2_HOOK_SPEC,
    MLLMU_ROOT,
    OUTPUT_ROOT,
    CHECKPOINT_ROOT,
    UNLEARNING_METHODS,
)
from models.loaders import load_model, load_llava_with_lora, load_blip2_with_lora, free_model
from models.hook_manager import extract_representations
from unlearning.methods import run_unlearning
from probes.cka_rsa import (
    compute_component_cka,
    compute_component_rsa,
    summarise_component_cka,
)
from data.datasets import (
    MLLMUBenchDataset,
    get_loaders,
    WikidataNeighborhoodBuilder,
    extract_entity_names,
)
from evaluation.recovery_test import RecoveryTester, run_all_correlations, compute_per_entity_residuals
from evaluation.neighborhood_eval import NeighborhoodEvaluator, compute_cnis
from visualization.plots import save_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("audit")


# ------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Two-Dimensional MLLM Audit")

    p.add_argument("--arch",       default="llava",
                   choices=["llava", "blip2"],
                   help="Model architecture to audit")

    p.add_argument("--methods",    nargs="+", default=["ga", "npo"],
                   help="Unlearning methods to run. Use 'all' for all 6.")

    p.add_argument("--data_root",  default=MLLMU_ROOT,
                   help="Path to MLLMU-Bench data")

    p.add_argument("--output_dir", default=OUTPUT_ROOT,
                   help="Output directory for results + plots")

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_steps",  type=int, default=100,
                   help="Unlearning optimisation steps")

    p.add_argument("--max_forget", type=int, default=None,
                   help="Limit forget set size (for debugging)")

    p.add_argument("--max_retain", type=int, default=None,
                   help="Limit retain set size")

    p.add_argument("--no_4bit",    action="store_true",
                   help="Disable 4-bit quantisation (requires more VRAM)")

    p.add_argument("--no_unlearn", action="store_true",
                   help="Skip unlearning; load from cached checkpoints")

    p.add_argument("--no_neighborhood", action="store_true",
                   help="Skip Wikidata neighborhood evaluation (saves time)")

    p.add_argument("--no_recovery", action="store_true",
                   help="Skip adversarial recovery probing (useful for retain-set CRP controls)")

    p.add_argument("--eval_split", choices=["forget", "retain"], default="forget",
                   help="Dataset split used for CKA/RSA representation extraction")

    p.add_argument("--debug",      action="store_true",
                   help="Debug mode: 2 methods, 8 forget samples, 20 steps")

    p.add_argument("--seed",       type=int, default=42)

    return p.parse_args()


# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_hook_spec(arch: str) -> Dict:
    return LLAVA_HOOK_SPEC if arch == "llava" else BLIP2_HOOK_SPEC


def adapter_path(output_dir: str, arch: str, method: str) -> str:
    return os.path.join(CHECKPOINT_ROOT, f"{arch}_{method}_adapter")


def results_path(output_dir: str, arch: str) -> str:
    return os.path.join(output_dir, f"{arch}_audit_results.json")


def save_results(results: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {path}")


# ------------------------------------------------------------------------
# Core per-method audit step
# ------------------------------------------------------------------------
def audit_one_method(
    method          : str,
    arch            : str,
    original_bundle : Dict[str, Any],
    forget_loader   ,
    retain_loader   ,
    forget_dataset  : MLLMUBenchDataset,
    hook_spec       : Dict,
    neighborhoods   : Dict,
    cfg             : ExperimentConfig,
    args            : argparse.Namespace,
    device          : str,
) -> Dict[str, Any]:
    """
    Full audit for a single unlearning method.
    Returns structured results dict.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  METHOD: {method.upper()}  ARCH: {arch.upper()}")
    logger.info(f"{'='*60}")

    adpt_path = adapter_path(cfg.output_dir, arch, method)

    # -- Step 1: Unlearning ----------------------------------------------------
    # Special control method: compare ORIGINAL against ORIGINAL without applying
    # any unlearning. This is used for original-model CNIS baselines and sanity
    # checks. We load a second copy so cleanup does not free original_bundle.
    if method.lower() in {"no_unlearn", "original", "baseline", "none"}:
        logger.info("Control method requested: loading an unchanged model copy")
        unlearned_bundle = load_model(arch, load_in_4bit=not args.no_4bit)

    elif args.no_unlearn and os.path.exists(adpt_path):
        logger.info(f"Loading cached unlearned model from {adpt_path}")
        unlearned_bundle = load_model(
            arch, load_in_4bit=not args.no_4bit, adapter_path=adpt_path
        )
    else:
        if arch == "llava":
            unlearned_bundle = load_llava_with_lora(
                cfg=cfg.unlearning, load_in_4bit=not args.no_4bit
            )
        else:
            unlearned_bundle = load_blip2_with_lora(
                cfg=cfg.unlearning, load_in_4bit=not args.no_4bit
            )

        # NPO needs reference model
        ref_bundle = original_bundle if method == "npo" else None

        unlearn_cfg       = UnlearningConfig(
            method=method,
            num_steps=cfg.unlearning.num_steps,
            batch_size=args.batch_size,
        )
        run_unlearning(
            method       = method,
            bundle       = unlearned_bundle,
            forget_loader= forget_loader,
            retain_loader= retain_loader,
            cfg          = unlearn_cfg,
            ref_bundle   = ref_bundle,
            device       = device,
            save_path    = adpt_path,
        )

    # -- Step 2: Extract representations --------------------------------------
    # Use a non-shuffled evaluation loader so original/unlearned representations
    # are row-aligned for per-entity residual similarity.
    chosen_loader = retain_loader if args.eval_split == "retain" else forget_loader
    logger.info(f"Representation audit split for CKA/RSA: {args.eval_split}")
    eval_loader = TorchDataLoader(
        chosen_loader.dataset,
        batch_size=chosen_loader.batch_size,
        shuffle=False,
        collate_fn=chosen_loader.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    logger.info("Extracting representations from ORIGINAL model ...")
    reps_orig, entity_ids_for_residuals = extract_representations(
        bundle     = original_bundle,
        dataloader = eval_loader,
        hook_spec  = hook_spec,
        device     = device,
        max_batches= None,
    )

    logger.info("Extracting representations from UNLEARNED model ...")
    reps_unl, entity_ids_unl = extract_representations(
        bundle     = unlearned_bundle,
        dataloader = eval_loader,
        hook_spec  = hook_spec,
        device     = device,
        max_batches= None,
    )

    if entity_ids_for_residuals != entity_ids_unl:
        logger.warning(
            "Entity order mismatch between original and unlearned extraction; "
            "per-entity residuals will use original ordering."
        )

    # -- Step 3: CKA + RSA per component --------------------------------------
    logger.info("Computing CKA ...")
    cka_results = compute_component_cka(reps_orig, reps_unl, use_linear=True)
    cka_summary = summarise_component_cka(cka_results)

    logger.info("Computing RSA ...")
    rsa_results = compute_component_rsa(reps_orig, reps_unl)

    logger.info(
        "CKA summary: "
        + "  ".join(f"{k}={v:.4f}" for k, v in cka_summary.items())
    )

    # -- Step 4: Recovery test -------------------------------------------------
    recovery_per_entity = {}
    correlation_results = {}
    per_entity_residuals = {}
    per_entity_correlations = {}

    if args.no_recovery:
        logger.info("Skipping recovery tests (--no_recovery)")
    else:
        logger.info("Running recovery tests ...")
        tester = RecoveryTester(
            bundle    = unlearned_bundle,
            device    = device,
            n_rephrase= cfg.recovery.n_rephrase,
        )
        recovery_per_entity = tester.run_on_forget_set(forget_dataset.samples)

        # Aggregate CKA per entity for correlation
        cka_per_entity: Dict[str, Dict[str, float]] = {}
        for sample in forget_dataset.samples:
            ent = sample["entity_name"]
            # Use the component summary (same for all samples from this run)
            cka_per_entity[ent] = cka_summary

        correlation_results = run_all_correlations(
            cka_per_entity     = cka_per_entity,
            recovery_per_entity= recovery_per_entity,
        )

        # Per-entity residual similarity + per-entity residual/recovery correlations.
        logger.info("Computing per-entity residual similarity ...")
        per_entity_result = compute_per_entity_residuals(
            reps_original   = reps_orig,
            reps_unlearned  = reps_unl,
            entity_ids      = entity_ids_for_residuals,
            recovery_scores = recovery_per_entity,
        )
        per_entity_residuals    = per_entity_result["per_entity"]
        per_entity_correlations = per_entity_result["correlations"]
        logger.info("Per-entity correlations: %s", per_entity_correlations)

    # -- Step 5: Neighborhood evaluation --------------------------------------
    cnis_scores = {}
    neighborhood_results = {}

    if not args.no_neighborhood and neighborhoods:
        logger.info("Running neighborhood evaluation ...")
        evaluator = NeighborhoodEvaluator(
            bundle        = unlearned_bundle,
            neighborhoods = neighborhoods,
            device        = device,
        )
        forget_names      = extract_entity_names(forget_dataset)
        neighborhood_results = evaluator.run_on_forget_set(forget_names)
        cnis_scores          = compute_cnis(neighborhood_results)

        logger.info(
            "CNIS scores: "
            + "  ".join(f"{k}={v:.3f}" for k, v in list(cnis_scores.items())[:5])
        )

    # -- Cleanup ---------------------------------------------------------------
    free_model(unlearned_bundle)
    torch.cuda.empty_cache()

    return {
        "method"               : method,
        "cka_results"          : {
            c: {str(l): v for l, v in layers.items()}
            for c, layers in cka_results.items()
        },
        "cka_summary"          : cka_summary,
        "rsa_results"          : {
            c: {str(l): list(v) for l, v in layers.items()}
            for c, layers in rsa_results.items()
        },
        "recovery_per_entity"  : recovery_per_entity,
        "correlation_results"  : correlation_results,
        "per_entity_residuals"  : per_entity_residuals,
        "per_entity_correlations": per_entity_correlations,
        "cnis_scores"          : cnis_scores,
        "neighborhood_results" : neighborhood_results,
    }


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    # Debug overrides
    if args.debug:
        logger.info("DEBUG MODE: limited samples + steps")
        # args.methods = args.methods[:2]  # disabled: allow all methods in debug
        args.max_forget = 8
        args.max_retain = 16
        args.num_steps  = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Resolve methods
    methods = UNLEARNING_METHODS if "all" in args.methods else args.methods
    arch    = args.arch

    cfg = ExperimentConfig()
    cfg.unlearning.num_steps = args.num_steps
    cfg.output_dir = args.output_dir

    hook_spec  = get_hook_spec(arch)
    output_dir = os.path.join(args.output_dir, arch)
    os.makedirs(output_dir, exist_ok=True)

    # -- Load original model ---------------------------------------------------
    logger.info(f"Loading ORIGINAL {arch.upper()} ...")
    original_bundle = load_model(arch, load_in_4bit=not args.no_4bit)

    processor = original_bundle["processor"]

    # -- Build data loaders ----------------------------------------------------
    logger.info("Building data loaders ...")
    forget_loader, retain_loader = get_loaders(
        root       = args.data_root,
        processor  = processor,
        arch       = arch,
        batch_size = args.batch_size,
        max_forget = args.max_forget,
        max_retain = args.max_retain,
    )

    forget_dataset = MLLMUBenchDataset(
        args.data_root, "forget", processor, arch, args.max_forget
    )

    # -- Build Wikidata neighborhoods ------------------------------------------
    neighborhoods = {}
    if not args.no_neighborhood:
        logger.info("Building Wikidata neighborhoods ...")
        nbhd_builder = WikidataNeighborhoodBuilder(cfg.neighborhood)
        entity_names = extract_entity_names(forget_dataset)
        neighborhoods = nbhd_builder.build_neighborhoods(entity_names[:20])  # limit
        logger.info(f"Built neighborhoods for {len(neighborhoods)} entities")

    # -- Per-method audit ------------------------------------------------------
    all_results: Dict[str, Any] = {
        "arch"   : arch,
        "methods": {},
    }

    # Accumulate for plotting
    cka_summary_by_method  : Dict[str, Dict[str, float]] = {}
    rsa_summary_by_method  : Dict[str, Dict[str, Any]]   = {}
    cka_layer_by_method    : Dict[str, Dict] = {}
    cnis_by_method         : Dict[str, Dict[str, float]] = {}
    all_correlations       : Dict[str, Any] = {}

    for method in methods:
        try:
            result = audit_one_method(
                method         = method,
                arch           = arch,
                original_bundle= original_bundle,
                forget_loader  = forget_loader,
                retain_loader  = retain_loader,
                forget_dataset = forget_dataset,
                hook_spec      = hook_spec,
                neighborhoods  = neighborhoods,
                cfg            = cfg,
                args           = args,
                device         = device,
            )

            all_results["methods"][method] = result

            # Collect for plotting
            cka_summary_by_method[method] = result["cka_summary"]
            cka_layer_by_method[method]   = result["cka_results"]
            rsa_summary_by_method[method] = {
                comp: layers.get(str(list(layers.keys())[-1])) if layers else None
                for comp, layers in result["rsa_results"].items()
            }
            cnis_by_method[method]        = result["cnis_scores"]
            all_correlations.update(result["correlation_results"])

        except Exception as e:
            logger.error(f"Method {method} FAILED: {e}", exc_info=True)
            all_results["methods"][method] = {"error": str(e)}

    # -- Save raw results ------------------------------------------------------
    save_results(all_results, results_path(output_dir, arch))

    # -- Generate all plots ----------------------------------------------------
    logger.info("Generating plots ...")
    try:
        save_all_plots(
            output_dir          = os.path.join(output_dir, "plots"),
            arch                = arch,
            cka_summary         = cka_summary_by_method,
            rsa_summary         = rsa_summary_by_method,
            correlation_results = all_correlations,
            cnis_by_method      = cnis_by_method,
            cka_layer_results   = cka_layer_by_method,
        )
    except Exception as e:
        logger.error(f"Plot generation failed: {e}", exc_info=True)

    # -- Print summary table ---------------------------------------------------
    print("\n" + "="*70)
    print(f"  AUDIT SUMMARY - {arch.upper()}")
    print("="*70)
    header = f"{'Method':<16} {'VE-CKA':>8} {'Bridge-CKA':>11} {'LB-CKA':>8} {'CNIS':>7}"
    print(header)
    print("-" * 70)
    for method in methods:
        if "error" in all_results["methods"].get(method, {}):
            print(f"{method:<16}  ERROR")
            continue
        cs = all_results["methods"][method]["cka_summary"]
        cnis_vals = list(all_results["methods"][method]["cnis_scores"].values())
        mean_cnis = np.mean([v for v in cnis_vals if not np.isnan(v)]) if cnis_vals else float("nan")
        print(
            f"{method:<16} "
            f"{cs.get('vision_encoder', float('nan')):>8.4f} "
            f"{cs.get('bridge', float('nan')):>11.4f} "
            f"{cs.get('language_backbone', float('nan')):>8.4f} "
            f"{mean_cnis:>7.3f}"
        )
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("Done OK")


if __name__ == "__main__":
    main()
