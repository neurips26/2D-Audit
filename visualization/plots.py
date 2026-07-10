"""
Visualization utilities for the two-dimensional residual knowledge audit.

Plots:
  1. CKA heatmap        - component × method matrix
  2. RSA heatmap        - same layout with Spearman ρ
  3. Recovery scatter   - CKA vs recovery score per component
  4. CNIS bar chart     - neighborhood integrity by method
  5. Component profile  - layer-wise CKA for a single entity
"""

import os
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (Windows safe)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)

COMPONENT_ORDER = ["vision_encoder", "bridge", "language_backbone"]
COMPONENT_LABELS = {
    "vision_encoder"    : "Vision encoder",
    "bridge"            : "Bridge module",
    "language_backbone" : "Language backbone",
}


# -- 1. CKA Heatmap ------------------------------------------------------------
def plot_cka_heatmap(
    cka_summary: Dict[str, Dict[str, float]],
    # cka_summary[method][component] -> float
    arch       : str = "llava",
    save_path  : Optional[str] = None,
    title      : Optional[str] = None,
) -> plt.Figure:
    """
    Heat-map: rows = methods, cols = components.
    Cell value = mean CKA (higher = more residual knowledge = worse forgetting).
    """
    methods    = sorted(cka_summary.keys())
    components = COMPONENT_ORDER

    data = np.full((len(methods), len(components)), fill_value=np.nan)
    for i, method in enumerate(methods):
        for j, comp in enumerate(components):
            val = cka_summary[method].get(comp, np.nan)
            data[i, j] = val

    fig, ax = plt.subplots(figsize=(7, max(3, 0.7 * len(methods) + 1.5)))

    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)

    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(
        [COMPONENT_LABELS.get(c, c) for c in components], fontsize=11
    )
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=11)

    # Annotate cells
    for i in range(len(methods)):
        for j in range(len(components)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, label="CKA (higher = more residual knowledge)")

    default_title = (
        f"Component-level CKA - {arch.upper()} "
        f"(forget set, post-unlearning)"
    )
    ax.set_title(title or default_title, fontsize=12, pad=12)
    ax.set_xlabel("Model component", fontsize=11)
    ax.set_ylabel("Unlearning method", fontsize=11)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"CKA heatmap saved to {save_path}")

    return fig


# -- 2. RSA Heatmap ------------------------------------------------------------
def plot_rsa_heatmap(
    rsa_summary: Dict[str, Dict[str, Tuple[float, float]]],
    # rsa_summary[method][component] -> (rho, pval)
    arch       : str = "llava",
    save_path  : Optional[str] = None,
) -> plt.Figure:
    """RSA Spearman ρ heatmap (same layout as CKA heatmap)."""
    methods    = sorted(rsa_summary.keys())
    components = COMPONENT_ORDER

    data = np.full((len(methods), len(components)), fill_value=np.nan)
    for i, method in enumerate(methods):
        for j, comp in enumerate(components):
            entry = rsa_summary[method].get(comp, None)
            if entry is not None:
                data[i, j] = entry[0]  # rho

    fig, ax = plt.subplots(figsize=(7, max(3, 0.7 * len(methods) + 1.5)))

    im = ax.imshow(data, aspect="auto", cmap="PuOr", vmin=-1, vmax=1)

    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(
        [COMPONENT_LABELS.get(c, c) for c in components], fontsize=11
    )
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=11)

    for i in range(len(methods)):
        for j in range(len(components)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=10)

    plt.colorbar(im, ax=ax, label="RSA Spearman ρ")
    ax.set_title(
        f"RSA (Representational Similarity) - {arch.upper()}", fontsize=12, pad=12
    )
    ax.set_xlabel("Model component", fontsize=11)
    ax.set_ylabel("Unlearning method", fontsize=11)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"RSA heatmap saved to {save_path}")

    return fig


# -- 3. Recovery Correlation Scatter ------------------------------------------
def plot_recovery_correlation(
    correlation_results: Dict[str, Any],
    # correlation_results from run_all_correlations()
    components         : Optional[List[str]] = None,
    save_path          : Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plots: CKA vs recovery score for each component.
    This is the key plot linking Dimension 1 to concrete leakage.
    Safely skips regression lines when values are constant or invalid.
    """
    if components is None:
        components = COMPONENT_ORDER

    fig, axes = plt.subplots(1, len(components), figsize=(5 * len(components), 4))
    if len(components) == 1:
        axes = [axes]

    for ax, comp in zip(axes, components):
        key = f"{comp}_x_mean"
        res = correlation_results.get(key, {})

        if "error" in res or not res:
            ax.text(
                0.5, 0.5, "No data",
                ha="center", va="center",
                transform=ax.transAxes
            )
            ax.set_title(COMPONENT_LABELS.get(comp, comp))
            ax.set_xlabel("CKA (residual similarity)", fontsize=11)
            ax.set_ylabel("Recovery score", fontsize=11)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            continue

        cka_vals = np.asarray(res.get("cka_vals", []), dtype=float)
        rec_vals = np.asarray(res.get("recovery_vals", []), dtype=float)

        pr = res.get("pearson_r", np.nan)
        sr = res.get("spearman_r", np.nan)
        n  = res.get("n", len(cka_vals))

        valid = np.isfinite(cka_vals) & np.isfinite(rec_vals)
        cka_plot = cka_vals[valid]
        rec_plot = rec_vals[valid]

        if len(cka_plot) == 0:
            ax.text(
                0.5, 0.5, "No valid points",
                ha="center", va="center",
                transform=ax.transAxes
            )
        else:
            ax.scatter(
                cka_plot,
                rec_plot,
                alpha=0.7,
                s=60,
                color="#E85D30",
                edgecolors="white",
                linewidths=0.5
            )

            # Regression line - skip if CKA/recovery values are constant.
            cka_unique = np.unique(np.round(cka_plot, 6))
            rec_unique = np.unique(np.round(rec_plot, 6))

            if len(cka_plot) > 2 and len(cka_unique) > 1 and len(rec_unique) > 1:
                try:
                    z = np.polyfit(cka_plot, rec_plot, 1)
                    p = np.poly1d(z)
                    xs = np.linspace(float(np.min(cka_plot)), float(np.max(cka_plot)), 100)
                    ax.plot(xs, p(xs), "k--", alpha=0.5, linewidth=1.5)
                except Exception as e:
                    logger.warning(f"Skipping regression line for {comp}: {e}")
            else:
                logger.info(
                    f"Skipping regression line for {comp}: constant or insufficient data"
                )

        ax.set_xlabel("CKA (residual similarity)", fontsize=11)
        ax.set_ylabel("Recovery score", fontsize=11)

        pr_text = "nan" if not np.isfinite(pr) else f"{pr:.2f}"
        sr_text = "nan" if not np.isfinite(sr) else f"{sr:.2f}"

        ax.set_title(
            f"{COMPONENT_LABELS.get(comp, comp)}\n"
            f"Pearson r={pr_text}  Spearman ρ={sr_text}  n={n}",
            fontsize=10,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "CKA vs Adversarial Recovery Score (per model component)",
        fontsize=12,
        y=1.02
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Recovery correlation plot saved to {save_path}")

    return fig


# -- 4. CNIS Bar Chart ---------------------------------------------------------
def plot_cnis_bars(
    cnis_by_method: Dict[str, Dict[str, float]],
    # cnis_by_method[method][entity] -> cnis_score
    save_path     : Optional[str] = None,
) -> plt.Figure:
    """Bar chart of mean CNIS per unlearning method."""
    from evaluation.neighborhood_eval import aggregate_cnis

    methods = sorted(cnis_by_method.keys())
    means   = []
    stds    = []

    for method in methods:
        agg = aggregate_cnis(cnis_by_method[method])
        means.append(agg["mean"] if not np.isnan(agg["mean"]) else 0.0)
        stds.append(agg["std"]   if not np.isnan(agg["std"])  else 0.0)

    fig, ax = plt.subplots(figsize=(max(5, len(methods) * 1.2), 4))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))
    bars   = ax.bar(methods, means, yerr=stds, capsize=4,
                    color=colors, edgecolor="white", alpha=0.85)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean CNIS (higher = better neighborhood integrity)", fontsize=11)
    ax.set_xlabel("Unlearning method", fontsize=11)
    ax.set_title(
        "Concept Neighbourhood Integrity Score (CNIS) by Unlearning Method",
        fontsize=12, pad=10
    )
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.4, label="chance")
    ax.legend(fontsize=9)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mean:.2f}",
            ha="center", va="bottom", fontsize=9
        )

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"CNIS bar chart saved to {save_path}")

    return fig


# -- 5. Layer-wise CKA Profile ------------------------------------------------
def plot_layer_cka_profile(
    cka_layer_results: Dict[str, Dict[str, Dict[int, float]]],
    # cka_layer_results[method][component][layer_idx] -> float
    arch            : str = "llava",
    save_path       : Optional[str] = None,
) -> plt.Figure:
    """
    Line plot of CKA per layer for each component and method.
    Shows WHERE in the depth the residual knowledge persists.
    """
    fig, axes = plt.subplots(1, len(COMPONENT_ORDER),
                             figsize=(5 * len(COMPONENT_ORDER), 4), sharey=True)

    methods = sorted(cka_layer_results.keys())
    cmap    = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for ax, comp in zip(axes, COMPONENT_ORDER):
        for i, method in enumerate(methods):
            layer_dict = cka_layer_results.get(method, {}).get(comp, {})
            if not layer_dict:
                continue
            xs = sorted(layer_dict.keys())
            ys = [layer_dict[x] for x in xs]
            ax.plot(xs, ys, marker="o", label=method,
                    color=cmap[i], linewidth=1.5, markersize=5)

        ax.set_xlabel("Layer index", fontsize=10)
        ax.set_ylabel("CKA" if ax == axes[0] else "", fontsize=10)
        ax.set_title(COMPONENT_LABELS.get(comp, comp), fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    axes[-1].legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.suptitle(
        f"Layer-wise CKA Profile - {arch.upper()} (forget set)",
        fontsize=12
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Layer CKA profile saved to {save_path}")

    return fig


# -- Save all plots for one architecture/method run ---------------------------
def save_all_plots(
    output_dir         : str,
    arch               : str,
    cka_summary        : Dict,
    rsa_summary        : Dict,
    correlation_results: Dict,
    cnis_by_method     : Dict,
    cka_layer_results  : Dict,
):
    os.makedirs(output_dir, exist_ok=True)

    plot_cka_heatmap(
        cka_summary, arch,
        save_path=os.path.join(output_dir, f"{arch}_cka_heatmap.png")
    )
    plot_rsa_heatmap(
        rsa_summary, arch,
        save_path=os.path.join(output_dir, f"{arch}_rsa_heatmap.png")
    )
    plot_recovery_correlation(
        correlation_results,
        save_path=os.path.join(output_dir, f"{arch}_recovery_correlation.png")
    )
    plot_cnis_bars(
        cnis_by_method,
        save_path=os.path.join(output_dir, f"{arch}_cnis_bars.png")
    )
    plot_layer_cka_profile(
        cka_layer_results, arch,
        save_path=os.path.join(output_dir, f"{arch}_layer_cka_profile.png")
    )

    plt.close("all")
    logger.info(f"All plots saved to {output_dir}")
