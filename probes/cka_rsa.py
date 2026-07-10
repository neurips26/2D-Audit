"""
Representation similarity probes:
  - Linear CKA  (fast, numerically stable)
  - Kernel CKA  (RBF, more expressive but slower)
  - RSA         (Spearman correlation of RDMs)

Reference:
  Kornblith et al., "Similarity of Neural Network Representations
  Revisited", ICML 2019.
"""

import torch
import numpy as np
from scipy.stats import spearmanr
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


# -- Centering helpers ----------------------------------------------------------
def _centre(X: torch.Tensor) -> torch.Tensor:
    """Row-centre X: subtract column means."""
    return X - X.mean(dim=0, keepdim=True)


def _double_centre(K: torch.Tensor) -> torch.Tensor:
    """Double-centre a kernel matrix (for RSA / kernel CKA)."""
    n   = K.shape[0]
    row = K.mean(dim=1, keepdim=True)
    col = K.mean(dim=0, keepdim=True)
    tot = K.mean()
    return K - row - col + tot


# -- Linear CKA ----------------------------------------------------------------
def linear_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    debiased: bool = True,
) -> float:
    """
    Compute linear CKA between representation matrices X and Y.

    Args:
        X: (n, p) representation matrix from model A
        Y: (n, q) representation matrix from model B
        debiased: use debiased HSIC estimator (recommended for small n)

    Returns:
        CKA value in [0, 1]  (1 = identical geometry, 0 = orthogonal)
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"

    X = _centre(X).double()
    Y = _centre(Y).double()

    if debiased:
        cka = _debiased_linear_cka(X, Y)
    else:
        gram_xy = (Y.T @ X).norm(p="fro") ** 2
        gram_xx = (X.T @ X).norm(p="fro")
        gram_yy = (Y.T @ Y).norm(p="fro")
        denom   = gram_xx * gram_yy
        cka     = (gram_xy / denom).item() if denom > 0 else 0.0

    return float(np.clip(cka, 0.0, 1.0))


def _debiased_linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Debiased linear CKA using the HSIC_1 estimator.
    Reduces bias for small sample sizes.
    """
    n = X.shape[0]
    if n < 4:
        return 0.0

    # Gram matrices
    KX = X @ X.T
    KY = Y @ Y.T

    hsic_xy = _debiased_hsic(KX, KY)
    hsic_xx = _debiased_hsic(KX, KX)
    hsic_yy = _debiased_hsic(KY, KY)

    denom = (hsic_xx * hsic_yy) ** 0.5
    return (hsic_xy / denom).item() if denom > 1e-12 else 0.0


def _debiased_hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Unbiased HSIC estimator (Song et al., 2012).
    K, L are (n, n) kernel matrices.
    """
    n   = K.shape[0]
    k_diag = torch.diag(K)
    l_diag = torch.diag(L)

    # zero out diagonals
    K = K - torch.diag(k_diag)
    L = L - torch.diag(l_diag)

    ones = torch.ones(n, device=K.device, dtype=K.dtype)

    hsic = (
        torch.trace(K @ L)
        + (ones @ K @ ones) * (ones @ L @ ones) / ((n - 1) * (n - 2))
        - 2 * (ones @ K @ L @ ones) / (n - 2)
    )
    return hsic / (n * (n - 3))


# -- Kernel (RBF) CKA ----------------------------------------------------------
def rbf_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
    sigma_x: Optional[float] = None,
    sigma_y: Optional[float] = None,
) -> float:
    """
    Kernel CKA using RBF kernel. Slower but captures nonlinear geometry.
    If sigma is None, use median pairwise distance heuristic.
    """
    def _rbf_kernel(Z: torch.Tensor, sigma: Optional[float]) -> torch.Tensor:
        dist = torch.cdist(Z.float(), Z.float(), p=2) ** 2
        if sigma is None:
            sigma = dist.median().item() ** 0.5 + 1e-8
        return torch.exp(-dist / (2 * sigma ** 2))

    KX = _rbf_kernel(X, sigma_x)
    KY = _rbf_kernel(Y, sigma_y)

    KX = _double_centre(KX.double())
    KY = _double_centre(KY.double())

    hsic_xy = (KX * KY).sum()
    hsic_xx = (KX * KX).sum()
    hsic_yy = (KY * KY).sum()

    denom = (hsic_xx * hsic_yy) ** 0.5
    cka   = (hsic_xy / denom).item() if denom > 1e-12 else 0.0
    return float(np.clip(cka, 0.0, 1.0))


# -- RSA -----------------------------------------------------------------------
def rsa_correlation(
    X: torch.Tensor,
    Y: torch.Tensor,
    metric: str = "cosine",
) -> Tuple[float, float]:
    """
    Representational Similarity Analysis.
    Computes pairwise RDMs for X and Y, then returns their Spearman correlation.

    Args:
        X: (n, p) representations from model A
        Y: (n, q) representations from model B
        metric: "cosine" | "euclidean" | "correlation"

    Returns:
        (rho, p_value) - Spearman correlation and p-value
    """
    rdm_x = _compute_rdm(X, metric)
    rdm_y = _compute_rdm(Y, metric)

    # upper triangle indices (exclude diagonal)
    idx   = np.triu_indices(rdm_x.shape[0], k=1)
    vec_x = rdm_x[idx]
    vec_y = rdm_y[idx]

    rho, pval = spearmanr(vec_x, vec_y)
    return float(rho), float(pval)


def _compute_rdm(X: torch.Tensor, metric: str = "cosine") -> np.ndarray:
    """Compute n×n representational dissimilarity matrix."""
    X_np = X.float().numpy()
    n    = X_np.shape[0]

    if metric == "cosine":
        # cosine distance = 1 - cosine similarity
        norm  = np.linalg.norm(X_np, axis=1, keepdims=True) + 1e-8
        X_n   = X_np / norm
        rdm   = 1.0 - (X_n @ X_n.T)
    elif metric == "euclidean":
        from scipy.spatial.distance import cdist
        rdm = cdist(X_np, X_np, metric="euclidean")
    elif metric == "correlation":
        from scipy.spatial.distance import cdist
        rdm = cdist(X_np, X_np, metric="correlation")
    else:
        raise ValueError(f"Unknown metric: {metric}")

    rdm = np.clip(rdm, 0.0, None)
    return rdm


# -- Component-level audit -----------------------------------------------------
def compute_component_cka(
    reps_original : Dict[str, Dict[int, torch.Tensor]],
    reps_unlearned: Dict[str, Dict[int, torch.Tensor]],
    use_linear    : bool = True,
) -> Dict[str, Dict[int, float]]:
    """
    Compute CKA between original and unlearned representations
    for every (component, layer) pair.

    A HIGH CKA value means the unlearned model still encodes
    the forgotten entity similarly to the original -> residual knowledge.

    Args:
        reps_original:  dict[component][layer] -> tensor(N, D)
        reps_unlearned: dict[component][layer] -> tensor(N, D)
        use_linear:     True = linear CKA, False = RBF CKA

    Returns:
        dict[component][layer] -> float CKA score
    """
    results: Dict[str, Dict[int, float]] = {}

    for component in reps_original:
        if component not in reps_unlearned:
            logger.warning(f"Component '{component}' missing from unlearned reps")
            continue

        results[component] = {}
        for layer_idx in reps_original[component]:
            if layer_idx not in reps_unlearned[component]:
                continue

            X = reps_original[component][layer_idx]
            Y = reps_unlearned[component][layer_idx]

            # align sample count
            n = min(X.shape[0], Y.shape[0])
            X, Y = X[:n], Y[:n]

            if n < 4:
                logger.warning(
                    f"Too few samples ({n}) for {component}[{layer_idx}], skipping"
                )
                continue

            try:
                if use_linear:
                    score = linear_cka(X, Y)
                else:
                    score = rbf_cka(X, Y)
                results[component][layer_idx] = score
                logger.debug(
                    f"CKA {component}[{layer_idx}] = {score:.4f}"
                )
            except Exception as e:
                logger.warning(
                    f"CKA failed for {component}[{layer_idx}]: {e}"
                )
                results[component][layer_idx] = float("nan")

    return results


def compute_component_rsa(
    reps_original : Dict[str, Dict[int, torch.Tensor]],
    reps_unlearned: Dict[str, Dict[int, torch.Tensor]],
    metric        : str = "cosine",
) -> Dict[str, Dict[int, Tuple[float, float]]]:
    """
    Compute RSA (Spearman rho, p-value) for every (component, layer) pair.
    Returns dict[component][layer] -> (rho, pval).
    """
    results: Dict[str, Dict[int, Tuple[float, float]]] = {}

    for component in reps_original:
        if component not in reps_unlearned:
            continue

        results[component] = {}
        for layer_idx in reps_original[component]:
            if layer_idx not in reps_unlearned[component]:
                continue

            X = reps_original[component][layer_idx]
            Y = reps_unlearned[component][layer_idx]
            n = min(X.shape[0], Y.shape[0])
            X, Y = X[:n], Y[:n]

            if n < 4:
                continue

            try:
                rho, pval = rsa_correlation(X, Y, metric)
                results[component][layer_idx] = (rho, pval)
            except Exception as e:
                logger.warning(f"RSA failed for {component}[{layer_idx}]: {e}")

    return results


def summarise_component_cka(
    cka_results: Dict[str, Dict[int, float]]
) -> Dict[str, float]:
    """
    Compute per-component mean CKA across sampled layers.
    Returns dict[component] -> mean_CKA.
    """
    summary = {}
    for comp, layers in cka_results.items():
        vals = [v for v in layers.values() if not np.isnan(v)]
        summary[comp] = float(np.mean(vals)) if vals else float("nan")
    return summary
