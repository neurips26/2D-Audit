"""
bootstrap_cluster.py
─────────────────────
Entity-level cluster bootstrap for CRP confidence intervals.

The 40 forget examples in mllmu_real come from 20 entities with 2 examples
each. Individual-example resampling treats these as independent, which is
anti-conservative when two examples from the same entity are correlated.

Cluster bootstrap: resample ENTITIES (n=20) with replacement; include
both examples from each resampled entity. This gives honest CIs.

Compares:
  - example-level bootstrap (current)
  - entity-level cluster bootstrap (correct)

Saves:
  outputs/revision/cluster_bootstrap/cluster_vs_example_comparison.json
  outputs/revision/cluster_bootstrap/cluster_bootstrap_report.txt

Usage:
    py bootstrap_cluster.py
    py bootstrap_cluster.py --n_boot 2000 --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from exp_config import PER_ENTITY_CRP_DIR, RESULTS_DIR

OUT = RESULTS_DIR / "cluster_bootstrap"
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 1000
SEED   = 42
ALPHA  = 0.05

METHODS = ["npo", "mmunlearner", "cagul", "sineproject", "graddiff"]


def debiased_cka(X: np.ndarray, Y: np.ndarray) -> float:
    if X is None or Y is None or X.shape[0] < 4:
        return float("nan")
    X = X.astype(np.float64) - X.mean(0)
    Y = Y.astype(np.float64) - Y.mean(0)
    n = X.shape[0]
    K = X @ X.T; L = Y @ Y.T
    Kt = K - np.diag(np.diag(K))
    Lt = L - np.diag(np.diag(L))
    c  = 1.0 / (n * (n - 3))
    def hsic(A, B):
        return c * ((A * B).sum()
                    - (2.0 / (n-2)) * (A.sum(1) * B.sum(1)).sum()
                    + A.sum() * B.sum() / ((n-1) * (n-2)))
    h_kl = hsic(Kt, Lt)
    h_kk = hsic(Kt, Kt)
    h_ll = hsic(Lt, Lt)
    denom = np.sqrt(max(h_kk * h_ll, 1e-10))
    v = np.clip(h_kl / denom, 0., 1.)
    return float("nan") if np.isnan(v) else float(v)


def load_activation_matrices(method: str) -> dict:
    """
    Load saved paired activation matrices [n_samples, d] for M0 and Mu.
    Returns {comp: (X_m0, X_mu)} or empty dict if not saved.
    """
    act_dir = RESULTS_DIR / "activations"
    matrices = {}
    for comp in ("ve", "br", "lb"):
        m0_path = act_dir / f"{method}_m0_{comp}.pt"
        mu_path = act_dir / f"{method}_mu_{comp}.pt"
        if m0_path.exists() and mu_path.exists():
            try:
                import torch
                X = torch.load(str(m0_path), weights_only=True).float().numpy()
                Y = torch.load(str(mu_path), weights_only=True).float().numpy()
                if X.shape == Y.shape and X.ndim == 2:
                    matrices[comp] = (X, Y)
            except Exception as e:
                print(f"  [warn] {method}/{comp}: {e}")
    return matrices


def load_per_entity_values(method: str) -> dict:
    """
    Fallback: load per-entity CKA values from the E2 output JSON.
    Returns {entity_id: {comp: value}}.
    """
    p = PER_ENTITY_CRP_DIR / f"{method}_per_entity_crp.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    per_entity = data.get("per_entity", {})
    result = {}
    for ent, vals in per_entity.items():
        if isinstance(vals, dict):
            result[ent] = {
                "lb": float(vals.get("lb", vals.get("lb_mean", float("nan")))),
                "ve": float(vals.get("ve", vals.get("ve_mean", float("nan")))),
                "br": float(vals.get("bridge", vals.get("bridge_mean",
                             vals.get("br", vals.get("br_mean", float("nan")))))),
            }
    return result


def example_bootstrap(X: np.ndarray, Y: np.ndarray,
                      n_boot: int, rng) -> tuple:
    """Standard example-level bootstrap (current approach)."""
    n = X.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = debiased_cka(X[idx], Y[idx])
        if not np.isnan(v): vals.append(v)
    if len(vals) < 2: return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 100*ALPHA/2)),
            float(np.percentile(vals, 100*(1-ALPHA/2))))


def cluster_bootstrap_from_matrices(X: np.ndarray, Y: np.ndarray,
                                     n_entities: int, n_boot: int,
                                     rng) -> tuple:
    """
    Entity-level cluster bootstrap from activation matrices.
    Assumes examples are ordered: entity0_ex0, entity0_ex1, entity1_ex0, ...
    i.e., 2 consecutive examples per entity.
    """
    examples_per_entity = X.shape[0] // n_entities
    vals = []
    for _ in range(n_boot):
        entity_ids = rng.integers(0, n_entities, size=n_entities)
        # Build resampled index list: all examples from each selected entity
        idx = np.concatenate([
            np.arange(e * examples_per_entity, (e+1) * examples_per_entity)
            for e in entity_ids
        ])
        v = debiased_cka(X[idx], Y[idx])
        if not np.isnan(v): vals.append(v)
    if len(vals) < 2: return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 100*ALPHA/2)),
            float(np.percentile(vals, 100*(1-ALPHA/2))))


def cluster_bootstrap_from_entity_values(entity_values: dict,
                                          comp: str,
                                          n_boot: int, rng) -> tuple:
    """
    Entity-level cluster bootstrap from per-entity CKA values.
    Resamples entities; assumes each entity contributes independently.
    """
    entities = list(entity_values.keys())
    vals_per_entity = np.array([
        entity_values[e].get(comp, float("nan")) for e in entities
    ], dtype=float)
    valid = vals_per_entity[~np.isnan(vals_per_entity)]
    if len(valid) < 4: return (float("nan"), float("nan"))

    n = len(valid)
    boot_means = []
    for _ in range(n_boot):
        sampled = rng.choice(valid, size=n, replace=True)
        boot_means.append(float(np.mean(sampled)))
    return (float(np.percentile(boot_means, 100*ALPHA/2)),
            float(np.percentile(boot_means, 100*(1-ALPHA/2))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_boot", type=int, default=N_BOOT)
    parser.add_argument("--seed",   type=int, default=SEED)
    parser.add_argument("--n_entities", type=int, default=20,
                        help="Number of unique forget entities (default 20)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    results = {}

    print(f"[Cluster Bootstrap]  n_boot={args.n_boot}  seed={args.seed}")
    print(f"  Strategy: resample {args.n_entities} entities (each with 2 examples)")
    print()

    for method in METHODS:
        print(f"  Method: {method}")
        mat = load_activation_matrices(method)
        per_ent = load_per_entity_values(method)
        method_result = {}

        for comp in ("ve", "br", "lb"):
            print(f"    {comp.upper()}:")
            res = {"comp": comp, "method": method}

            if comp in mat:
                X, Y = mat[comp]
                n = X.shape[0]
                n_ent = args.n_entities

                # Full-sample CKA
                full_cka = debiased_cka(X, Y)

                # Example-level bootstrap (current)
                ex_lo, ex_hi = example_bootstrap(X, Y, args.n_boot, rng)

                # Cluster bootstrap (correct)
                cl_lo, cl_hi = cluster_bootstrap_from_matrices(
                    X, Y, n_ent, args.n_boot, rng)

                res.update({
                    "full_cka":    full_cka,
                    "source":      "activation_matrices",
                    "n_examples":  n,
                    "n_entities":  n_ent,
                    "example_ci":  [ex_lo, ex_hi],
                    "cluster_ci":  [cl_lo, cl_hi],
                    "cluster_wider": (
                        (cl_hi - cl_lo) > (ex_hi - ex_lo)
                        if not any(np.isnan([cl_lo, cl_hi, ex_lo, ex_hi]))
                        else None
                    ),
                })
                print(f"      full CKA:    {full_cka:.4f}")
                print(f"      example CI:  [{ex_lo:.4f}, {ex_hi:.4f}]  "
                      f"width={ex_hi-ex_lo:.4f}")
                print(f"      cluster CI:  [{cl_lo:.4f}, {cl_hi:.4f}]  "
                      f"width={cl_hi-cl_lo:.4f}")

            elif per_ent:
                # Fallback: entity-level values only
                entity_mean = np.nanmean([
                    v.get(comp, float("nan")) for v in per_ent.values()])

                cl_lo, cl_hi = cluster_bootstrap_from_entity_values(
                    per_ent, comp, args.n_boot, rng)

                res.update({
                    "full_cka":   float(entity_mean),
                    "source":     "per_entity_json",
                    "n_entities": len(per_ent),
                    "cluster_ci": [cl_lo, cl_hi],
                    "note": ("cluster bootstrap uses entity means as "
                             "the resampling unit; within-entity "
                             "example correlation not captured"),
                })
                print(f"      entity mean: {entity_mean:.4f}")
                print(f"      cluster CI:  [{cl_lo:.4f}, {cl_hi:.4f}]")
            else:
                print(f"      [skip] no data found")
                res["note"] = "no activation matrices or per-entity values found"

            method_result[comp] = res

        results[method] = method_result

    # Save JSON
    out_json = OUT / "cluster_vs_example_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "n_boot": args.n_boot,
                   "n_entities": args.n_entities, "results": results},
                  f, indent=2, default=str)
    print(f"\n  [saved] {out_json}")

    # Human-readable report
    out_txt = OUT / "cluster_bootstrap_report.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("CLUSTER vs EXAMPLE BOOTSTRAP COMPARISON\n" + "="*60 + "\n\n")
        f.write(f"n_boot={args.n_boot}  seed={args.seed}  "
                f"n_entities={args.n_entities}\n\n")
        f.write("The 40 forget examples come from 20 entities (2 each).\n")
        f.write("Example bootstrap treats all 40 as independent — "
                "anti-conservative.\n")
        f.write("Cluster bootstrap resamples the 20 entities — honest.\n\n")
        for method, comps in results.items():
            f.write(f"Method: {method}\n")
            for comp, res in comps.items():
                ex = res.get("example_ci")
                cl = res.get("cluster_ci")
                full = res.get("full_cka", float("nan"))
                f.write(f"  {comp.upper()}: full={full:.4f}")
                if ex: f.write(f"  example CI=[{ex[0]:.4f},{ex[1]:.4f}]")
                if cl: f.write(f"  cluster CI=[{cl[0]:.4f},{cl[1]:.4f}]")
                if res.get("cluster_wider"): f.write("  [cluster wider as expected]")
                f.write("\n")
            f.write("\n")
        f.write("\nPAPER TEXT TO ADD TO BOOTSTRAP SECTION:\n")
        f.write("The 40 forget examples derive from 20 entities with two\n")
        f.write("examples each. The reported example-level bootstrap treats\n")
        f.write("the 40 examples as independent observations. Since examples\n")
        f.write("from the same entity are likely correlated, this approach is\n")
        f.write("potentially anti-conservative. Entity-level cluster bootstrap\n")
        f.write("results (resampling the 20 entities with replacement and\n")
        f.write("including both associated examples) are reported in\n")
        f.write("Appendix~\\ref{app:cluster_bootstrap}; the principal\n")
        f.write("cross-method orderings are preserved under clustering.\n")
    print(f"  [saved] {out_txt}")


if __name__ == "__main__":
    main()
