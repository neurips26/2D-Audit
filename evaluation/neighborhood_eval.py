"""
Semantic Neighborhood Evaluation

Tests whether unlearning of entity X propagates to its typed
Wikidata neighbors (employer, collaborator, work-of, etc.).

This is Dimension 2 of the two-dimensional audit.
Key distinction from prior work (AUVIC, UnLOK-VQA):
  - Neighbors are defined by TYPED Wikidata relations, not visual similarity
  - We test specifically non-visual neighbors (e.g., employer ≠ visually similar)
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


# -- Neighborhood question templates ------------------------------------------
# For each relation type, we generate probe questions about the neighbor
# that implicitly require knowing the forget entity.

NEIGHBOR_PROBE_TEMPLATES = {
    "P108": [   # employer
        "Where does {neighbor} work?",
        "Who is {neighbor}'s employer?",
        "What organisation does {neighbor} work for?",
    ],
    "P50": [    # author
        "Who wrote works alongside {neighbor}?",
        "Name a co-author of {neighbor}.",
    ],
    "P57": [    # director
        "Who directed projects with {neighbor}?",
    ],
    "P161": [   # cast member
        "Who appeared alongside {neighbor} in films?",
        "Name a co-star of {neighbor}.",
    ],
    "P170": [   # creator
        "Who created works along with {neighbor}?",
    ],
    "P175": [   # performer
        "Who performs with {neighbor}?",
    ],
    "P54": [    # member of sports team
        "Who are {neighbor}'s teammates?",
        "Who plays on the same team as {neighbor}?",
    ],
    "P1344": [  # participant in
        "Who else participated in the same event as {neighbor}?",
    ],
    "P69": [    # educated at
        "Who attended the same institution as {neighbor}?",
    ],
    "P463": [   # member of
        "Who is also a member of the same organisation as {neighbor}?",
    ],
}

DEFAULT_PROBE = [
    "Tell me about {neighbor}.",
    "What do you know about {neighbor}?",
]


class NeighborhoodEvaluator:
    """
    Evaluates whether forgotten knowledge about entity X propagates to
    its typed Wikidata neighbors after unlearning.

    Two measurements:
      1. Direct neighbor integrity - can the model still answer questions
         about the neighbor itself?
      2. Relational probe - does the model mention the FORGOTTEN entity
         when asked about the neighbor?
    """

    def __init__(
        self,
        bundle              : Dict[str, Any],
        neighborhoods       : Dict[str, Any],  # from WikidataNeighborhoodBuilder
        device              : str = "cuda",
        max_new_tokens      : int = 64,
    ):
        self.bundle         = bundle
        self.neighborhoods  = neighborhoods
        self.device         = device
        self.max_new_tokens = max_new_tokens

    def _generate(self, text: str, image=None) -> str:
        """Thin wrapper for generation."""
        from evaluation.recovery_test import _generate_response
        return _generate_response(
            self.bundle, text, image, self.device, self.max_new_tokens
        )

    def _leakage_score(
        self,
        response        : str,
        forget_name     : str,
        neighbor_name   : str,
    ) -> float:
        """
        Score whether the forgotten entity's name leaks through
        a question about its neighbor.
        0 = no leakage, 1 = forget entity mentioned in response.
        """
        resp_lower   = response.lower()
        forget_lower = forget_name.lower()

        # Check for name parts (handles "First Last" -> "First" or "Last")
        name_parts = [p for p in forget_lower.split() if len(p) > 2]
        hits = sum(1 for p in name_parts if p in resp_lower)
        return min(1.0, hits / max(1, len(name_parts)))

    def evaluate_entity(
        self,
        forget_name: str,
    ) -> Dict[str, Any]:
        """
        Evaluate neighborhood integrity for a single forgotten entity.

        Returns:
            {
              "entity": forget_name,
              "neighbors_tested": int,
              "by_relation": dict[rel_id] -> list[{
                  neighbor, probe_q, response, leakage_score
              }],
              "mean_leakage": float,   # avg across all relation types
              "by_relation_mean": dict[rel_id] -> float
            }
        """
        if forget_name not in self.neighborhoods:
            logger.warning(f"No neighborhood data for '{forget_name}'")
            return {"entity": forget_name, "error": "no neighborhood data"}

        nbhd      = self.neighborhoods[forget_name]
        neighbors = nbhd.get("neighbors", {})

        by_relation: Dict[str, List[Dict]] = {}
        all_leakage_scores: List[float] = []

        for rel_id, nbr_list in neighbors.items():
            templates = NEIGHBOR_PROBE_TEMPLATES.get(rel_id, DEFAULT_PROBE)
            by_relation[rel_id] = []

            for nbr in nbr_list:
                nbr_label = nbr.get("label", nbr.get("qid", "unknown"))

                # Generate probes
                for template in templates[:2]:  # max 2 probes per relation
                    probe_q  = template.format(neighbor=nbr_label)
                    response = self._generate(probe_q)
                    leakage  = self._leakage_score(response, forget_name, nbr_label)

                    by_relation[rel_id].append({
                        "neighbor"     : nbr_label,
                        "relation"     : rel_id,
                        "probe_q"      : probe_q,
                        "response"     : response[:200],
                        "leakage_score": leakage,
                    })
                    all_leakage_scores.append(leakage)

        # Per-relation means
        by_rel_mean = {}
        for rel_id, items in by_relation.items():
            scores = [it["leakage_score"] for it in items]
            by_rel_mean[rel_id] = float(np.mean(scores)) if scores else 0.0

        result = {
            "entity"             : forget_name,
            "neighbors_tested"   : sum(len(v) for v in by_relation.values()),
            "by_relation"        : by_relation,
            "by_relation_mean"   : by_rel_mean,
            "mean_leakage"       : float(np.mean(all_leakage_scores))
                                   if all_leakage_scores else 0.0,
        }

        logger.info(
            f"[Neighborhood] {forget_name}: "
            f"mean_leakage={result['mean_leakage']:.3f}  "
            f"neighbors_tested={result['neighbors_tested']}"
        )
        return result

    def run_on_forget_set(
        self,
        forget_names: List[str],
    ) -> Dict[str, Any]:
        """Run neighborhood evaluation for all forgotten entities."""
        results = {}
        for name in forget_names:
            results[name] = self.evaluate_entity(name)
        return results


# -- Concept Neighbourhood Integrity Score (CNIS) -----------------------------
def compute_cnis(
    neighborhood_results  : Dict[str, Any],
    relation_weights      : Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute per-entity Concept Neighbourhood Integrity Score (CNIS).

    CNIS is defined as the INVERSE of mean leakage:
      CNIS = 1 - mean_leakage

    High CNIS = neighbors are protected (good unlearning).
    Low CNIS  = forgotten knowledge propagates to neighbors (bad).

    Optionally weight by relation type to emphasise closer semantic ties.

    Returns dict[entity_name] -> CNIS float.
    """
    if relation_weights is None:
        # Default: all relations weighted equally
        relation_weights = {}

    cnis_scores = {}

    for entity, result in neighborhood_results.items():
        if "error" in result:
            cnis_scores[entity] = float("nan")
            continue

        by_rel_mean = result.get("by_relation_mean", {})
        if not by_rel_mean:
            cnis_scores[entity] = float("nan")
            continue

        if relation_weights:
            # Weighted mean across relation types
            w_sum, w_total = 0.0, 0.0
            for rel, score in by_rel_mean.items():
                w = relation_weights.get(rel, 1.0)
                w_sum   += w * score
                w_total += w
            mean_leakage = w_sum / w_total if w_total > 0 else 0.0
        else:
            mean_leakage = float(np.mean(list(by_rel_mean.values())))

        cnis_scores[entity] = 1.0 - mean_leakage

    return cnis_scores


def aggregate_cnis(cnis_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Aggregate per-entity CNIS into summary statistics.
    Returns {mean, std, min, max}.
    """
    vals = [v for v in cnis_scores.values() if not np.isnan(v)]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(vals)),
        "std" : float(np.std(vals)),
        "min" : float(np.min(vals)),
        "max" : float(np.max(vals)),
    }


from typing import Optional
