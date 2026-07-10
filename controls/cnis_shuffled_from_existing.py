import json
import math
import random
import re
import unicodedata
from pathlib import Path

INFILE = Path("outputs/llava_original_cnis/llava/llava_audit_results.json")
OUTDIR = Path("outputs/controls")
OUTDIR.mkdir(parents=True, exist_ok=True)

def norm(s):
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def score_leakage(response, forget_name, neighbor):
    """
    Match the evaluator's leakage idea approximately:
    score is high if response mentions the forgotten entity or neighbor.
    For shuffled control, we preserve the response but score against wrong neighbor labels.
    """
    r = norm(response)
    f = norm(forget_name)
    n = norm(neighbor)

    if not r:
        return 0.0
    if f and f in r:
        return 1.0
    if n and n in r:
        return 0.5
    return 0.0

def cnis_from_leakage(scores):
    if not scores:
        return float("nan")
    return 1.0 - (sum(scores) / len(scores))

data = json.loads(INFILE.read_text(encoding="utf-8"))
method_block = data["methods"]["no_unlearn"]
neighborhood_results = method_block["neighborhood_results"]

rng = random.Random(42)

per_entity = {}
all_correct_cnis = []
all_shuffled_cnis = []

for entity, result in neighborhood_results.items():
    by_relation = result.get("by_relation", {})
    records = []

    for rel_id, items in by_relation.items():
        for it in items:
            records.append({
                "relation": rel_id,
                "neighbor": it.get("neighbor", ""),
                "response": it.get("response", ""),
                "original_leakage": float(it.get("leakage_score", 0.0)),
            })

    if len(records) < 2:
        continue

    correct_scores = [r["original_leakage"] for r in records]
    correct_cnis = cnis_from_leakage(correct_scores)

    # shuffle neighbor labels across records, preserving responses
    idx = list(range(len(records)))
    shuf = idx[:]
    tries = 0
    while shuf == idx and tries < 100:
        rng.shuffle(shuf)
        tries += 1

    shuffled_scores = []
    for i, rec in enumerate(records):
        wrong_neighbor = records[shuf[i]]["neighbor"]
        shuffled_scores.append(
            score_leakage(rec["response"], entity, wrong_neighbor)
        )

    shuffled_cnis = cnis_from_leakage(shuffled_scores)

    if not math.isnan(correct_cnis) and not math.isnan(shuffled_cnis):
        per_entity[entity] = {
            "n_records": len(records),
            "correct_cnis": correct_cnis,
            "shuffled_cnis": shuffled_cnis,
            "delta_correct_minus_shuffled": correct_cnis - shuffled_cnis,
        }
        all_correct_cnis.append(correct_cnis)
        all_shuffled_cnis.append(shuffled_cnis)

correct_mean = sum(all_correct_cnis) / len(all_correct_cnis)
shuffled_mean = sum(all_shuffled_cnis) / len(all_shuffled_cnis)
delta = correct_mean - shuffled_mean

out = {
    "source": str(INFILE),
    "method": "no_unlearn",
    "n_entities": len(per_entity),
    "correct_relation_cnis": correct_mean,
    "shuffled_relation_cnis": shuffled_mean,
    "delta_correct_minus_shuffled": delta,
    "per_entity": per_entity,
}

outpath = OUTDIR / "cnis_shuffled_from_existing_no_unlearn.json"
outpath.write_text(json.dumps(out, indent=2), encoding="utf-8")

print("=== CNIS SHUFFLED CONTROL FROM EXISTING NO-UNLEARN ===")
print(f"N entities:             {len(per_entity)}")
print(f"Correct CNIS:           {correct_mean:.4f}")
print(f"Shuffled CNIS:          {shuffled_mean:.4f}")
print(f"Delta correct-shuffled: {delta:+.4f}")
print("Saved:", outpath)

if abs(delta) >= 0.05:
    print("INTERPRETATION: Meaningful difference. Potentially reportable.")
elif abs(delta) >= 0.02:
    print("INTERPRETATION: Small difference. Report cautiously.")
else:
    print("INTERPRETATION: No meaningful difference. Do not use as positive control.")
