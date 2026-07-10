import json
import random
import re
import unicodedata
from pathlib import Path
from collections import defaultdict

ROOT = Path("outputs")
OUT = Path("outputs/controls")
OUT.mkdir(parents=True, exist_ok=True)

def norm(s):
    s = str(s).lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def score_response(response, label, aliases=None):
    r = norm(response)
    labs = [norm(label)] + [norm(a) for a in (aliases or [])]
    labs = [x for x in labs if x]
    for lab in labs:
        if r == lab:
            return 1.0
        if lab in r or r in lab:
            return 0.5
    return 0.0

ENTITY_KEYS = ["entity", "entity_id", "name", "target", "subject", "person"]
REL_KEYS = ["relation", "relation_name", "property", "predicate", "relation_type"]
LABEL_KEYS = ["neighbor_label", "neighbour_label", "label", "answer", "target_label", "correct_answer"]
ALIAS_KEYS = ["aliases", "neighbor_aliases", "neighbour_aliases"]
RESP_KEYS = ["model_response", "response", "prediction", "output", "generated_text"]
SCORE_KEYS = ["score", "cnis_score", "q", "match_score"]

def walk(obj, path=""):
    if isinstance(obj, dict):
        yield obj, path
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")

def first(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None

records = []

for f in ROOT.rglob("*.json"):
    if "control" in str(f).lower():
        continue
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        continue

    for d, jpath in walk(data):
        if not isinstance(d, dict):
            continue

        ent = first(d, ENTITY_KEYS)
        rel = first(d, REL_KEYS)
        lab = first(d, LABEL_KEYS)
        resp = first(d, RESP_KEYS)
        aliases = first(d, ALIAS_KEYS)
        score = first(d, SCORE_KEYS)

        if ent is not None and rel is not None and lab is not None and resp is not None:
            if aliases is None:
                aliases = []
            if not isinstance(aliases, list):
                aliases = [aliases]
            if score is None or not isinstance(score, (int, float)):
                score = score_response(resp, lab, aliases)

            records.append({
                "file": str(f),
                "json_path": jpath,
                "entity": str(ent),
                "relation": str(rel),
                "label": str(lab),
                "aliases": [str(a) for a in aliases],
                "response": str(resp),
                "score": float(score)
            })

print("=== CNIS PER-RELATION RECORD SEARCH ===")
print(f"Records found: {len(records)}")

Path(OUT / "cnis_candidate_relation_records.json").write_text(
    json.dumps(records, indent=2), encoding="utf-8"
)

if len(records) == 0:
    print("No per-relation CNIS records found.")
    print("Your CNIS logs are probably aggregate-only.")
    print("Rerun CNIS with per-relation response logging, then run this script again.")
    raise SystemExit(0)

# Group by entity
by_entity = defaultdict(list)
for r in records:
    by_entity[r["entity"]].append(r)

eligible = {e: rs for e, rs in by_entity.items() if len(rs) >= 2}
print(f"Entities with >=2 relation records: {len(eligible)}")

if not eligible:
    print("Not enough multi-relation entities for shuffled control.")
    raise SystemExit(0)

rng = random.Random(42)
per_entity = {}

for ent, rs in eligible.items():
    correct_scores = [float(r["score"]) for r in rs]
    correct = sum(correct_scores) / len(correct_scores)

    idx = list(range(len(rs)))
    shuf = idx[:]
    tries = 0
    while shuf == idx and tries < 100:
        rng.shuffle(shuf)
        tries += 1

    shuffled_scores = []
    for i, r in enumerate(rs):
        wrong = rs[shuf[i]]
        shuffled_scores.append(
            score_response(r["response"], wrong["label"], wrong.get("aliases", []))
        )

    shuffled = sum(shuffled_scores) / len(shuffled_scores)
    per_entity[ent] = {
        "n_relations": len(rs),
        "correct_cnis": correct,
        "shuffled_cnis": shuffled,
        "delta": correct - shuffled
    }

correct_mean = sum(x["correct_cnis"] for x in per_entity.values()) / len(per_entity)
shuffled_mean = sum(x["shuffled_cnis"] for x in per_entity.values()) / len(per_entity)
delta = correct_mean - shuffled_mean

out = {
    "n_entities": len(per_entity),
    "correct_relation_cnis": correct_mean,
    "shuffled_relation_cnis": shuffled_mean,
    "delta": delta,
    "per_entity": per_entity
}

Path(OUT / "cnis_shuffled_relation_control.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8"
)

print()
print("=== CNIS SHUFFLED-RELATION CONTROL ===")
print(f"N entities:              {len(per_entity)}")
print(f"Correct-relation CNIS:   {correct_mean:.4f}")
print(f"Shuffled-relation CNIS:  {shuffled_mean:.4f}")
print(f"Delta:                   {delta:+.4f}")
print()
print(f"Saved: {OUT / 'cnis_shuffled_relation_control.json'}")

if delta >= 0.05:
    print("INTERPRETATION: Useful control. Correct CNIS is meaningfully higher than shuffled CNIS.")
elif delta >= 0.02:
    print("INTERPRETATION: Weak-to-moderate control. Report cautiously if needed.")
else:
    print("INTERPRETATION: Weak control. Do not use as positive evidence; mention only as limitation if asked.")
