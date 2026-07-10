import json
import math
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path("outputs")
OUT = Path("outputs/controls")
OUT.mkdir(parents=True, exist_ok=True)

RESPONSE_KEYS = {
    "response", "answer", "prediction", "generated_text",
    "model_response", "output", "completion"
}

ENTITY_KEYS = {
    "entity", "entity_id", "name", "target", "person",
    "subject", "id", "sample_id", "qid"
}

LB_KEYS_HINTS = [
    "lb", "language", "language_backbone", "lb_cka",
    "lb_sim", "language_backbone_cka", "residual"
]

def walk(obj, path=""):
    if isinstance(obj, dict):
        yield obj, path
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")

def get_first(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None

def word_len(x):
    return len(str(x).strip().split())

def maybe_method_from_path(p):
    s = str(p).lower()
    for m in ["no_unlearn", "nounlearn", "orig", "original", "ga", "npo", "mmunlearner", "mmun", "cagul", "manu"]:
        if m in s:
            return m
    return "unknown"

response_records = []
lb_records = []

for f in ROOT.rglob("*.json"):
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        continue

    method = maybe_method_from_path(f)

    for d, jpath in walk(data):
        if not isinstance(d, dict):
            continue

        resp_key = next((k for k in d.keys() if k in RESPONSE_KEYS), None)
        if resp_key:
            ent = get_first(d, ENTITY_KEYS)
            response_records.append({
                "file": str(f),
                "json_path": jpath,
                "method": method,
                "entity": str(ent) if ent is not None else "",
                "response_key": resp_key,
                "length_words": word_len(d.get(resp_key, "")),
                "response_preview": str(d.get(resp_key, ""))[:120]
            })

        # Search for per-entity LB-like scalar fields
        for k, v in d.items():
            kl = str(k).lower()
            if any(h in kl for h in LB_KEYS_HINTS):
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    ent = get_first(d, ENTITY_KEYS)
                    lb_records.append({
                        "file": str(f),
                        "json_path": jpath,
                        "method": method,
                        "entity": str(ent) if ent is not None else "",
                        "key": k,
                        "value": float(v)
                    })

# Save availability reports
Path(OUT / "response_length_records.json").write_text(
    json.dumps(response_records, indent=2), encoding="utf-8"
)
Path(OUT / "lb_scalar_records.json").write_text(
    json.dumps(lb_records, indent=2), encoding="utf-8"
)

print("=== GENERATION LENGTH AVAILABILITY ===")
print(f"JSON files searched: {len(list(ROOT.rglob('*.json')))}")
print(f"Response records found: {len(response_records)}")
print(f"LB-like scalar records found: {len(lb_records)}")
print()

by_file = defaultdict(int)
for r in response_records:
    by_file[r["file"]] += 1

print("Top response files:")
for f, n in sorted(by_file.items(), key=lambda x: -x[1])[:20]:
    print(f"  {n:4d}  {f}")

print()
by_lb_file = defaultdict(int)
for r in lb_records:
    by_lb_file[r["file"]] += 1

print("Top LB-like scalar files:")
for f, n in sorted(by_lb_file.items(), key=lambda x: -x[1])[:20]:
    print(f"  {n:4d}  {f}")

print()
print("Saved:")
print(f"  {OUT / 'response_length_records.json'}")
print(f"  {OUT / 'lb_scalar_records.json'}")
print()

if not response_records:
    print("RESULT: No usable response fields found. Length-confound correlation cannot be run from saved logs.")
elif not lb_records:
    print("RESULT: Responses found, but no per-entity LB residual/CKA scalar fields found.")
    print("You can report that per-entity generation-length correlation was not available from saved logs.")
else:
    print("RESULT: Some responses and LB-like fields were found.")
    print("Next step: inspect the saved JSON reports and confirm whether entity IDs align.")
