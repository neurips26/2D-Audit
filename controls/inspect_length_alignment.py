import json
from pathlib import Path
from collections import Counter, defaultdict

OUT = Path("outputs/controls")

resp_path = OUT / "response_length_records.json"
lb_path   = OUT / "lb_scalar_records.json"

responses = json.loads(resp_path.read_text(encoding="utf-8"))
lbs       = json.loads(lb_path.read_text(encoding="utf-8"))

print("=== RESPONSE METHODS ===")
print(Counter(r["method"] for r in responses))

print("\n=== LB METHODS ===")
print(Counter(r["method"] for r in lbs))

print("\n=== RESPONSE ENTITY EXAMPLES ===")
for r in responses[:20]:
    print({
        "method": r["method"],
        "entity": r["entity"],
        "file": r["file"],
        "length": r["length_words"],
        "preview": r["response_preview"][:60]
    })

print("\n=== LB ENTITY EXAMPLES ===")
for r in lbs[:40]:
    print({
        "method": r["method"],
        "entity": r["entity"],
        "key": r["key"],
        "value": r["value"],
        "file": r["file"]
    })

# Check non-empty entity IDs
resp_nonempty = [r for r in responses if r.get("entity")]
lb_nonempty   = [r for r in lbs if r.get("entity")]

print("\n=== ENTITY AVAILABILITY ===")
print("Response records with entity:", len(resp_nonempty), "/", len(responses))
print("LB records with entity:", len(lb_nonempty), "/", len(lbs))

print("\nTop response files:")
for f, n in Counter(r["file"] for r in responses).most_common(10):
    print(n, f)

print("\nTop LB files:")
for f, n in Counter(r["file"] for r in lbs).most_common(10):
    print(n, f)
