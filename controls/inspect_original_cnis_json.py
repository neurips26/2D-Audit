import json
from pathlib import Path

p = Path("outputs/llava_original_cnis/llava/llava_audit_results.json")
d = json.loads(p.read_text(encoding="utf-8"))

print("Top keys:", d.keys())

if "methods" in d:
    print("Methods:", d["methods"].keys())
    for m, v in d["methods"].items():
        print("\nMETHOD", m)
        print("keys:", v.keys())
        if "neighborhood_results" in v:
            print("neighborhood_results type:", type(v["neighborhood_results"]))
            print("first keys:", list(v["neighborhood_results"].keys())[:5])
        if "cnis_scores" in v:
            print("cnis_scores:", v["cnis_scores"])
else:
    print("No methods key")
    print(json.dumps(d, indent=2)[:2000])
