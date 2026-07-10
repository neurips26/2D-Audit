"""
inspect_blip2_schema.py
────────────────────────
Opens the three BLIP-2 audit JSON files and prints every key path
and scalar value inside data["methods"] for every method.

Prints exact paths such as:
  methods.ga.crp.ve
  methods.ga.aggregate.bridge_cka
  methods.ga.results.language_backbone

Does NOT:
  - infer or rename metrics
  - generate tables or prose
  - use placeholders
  - overwrite any result files

Outputs:
  outputs/revision/blip2_exact_schema.txt
  outputs/revision/blip2_exact_schema.json

Usage:
    py inspect_blip2_schema.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp_config import ROOT, RESULTS_DIR

BLIP2_FILES = [
    ROOT / "outputs" / "blip2" / "blip2_audit_results.json",
    ROOT / "outputs" / "smoke_blip2" / "blip2" / "blip2_audit_results.json",
    ROOT / "outputs" / "blip2_mllmu" / "blip2" / "blip2_audit_results.json",
]

OUTPUT_TXT  = RESULTS_DIR / "blip2_exact_schema.txt"
OUTPUT_JSON = RESULTS_DIR / "blip2_exact_schema.json"


def repr_value(val, max_list: int = 5) -> str:
    """
    Compact representation of any value.
    Lists/tuples: show length and first max_list items.
    Dicts: show keys only.
    Scalars: show directly.
    """
    if isinstance(val, dict):
        return f"dict(keys={list(val.keys())})"
    if isinstance(val, (list, tuple)):
        preview = [repr_value(v) for v in val[:max_list]]
        suffix  = f"...+{len(val)-max_list}" if len(val) > max_list else ""
        return f"{type(val).__name__}[{len(val)}]=[{', '.join(preview)}{suffix}]"
    if isinstance(val, float):
        return f"{val:.6g}"
    return repr(val)


def walk(obj, prefix: str, lines: list, schema: dict, depth: int = 0) -> None:
    """
    Recursively walk obj, recording every key path and scalar value.
    Stops descending into lists of non-dict items (prints summary instead).
    """
    if depth > 12:
        lines.append(f"{prefix}: [max depth reached]")
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                lines.append(f"{full_key}:  {{dict, {len(v)} keys: {list(v.keys())}}}")
                schema[full_key] = {"type": "dict", "keys": list(v.keys())}
                walk(v, full_key, lines, schema, depth + 1)
            elif isinstance(v, list):
                lines.append(f"{full_key}:  list[{len(v)}]  "
                              f"first_item_type={type(v[0]).__name__ if v else 'empty'}")
                schema[full_key] = {"type": "list", "len": len(v)}
                if v and isinstance(v[0], dict):
                    lines.append(f"{full_key}[0]:  "
                                 f"{{dict, keys={list(v[0].keys())}}}")
                    walk(v[0], f"{full_key}[0]", lines, schema, depth + 1)
                elif v:
                    lines.append(f"{full_key}[0..{min(4,len(v)-1)}]:  "
                                 f"{repr_value(v[:5])}")
            else:
                lines.append(f"{full_key}:  {repr_value(v)}")
                schema[full_key] = {"type": type(v).__name__, "value": repr_value(v)}
    else:
        lines.append(f"{prefix}:  {repr_value(obj)}")


def inspect_file(path: Path) -> dict:
    """
    Full inspection of one BLIP-2 audit JSON.
    Returns inspection result dict.
    """
    result = {
        "path":    str(path),
        "exists":  path.exists(),
        "methods": {},
        "top_keys": [],
        "representative_method": None,
        "representative_json":   None,
        "schema": {},
        "lines": [],
    }

    if not path.exists():
        result["lines"].append(f"[NOT FOUND] {path}")
        return result

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    result["top_keys"] = list(data.keys()) if isinstance(data, dict) else ["[list]"]
    result["lines"].append(f"  Top-level keys: {result['top_keys']}")

    if not isinstance(data, dict) or "methods" not in data:
        result["lines"].append(
            f"  [WARN] No 'methods' key found at top level. "
            f"Top-level keys: {result['top_keys']}"
        )
        # Try treating top-level as method dict
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and k not in ("arch","architecture",
                                                       "dataset","config","meta","info"):
                    result["methods"][k] = v
            if result["methods"]:
                result["lines"].append(
                    f"  Treating top-level as method dict: "
                    f"{list(result['methods'].keys())}"
                )
        return result

    methods_dict = data["methods"]
    result["lines"].append(
        f"  data['methods'] keys: {list(methods_dict.keys())}"
    )
    result["methods"] = methods_dict

    # Walk every method
    for method_name, method_rec in methods_dict.items():
        result["lines"].append(f"\n  === METHOD: {method_name} ===")
        walk(method_rec, f"methods.{method_name}", result["lines"],
             result["schema"])

    # Print full JSON for one representative method
    rep_method = next(iter(methods_dict), None)
    if rep_method:
        result["representative_method"] = rep_method
        result["representative_json"]   = methods_dict[rep_method]
        result["lines"].append(
            f"\n  === FULL JSON for representative method: {rep_method} ==="
        )
        result["lines"].append(
            json.dumps(methods_dict[rep_method], indent=4, default=str)
        )

    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_lines   = []

    for path in BLIP2_FILES:
        header = f"\n{'='*70}\nFILE: {path}\n{'='*70}"
        print(header)
        all_lines.append(header)

        result = inspect_file(path)
        all_results[str(path)] = {
            "exists":      result["exists"],
            "top_keys":    result["top_keys"],
            "method_names": list(result["methods"].keys()),
            "schema":      result["schema"],
            "representative_method": result["representative_method"],
            "representative_json":   result["representative_json"],
        }

        for line in result["lines"]:
            print(line)
            all_lines.append(line)

        if not result["exists"]:
            print("  [FILE NOT FOUND — skipping]")

    # Summary
    summary_lines = [
        "\n" + "="*70,
        "SUMMARY: KEY PATHS FOUND ACROSS ALL FILES",
        "="*70,
    ]
    all_paths = set()
    for path_key, res in all_results.items():
        for schema_key in res["schema"]:
            all_paths.add(schema_key)

    for p in sorted(all_paths):
        # Show value from first file that has it
        for path_key, res in all_results.items():
            if p in res["schema"]:
                info = res["schema"][p]
                summary_lines.append(
                    f"  {p:<60}  "
                    f"type={info.get('type','?'):<12}  "
                    f"val={info.get('value','[dict/list]')}"
                )
                break

    summary_lines.append(
        "\nNEXT STEP: identify the exact paths for ve, br, lb metrics "
        "from this output and add them to the aliases in fix2_diagnostic_range.py"
    )

    for line in summary_lines:
        print(line)
    all_lines.extend(summary_lines)

    # Save TXT
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))
    print(f"\n[saved] {OUTPUT_TXT}")

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"[saved] {OUTPUT_JSON}")

    print("\nRun this first. Provide the output to get the final BLIP-2 extractor.")


if __name__ == "__main__":
    main()
