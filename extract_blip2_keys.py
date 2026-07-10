"""
extract_blip2_keys.py
──────────────────────
Reads the three BLIP-2 audit JSONs and prints ONLY what is needed
to write the final extractor:

  1. For every method in data["methods"], prints every leaf key path
     that has a numeric (float/int) value.
  2. Groups them by value range to distinguish CKA (0-1) from
     accuracy (0-1) from counts (>1).
  3. Prints one compact table per file showing:
       method -> path -> value
  4. Saves outputs/revision/blip2_useful_keys.json
     (small, pasteable)

Usage:
    py extract_blip2_keys.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from exp_config import ROOT, RESULTS_DIR

BLIP2_FILES = [
    ROOT / "outputs" / "blip2"          / "blip2_audit_results.json",
    ROOT / "outputs" / "smoke_blip2"    / "blip2" / "blip2_audit_results.json",
    ROOT / "outputs" / "blip2_mllmu"    / "blip2" / "blip2_audit_results.json",
]


def collect_leaves(obj, prefix="") -> list:
    """
    Return list of (dotted_path, value) for every numeric leaf.
    Descends into dicts. Skips lists longer than 10 items (activations).
    """
    results = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                results.extend(collect_leaves(v, path))
            elif isinstance(v, list):
                if len(v) <= 10:
                    for i, item in enumerate(v):
                        results.extend(collect_leaves(item, f"{path}[{i}]"))
                else:
                    # Large list — just note the length and type
                    if v and isinstance(v[0], (int, float)):
                        results.append((path, f"[numeric list, len={len(v)}, "
                                              f"first={v[0]:.4f}]"))
            elif isinstance(v, (int, float)):
                results.append((path, v))
            elif isinstance(v, str):
                # Include strings that look like checkpoint paths or method IDs
                if any(kw in k.lower() for kw in ("ckpt","checkpoint","method",
                                                    "arch","seed","split","path")):
                    results.append((path, v))
    return results


def inspect_file(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "path": str(path)}

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    out = {"exists": True, "path": str(path), "methods": {}}

    # Find method dict — handle nested or flat
    if isinstance(data, dict) and "methods" in data:
        methods_src = data["methods"]
    elif isinstance(data, dict):
        methods_src = {k: v for k, v in data.items()
                       if isinstance(v, dict) and k not in
                       ("arch","architecture","dataset","config","meta","info")}
    else:
        return {"exists": True, "path": str(path), "error": "not a dict"}

    for method_name, method_rec in methods_src.items():
        if not isinstance(method_rec, dict):
            continue
        leaves = collect_leaves(method_rec, prefix=method_name)
        out["methods"][method_name] = leaves

    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_out   = {}
    all_lines = []

    for path in BLIP2_FILES:
        label = path.parent.name + "/" + path.name
        print(f"\n{'='*60}")
        print(f"FILE: {label}")
        print(f"{'='*60}")

        result = inspect_file(path)
        all_out[str(path)] = result

        if not result.get("exists"):
            print(f"  [NOT FOUND]")
            all_lines.append(f"\n{label}: NOT FOUND")
            continue

        if "error" in result:
            print(f"  [ERROR] {result['error']}")
            all_lines.append(f"\n{label}: ERROR {result['error']}")
            continue

        if not result["methods"]:
            print(f"  [NO METHODS FOUND]")
            all_lines.append(f"\n{label}: NO METHODS")
            continue

        file_lines = [f"\n{label}"]
        for method, leaves in result["methods"].items():
            print(f"\n  {method}:")
            file_lines.append(f"  {method}:")
            if not leaves:
                print(f"    [no numeric leaves]")
                file_lines.append(f"    [no numeric leaves]")
                continue
            for path_str, val in leaves:
                val_str = f"{val:.6f}" if isinstance(val, float) else str(val)
                line    = f"    {path_str:<55}  {val_str}"
                print(line)
                file_lines.append(line)
        all_lines.extend(file_lines)

    # Compact cross-file summary: unique paths with values
    print(f"\n{'='*60}")
    print("UNIQUE NUMERIC PATHS ACROSS ALL FILES")
    print(f"{'='*60}")
    all_lines.append("\n\nUNIQUE NUMERIC PATHS ACROSS ALL FILES")

    seen = {}   # path -> (value, file)
    for fpath, result in all_out.items():
        for method, leaves in result.get("methods", {}).items():
            for path_str, val in leaves:
                # Strip method prefix for cross-method comparison
                generic = path_str.split(".", 1)[1] if "." in path_str else path_str
                if generic not in seen and isinstance(val, float):
                    seen[generic] = (val, Path(fpath).name, method)

    for generic, (val, fname, method) in sorted(seen.items()):
        line = f"  {generic:<50}  {val:.4f}  (from {method} in {fname})"
        print(line)
        all_lines.append(line)

    # Save compact JSON
    compact = {}
    for fpath, result in all_out.items():
        fname = Path(fpath).name
        compact[fname] = {}
        for method, leaves in result.get("methods", {}).items():
            compact[fname][method] = {
                path_str: val for path_str, val in leaves
                if isinstance(val, float)
            }

    out_json = RESULTS_DIR / "blip2_useful_keys.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2, default=str)
    print(f"\n[saved] {out_json}")

    out_txt = RESULTS_DIR / "blip2_useful_keys.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))
    print(f"[saved] {out_txt}")

    # Print file size so user knows if it is pasteable
    size = out_json.stat().st_size
    print(f"\nJSON size: {size} bytes  "
          f"({'pasteable' if size < 50_000 else 'large — paste blip2_useful_keys.txt instead'})")
    print("\nPaste the console output above OR the contents of:")
    print(f"  {out_json}")


if __name__ == "__main__":
    main()
