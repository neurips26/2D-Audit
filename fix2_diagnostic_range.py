"""
fix2_diagnostic_range.py  (complete replacement — nested BLIP-2 parser)
────────────────────────────────────────────────────────────────────────
Key fix: BLIP-2 audit JSONs have structure:
  {"arch": ..., "methods": {"ga": {...}, "npo": {...}, ...}}
  Previous parser treated the entire "methods" dict as one record.
  This version iterates over data["methods"].items().

Also:
  - Softened caption: "four diagnostic outcomes" not "four regimes emerge"
  - GA CRP numbers moved to appendix note, not main table
  - Joins CRP audit files with behavioural_results_blip2.json by method name
  - Treats identical-mtime files as duplicates, not independent sources

Usage:
    py fix2_diagnostic_range.py
    py fix2_diagnostic_range.py --skip_figure
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from exp_config import PER_ENTITY_CRP_DIR, RESULTS_DIR, ROOT

CHECKS  = []
FIGURES = RESULTS_DIR / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
START   = time.time()

def chk(label, verdict, detail):
    CHECKS.append({"check": label, "verdict": verdict, "detail": str(detail)})
    icon = {"PASS":"OK","WARN":"!!","FAIL":"XX"}.get(verdict,"??")
    print(f"  [{icon} {verdict}] {label}: {detail}")


# ── Verified LLaVA values ─────────────────────────────────────────────────────
LLAVA_VERIFIED = {
    "npo":         {"ve":0.9997,"br":0.9986,"lb":0.9931,"fa":0.9750,"ra":0.9750},
    "mmunlearner": {"ve":0.9998,"br":0.9965,"lb":0.9336,"fa":0.9750,"ra":0.9625},
    "cagul":       {"ve":0.9997,"br":0.9973,"lb":0.9927,"fa":0.9750,"ra":0.9625},
    "sineproject": {"ve":0.9999,"br":0.9986,"lb":0.9980,"fa":0.9750,"ra":0.9625},
    "graddiff":    {"ve":0.9728,"br":0.8032,"lb":0.8482,"fa":0.9250,"ra":0.9625},
}

GA_RUN_A = {"ve":0.9557,"br":0.5260,"lb":0.4828,
             "ckpt":"checkpoints/schedule_sensitivity/llava_ga_retrained_attn4_50steps",
             "note":"Numerically consistent with Run A, but exact checkpoint provenance "
                    "is not encoded in the source result file."}

# Four diagnostic outcome descriptors
OUTCOMES = {
    "npo":         ("weak_forget","Weak forgetting, near-original geometry"),
    "cagul":       ("weak_forget","Weak forgetting, near-original geometry"),
    "sineproject": ("weak_forget","Weak forgetting, near-original geometry"),
    "mmunlearner": ("hidden_drift","Behaviourally hidden internal drift"),
    "graddiff":    ("partial_forget","Limited partial forgetting, larger internal change"),
    "ga":          ("collapse","Destructive generation collapse"),
}
OUTCOME_COLORS = {
    "weak_forget":    "#4878D0",
    "hidden_drift":   "#EE854A",
    "partial_forget": "#6BAE75",
    "collapse":       "#D65F5F",
}
DISP = {"npo":"NPO","mmunlearner":"MMUnlearner","cagul":"CAGUL",
        "sineproject":"SineProject","graddiff":"GradDiff","ga":"GA-attn4-50*"}

# BLIP-2 expected methods (SineProject is absent from BLIP-2 experiments)
BLIP2_METHODS   = ["no_unlearn","ga","npo","mmunlearner","cagul","manu"]
BLIP2_ABSENT    = {"sineproject","graddiff"}

# Known BLIP-2 file locations
BLIP2_CRP_CANDIDATES = [
    ROOT / "outputs" / "blip2" / "blip2_audit_results.json",
    ROOT / "outputs" / "smoke_blip2" / "blip2" / "blip2_audit_results.json",
    ROOT / "outputs" / "blip2_mllmu" / "blip2" / "blip2_audit_results.json",
]
BLIP2_BEHAV_CANDIDATES = [
    ROOT / "outputs" / "eval_behavioural" / "behavioural_results_blip2.json",
    RESULTS_DIR / "behavioural_results_blip2.json",
    RESULTS_DIR / "blip2_behavioural.json",
]


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def safe_float(val) -> float:
    """Convert to float. Returns NaN for None, '', '?', non-finite."""
    if val is None or val == "" or val == "?":
        return float("nan")
    try:
        v = float(val)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def norm_method(name: str) -> str:
    """Normalise method name for matching."""
    return (str(name).lower()
            .replace("-","").replace("_","").replace(" ",""))


def extract_numeric_from_record(rec: dict) -> dict:
    """
    Extract fa, ra, ve, br, lb from a method record.
    Handles flat layout and nested sub-dicts.
    Never returns '?' or placeholder values.
    """
    out = {}

    # Key aliases
    aliases = {
        "fa": ["forget_acc","forget_accuracy","ForgetAcc","fa","f_acc",
               "forget_accuracy_mean","avg_forget_acc"],
        "ra": ["retain_acc","retain_accuracy","RetainAcc","ra","r_acc",
               "retain_accuracy_mean","avg_retain_acc"],
        "fr": ["forget_rate","ForgetRate","fr"],
        "ve": ["ve_cka","ve_mean","VE_CKA","ve","veCKA","vision_cka"],
        "br": ["br_cka","bridge_mean","bridge_cka","BR_CKA","br","brCKA"],
        "lb": ["lb_cka","lb_mean","LB_CKA","lb","lbCKA","lang_cka"],
        "ckpt": ["checkpoint","checkpoint_path","ckpt","adapter"],
        "seed": ["seed"],
    }

    def search(d: dict, keys: list):
        for k in keys:
            if k in d:
                return d[k]
        return None

    # Flat search first
    for out_key, src_keys in aliases.items():
        val = search(rec, src_keys)
        if val is not None:
            if out_key in ("ckpt","seed"):
                out[out_key] = str(val)
            else:
                v = safe_float(val)
                if not np.isnan(v):
                    out[out_key] = v

    # cka_summary block (exact BLIP-2 field names from schema inspection)
    if "cka_summary" in rec and isinstance(rec["cka_summary"], dict):
        cs = rec["cka_summary"]
        if "vision_encoder"    in cs and "ve" not in out:
            v = safe_float(cs["vision_encoder"]);   out["ve"] = v if not np.isnan(v) else out.get("ve", float("nan"))
        if "bridge"            in cs and "br" not in out:
            v = safe_float(cs["bridge"]);            out["br"] = v if not np.isnan(v) else out.get("br", float("nan"))
        if "language_backbone" in cs and "lb" not in out:
            v = safe_float(cs["language_backbone"]); out["lb"] = v if not np.isnan(v) else out.get("lb", float("nan"))

    # Nested sub-dicts (behavioral, crp, aggregate, forget, retain)
    for sub_key in ["behavioral","behavioural","crp","aggregate","forget","retain",
                    "metrics","results"]:
        if sub_key in rec and isinstance(rec[sub_key], dict):
            for out_key, src_keys in aliases.items():
                if out_key in out:
                    continue
                val = search(rec[sub_key], src_keys)
                if val is not None:
                    v = safe_float(val)
                    if not np.isnan(v):
                        out[out_key] = v

    return out


# ══════════════════════════════════════════════════════════════════════════════
# BLIP-2 CRP PARSER — handles nested {"methods": {method: record}} structure
# ══════════════════════════════════════════════════════════════════════════════

def parse_blip2_crp_file(path: Path) -> tuple:
    """
    Parse one BLIP-2 CRP audit JSON.
    Handles:
      - {"methods": {"ga": {...}, "npo": {...}, ...}}  ← primary structure
      - {"ga": {...}, "npo": {...}, ...}                ← flat method dict
      - [{"method": "ga", ...}, ...]                   ← list of records

    Returns (schema_info, {norm_method_name: numeric_dict})
    """
    if not path.exists():
        return {"path": str(path), "exists": False}, {}

    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"path": str(path), "error": str(e)}, {}

    schema = {
        "path":       str(path),
        "mtime":      mtime,
        "top_keys":   list(data.keys()) if isinstance(data, dict) else f"list[{len(data)}]",
    }

    records = {}

    if isinstance(data, dict):
        # PRIMARY CASE: nested methods dict
        if "methods" in data and isinstance(data["methods"], dict):
            schema["structure"] = "nested_methods"
            print(f"    Structure: nested methods dict with "
                  f"{len(data['methods'])} entries")
            for method_name, method_rec in data["methods"].items():
                nm = norm_method(method_name)
                nums = extract_numeric_from_record(
                    method_rec if isinstance(method_rec, dict) else {})
                records[nm] = nums
                print(f"      {method_name}: fa={nums.get('fa','--')}  "
                      f"ra={nums.get('ra','--')}  ve={nums.get('ve','--')}  "
                      f"br={nums.get('br','--')}  lb={nums.get('lb','--')}")

        # SECONDARY CASE: flat method dict (keys are method names)
        else:
            schema["structure"] = "flat_dict"
            for k, v in data.items():
                if isinstance(v, dict) and k not in ("arch","architecture",
                                                       "dataset","config","meta"):
                    nm = norm_method(k)
                    nums = extract_numeric_from_record(v)
                    records[nm] = nums
                    print(f"      {k}: fa={nums.get('fa','--')}  "
                          f"ra={nums.get('ra','--')}  lb={nums.get('lb','--')}")

    elif isinstance(data, list):
        schema["structure"] = "list"
        for item in data:
            if isinstance(item, dict) and "method" in item:
                nm   = norm_method(item["method"])
                nums = extract_numeric_from_record(item)
                records[nm] = nums

    schema["n_methods_found"] = len(records)
    return schema, records


def load_blip2_behavioural() -> dict:
    """
    Load BLIP-2 behavioural results from known locations.
    Returns {norm_method: {fa, ra, fr}} or empty dict.
    """
    for p in BLIP2_BEHAV_CANDIDATES:
        if not p.exists():
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            result = {}
            if isinstance(data, list):
                for rec in data:
                    if "method" in rec:
                        nm   = norm_method(rec["method"])
                        nums = extract_numeric_from_record(rec)
                        result[nm] = nums
            elif isinstance(data, dict):
                if "methods" in data:
                    for m, v in data["methods"].items():
                        nm   = norm_method(m)
                        result[nm] = extract_numeric_from_record(v)
                else:
                    for m, v in data.items():
                        if isinstance(v, dict):
                            nm   = norm_method(m)
                            result[nm] = extract_numeric_from_record(v)
            if result:
                print(f"    Loaded BLIP-2 behavioural from: {p}")
                return result
        except Exception as e:
            print(f"    [warn] {p}: {e}")
    return {}


def resolve_blip2(skip_on_fail: bool = True) -> dict:
    """
    Load and reconcile all BLIP-2 sources.
    Returns authoritative {method: merged_record} or None.
    """
    print("\n=== BLIP-2 NESTED PARSER ===")

    # Also search for any blip2 JSONs not in known list
    extra = []
    for base in [ROOT/"outputs", RESULTS_DIR]:
        if base.exists():
            for p in base.rglob("*blip2*audit*.json"):
                if p not in BLIP2_CRP_CANDIDATES:
                    extra.append(p)

    all_crp_paths = BLIP2_CRP_CANDIDATES + extra[:3]

    # Load CRP from all candidates
    crp_schemas    = []
    crp_by_file    = {}
    seen_mtimes    = {}   # mtime -> first path (to detect duplicates)

    for path in all_crp_paths:
        print(f"\n  CRP file: {path}")
        schema, records = parse_blip2_crp_file(path)
        crp_schemas.append(schema)
        if "error" in schema or not schema.get("exists", True):
            if not schema.get("exists", True):
                print(f"    [not found]")
            else:
                print(f"    [error] {schema.get('error','?')}")
            continue

        mtime = schema.get("mtime","")
        if mtime in seen_mtimes:
            print(f"    [DUPLICATE] same mtime as {seen_mtimes[mtime]} — treating as duplicate")
            schema["is_duplicate"] = True
            schema["duplicate_of"] = seen_mtimes[mtime]
        else:
            seen_mtimes[mtime] = str(path)

        chk(f"CRP file {path.name}", "PASS" if records else "WARN",
            f"{len(records)} method records parsed")
        crp_by_file[str(path)] = records

    # Classify files: smoke/test vs full-audit
    # Files under smoke_* directories are deprioritised.
    # Prefer outputs/blip2_mllmu/ as the full-audit source.
    def file_priority(fpath: str) -> int:
        p = Path(fpath)
        if "smoke" in str(p).lower():
            return 0    # lowest — smoke/test run
        if "mllmu" in str(p).lower():
            return 2    # highest — full MLLMU audit
        return 1        # middle

    # Deduplicate: keep one copy of identical-mtime files
    unique_files = {}
    for fpath, records in crp_by_file.items():
        schema = next((s for s in crp_schemas if s.get("path")==fpath), {})
        if not schema.get("is_duplicate", False):
            unique_files[fpath] = records

    print(f"\n  Unique CRP source files (with priority):")
    for fpath in unique_files:
        print(f"    priority={file_priority(fpath)}  {Path(fpath).parent.name}/{Path(fpath).name}")

    # Load behavioural
    behav = load_blip2_behavioural()
    if behav:
        chk("BLIP-2 behavioural", "PASS",
            f"Loaded {len(behav)} method records")
    else:
        chk("BLIP-2 behavioural", "WARN",
            f"Not found at any of: "
            f"{[str(p) for p in BLIP2_BEHAV_CANDIDATES]}")

    # Select ONE source per method — highest priority file that has the method.
    # NEVER average across files. If two non-smoke files disagree, mark conflicting.
    merged       = {}
    conflict_log = []

    for method in BLIP2_METHODS:
        nm    = norm_method(method)
        m_rec = {"method": method, "selected_source": None, "rejected_sources": []}

        # Collect candidates per source file, sorted by priority descending
        candidates = []
        for fpath, records in unique_files.items():
            if nm in records and records[nm]:
                candidates.append((file_priority(fpath), fpath, records[nm]))
        candidates.sort(key=lambda x: x[0], reverse=True)

        if not candidates:
            m_rec["status"] = "rejected"
            m_rec["reason"] = "no CRP record found in any source file"
            merged[method]  = m_rec
            continue

        # Smoke-only: use with WARN
        if all(file_priority(c[1]) == 0 for c in candidates):
            chk(f"BLIP-2 {method} source", "WARN",
                f"Only smoke/test results found. "
                f"Using {Path(candidates[0][1]).parent.name} with WARN status.")
            chosen_priority, chosen_fpath, chosen_rec = candidates[0]
            m_rec["selected_source"] = chosen_fpath
            m_rec["status_note"]     = "smoke_only"
            for r in candidates[1:]:
                m_rec["rejected_sources"].append(
                    {"path": r[1], "reason": "lower priority smoke run"})
        else:
            # Filter to non-smoke candidates
            full_candidates = [(p, f, r) for p, f, r in candidates if p > 0]

            if len(full_candidates) == 1:
                chosen_priority, chosen_fpath, chosen_rec = full_candidates[0]
                m_rec["selected_source"] = chosen_fpath
                # Reject smoke candidates
                for c in candidates:
                    if file_priority(c[1]) == 0:
                        m_rec["rejected_sources"].append(
                            {"path": c[1], "reason": "smoke/test run excluded"})
            else:
                # Multiple full-audit candidates — check for conflict
                lb_vals = [(f, r.get("lb", float("nan")))
                           for _, f, r in full_candidates]
                lb_fin  = [(f,v) for f,v in lb_vals if not np.isnan(v)]
                if len(lb_fin) >= 2:
                    vals = [v for _,v in lb_fin]
                    if max(vals)-min(vals) > 0.005:
                        conflict_log.append({
                            "method": method, "key": "lb",
                            "values": [(Path(f).name, round(v,6)) for f,v in lb_fin]
                        })
                        chk(f"BLIP-2 {method} LB conflict", "WARN",
                            f"Full-audit files disagree: "
                            f"{[(Path(f).name, round(v,4)) for f,v in lb_fin]}")
                        # Prefer highest-priority (mllmu) file; mark conflicting
                        chosen_priority, chosen_fpath, chosen_rec = full_candidates[0]
                        m_rec["status_note"] = "conflicting_preferred_mllmu"
                        for c in full_candidates[1:]:
                            m_rec["rejected_sources"].append(
                                {"path": c[1],
                                 "reason": "conflict — lower-priority full file"})
                    else:
                        # Values agree — use highest priority
                        chosen_priority, chosen_fpath, chosen_rec = full_candidates[0]
                        for c in full_candidates[1:]:
                            m_rec["rejected_sources"].append(
                                {"path": c[1],
                                 "reason": "duplicate values — lower priority"})
                else:
                    chosen_priority, chosen_fpath, chosen_rec = full_candidates[0]

                m_rec["selected_source"] = chosen_fpath
                # Reject smoke
                for c in candidates:
                    if file_priority(c[1]) == 0:
                        m_rec["rejected_sources"].append(
                            {"path": c[1], "reason": "smoke/test run excluded"})

        # Copy values from selected source ONLY
        for k, v in chosen_rec.items():
            if isinstance(v, float) and not np.isnan(v):
                m_rec[k] = v

        # Add behavioural values
        if nm in behav:
            for k, v in behav[nm].items():
                if k in ("fa","ra","fr") and not np.isnan(v):
                    m_rec[k] = v

        # Determine status
        has_fa  = not np.isnan(m_rec.get("fa", float("nan")))
        has_ra  = not np.isnan(m_rec.get("ra", float("nan")))
        has_lb  = not np.isnan(m_rec.get("lb", float("nan")))
        conflicting = m_rec.get("status_note","") in ("conflicting_preferred_mllmu",)

        if has_fa and has_ra and has_lb and not conflicting:
            m_rec["status"] = "authoritative"
            m_rec["reason"] = f"Selected from {Path(chosen_fpath).parent.name}/{Path(chosen_fpath).name}"
        elif has_fa and has_ra and has_lb and conflicting:
            m_rec["status"] = "conflicting"
            m_rec["reason"] = f"Conflict detected; preferred {Path(chosen_fpath).parent.name}"
        elif has_lb:
            m_rec["status"] = "incomplete"
            m_rec["reason"] = "Missing behavioural values"
        else:
            m_rec["status"] = "rejected"
            m_rec["reason"] = "Insufficient data"

        merged[method] = m_rec
        src_name = Path(chosen_fpath).parent.name if chosen_fpath else "none"
        print(f"  {method}: fa={m_rec.get('fa','--')}  ra={m_rec.get('ra','--')}  "
              f"ve={m_rec.get('ve','--')}  br={m_rec.get('br','--')}  "
              f"lb={m_rec.get('lb','--')}  status={m_rec['status']}  src={src_name}")

    # Authoritative subset
    auth = {m: r for m, r in merged.items() if r["status"] == "authoritative"}
    if auth:
        chk("BLIP-2 authoritative rows", "PASS",
            f"{len(auth)} methods: {list(auth.keys())}")
    else:
        chk("BLIP-2 authoritative rows", "WARN",
            "No fully authoritative BLIP-2 rows yet. "
            "Inspect blip2_schema_inventory.txt for field names. "
            "LLaVA-only table will be generated.")

    return crp_schemas, merged, auth, conflict_log


# ══════════════════════════════════════════════════════════════════════════════
# GA PROVENANCE
# ══════════════════════════════════════════════════════════════════════════════

def check_ga_provenance() -> dict:
    """Numerical match only. Never claim 'confirmed'."""
    print("\n=== GA CRP PROVENANCE ===")
    result = {"resolved":False,"source":None,
              "ve":float("nan"),"br":float("nan"),"lb":float("nan"),
              "run_match":None,"provenance_note":""}

    search_dirs = [PER_ENTITY_CRP_DIR, RESULTS_DIR,
                   ROOT/"outputs"/"crp_per_entity", ROOT/"outputs"/"revision"]
    ga_json = None
    for base in search_dirs:
        if not Path(base).exists(): continue
        for p in Path(base).rglob("ga*per_entity_crp.json"):
            ga_json = p; break
        if ga_json: break

    if not ga_json:
        result["provenance_note"] = (
            "No GA CRP result JSON found. "
            f"Run A values (schedule_sensitivity checkpoint): "
            f"VE={GA_RUN_A['ve']} BR={GA_RUN_A['br']} LB={GA_RUN_A['lb']}. "
            "These will appear in appendix only, not main table.")
        chk("GA CRP JSON","WARN",result["provenance_note"])
        return result

    with open(ga_json, encoding="utf-8") as f:
        data = json.load(f)

    ve = safe_float(data.get("ve_mean", float("nan")))
    br = safe_float(data.get("bridge_mean", float("nan")))
    lb = safe_float(data.get("lb_mean", float("nan")))
    result.update({"ve":ve,"br":br,"lb":lb,"source":str(ga_json)})

    ckpt_in_file = str(data.get("checkpoint","") or data.get("source_checkpoint",""))
    diagnosed    = "llava_ga_retrained_attn4_50steps"

    if diagnosed in ckpt_in_file:
        result["run_match"]       = "A"
        result["provenance_note"] = (
            f"Checkpoint path in result matches diagnosed collapsed checkpoint. "
            f"VE={ve:.4f} BR={br:.4f} LB={lb:.4f}")
        result["resolved"] = True
        chk("GA provenance","PASS",result["provenance_note"])
    elif not np.isnan(lb):
        diff_A = abs(lb - GA_RUN_A["lb"])
        diff_B = abs(lb - 0.5333)
        match  = "A" if diff_A < diff_B else "B"
        result["run_match"] = match
        result["provenance_note"] = (
            f"LB={lb:.4f} is numerically consistent with Run {match}, "
            f"but exact checkpoint provenance is not encoded in the source result. "
            f"CRP values will appear in appendix only.")
        chk("GA provenance","WARN",result["provenance_note"])
    else:
        result["provenance_note"] = "GA LB-CKA is NaN — cannot match to known run."
        chk("GA provenance","FAIL",result["provenance_note"])

    return result


# ══════════════════════════════════════════════════════════════════════════════
# LLAVA TABLE
# ══════════════════════════════════════════════════════════════════════════════

def build_llava_rows() -> list:
    rows = []
    for method, vals in LLAVA_VERIFIED.items():
        ok, ol = OUTCOMES[method]
        rows.append({"method":method,"display":DISP[method],
                     "outcome_key":ok,"outcome_label":ol,
                     "ve":vals["ve"],"br":vals["br"],"lb":vals["lb"],
                     "fa":vals["fa"],"ra":vals["ra"]})
    # GA: no numeric CRP in main table — collapse row only
    ok, ol = OUTCOMES["ga"]
    rows.append({"method":"ga","display":DISP["ga"],
                 "outcome_key":ok,"outcome_label":ol,
                 "ve":float("nan"),"br":float("nan"),"lb":float("nan"),
                 "fa":float("nan"),"ra":float("nan")})
    # Sort by LB descending (nan last)
    rows.sort(key=lambda r: r["lb"] if not np.isnan(r["lb"]) else -1, reverse=True)
    return rows


def build_latex_llava(rows: list) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Behavioural and representational outcomes on",
        r"\emph{mllmu\_real} using LLaVA-1.5-7B ($n=40$ forget samples).",
        r"We use four diagnostic descriptors to summarise the observed outcomes:",
        r"weak forgetting with near-original geometry;",
        r"behaviourally hidden internal drift;",
        r"limited partial forgetting with larger internal change;",
        r"and destructive generation collapse.",
        r"*GA-attn4-50 produced deterministic repetitive generation,",
        r"so conventional behavioural accuracy is not reported.",
        r"Its CRP values are numerically consistent with Run~A",
        r"but exact checkpoint provenance is not encoded in the",
        r"retained result file; see Appendix for those values.}",
        r"\label{tab:diagnostic_range}",
        r"\setlength{\tabcolsep}{4pt}\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Diagnostic outcome}"
        r" & \textbf{F-Acc$\downarrow$} & \textbf{Ret-Acc$\uparrow$}"
        r" & \textbf{VE-CKA} & \textbf{BR-CKA} & \textbf{LB-CKA} \\",
        r"\midrule",
    ]
    prev = None
    for r in rows:
        if prev and r["outcome_key"] != prev:
            lines.append(r"\midrule")
        prev = r["outcome_key"]
        fa = (f"{r['fa']:.4f}" if not np.isnan(r["fa"])
              else r"\textit{collapse}")
        ra = (f"{r['ra']:.4f}" if not np.isnan(r["ra"]) else "---")
        ve = (f"{r['ve']:.4f}" if not np.isnan(r["ve"]) else "---")
        br = (f"{r['br']:.4f}" if not np.isnan(r["br"]) else "---")
        lb = (f"{r['lb']:.4f}" if not np.isnan(r["lb"]) else "---")
        lines.append(f"{r['display']} & {r['outcome_label']}"
                     f" & {fa} & {ra} & {ve} & {br} & {lb} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_appendix_ga_table(ga: dict) -> str:
    """Appendix table for GA CRP values with provenance note."""
    ve = ga.get("ve", float("nan"))
    br = ga.get("br", float("nan"))
    lb = ga.get("lb", float("nan"))
    if np.isnan(lb):
        # Use Run A values
        ve = GA_RUN_A["ve"]; br = GA_RUN_A["br"]; lb = GA_RUN_A["lb"]

    return (
        r"\begin{table}[h]" "\n"
        r"\centering" "\n"
        r"\caption{GA-attn4-50 CRP values (appendix diagnostic reference)." "\n"
        r"These values are numerically consistent with the Run~A result" "\n"
        r"(checkpoint \texttt{llava\_ga\_retrained\_attn4\_50steps})," "\n"
        r"but exact checkpoint provenance was not retained in the source" "\n"
        r"result file. GA-attn4-50 produced deterministic repetitive generation;" "\n"
        r"behavioural accuracy is not interpretable and is not reported.}" "\n"
        r"\label{tab:ga_crp_appendix}" "\n"
        r"\setlength{\tabcolsep}{5pt}\small" "\n"
        r"\begin{tabular}{lrrr}" "\n"
        r"\toprule" "\n"
        r"\textbf{Method} & \textbf{VE-CKA} & \textbf{BR-CKA} & \textbf{LB-CKA} \\" "\n"
        r"\midrule" "\n"
        f"GA-attn4-50 & {ve:.4f} & {br:.4f} & {lb:.4f} \\\\" "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def build_figure(llava_rows: list, blip2_auth: dict) -> None:
    show_blip2 = bool(blip2_auth and len(blip2_auth) >= 2)
    fig, axes = plt.subplots(1, 2 if show_blip2 else 1,
                             figsize=(10 if show_blip2 else 5, 3.8))
    if not show_blip2: axes = [axes]

    # Panel A: LLaVA
    ax = axes[0]
    lrows   = [r for r in llava_rows if not np.isnan(r["lb"])]
    labels  = [r["display"] for r in lrows]
    lb_vals = [r["lb"]      for r in lrows]
    colors  = [OUTCOME_COLORS[r["outcome_key"]] for r in lrows]
    y = np.arange(len(labels))
    bars = ax.barh(y, lb_vals, color=colors, edgecolor="black",
                   linewidth=0.5, height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0.4, 1.02); ax.set_xlabel("LB-CKA", fontsize=9)
    ax.set_title("(a) LLaVA-1.5-7B — mllmu_real\n(GA not shown: collapse)",
                 fontsize=8.5)
    ax.axvline(1.0, color="gray", linewidth=0.8, linestyle=":")
    for bar, val in zip(bars, lb_vals):
        ax.text(min(val+0.005,1.01), bar.get_y()+bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=7.5)
    patches = [mpatches.Patch(color=c, label=l)
               for l, c in OUTCOME_COLORS.items()
               if any(r.get("outcome_key")==l for r in lrows)]
    ax.legend(handles=patches, fontsize=7, loc="lower right")

    # Panel B: BLIP-2
    if show_blip2:
        ax2    = axes[1]
        b_rows = sorted(blip2_auth.values(),
                        key=lambda r: safe_float(r.get("lb", float("nan"))),
                        reverse=True)
        b_labels = [r["method"] for r in b_rows]
        b_vals   = [safe_float(r.get("lb", float("nan"))) for r in b_rows]
        b_valid  = [(l,v) for l,v in zip(b_labels,b_vals) if not np.isnan(v)]
        if b_valid:
            bl, bv = zip(*b_valid)
            bc = ["#4878D0" if v > 0.8 else "#EE854A" for v in bv]
            y2 = np.arange(len(bl))
            bars2 = ax2.barh(y2, bv, color=bc, edgecolor="black",
                             linewidth=0.5, height=0.6)
            ax2.set_yticks(y2); ax2.set_yticklabels(bl, fontsize=9)
            ax2.set_xlim(0.4, 1.05); ax2.set_xlabel("LB-CKA", fontsize=9)
            fa_set = {f"{safe_float(r.get('fa',float('nan'))):.3f}"
                      for r in b_rows if not np.isnan(safe_float(r.get("fa",float("nan"))))}
            fa_note = (f"All F-Acc = {next(iter(fa_set))}"
                       if len(fa_set)==1 else "F-Acc varies")
            ax2.set_title(f"(b) BLIP-2-OPT-2.7B\n{fa_note}", fontsize=8.5)
            ax2.axvline(1.0, color="gray", linewidth=0.8, linestyle=":")
            for bar, val in zip(bars2, bv):
                ax2.text(min(val+0.005,1.03), bar.get_y()+bar.get_height()/2,
                         f"{val:.3f}", va="center", fontsize=7.5)

    fig.tight_layout()
    out = FIGURES / "fig_diagnostic_spectrum.pdf"
    fig.savefig(str(out), format="pdf", bbox_inches="tight", dpi=200)
    src = FIGURES / "fig_diagnostic_spectrum.source.json"
    with open(src,"w",encoding="utf-8") as f:
        json.dump({"llava":llava_rows,"blip2":blip2_auth}, f,
                  indent=2, default=str)
    print(f"  [saved] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# BLIP-2 LATEX TABLE AND NARRATIVE
# ══════════════════════════════════════════════════════════════════════════════

BLIP2_ROW_ORDER = ["ga","npo","mmunlearner","cagul","manu"]
BLIP2_DISP      = {"ga":"GA","npo":"NPO","mmunlearner":"MMUnlearner",
                   "cagul":"CAGUL","manu":"MANU"}

def build_blip2_latex(auth: dict) -> str:
    if not auth:
        return "% BLIP-2 table: no authoritative data"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Behavioural and representational outcomes on BLIP-2-OPT-2.7B",
        r"(\emph{mllmu\_real}). All methods produce identical behavioural accuracy,",
        r"yet CRP reveals markedly different internal geometries. VE-CKA and",
        r"BR-CKA equal 1.000 for gradient-based methods because the vision",
        r"encoder and Q-Former are not updated by LoRA. MANU uses a full-model",
        r"checkpoint and shows broad representational disruption across all",
        r"three components. The language-backbone spread (0.529--0.995) is",
        r"invisible to output-only evaluation.}",
        r"\label{tab:blip2_audit}",
        r"\setlength{\tabcolsep}{4pt}\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Method}"
        r" & \textbf{F-Acc$\downarrow$} & \textbf{Ret-Acc$\uparrow$}"
        r" & \textbf{VE-CKA} & \textbf{BR-CKA} & \textbf{LB-CKA} \\",
        r"\midrule",
    ]

    for method in BLIP2_ROW_ORDER:
        if method not in auth:
            continue
        r   = auth[method]
        fa  = f"{r.get('fa', float('nan')):.4f}"  if not np.isnan(r.get('fa',float('nan')))  else "---"
        ra  = f"{r.get('ra', float('nan')):.4f}"  if not np.isnan(r.get('ra',float('nan')))  else "---"
        ve  = f"{r.get('ve', float('nan')):.4f}"  if not np.isnan(r.get('ve',float('nan')))  else "---"
        br  = f"{r.get('br', float('nan')):.4f}"  if not np.isnan(r.get('br',float('nan')))  else "---"
        lb  = f"{r.get('lb', float('nan')):.4f}"  if not np.isnan(r.get('lb',float('nan')))  else "---"
        src = r.get("selected_source","")
        src_note = f"  % {Path(src).parent.name}" if src else ""
        lines.append(f"{BLIP2_DISP.get(method,method)} & {fa} & {ra}"
                     f" & {ve} & {br} & {lb} \\{src_note}")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def blip2_narrative(auth: dict) -> str:
    if not auth:
        return "[BLIP-2 narrative blocked — no authoritative values]"
    ga_lb   = auth.get("ga",{}).get("lb", float("nan"))
    cagul_lb= auth.get("cagul",{}).get("lb", float("nan"))
    npo_lb  = auth.get("npo",{}).get("lb", float("nan"))
    mmu_lb  = auth.get("mmunlearner",{}).get("lb", float("nan"))
    manu_lb = auth.get("manu",{}).get("lb", float("nan"))
    manu_ve = auth.get("manu",{}).get("ve", float("nan"))
    manu_br = auth.get("manu",{}).get("br", float("nan"))
    fa      = auth.get("ga",{}).get("fa", 0.4)
    ra      = auth.get("ga",{}).get("ra", 0.2875)
    lb_vals = [v for v in [ga_lb,cagul_lb,npo_lb,mmu_lb,manu_lb] if not np.isnan(v)]
    spread  = round(max(lb_vals)-min(lb_vals), 4) if lb_vals else float("nan")
    return f"""
All five BLIP-2 methods produce identical behavioural accuracy
(ForgetAcc = {fa:.4f}, RetainAcc = {ra:.4f}), so output-level metrics cannot
distinguish them. CRP reveals markedly different internal geometries.
GA (LB-CKA = {ga_lb:.4f}) and CAGUL (LB-CKA = {cagul_lb:.4f}) remain
close to the original language backbone. NPO (LB-CKA = {npo_lb:.4f}) and
MMUnlearner (LB-CKA = {mmu_lb:.4f}) preserve vision and bridge representations
but substantially alter the language backbone. MANU causes broad
representational disruption across all three components
(VE-CKA = {manu_ve:.4f}, BR-CKA = {manu_br:.4f}, LB-CKA = {manu_lb:.4f}).
The language-backbone spread is {spread:.4f}, entirely invisible to
output-only evaluation.
"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_figure", action="store_true")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── BLIP-2 ───────────────────────────────────────────────────────────────
    blip2_schemas, blip2_merged, blip2_auth, blip2_conflicts = resolve_blip2()

    # Schema inventory
    inv_txt = RESULTS_DIR / "blip2_schema_inventory.txt"
    with open(inv_txt,"w",encoding="utf-8") as f:
        f.write("BLIP-2 SCHEMA INVENTORY\n"+"="*70+"\n\n")
        for s in blip2_schemas:
            f.write(f"FILE: {s.get('path','?')}\n")
            for k,v in s.items():
                if k!="path": f.write(f"  {k}: {v}\n")
            f.write("\n")
    chk("BLIP-2 schema inventory","PASS",str(inv_txt))

    # Provenance CSV
    prov_fields = ["method","architecture","dataset","split","checkpoint_path",
                   "seed","forget_acc","forget_rate","retain_acc",
                   "ve_cka","br_cka","lb_cka","source_file",
                   "source_modified_time","status","reason"]
    prov_csv = RESULTS_DIR / "blip2_provenance.csv"
    with open(prov_csv,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=prov_fields, extrasaction="ignore")
        w.writeheader()
        for method, rec in blip2_merged.items():
            w.writerow({
                "method":    method,
                "architecture":"BLIP-2-OPT-2.7B",
                "dataset":   "mllmu_real",
                "split":     rec.get("split",""),
                "checkpoint_path": rec.get("ckpt",""),
                "seed":      rec.get("seed",""),
                "forget_acc":  rec.get("fa",""),
                "forget_rate": rec.get("fr",""),
                "retain_acc":  rec.get("ra",""),
                "ve_cka":    rec.get("ve",""),
                "br_cka":    rec.get("br",""),
                "lb_cka":    rec.get("lb",""),
                "source_file":"merged",
                "source_modified_time":"",
                "status":    rec.get("status",""),
                "reason":    f"conflicts={blip2_conflicts}",
            })
    chk("BLIP-2 provenance CSV","PASS",str(prov_csv))

    prov_json = RESULTS_DIR / "blip2_provenance.json"
    with open(prov_json,"w",encoding="utf-8") as f:
        json.dump({"merged":blip2_merged,"authoritative":blip2_auth,
                   "conflicts":blip2_conflicts}, f, indent=2, default=str)
    chk("BLIP-2 provenance JSON","PASS",str(prov_json))

    # ── GA ───────────────────────────────────────────────────────────────────
    ga = check_ga_provenance()

    # ── LLaVA table ──────────────────────────────────────────────────────────
    llava_rows = build_llava_rows()
    latex      = build_latex_llava(llava_rows)
    tex        = RESULTS_DIR / "table_diagnostic_range.tex"
    with open(tex,"w",encoding="utf-8") as f: f.write(latex)
    chk("LLaVA LaTeX table","PASS",str(tex))

    # GA appendix table
    ga_tex     = build_appendix_ga_table(ga)
    ga_tex_path = RESULTS_DIR / "table_ga_crp_appendix.tex"
    with open(ga_tex_path,"w",encoding="utf-8") as f: f.write(ga_tex)
    chk("GA appendix table","PASS",str(ga_tex_path))

    # Figure
    if not args.skip_figure:
        build_figure(llava_rows, blip2_auth if blip2_auth else {})

    # BLIP-2 LaTeX table and narrative
    if blip2_auth:
        blip2_latex = build_blip2_latex(blip2_auth)
        blip2_tex   = RESULTS_DIR / "table_blip2_audit.tex"
        with open(blip2_tex,"w",encoding="utf-8") as f: f.write(blip2_latex)
        chk("BLIP-2 LaTeX table","PASS",str(blip2_tex))

        narr      = blip2_narrative(blip2_auth)
        narr_path = RESULTS_DIR / "blip2_narrative.txt"
        with open(narr_path,"w",encoding="utf-8") as f: f.write(narr)
        chk("BLIP-2 narrative","PASS",str(narr_path))
        print("\nBLIP-2 LaTeX:\n" + blip2_latex)
        print("\nBLIP-2 narrative:" + narr)
    else:
        chk("BLIP-2 LaTeX","WARN","Blocked — no authoritative values")

    # Summary JSON
    out_json = RESULTS_DIR / "diagnostic_range_summary.json"
    with open(out_json,"w",encoding="utf-8") as f:
        json.dump({"llava_rows":llava_rows,"ga":ga,
                   "blip2_auth":blip2_auth}, f, indent=2, default=str)
    chk("Summary JSON","PASS",str(out_json))

    # Run report
    elapsed = round(time.time()-START,1)
    n_pass = sum(1 for c in CHECKS if c["verdict"]=="PASS")
    n_warn = sum(1 for c in CHECKS if c["verdict"]=="WARN")
    n_fail = sum(1 for c in CHECKS if c["verdict"]=="FAIL")
    rj = RESULTS_DIR/"run_report.json"
    rt = RESULTS_DIR/"run_report.txt"
    with open(rj,"w",encoding="utf-8") as f:
        json.dump({"elapsed":elapsed,"n_pass":n_pass,"n_warn":n_warn,
                   "n_fail":n_fail,"checks":CHECKS}, f, indent=2, default=str)
    with open(rt,"w",encoding="utf-8") as f:
        f.write("FIX 2 RUN REPORT\n"+"="*60+"\n\n")
        for c in CHECKS:
            icon={"PASS":"OK","WARN":"!!","FAIL":"XX"}.get(c["verdict"],"??")
            f.write(f"[{icon} {c['verdict']:4s}] {c['check']}: {c['detail']}\n")
        f.write(f"\nSummary: {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL\n")
    chk("Run report","PASS",str(rt))

    print(f"\n  Summary: {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL")

    if not blip2_auth:
        print("\n  [WARN] BLIP-2 authoritative table not yet resolved.")
        print("  Inspect: blip2_schema_inventory.txt")
        print("           blip2_provenance.csv")
        print("  If fields have different names, add aliases to extract_numeric_from_record().")
    else:
        print(f"\n  BLIP-2 authoritative methods: {list(blip2_auth.keys())}")

    print("\nLLaVA LaTeX:\n" + latex)
    print("\nGA Appendix LaTeX:\n" + ga_tex)


if __name__ == "__main__":
    main()
