"""
mllmu_bench_preflight.py
Strict, no-GPU preflight for a possible MLLMU-Bench evaluation.

It does not invent dataset paths or run an evaluation. It discovers candidate
paths from config modules and ROOT/data, validates structure, images, QA fields,
checkpoint coverage, and authoritative evaluator availability, then writes a
machine-readable go/no-go report.

Usage:
    py .\mllmu_bench_preflight.py
"""

from __future__ import annotations
import hashlib, inspect, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_config

OUT = exp_config.RESULTS_DIR / "mllmu_bench_preflight"
OUT.mkdir(parents=True, exist_ok=True)
CHECKS = []
METHODS = ["npo", "mmunlearner", "cagul", "sineproject", "graddiff"]

def chk(label, verdict, detail):
    CHECKS.append({"check":label,"verdict":verdict,"detail":str(detail)})
    icon={"PASS":"OK","WARN":"!!","FAIL":"XX"}.get(verdict,"??")
    print(f"  [{icon} {verdict}] {label}: {detail}")

def candidate_dirs():
    out=[]
    for module_name in ("eval_config","exp_config","config"):
        try:
            mod=__import__(module_name)
        except Exception:
            continue
        for name in dir(mod):
            if "MLLMU" in name.upper() and "REAL" not in name.upper():
                value=getattr(mod,name)
                if isinstance(value,(str,Path)):
                    p=Path(value)
                    out.append((f"{module_name}.{name}",p))
    root=Path(exp_config.ROOT)
    for rel in (
        "data/mllmu_bench","data/MLLMU_Bench","data/MLLMU-Bench",
        "data/mllmu","datasets/mllmu_bench","datasets/MLLMU-Bench"
    ):
        out.append((f"discovery:{rel}",root/rel))
    seen=set(); unique=[]
    for source,p in out:
        rp=str(p.expanduser().resolve())
        if rp not in seen:
            seen.add(rp); unique.append((source,Path(rp)))
    return unique

def scan_split(path):
    if not path.exists() or not path.is_dir():
        return None
    ann=path/"annotations.json"
    items=[]
    if ann.exists():
        try:
            raw=json.loads(ann.read_text(encoding="utf-8"))
            if isinstance(raw,list):
                for r in raw:
                    img=Path(r.get("image",""))
                    if not img.is_absolute(): img=path/img
                    items.append({
                        "image":img,"question":r.get("question"),
                        "answer":r.get("answer",r.get("gt")),
                    })
        except Exception as e:
            return {"error":f"annotations parse failed: {e}"}
    else:
        for d in sorted(path.iterdir()):
            if not d.is_dir(): continue
            imgs=list(d.glob("*.jpg"))+list(d.glob("*.jpeg"))+list(d.glob("*.png"))
            js=list(d.glob("*.json"))
            if not imgs or not js: continue
            try:
                q=json.loads(js[0].read_text(encoding="utf-8"))
                for r in (q if isinstance(q,list) else [q]):
                    items.append({"image":imgs[0],"question":r.get("question"),
                                  "answer":r.get("answer",r.get("gt"))})
            except Exception:
                pass
    missing_images=sum(not x["image"].exists() for x in items)
    missing_q=sum(not x["question"] for x in items)
    missing_a=sum(not x["answer"] for x in items)
    return {
        "path":str(path.resolve()),"n_items":len(items),
        "missing_images":missing_images,"missing_questions":missing_q,
        "missing_answers":missing_a,
    }

cands=candidate_dirs()
existing=[(s,p) for s,p in cands if p.exists()]
if not existing:
    chk("dataset discovery","FAIL",
        "No MLLMU-Bench-like directory found. Candidates: "+
        "; ".join(f"{s}={p}" for s,p in cands))
    dataset_root=None
else:
    chk("dataset discovery","PASS",
        "; ".join(f"{s}={p}" for s,p in existing))
    dataset_root=existing[0][1]

split_results={}
if dataset_root:
    names=["forget","retain","test","validation","val","train"]
    for name in names:
        p=dataset_root/name
        r=scan_split(p)
        if r: split_results[name]=r
    if not split_results:
        direct=scan_split(dataset_root)
        if direct: split_results["root"]=direct

    if not split_results:
        chk("dataset structure","FAIL","No parseable split/root structure")
    else:
        bad=[]
        for name,r in split_results.items():
            if "error" in r or r["n_items"]==0 or r["missing_images"] or r["missing_questions"] or r["missing_answers"]:
                bad.append((name,r))
        if bad:
            chk("dataset integrity","FAIL",bad)
        else:
            chk("dataset integrity","PASS",split_results)

# Check methods
adapters=getattr(exp_config,"LLAVA_ADAPTERS",{})
coverage={}
for m in METHODS:
    p=adapters.get(m)
    ok=p is not None and Path(str(p)).exists()
    coverage[m]={"path":str(p) if p is not None else None,"exists":ok}
missing=[m for m,v in coverage.items() if not v["exists"]]
if missing:
    chk("checkpoint coverage","FAIL",f"Missing: {missing}")
else:
    chk("checkpoint coverage","PASS","all five valid LLaVA checkpoints exist")

# Authoritative evaluator availability
try:
    import eval_utils
    required=["load_llava_model","run_llava_inference","score_response"]
    absent=[x for x in required if not hasattr(eval_utils,x)]
    if absent:
        chk("authoritative evaluator","FAIL",f"missing functions: {absent}")
        evaluator=None
    else:
        evaluator={
            "module":str(Path(eval_utils.__file__).resolve()),
            "functions":{x:str(inspect.signature(getattr(eval_utils,x))) for x in required}
        }
        chk("authoritative evaluator","PASS",evaluator)
except Exception as e:
    evaluator=None
    chk("authoritative evaluator","FAIL",repr(e))

npass=sum(c["verdict"]=="PASS" for c in CHECKS)
nwarn=sum(c["verdict"]=="WARN" for c in CHECKS)
nfail=sum(c["verdict"]=="FAIL" for c in CHECKS)
verdict="FAIL" if nfail else ("WARN" if nwarn else "PASS")
report={
    "overall_verdict":verdict,
    "checks":CHECKS,
    "dataset_root":str(dataset_root) if dataset_root else None,
    "splits":split_results,
    "checkpoint_coverage":coverage,
    "evaluator":evaluator,
    "safe_to_run_full_evaluation": verdict=="PASS",
    "next_action":(
        "Run full MLLMU-Bench evaluation only after a PASS."
        if verdict=="PASS" else
        "Do not run full evaluation. Resolve all FAIL checks first."
    ),
}
(OUT/"preflight_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
(OUT/"preflight_report.txt").write_text(
    "\n".join([
        "MLLMU-BENCH PREFLIGHT","="*72,
        f"OVERALL VERDICT: {verdict}",
        f"Dataset root: {report['dataset_root']}",
        f"Safe to run full evaluation: {report['safe_to_run_full_evaluation']}",
        report["next_action"],
    ]),encoding="utf-8")
print(f"\nOVERALL VERDICT: {verdict}")
print(f"Report: {OUT/'preflight_report.json'}")
raise SystemExit(0 if verdict=="PASS" else 1)
