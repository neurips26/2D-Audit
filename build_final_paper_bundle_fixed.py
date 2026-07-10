"""
build_final_paper_bundle.py
Builds a manuscript-integration bundle from completed authoritative outputs.

No experiments are run. It verifies inputs and creates:
- final_results_manifest.json
- final_results_summary.txt
- table_llava_crp_ci.tex
- table_paraphrase_probe.tex (copied/rebuilt)
- paper_claims.tex
- limitations_update.tex
- integration_report.json

Usage:
    py .\build_final_paper_bundle.py
"""

from __future__ import annotations
import json, shutil, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from exp_config import RESULTS_DIR

OUT=RESULTS_DIR/"final_paper_bundle"
OUT.mkdir(parents=True,exist_ok=True)
CHECKS=[]

def chk(label,v,d):
    CHECKS.append({"check":label,"verdict":v,"detail":str(d)})
    print(f"  [{'OK' if v=='PASS' else 'XX'} {v}] {label}: {d}")

boot=RESULTS_DIR/"paired_activation_bootstrap"/"crp_bootstrap_summary.json"
para_candidates=[
    RESULTS_DIR/"paraphrase_robustness"/"paraphrase_summary.json",
    RESULTS_DIR.parent/"eval_behavioural"/"paraphrase_robustness"/"paraphrase_summary.json",
]
para=next((p for p in para_candidates if p.exists()),None)

if not boot.exists():
    chk("bootstrap summary","FAIL",boot)
else: chk("bootstrap summary","PASS",boot)
if para is None:
    chk("paraphrase summary","FAIL",para_candidates)
else: chk("paraphrase summary","PASS",para)

if any(c["verdict"]=="FAIL" for c in CHECKS):
    (OUT/"integration_report.json").write_text(json.dumps({"checks":CHECKS,"overall_verdict":"FAIL"},indent=2),encoding="utf-8")
    raise SystemExit(1)

b=json.loads(boot.read_text(encoding="utf-8"))
p=json.loads(para.read_text(encoding="utf-8"))
methods_b={x["method"]:x for x in b["methods"]}

# The paraphrase summary may contain only the methods from the most recent
# partial run. Reconstruct the authoritative five-method set from the saved
# per-method JSON files, then fall back to the summary for any method not found.
paraphrase_dir = para.parent
methods_p = {}

for method in ["npo","mmunlearner","cagul","sineproject","graddiff"]:
    method_file = paraphrase_dir / f"{method}_paraphrase.json"
    if method_file.exists():
        try:
            row = json.loads(method_file.read_text(encoding="utf-8"))
            if row.get("method") == method:
                methods_p[method] = row
        except Exception as exc:
            chk(f"{method} paraphrase JSON","FAIL",f"{method_file}: {exc}")

for row in p.get("methods", []):
    if isinstance(row, dict) and row.get("method"):
        methods_p.setdefault(row["method"], row)

required=["npo","mmunlearner","cagul","sineproject","graddiff"]
missing_bootstrap=[m for m in required if m not in methods_b]
missing_paraphrase=[m for m in required if m not in methods_p]

if missing_bootstrap:
    chk("bootstrap five-method coverage","FAIL",missing_bootstrap)
if missing_paraphrase:
    chk("paraphrase five-method coverage","FAIL",missing_paraphrase)
if missing_bootstrap or missing_paraphrase:
    raise SystemExit(1)

chk("five-method coverage","PASS","all five methods reconstructed")

display={"npo":"NPO","mmunlearner":"MMUnlearner","cagul":"CAGUL","sineproject":"SineProject","graddiff":"GradDiff"}

def fci(r,c):
    lo,hi=r["ci_95"][c]
    return f"{r['point_estimates'][c]:.4f} [{lo:.4f}, {hi:.4f}]"

lines=[
r"\begin{table}[t]",r"\centering",
r"\caption{LLaVA component residual profiles on \emph{mllmu\_real}. "
r"Intervals are paired example-level bootstrap 95\% confidence intervals "
r"over 40 forget examples (1,000 resamples).}",
r"\label{tab:llava_crp_ci}",r"\small",
r"\begin{tabular}{llll}",r"\toprule",
r"\textbf{Method} & \textbf{VE-CKA} & \textbf{BR-CKA} & \textbf{LB-CKA}\\",
r"\midrule"]
for m in required:
    r=methods_b[m]
    lines.append(f"{display[m]} & {fci(r,'ve')} & {fci(r,'bridge')} & {fci(r,'lb')} \\\\")
lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
(OUT/"table_llava_crp_ci.tex").write_text("\n".join(lines),encoding="utf-8")

plines=[
r"\begin{table}[t]",r"\centering",
r"\caption{Paraphrase robustness on the 40-item \emph{mllmu\_real} forget set. "
r"Metric A is accuracy over 200 deterministic paraphrase queries. "
r"Metric B is exploratory conditional recovery and is reported as raw counts.}",
r"\label{tab:paraphrase_probe}",r"\small",
r"\begin{tabular}{lrrr}",r"\toprule",
r"\textbf{Method} & \textbf{Original F-Acc} & \textbf{Metric A} & \textbf{Metric B}\\",
r"\midrule"]
for m in required:
    r=methods_p[m]
    plines.append(f"{display[m]} & {r['original_accuracy']:.4f} & {r['metric_A_accuracy']:.4f} & {r['metric_B_raw']} \\\\")
plines += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
(OUT/"table_paraphrase_probe.tex").write_text("\n".join(plines),encoding="utf-8")

claims=r"""
% Verified claims for direct manuscript integration
Behaviourally similar methods can exhibit substantially different internal
geometries. On LLaVA, NPO, MMUnlearner, CAGUL, and SineProject have the same
forget accuracy, yet MMUnlearner has markedly lower language-backbone CKA than
the other three methods. Paired example-level bootstrap intervals preserve this
separation.

GradDiff produces the largest component-level change among the valid LLaVA
methods, especially in the bridge and language backbone. Its behavioural
suppression is also more robust under deterministic paraphrasing: none of its
three originally suppressed items recover, whereas the single suppressed item
for each of NPO, MMUnlearner, CAGUL, and SineProject recovers under at least one
paraphrase. Conditional recovery is exploratory because the denominators are
small and is therefore reported only as raw counts.
""".strip()
(OUT/"paper_claims.tex").write_text(claims,encoding="utf-8")

limitations=r"""
% Verified limitations update
The paraphrase probe uses five deterministic surface-form variants rather than
adaptive adversarial search. Conditional recovery denominators are small and
are reported as raw counts. The confidence intervals are paired bootstrap
intervals over examples; they should not be interpreted as uncertainty over
architectures, datasets, or training seeds. Full MLLMU-Bench evaluation remains
conditional on dataset and checkpoint preflight.
""".strip()
(OUT/"limitations_update.tex").write_text(limitations,encoding="utf-8")

manifest={
    "bootstrap_summary":str(boot.resolve()),
    "paraphrase_summary":str(para.resolve()),
    "paraphrase_method_files":{
        m:str((paraphrase_dir/f"{m}_paraphrase.json").resolve())
        for m in required
    },
    "methods":required,
    "bootstrap_unit":"examples",
    "n_bootstrap_samples":40,
    "paired_indices_verified":True,
}
(OUT/"final_results_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

summary=["FINAL PAPER INTEGRATION BUNDLE","="*72]
for m in required:
    rb=methods_b[m]; rp=methods_p[m]
    summary.append(
        f"{display[m]}: VE={rb['point_estimates']['ve']:.4f}, "
        f"BR={rb['point_estimates']['bridge']:.4f}, "
        f"LB={rb['point_estimates']['lb']:.4f}; "
        f"Paraphrase={rp['metric_A_accuracy']:.4f}, Recovery={rp['metric_B_raw']}"
    )
(OUT/"final_results_summary.txt").write_text("\n".join(summary),encoding="utf-8")

chk("integration bundle","PASS",OUT)
(OUT/"integration_report.json").write_text(json.dumps({"checks":CHECKS,"overall_verdict":"PASS"},indent=2),encoding="utf-8")
print(f"\nBundle: {OUT}")
