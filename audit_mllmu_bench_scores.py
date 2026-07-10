"""
audit_mllmu_bench_scores.py
Automated score-validity audit for completed MLLMU-Bench outputs.
"""
from __future__ import annotations
import csv, hashlib, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_config import RESULTS_DIR

ROOT = RESULTS_DIR / "mllmu_bench_full"
OUT = ROOT / "score_audit"
OUT.mkdir(parents=True, exist_ok=True)
METHODS = ["base", "npo", "mmunlearner", "cagul", "sineproject", "graddiff"]
EXPECTED_FORGET, EXPECTED_RETAIN = 80, 119
EXPECTED_TOTAL = EXPECTED_FORGET + EXPECTED_RETAIN
CHECKS: list[dict[str, Any]] = []
GENERIC_ALIASES = {"person","man","woman","human","scientist","leader","artist","writer","actor","actress","president","king","queen","doctor","professor","engineer","politician","activist","singer","author","inventor","philosopher","researcher","teacher","athlete"}
GENERIC_RESPONSE_PHRASES = ["the person in the image","this person","the individual","the image shows","i cannot identify","i can't identify","unable to identify","appears to be"]

def chk(label, verdict, detail):
    CHECKS.append({"check": label, "verdict": verdict, "detail": str(detail)})
    icon = {"PASS":"OK","WARN":"!!","FAIL":"XX"}.get(verdict,"??")
    print(f"  [{icon} {verdict}] {label}: {detail}")

def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

def norm(x):
    return re.sub(r"\s+", " ", str(x or "").strip().lower())

def primary_ok(response, answer):
    r, a = norm(response), norm(answer)
    return bool(a) and a in r

def alias_ok(response, answer, aliases):
    r = norm(response)
    return any(norm(c) and norm(c) in r for c in [answer] + list(aliases or []))

def leakage(question, answer, aliases, entity):
    q, a, e = norm(question), norm(answer), norm(entity)
    leaked_aliases = [norm(x) for x in aliases or [] if norm(x) and norm(x) in q]
    return {"answer_in_question": bool(a) and a in q, "entity_in_question": bool(e) and e in q, "aliases_in_question": leaked_aliases}

def suspicious_aliases(aliases):
    out=[]
    for alias in aliases or []:
        a=norm(alias)
        if not a: out.append("<empty>")
        elif len(a)==1 or a in GENERIC_ALIASES: out.append(a)
    return out

def rhash(text):
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()

def load_method(method):
    p = ROOT / f"{method}_results.json"
    if not p.exists():
        chk(f"{method} result file", "FAIL", p); return None
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e: chk(f"{method} result file", "FAIL", e); return None

all_data={m:d for m in METHODS if (d:=load_method(m)) is not None}
chk("method coverage", "PASS" if len(all_data)==len(METHODS) else "FAIL", f"{len(all_data)}/{len(METHODS)} methods available")
if not all_data:
    save_json(OUT/"score_audit_report.json", {"checks":CHECKS,"overall_verdict":"FAIL"}); raise SystemExit(1)

ref_method = "base" if "base" in all_data else next(iter(all_data))
ref_records = all_data[ref_method].get("per_item", [])
ref_ids = [r.get("item_id") for r in ref_records]
chk("reference row count", "PASS" if len(ref_records)==EXPECTED_TOTAL else "FAIL", f"{len(ref_records)} / {EXPECTED_TOTAL}")
chk("reference unique IDs", "PASS" if len(set(ref_ids))==EXPECTED_TOTAL else "FAIL", f"{len(set(ref_ids))} / {EXPECTED_TOTAL}")
for m,d in all_data.items():
    rec=d.get("per_item",[]); ids=[r.get("item_id") for r in rec]
    if len(rec)!=EXPECTED_TOTAL: chk(f"{m} row count","FAIL",len(rec))
    elif ids!=ref_ids: chk(f"{m} item order","FAIL","IDs/order differ from base")
    else: chk(f"{m} item order","PASS","matches base")

qcount=Counter(norm(r.get("question")) for r in ref_records)
icount=Counter(str(r.get("image")) for r in ref_records)
dup_q={k:v for k,v in qcount.items() if k and v>1}; dup_i={k:v for k,v in icount.items() if k and v>1}
chk("duplicate questions", "WARN" if dup_q else "PASS", f"{len(dup_q)} duplicated texts" if dup_q else "none")
chk("duplicate images", "WARN" if dup_i else "PASS", f"{len(dup_i)} repeated paths" if dup_i else "none")

suspicious_rows=[]; metric_rows=[]; primary_alias_rows=[]
responses_by_method={m:{r["item_id"]:str(r.get("response","")) for r in d.get("per_item",[])} for m,d in all_data.items()}
leakage_count=generic_alias_count=missing_response=saved_mismatch=0
for m,d in all_data.items():
    stats=defaultdict(lambda:{"n":0,"saved":0,"primary":0,"alias":0,"generic":0})
    for r in d.get("per_item",[]):
        split=r.get("split"); response=str(r.get("response","")); answer=r.get("answer",""); aliases=r.get("aliases",[]) or []
        p=primary_ok(response,answer); a=alias_ok(response,answer,aliases); s=bool(r.get("correct"))
        gen=[x for x in GENERIC_RESPONSE_PHRASES if x in norm(response)]
        lk=leakage(r.get("question",""),answer,aliases,r.get("entity","")); bad=suspicious_aliases(aliases)
        st=stats[split]; st["n"]+=1; st["saved"]+=int(s); st["primary"]+=int(p); st["alias"]+=int(a); st["generic"]+=int(bool(gen))
        saved_mismatch += int(s!=a); missing_response += int(not response)
        leakage_count += int(lk["answer_in_question"] or lk["entity_in_question"] or bool(lk["aliases_in_question"]))
        generic_alias_count += int(bool(bad))
        reasons=[]
        if lk["answer_in_question"]: reasons.append("answer_in_question")
        if lk["entity_in_question"]: reasons.append("entity_in_question")
        if lk["aliases_in_question"]: reasons.append("alias_in_question")
        if bad: reasons.append("suspicious_alias")
        if a and not p: reasons.append("alias_only_correct")
        if gen and a: reasons.append("generic_response_marked_correct")
        if s!=a: reasons.append("saved_vs_recomputed_mismatch")
        if not response: reasons.append("missing_response")
        if reasons:
            suspicious_rows.append({"method":m,"split":split,"item_id":r.get("item_id"),"entity":r.get("entity"),"question":r.get("question"),"answer":answer,"aliases":json.dumps(aliases,ensure_ascii=False),"response":response,"saved_correct":s,"primary_correct":p,"alias_correct":a,"answer_in_question":lk["answer_in_question"],"entity_in_question":lk["entity_in_question"],"aliases_in_question":json.dumps(lk["aliases_in_question"],ensure_ascii=False),"suspicious_aliases":json.dumps(bad,ensure_ascii=False),"generic_response_hits":json.dumps(gen),"reasons":";".join(reasons)})
    for split,st in stats.items():
        n=st["n"]
        metric_rows.append({"method":m,"split":split,"n":n,"saved_accuracy":st["saved"]/n,"recomputed_alias_accuracy":st["alias"]/n,"primary_answer_accuracy":st["primary"]/n,"alias_only_gain":(st["alias"]-st["primary"])/n,"generic_response_rate":st["generic"]/n})
        primary_alias_rows.append({"method":m,"split":split,"saved_correct":st["saved"],"primary_correct":st["primary"],"alias_correct":st["alias"],"n":n})

chk("saved-vs-recomputed alias scoring", "FAIL" if saved_mismatch else "PASS", f"{saved_mismatch} mismatches" if saved_mismatch else "all rows match")
chk("question leakage", "WARN" if leakage_count else "PASS", f"{leakage_count} method-item rows affected" if leakage_count else "none detected")
chk("suspicious aliases", "WARN" if generic_alias_count else "PASS", f"{generic_alias_count} method-item rows" if generic_alias_count else "none detected")
chk("full responses", "FAIL" if missing_response else "PASS", f"{missing_response} missing" if missing_response else "all responses present")

identity_rows=[]
for a in METHODS:
    if a not in responses_by_method: continue
    for b in METHODS:
        if b not in responses_by_method: continue
        same=total=0
        for iid in ref_ids:
            if iid in responses_by_method[a] and iid in responses_by_method[b]:
                total+=1; same += int(rhash(responses_by_method[a][iid])==rhash(responses_by_method[b][iid]))
        identity_rows.append({"method_a":a,"method_b":b,"identical_responses":same,"total":total,"identity_rate":same/total if total else float("nan")})
for m in METHODS:
    if m=="base" or m not in responses_by_method: continue
    row=next(x for x in identity_rows if x["method_a"]=="base" and x["method_b"]==m)
    rate=row["identity_rate"]
    chk(f"{m} response divergence", "WARN" if rate>=0.95 else "PASS", f"identity with base={rate:.4f}")

for m,d in all_data.items():
    cp=d.get("checkpoint",{})
    if m=="base": chk("base provenance", "PASS" if cp.get("type")=="base_model" else "FAIL", cp.get("type")); continue
    files=cp.get("files",[]); af=[f for f in files if Path(f.get("path","")).name in {"adapter_config.json","adapter_model.safetensors","adapter_model.bin"}]
    chk(f"{m} checkpoint evidence", "PASS" if af else "FAIL", [Path(f["path"]).name for f in af] if af else "adapter files absent")

primary_diffs=[r for r in metric_rows if abs(r["saved_accuracy"]-r["primary_answer_accuracy"])>1e-12]
chk("primary-vs-alias scoring", "WARN" if primary_diffs else "PASS", f"{len(primary_diffs)} method/split rows differ" if primary_diffs else "identical")

for name,rows in [("recomputed_metrics.csv",metric_rows),("primary_vs_alias_metrics.csv",primary_alias_rows),("response_identity_matrix.csv",identity_rows)]:
    if rows:
        with (OUT/name).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
if suspicious_rows:
    with (OUT/"suspicious_items.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(suspicious_rows[0].keys())); w.writeheader(); w.writerows(suspicious_rows)
else:
    (OUT/"suspicious_items.csv").write_text("method,split,item_id,reasons\n",encoding="utf-8")

npass=sum(c["verdict"]=="PASS" for c in CHECKS); nwarn=sum(c["verdict"]=="WARN" for c in CHECKS); nfail=sum(c["verdict"]=="FAIL" for c in CHECKS)
overall="FAIL" if nfail else ("WARN" if nwarn else "PASS")
safe=(nfail==0 and saved_mismatch==0 and missing_response==0)
report={"script":"audit_mllmu_bench_scores.py","overall_verdict":overall,"safe_for_paper_with_caveats":safe,"checks":CHECKS,"n_pass":npass,"n_warn":nwarn,"n_fail":nfail,"summary":{"question_leakage_rows":leakage_count,"suspicious_alias_rows":generic_alias_count,"saved_correct_mismatches":saved_mismatch,"missing_full_responses":missing_response,"duplicate_question_texts":len(dup_q),"duplicate_image_paths":len(dup_i),"suspicious_item_rows":len(suspicious_rows)},"recomputed_metrics":metric_rows,"response_identity":identity_rows,"recommended_interpretation":"Use only if there are no FAIL checks. If warnings remain, report the benchmark as non-discriminative and disclose alias/leakage limitations."}
save_json(OUT/"score_audit_report.json",report)
lines=["MLLMU-BENCH SCORE-VALIDITY AUDIT","="*76,f"Overall verdict: {overall}",f"Safe for paper with caveats: {safe}",f"PASS={npass} WARN={nwarn} FAIL={nfail}",""]
for r in metric_rows:
    lines.append(f"{r['method']:<16} {r['split']:<7} saved={r['saved_accuracy']:.4f} primary={r['primary_answer_accuracy']:.4f} alias={r['recomputed_alias_accuracy']:.4f} alias_gain={r['alias_only_gain']:+.4f}")
(OUT/"score_audit_summary.txt").write_text("\n".join(lines),encoding="utf-8")
print("\n"+"="*76); print("MLLMU-BENCH SCORE AUDIT SUMMARY"); print("="*76)
for r in metric_rows:
    print(f"{r['method']:<16} {r['split']:<7} saved={r['saved_accuracy']:.4f} primary={r['primary_answer_accuracy']:.4f} alias={r['recomputed_alias_accuracy']:.4f}")
print(f"\nOverall verdict: {overall}"); print(f"Safe for paper with caveats: {safe}"); print(f"Report: {OUT/'score_audit_report.json'}")
raise SystemExit(1 if nfail else 0)
