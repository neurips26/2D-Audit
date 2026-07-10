from pathlib import Path
import json
import subprocess
import sys

root = Path.cwd()
result_dir = root / "outputs" / "revision" / "unlok_vqa_fixed"
log_dir = root / "outputs" / "revision" / "unlok_400_logs"
log_dir.mkdir(parents=True, exist_ok=True)

methods = [
    "ga_attn4_50",
    "npo",
    "mmunlearner",
    "cagul",
    "sineproject",
]

def completed_methods():
    done = set()

    if not result_dir.exists():
        return done

    for path in result_dir.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue

        rows = data if isinstance(data, list) else [data]

        if isinstance(data, dict) and isinstance(data.get("results"), list):
            rows += data["results"]

        for row in rows:
            if not isinstance(row, dict):
                continue

            method = row.get("method")
            n_forget = int(row.get("n_forget", 0) or 0)
            n_retain = int(row.get("n_retain", 0) or 0)

            if method in methods and n_forget >= 400 and n_retain >= 400:
                done.add(method)

    return done


done = completed_methods()
failures = []

print("=" * 80)
print("RESUMABLE FULL UNLOK 400/400 RUN")
print("=" * 80)

for method in methods:
    if method in done:
        print(f"[SKIP] {method}: valid 400/400 result already exists.")
        continue

    log_path = log_dir / f"{method}_400x400.log"

    command = [
        sys.executable,
        "-u",
        "E4_eval_unlok_fixed.py",
        "--methods", method,
        "--n_forget", "400",
        "--n_retain", "400",
    ]

    print(f"\n[RUN] {method}")
    print("Command:", " ".join(command))
    print("Log:", log_path)

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()

        exit_code = process.wait()

    if exit_code != 0:
        failures.append({
            "method": method,
            "exit_code": exit_code,
            "log": str(log_path),
        })
        print(f"[FAIL] {method}: exit code {exit_code}")
        break

    done_after = completed_methods()

    if method not in done_after:
        failures.append({
            "method": method,
            "exit_code": 0,
            "log": str(log_path),
            "reason": "Process exited successfully but no valid 400/400 result was found.",
        })
        print(f"[FAIL] {method}: output validation failed.")
        break

    print(f"[PASS] {method}: validated 400/400 result.")

print("\n" + "=" * 80)

if failures:
    report = log_dir / "full_unlok_400_status.json"
    report.write_text(
        json.dumps(
            {
                "overall_verdict": "FAIL",
                "completed": sorted(completed_methods()),
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("OVERALL VERDICT: FAIL")
    print("REPORT:", report)
    raise SystemExit(1)

report = log_dir / "full_unlok_400_status.json"
report.write_text(
    json.dumps(
        {
            "overall_verdict": "PASS",
            "completed": sorted(completed_methods()),
        },
        indent=2,
    ),
    encoding="utf-8",
)

print("OVERALL VERDICT: PASS")
print("COMPLETED:", ", ".join(sorted(completed_methods())))
print("REPORT:", report)
