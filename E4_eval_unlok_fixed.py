from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_config import (
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_N,
    DEVICE,
    LLAVA_ADAPTERS,
    LLAVA_LB_LAYERS,
    LLAVA_VE_LAYERS,
    MAX_NEW_TOKENS,
    RESULTS_DIR,
    UNLOK_FORGET_DIR,
    UNLOK_RETAIN_DIR,
)
from E2_run_audit_per_entity import debiased_cka, get_activations, load_llava

OUTDIR = RESULTS_DIR / "unlok_vqa_fixed"
OUTDIR.mkdir(parents=True, exist_ok=True)
METHODS = ["ga_retrained", "npo", "mmunlearner", "cagul", "sineproject"]


def load_split(path: Path, limit: int | None) -> list[dict]:
    ann = path / "annotations.json"
    if not ann.exists():
        raise FileNotFoundError(f"Missing annotations: {ann}")
    rows = json.loads(ann.read_text(encoding="utf-8"))
    rows = [x for x in rows if Path(x["image"]).exists()]
    return rows[:limit] if limit else rows


def score(response: str, answer: str, aliases: list[str] | None = None) -> bool:
    r = response.lower().strip()
    candidates = [answer.lower()] + [str(a).lower() for a in (aliases or [])]
    return any(c and c in r for c in candidates)


def infer(model, processor, image: Image.Image, question: str) -> str:
    prompt = f"USER: <image>\n{question} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=False,
        )
    text = processor.decode(out[0], skip_special_tokens=True)
    return text.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in text else text.strip()


def crop_attack(image: Image.Image) -> Image.Image:
    w, h = image.size
    dx, dy = max(1, int(0.08 * w)), max(1, int(0.08 * h))
    return image.crop((dx, dy, w - dx, h - dy)).resize((w, h))


def perturb_attack(image: Image.Image) -> Image.Image:
    # Deterministic, mild photometric perturbation.
    return ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(0.94)).enhance(1.08)


def stack_hook(rows: list[dict], key: str) -> torch.Tensor | None:
    vals = [r[key] for r in rows if key in r]
    if len(vals) != len(rows) or not vals:
        return None
    return torch.stack(vals)


def component_estimate(base_rows: list[dict], mu_rows: list[dict], keys: list[str]) -> float:
    vals = []
    for key in keys:
        x, y = stack_hook(base_rows, key), stack_hook(mu_rows, key)
        if x is not None and y is not None and x.shape == y.shape:
            value = debiased_cka(x, y)
            if math.isfinite(value):
                vals.append(value)
    return float(np.mean(vals)) if vals else float("nan")


def component_bootstrap(
    base_rows: list[dict],
    mu_rows: list[dict],
    keys: list[str],
    seed: int,
) -> list[float]:
    pairs = []
    for key in keys:
        x, y = stack_hook(base_rows, key), stack_hook(mu_rows, key)
        if x is not None and y is not None and x.shape == y.shape:
            pairs.append((x, y))
    if not pairs or pairs[0][0].shape[0] < 4:
        return [float("nan"), float("nan")]
    n = pairs[0][0].shape[0]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(BOOTSTRAP_N):
        idx = torch.tensor(rng.integers(0, n, size=n), dtype=torch.long)
        vals = [debiased_cka(x[idx], y[idx]) for x, y in pairs]
        vals = [v for v in vals if math.isfinite(v)]
        if vals:
            boots.append(float(np.mean(vals)))
    if len(boots) < 2:
        return [float("nan"), float("nan")]
    lo = float(np.quantile(boots, BOOTSTRAP_ALPHA / 2))
    hi = float(np.quantile(boots, 1 - BOOTSTRAP_ALPHA / 2))
    return [lo, hi]


def compute_matrix_crp(base_rows: list[dict], mu_rows: list[dict], seed: int) -> dict:
    ve_keys = [f"ve_{i}" for i in LLAVA_VE_LAYERS]
    lb_keys = [f"lb_{i}" for i in LLAVA_LB_LAYERS]
    return {
        "metric": "debiased_linear_cka_matrix_forward_pass",
        "ve_cka": component_estimate(base_rows, mu_rows, ve_keys),
        "bridge_cka": component_estimate(base_rows, mu_rows, ["bridge"]),
        "lb_cka": component_estimate(base_rows, mu_rows, lb_keys),
        "ve_ci_95": component_bootstrap(base_rows, mu_rows, ve_keys, seed + 11),
        "bridge_ci_95": component_bootstrap(base_rows, mu_rows, ["bridge"], seed + 23),
        "lb_ci_95": component_bootstrap(base_rows, mu_rows, lb_keys, seed + 37),
    }


def evaluate(method: str, forget: list[dict], retain: list[dict], seed: int) -> dict:
    ckpt = LLAVA_ADAPTERS.get(method)
    if method != "no_unlearn" and (ckpt is None or not Path(ckpt).exists()):
        raise FileNotFoundError(f"Missing checkpoint for {method}: {ckpt}")

    m0, processor = load_llava(None)
    mu, _ = load_llava(ckpt)

    direct = rephrase_ok = rephrase_n = crop_ok = perturb_ok = 0
    base_acts, mu_acts = [], []

    for i, item in enumerate(forget, 1):
        image = Image.open(item["image"]).convert("RGB")
        answer, aliases = item["answer"], item.get("aliases")

        direct += int(score(infer(mu, processor, image, item["question"]), answer, aliases))

        questions = item.get("rephrase_questions") or item.get("rephrases") or []
        for question in questions:
            rephrase_n += 1
            rephrase_ok += int(score(infer(mu, processor, image, question), answer, aliases))

        crop_ok += int(score(infer(mu, processor, crop_attack(image), item["question"]), answer, aliases))
        perturb_ok += int(score(infer(mu, processor, perturb_attack(image), item["question"]), answer, aliases))

        base_acts.append(get_activations(m0, processor, item))
        mu_acts.append(get_activations(mu, processor, item))
        print(f"[{method}] forget {i}/{len(forget)}")

    retain_ok = 0
    for i, item in enumerate(retain, 1):
        image = Image.open(item["image"]).convert("RGB")
        retain_ok += int(score(infer(mu, processor, image, item["question"]), item["answer"], item.get("aliases")))
        if i % 25 == 0 or i == len(retain):
            print(f"[{method}] retain {i}/{len(retain)}")

    crp = compute_matrix_crp(base_acts, mu_acts, seed)
    result = {
        "method": method,
        "dataset": "unlok_vqa",
        "n_forget": len(forget),
        "n_retain": len(retain),
        "forget_acc": direct / len(forget),
        "forget_rate": 1 - direct / len(forget),
        "retain_acc": retain_ok / len(retain) if retain else float("nan"),
        "attacks": {
            "direct": direct / len(forget),
            "rephrase": rephrase_ok / rephrase_n if rephrase_n else float("nan"),
            "crop": crop_ok / len(forget),
            "perturb": perturb_ok / len(forget),
            "rephrase_n": rephrase_n,
        },
        **crp,
    }

    del m0, mu
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--n_forget", type=int, default=100)
    parser.add_argument("--n_retain", type=int, default=105)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checks = []
    def check(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    try:
        forget = load_split(UNLOK_FORGET_DIR, args.n_forget)
        retain = load_split(UNLOK_RETAIN_DIR, args.n_retain)
        check("Forget split loaded", len(forget) >= 4, f"n={len(forget)}")
        check("Retain split loaded", len(retain) > 0, f"n={len(retain)}")

        results = []
        for method in args.methods:
            result = evaluate(method, forget, retain, args.seed)
            results.append(result)
            path = OUTDIR / f"{method}_unlok_fixed.json"
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            check(f"{method} matrix CKA", result["metric"] == "debiased_linear_cka_matrix_forward_pass", result["metric"])
            check(f"{method} finite CRP", all(math.isfinite(result[k]) for k in ("ve_cka", "bridge_cka", "lb_cka")), {k: result[k] for k in ("ve_cka", "bridge_cka", "lb_cka")})

        (OUTDIR / "unlok_fixed_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    except Exception as exc:
        check("Execution", False, f"{type(exc).__name__}: {exc}")

    overall = "PASS" if checks and all(c["passed"] for c in checks) else "FAIL"
    report = {"overall": overall, "checks": checks}
    (OUTDIR / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 88)
    print("UNLOK FIXED EVALUATION VERDICT")
    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print(f"OVERALL VERDICT: {overall}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
