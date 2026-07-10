# GradDiff — safe first run

Copy these files into the `2dunl` project root:

- `train_graddiff_llava_fixed.py`
- `run_graddiff_smoke.ps1`

The Python script imports your existing `exp_config.py`; do not replace the master config yet.

## 1. Smoke test

```powershell
Copy-Item .\train_graddiff_llava_fixed.py .\train_graddiff_llava.py -Force
.\run_graddiff_smoke.ps1
```

Expected final line:

```text
OVERALL VERDICT: PASS
```

## 2. First controlled 50-step run

```powershell
py -u .\train_graddiff_llava.py --steps 50 --save_steps 5 10 20 50 --lambda_retain 1.0 --lr 1e-4 --seed 42
```

The checkpoint path includes learning rate, retain weight, and seed, so later sweep runs do not overwrite it.

## 3. Full behavioral check

The script runs the full available forget and retain splits automatically unless `--skip_final_eval` is passed. To re-check a checkpoint:

```powershell
py -u .\train_graddiff_llava.py --check_ckpt "<exact checkpoint path>" --eval_forget_n 0 --eval_retain_n 0
```

Predefined calibration threshold:

- Forget accuracy <= 0.50
- Retain accuracy >= 0.80

This is reported as `PARTIAL_SELECTIVE_CALIBRATION`, not as proof of perfect selective unlearning.

## Safeguards included

- answer-only loss masking
- deterministic seeds
- finite-loss and gradient checks
- gradient clipping
- bounded retries
- unique checkpoint names
- full prediction logs
- PASS/WARN/FAIL text and JSON reports
