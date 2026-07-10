# GradDiff paired flip analysis

Copy these two files into the project root:

- `analyze_graddiff_flips.py`
- `run_graddiff_flip_analysis.ps1`

Run:

```powershell
.\run_graddiff_flip_analysis.ps1
```

The script auto-discovers:

```text
outputs\revision\graddiff\base_model
outputs\revision\graddiff\...\graddiff_llava_*steps
```

It expects each result directory to contain:

```text
forget_predictions.json
retain_predictions.json
```

Outputs are written to:

```text
outputs\revision\graddiff\paired_flip_analysis
```

The key files are:

- `paired_flip_report.txt`
- `paired_flip_summary.json`
- one forget CSV and one retain CSV per checkpoint

Interpretation:

- `successful_forgetting`: base correct, checkpoint incorrect
- `unexpected_recovery`: base incorrect, checkpoint correct
- `utility_damage`: base correct, checkpoint incorrect
- `utility_improvement`: base incorrect, checkpoint correct
