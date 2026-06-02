# `code/` — UIE comparison scripts

Automation for comparing image-enhancement methods against hand-edited images
via YOLO classification. Full workflow: [`documents/comparison_UIE.md`](../documents/comparison_UIE.md).

## One-command pipeline

[`compare_uie.py`](compare_uie.py) runs SOP steps 4 + 7–13 from a single YAML
config: crop matched patches → train a model per dataset → evaluate → compare.

```bash
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r ../requirements.txt

cp compare_uie.example.yaml my_run.local.yaml   # edit paths
python compare_uie.py my_run.local.yaml
```

Flags: `--reuse-datasets`, `--reuse-runs` (skip cropping/training if outputs
already exist), `--skip-training` (require pre-trained runs).

The pipeline package lives in [`uie_pipeline/`](uie_pipeline/):

| Module | Responsibility |
|---|---|
| `config.py` | Load & validate the YAML run config |
| `annotations.py` | Parse the Toolbox annotation JSON; crop patches |
| `dataset.py` | Build matched datasets with one shared stratified split |
| `training.py` | Train `YOLO*-cls` via the Ultralytics Python API |
| `evaluation.py` | Score the test split → `metrics_report.csv` |

## Standalone helper scripts

These remain usable on their own (interactive prompts) and are also imported by
the pipeline:

| Script | Purpose |
|---|---|
| [`update_annotation_paths.py`](update_annotation_paths.py) | Remap annotation JSON image paths to a UIE folder (manual workflow) |
| [`match_dataset_split.py`](match_dataset_split.py) | Apply a template dataset's train/val/test split to another dataset |
| [`ML_metrics_comparison.py`](ML_metrics_comparison.py) | Terminal side-by-side of two runs — `main()` prompts; `compare_runs(specs)` is importable |
| [`model_comparison_workbook_w_summary.py`](model_comparison_workbook_w_summary.py) | Full Excel workbook for 2+ runs — `main()` prompts; `build_workbook(models, baseline, out_dir)` is importable |
