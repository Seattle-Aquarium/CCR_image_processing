#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_uie.py
==============

One command to run the entire "Compare UIE Output vs Hand-Edited" SOP
(documents/comparison_UIE.md) from a single YAML config.

Given the hand-edited annotation JSON and one image folder per enhancement
method, it will:

  1. Crop matched classification patches for every method, using one shared,
     class-stratified train/val/test split           (SOP steps 4, 7-10)
  2. Train a YOLO*-cls model on each dataset           (SOP step 11)
  3. Evaluate each on the test split -> metrics_report.csv  (SOP step 12)
  4. Compare all models: terminal table + model_comparison.xlsx  (SOP step 13)

Usage
-----
    python code/compare_uie.py code/compare_uie.example.yaml

    # re-run only the comparison/report from existing runs:
    python code/compare_uie.py config.yaml --reuse-datasets --reuse-runs

Run inside the project virtualenv (see requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Make sibling modules + the uie_pipeline package importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uie_pipeline import annotations as anno
from uie_pipeline import config as cfg_mod
from uie_pipeline import dataset as ds
from uie_pipeline import evaluation as ev
from uie_pipeline import training as tr


def _hr(title: str = ""):
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}" if title else f"\n{line}")


def _dataset_is_built(root: str) -> bool:
    return all(
        os.path.isdir(os.path.join(root, s)) and os.listdir(os.path.join(root, s))
        for s in ds.SPLITS
    )


def _run_is_trained(run_dir: str) -> dict | None:
    best = os.path.join(run_dir, "weights", "best.pt")
    results = os.path.join(run_dir, "results.csv")
    if os.path.isfile(best) and os.path.isfile(results):
        return {
            "name": os.path.basename(run_dir),
            "run_dir": run_dir,
            "weights": best,
            "results_csv": results,
        }
    return None


def run(config_path: str, reuse_datasets: bool, reuse_runs: bool,
        skip_training: bool) -> None:
    cfg = cfg_mod.load_config(config_path)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # --- Load annotations -------------------------------------------------- #
    _hr("1/4  Annotations")
    annos, stats = anno.load_annotations(cfg.annotations)
    print(f"Loaded {len(annos)} usable annotations from {cfg.annotations}")
    print(f"  annotation types seen: {', '.join(stats['seen_types'])}")
    if stats["unsupported_type"]:
        print(f"  WARNING: skipped {stats['unsupported_type']} unsupported "
              f"(e.g. mask) annotations")
    if stats["no_box"]:
        print(f"  WARNING: skipped {stats['no_box']} annotations with no usable box")
    if stats["generated_ids"]:
        print(f"  NOTE: generated ids for {stats['generated_ids']} annotations "
              f"that had none")
    if not annos:
        raise SystemExit("No usable annotations found — nothing to do.")

    # --- Shared split ------------------------------------------------------ #
    assignment = ds.assign_splits(annos, cfg.split)
    counts = ds.split_counts(assignment)
    n_labels = len({a.label for a in annos})
    print(f"Shared split (seed={cfg.split.seed}) across {n_labels} classes: "
          f"train={counts['train']}  val={counts['val']}  test={counts['test']}")

    # --- Build datasets ---------------------------------------------------- #
    _hr("2/4  Build matched datasets")
    dataset_roots: dict = {}
    for spec in cfg.datasets:
        root = os.path.join(cfg.output_dir, "datasets", spec.name)
        if reuse_datasets and _dataset_is_built(root):
            print(f"[{spec.name}] reusing existing dataset at {root}")
            dataset_roots[spec.name] = root
            continue

        print(f"[{spec.name}] cropping patches from {spec.images_dir} ...")
        report = ds.build_dataset(
            spec, annos, assignment, cfg.output_dir,
            patch_format=cfg.patch.format, patch_quality=cfg.patch.quality,
        )
        dataset_roots[spec.name] = report["root"]
        print(f"    wrote {report['written']} patches  "
              f"(train={report['counts']['train']}  "
              f"val={report['counts']['val']}  test={report['counts']['test']})")
        if report["missing_images"]:
            print(f"    WARNING: {len(report['missing_images'])} source image(s) "
                  f"not found in {spec.images_dir}; "
                  f"{report['missing_annotations']} annotations skipped")
            for name in report["missing_images"][:10]:
                print(f"        - {name}")
            if len(report["missing_images"]) > 10:
                print(f"        ... and {len(report['missing_images']) - 10} more")

    # --- Train ------------------------------------------------------------- #
    _hr("3/4  Train + evaluate")
    runs: dict = {}
    evals: dict = {}
    for spec in cfg.datasets:
        root = dataset_roots[spec.name]
        run_dir = os.path.join(cfg.output_dir, "runs", spec.name)

        existing = _run_is_trained(run_dir)
        if (reuse_runs or skip_training) and existing:
            print(f"[{spec.name}] reusing trained run at {run_dir}")
            run_info = existing
        elif skip_training:
            raise SystemExit(
                f"--skip-training set but no trained run found for '{spec.name}' "
                f"at {run_dir}"
            )
        else:
            print(f"[{spec.name}] training {cfg.training.model} "
                  f"({cfg.training.epochs} epochs) ...")
            run_info = tr.train_model(root, spec.name, cfg.output_dir, cfg.training)
            if not run_info["weights"]:
                raise SystemExit(f"training produced no weights for '{spec.name}'")

        runs[spec.name] = run_info

        # Evaluate -> metrics_report.csv
        metrics_path = os.path.join(run_dir, "test", "metrics_report.csv")
        if (reuse_runs or skip_training) and os.path.isfile(metrics_path):
            print(f"[{spec.name}] reusing metrics_report.csv")
            evals[spec.name] = {"name": spec.name, "metrics_csv": metrics_path}
        else:
            print(f"[{spec.name}] evaluating on test split ...")
            eval_info = ev.evaluate_model(run_info, root, cfg.training)
            print(f"    test accuracy {eval_info['test_accuracy']:.4f} "
                  f"on {eval_info['test_samples']} samples")
            evals[spec.name] = eval_info

    # --- Compare ----------------------------------------------------------- #
    _hr("4/4  Compare models")
    comparison_dir = os.path.join(cfg.output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    import ML_metrics_comparison as mlc
    import model_comparison_workbook_w_summary as wb

    names = [s.name for s in cfg.datasets]
    baseline = cfg.baseline.name

    # Terminal side-by-side (the SOP's quick 2-model check): baseline vs each variant.
    for variant in cfg.variants:
        specs = [
            {"name": baseline,
             "results_path": runs[baseline]["results_csv"],
             "metrics_path": evals[baseline]["metrics_csv"]},
            {"name": variant.name,
             "results_path": runs[variant.name]["results_csv"],
             "metrics_path": evals[variant.name]["metrics_csv"]},
        ]
        mlc.compare_runs(specs)

    # Full Excel workbook across all models.
    models = [
        wb.ModelFiles(
            name=n,
            results_csv=runs[n]["results_csv"],
            metrics_csv=evals[n]["metrics_csv"],
        )
        for n in names
    ]
    xlsx = wb.build_workbook(models, baseline_name=baseline, out_dir=comparison_dir)

    # --- Run manifest ------------------------------------------------------ #
    manifest = {
        "config": os.path.abspath(config_path),
        "baseline": baseline,
        "split": {"train": cfg.split.train, "val": cfg.split.val,
                  "test": cfg.split.test, "seed": cfg.split.seed,
                  "counts": counts},
        "models": {
            n: {
                "dataset": dataset_roots[n],
                "run_dir": runs[n]["run_dir"],
                "results_csv": runs[n]["results_csv"],
                "metrics_csv": evals[n]["metrics_csv"],
            }
            for n in names
        },
        "workbook": xlsx,
    }
    manifest_path = os.path.join(cfg.output_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    _hr("Done")
    print(f"Workbook : {xlsx}")
    print(f"Manifest : {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Automate the UIE-vs-hand-edited classification comparison SOP."
    )
    parser.add_argument("config", help="path to the YAML run config")
    parser.add_argument("--reuse-datasets", action="store_true",
                        help="skip patch cropping where a dataset already exists")
    parser.add_argument("--reuse-runs", action="store_true",
                        help="skip training/eval where a trained run already exists")
    parser.add_argument("--skip-training", action="store_true",
                        help="require pre-existing trained runs; never train (implies "
                             "--reuse-runs)")
    args = parser.parse_args()

    run(
        config_path=args.config,
        reuse_datasets=args.reuse_datasets,
        reuse_runs=args.reuse_runs,
        skip_training=args.skip_training,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:  # noqa: BLE001 - surface a clean message to the user
        print(f"\nERROR: {e}")
        sys.exit(1)
