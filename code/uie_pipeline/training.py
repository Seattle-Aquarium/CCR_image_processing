"""
training.py
===========

Train a YOLO classification model on a prepared dataset (SOP step 11) using the
Ultralytics Python API -- the programmatic equivalent of
"Ultralytics -> Train Model -> Classify" in the Toolbox.

Ultralytics writes ``results.csv`` (epoch, metrics/accuracy_top1,
metrics/accuracy_top5, train/loss, val/loss, ...) into the run directory, which
is exactly what the comparison scripts consume.

``ultralytics``/``torch`` are imported lazily so the rest of the pipeline
(dataset export, etc.) can run and be tested without them installed.
"""

from __future__ import annotations

import os


def train_model(dataset_root: str, run_name: str, output_root: str, training) -> dict:
    """
    Train one classification model.

    Parameters
    ----------
    dataset_root : path containing train/ val/ test/ class folders
    run_name     : friendly model name (used as the Ultralytics run name)
    output_root  : pipeline output dir; runs land under <output_root>/runs/
    training     : a config.TrainingSpec

    Returns a dict with the run directory, the best-weights path, and the
    results.csv path.
    """
    from ultralytics import YOLO

    project = os.path.join(output_root, "runs")
    os.makedirs(project, exist_ok=True)

    model = YOLO(training.model)
    model.train(
        data=dataset_root,
        epochs=training.epochs,
        imgsz=training.imgsz,
        batch=training.batch,
        device=(training.device or None),
        patience=training.patience,
        workers=training.workers,
        project=project,
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    run_dir = os.path.join(project, run_name)
    best = os.path.join(run_dir, "weights", "best.pt")
    results_csv = os.path.join(run_dir, "results.csv")

    return {
        "name": run_name,
        "run_dir": run_dir,
        "weights": best if os.path.isfile(best) else None,
        "results_csv": results_csv if os.path.isfile(results_csv) else None,
    }
