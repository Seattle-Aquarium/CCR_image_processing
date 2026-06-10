"""
evaluation.py
=============

Score a trained classification model on the test split and write a
``metrics_report.csv`` whose columns match the CoralNet-Toolbox output, so the
existing comparison scripts consume it unchanged (SOP step 12).

Columns written (per class, one row each):

    Class, Total Samples, True Positives, False Positives, False Negatives,
    True Negatives, Precision, Recall, F1 Score, Specificity,
    Balanced Accuracy, Class Distribution (%)

Macro/aggregate values are intentionally *not* written as rows -- the
comparison scripts compute macro aggregates themselves (and exclude
zero-sample classes when doing so).
"""

from __future__ import annotations

import csv
import json
import os

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

METRICS_COLUMNS = [
    "Class", "Total Samples",
    "True Positives", "False Positives", "False Negatives", "True Negatives",
    "Precision", "Recall", "F1 Score", "Specificity",
    "Balanced Accuracy", "Class Distribution (%)",
]


def _list_test_images(test_dir: str) -> tuple[list, list]:
    """Return (paths, true_labels) for every image under test/<label>/."""
    paths, labels = [], []
    if not os.path.isdir(test_dir):
        return paths, labels
    for label in sorted(os.listdir(test_dir)):
        label_dir = os.path.join(test_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in sorted(os.listdir(label_dir)):
            if fname.lower().endswith(IMAGE_EXTS):
                paths.append(os.path.join(label_dir, fname))
                labels.append(label)
    return paths, labels


def _predict_labels(weights: str, image_paths: list, idx_to_name: dict,
                    device: str, imgsz: int, batch: int = 64) -> list:
    """Top-1 predicted label for each image, using the trained model."""
    from ultralytics import YOLO

    model = YOLO(weights)
    preds: list = []
    for start in range(0, len(image_paths), batch):
        chunk = image_paths[start:start + batch]
        results = model.predict(
            chunk, imgsz=imgsz, device=(device or None), verbose=False
        )
        for r in results:
            top1 = int(r.probs.top1)
            preds.append(idx_to_name.get(top1, str(top1)))
    return preds


def _confusion_counts(y_true: list, y_pred: list, classes: list) -> dict:
    """Per-class TP/FP/FN/TN from paired true/predicted label lists."""
    counts = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "support": 0} for c in classes}
    total = len(y_true)
    support = {c: 0 for c in classes}
    for t in y_true:
        if t in support:
            support[t] += 1

    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        tn = total - tp - fp - fn
        counts[c] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "support": support[c]}
    return counts


def _metrics_rows(counts: dict, classes: list, total: int) -> list:
    rows = []
    for c in classes:
        cc = counts[c]
        tp, fp, fn, tn = cc["tp"], cc["fp"], cc["fn"], cc["tn"]
        support = cc["support"]

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0          # sensitivity
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        balanced_acc = (recall + specificity) / 2.0
        distribution = (100.0 * support / total) if total else 0.0

        rows.append({
            "Class": c,
            "Total Samples": support,
            "True Positives": tp,
            "False Positives": fp,
            "False Negatives": fn,
            "True Negatives": tn,
            "Precision": round(precision, 6),
            "Recall": round(recall, 6),
            "F1 Score": round(f1, 6),
            "Specificity": round(specificity, 6),
            "Balanced Accuracy": round(balanced_acc, 6),
            "Class Distribution (%)": round(distribution, 4),
        })
    return rows


def write_metrics_report(rows: list, out_dir: str) -> str:
    """Write metrics_report.csv into out_dir; return its path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "metrics_report.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRICS_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def evaluate_model(run: dict, dataset_root: str, training) -> dict:
    """
    Run the trained model over the test split and write metrics_report.csv.

    ``run`` is the dict returned by training.train_model (needs 'weights' and
    'run_dir').  Returns a dict with the metrics_report.csv path and overall
    test accuracy.
    """
    weights = run.get("weights")
    if not weights:
        raise FileNotFoundError(f"no trained weights for run '{run.get('name')}'")

    test_dir = os.path.join(dataset_root, "test")
    image_paths, y_true = _list_test_images(test_dir)
    if not image_paths:
        raise ValueError(f"no test images found under {test_dir}")

    # Class index -> name from the trained model so predictions map back to labels.
    from ultralytics import YOLO
    idx_to_name = YOLO(weights).names

    y_pred = _predict_labels(
        weights, image_paths, idx_to_name,
        device=training.device, imgsz=training.imgsz, batch=training.batch,
    )

    classes = sorted(set(y_true) | set(y_pred))
    total = len(y_true)
    counts = _confusion_counts(y_true, y_pred, classes)
    rows = _metrics_rows(counts, classes, total)

    test_out = os.path.join(run["run_dir"], "test")
    metrics_csv = write_metrics_report(rows, test_out)

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total if total else 0.0

    # Also drop a small json summary alongside, for convenience.
    with open(os.path.join(test_out, "metrics_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {"test_samples": total, "test_accuracy": round(accuracy, 6),
             "n_classes": len(classes)},
            fh, indent=2,
        )

    return {
        "name": run["name"],
        "metrics_csv": metrics_csv,
        "test_accuracy": accuracy,
        "test_samples": total,
    }
