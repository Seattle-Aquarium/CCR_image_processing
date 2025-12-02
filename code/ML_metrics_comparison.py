#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
compare_image_workflows_metrics.py

Interactively compare two CoralNet-Toolbox training runs
(e.g., Hand-edited vs UIE images) using:

From results.csv:
    - Best validation top-1 accuracy
    - Best validation top-5 accuracy
    - Corresponding validation loss

From metrics_report.csv:
    - Macro F1
    - Macro balanced accuracy
    - Weighted F1
"""

import pandas as pd
import os


# ---------- Helper functions ----------

def summarize_results(results_path):
    """Return best epoch, top-1, top-5 and val loss from a results.csv file."""
    df = pd.read_csv(results_path)

    # Ensure numeric types
    for col in ["metrics/accuracy_top1", "metrics/accuracy_top5", "val/loss"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Find row(s) with max top-1 accuracy
    max_top1 = df["metrics/accuracy_top1"].max()
    best_rows = df[df["metrics/accuracy_top1"] == max_top1]

    # Break ties by smallest validation loss
    min_val_loss = best_rows["val/loss"].min()
    best_row = best_rows[best_rows["val/loss"] == min_val_loss].iloc[0]

    return {
        "epoch": int(best_row["epoch"]),
        "best_top1": float(best_row["metrics/accuracy_top1"]),
        "best_top5": float(best_row["metrics/accuracy_top5"]),
        "val_loss_at_best": float(best_row["val/loss"]),
    }


def summarize_metrics(metrics_path):
    """Return macro F1, macro balanced accuracy, and weighted F1 from metrics_report.csv."""
    df = pd.read_csv(metrics_path)

    f1 = pd.to_numeric(df["F1 Score"], errors="coerce")
    bal = pd.to_numeric(df["Balanced Accuracy"], errors="coerce")
    support = pd.to_numeric(df["Total Samples"], errors="coerce")

    macro_f1 = float(f1.mean())
    macro_bal = float(bal.mean())
    weighted_f1 = float((f1 * support).sum() / support.sum())

    return {
        "macro_f1": macro_f1,
        "macro_balanced_accuracy": macro_bal,
        "weighted_f1": weighted_f1,
    }


def prompt_dataset(label):
    """Prompt user for file paths for one dataset (results + metrics)."""
    print(f"\n=== {label} dataset ===")
    results_path = input("Path to results.csv: ").strip().strip('"')
    metrics_path = input("Path to metrics_report.csv: ").strip().strip('"')

    if not os.path.isfile(results_path):
        raise FileNotFoundError(f"results.csv not found at: {results_path}")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"metrics_report.csv not found at: {metrics_path}")

    return results_path, metrics_path


# ---------- Main ----------

def main():
    print("Compare two CoralNet-Toolbox training runs (e.g., Hand-edited vs UIE).")
    print("You’ll be prompted for the results.csv and metrics_report.csv for each.\n")

    dataset_infos = []

    for label in ["Dataset 1", "Dataset 2"]:
        name = input(f"\nFriendly name for {label} (e.g., 'Hand edited', 'UIE'): ").strip()
        if not name:
            name = label

        results_path, metrics_path = prompt_dataset(label)

        res_summary = summarize_results(results_path)
        met_summary = summarize_metrics(metrics_path)

        dataset_infos.append({
            "name": name,
            **res_summary,
            **met_summary
        })

    # Print comparison table
    print("\n================= Comparison =================")
    header = (
        f"{'Dataset':<20}"
        f"{'Best epoch':>11}  "
        f"{'Top-1':>8}  "
        f"{'Top-5':>8}  "
        f"{'Val loss':>10}  "
        f"{'Macro F1':>10}  "
        f"{'Macro BalAcc':>13}  "
        f"{'Weighted F1':>12}"
    )
    print(header)
    print("-" * len(header))

    for info in dataset_infos:
        print(
            f"{info['name']:<20}"
            f"{info['epoch']:>11d}  "
            f"{info['best_top1']:>8.4f}  "
            f"{info['best_top5']:>8.4f}  "
            f"{info['val_loss_at_best']:>10.4f}  "
            f"{info['macro_f1']:>10.4f}  "
            f"{info['macro_balanced_accuracy']:>13.4f}  "
            f"{info['weighted_f1']:>12.4f}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
