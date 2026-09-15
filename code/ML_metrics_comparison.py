#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML_metrics_comparison.py

Interactively compare two CoralNet-Toolbox training runs
(e.g., Hand-edited vs UIE images) using:

From results.csv:
    - Best validation top-1 accuracy
    - Best validation top-5 accuracy
    - Corresponding validation loss

From metrics_report.csv (per-class, excluding zero-sample classes):
    - Macro F1        — equal weight per class; robust for imbalanced datasets
    - Macro Balanced Accuracy
    - Per-class comparison: F1 for each model, delta, and winner per class
"""

import pandas as pd
import os


# ---------- Helper functions ----------

def summarize_results(results_path):
    """Return best epoch, top-1, top-5 and val loss from a results.csv file."""
    df = pd.read_csv(results_path)

    for col in ["metrics/accuracy_top1", "metrics/accuracy_top5", "val/loss"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    max_top1 = df["metrics/accuracy_top1"].max()
    best_rows = df[df["metrics/accuracy_top1"] == max_top1]

    min_val_loss = best_rows["val/loss"].min()
    best_row = best_rows[best_rows["val/loss"] == min_val_loss].iloc[0]

    return {
        "epoch": int(best_row["epoch"]),
        "best_top1": float(best_row["metrics/accuracy_top1"]),
        "best_top5": float(best_row["metrics/accuracy_top5"]),
        "val_loss_at_best": float(best_row["val/loss"]),
    }


def summarize_metrics(metrics_path):
    """
    Return macro F1, macro balanced accuracy, and per-class table from
    metrics_report.csv.  Zero-sample classes (e.g. 'background') are
    excluded from aggregate averages so they don't depress macro scores.
    """
    df = pd.read_csv(metrics_path)

    label_col = next(
        (c for c in df.columns if c.strip().lower() in ("class", "label", "name")),
        None,
    )

    for col in ["Precision", "Recall", "F1 Score", "Balanced Accuracy", "Total Samples"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    support = df["Total Samples"] if "Total Samples" in df.columns else pd.Series([1] * len(df))
    valid = df[support > 0].copy()

    macro_f1 = float(valid["F1 Score"].mean())
    macro_bal = (
        float(valid["Balanced Accuracy"].mean())
        if "Balanced Accuracy" in valid.columns
        else None
    )

    keep = [c for c in [label_col, "Total Samples", "Precision", "Recall", "F1 Score"]
            if c and c in df.columns]
    per_class = df[keep].copy()

    return {
        "macro_f1": macro_f1,
        "macro_balanced_accuracy": macro_bal,
        "per_class": per_class,
        "label_col": label_col,
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


def overall_winner(info_a, info_b):
    """
    Determine the overall better model.
    Primary criterion: macro F1.  Tiebreaker: top-1 accuracy.
    Returns (winner_name, reason_string).
    """
    f1_a, f1_b = info_a["macro_f1"], info_b["macro_f1"]
    name_a, name_b = info_a["name"], info_b["name"]

    if abs(f1_a - f1_b) < 0.001:
        # F1 is essentially tied — fall back to top-1
        t1_a, t1_b = info_a["best_top1"], info_b["best_top1"]
        if abs(t1_a - t1_b) < 0.001:
            return "tie", "macro F1 and top-1 accuracy are equivalent"
        winner = name_a if t1_a > t1_b else name_b
        return winner, "macro F1 tied; decided by top-1 accuracy"

    winner = name_a if f1_a > f1_b else name_b
    delta = abs(f1_a - f1_b)
    return winner, f"macro F1 higher by {delta:.4f}"


def print_comparison_table(info_a, info_b):
    """
    Print a single side-by-side per-class F1 table with a Winner column.
    Rows are sorted by absolute F1 difference (biggest gaps first).
    Returns (wins_a, wins_b, ties).
    """
    name_a, name_b = info_a["name"], info_b["name"]
    lc_a = info_a["label_col"]
    lc_b = info_b["label_col"]

    pc_a = info_a["per_class"].rename(columns={
        lc_a: "Class", "F1 Score": "F1_a", "Precision": "P_a", "Recall": "R_a"
    })
    pc_b = info_b["per_class"].rename(columns={
        lc_b: "Class", "F1 Score": "F1_b", "Precision": "P_b", "Recall": "R_b"
    })

    merged = pd.merge(
        pc_a[["Class", "Total Samples", "P_a", "R_a", "F1_a"]],
        pc_b[["Class", "P_b", "R_b", "F1_b"]],
        on="Class", how="outer",
    )

    merged["delta_f1"] = merged["F1_b"] - merged["F1_a"]
    merged["abs_delta"] = merged["delta_f1"].abs()

    TIE_TOL = 0.005

    def _winner(row):
        if abs(row["delta_f1"]) < TIE_TOL:
            return "tie"
        return name_b if row["delta_f1"] > 0 else name_a

    merged["Winner"] = merged.apply(_winner, axis=1)
    merged = merged.sort_values("abs_delta", ascending=False)

    wins_a = int((merged["Winner"] == name_a).sum())
    wins_b = int((merged["Winner"] == name_b).sum())
    ties   = int((merged["Winner"] == "tie").sum())

    print(f"\n--- Per-class comparison  (A = {name_a},  B = {name_b}) ---")
    print(f"    Sorted by absolute F1 difference (largest gaps first).")
    print(f"    ΔF1% = B minus A  (positive = B better, negative = A better)\n")

    header = (
        f"{'Class':<16}"
        f"{'Smpl':>5}  "
        f"{f'P-A':>6}  "
        f"{f'R-A':>6}  "
        f"{f'F1-A':>6}  "
        f"{f'P-B':>6}  "
        f"{f'R-B':>6}  "
        f"{f'F1-B':>6}  "
        f"{'ΔF1%':>6}  "
        f"{'Winner'}"
    )
    print(header)
    print("-" * len(header))

    for _, row in merged.iterrows():
        samples = int(row["Total Samples"]) if pd.notna(row.get("Total Samples")) else 0
        p_a  = row.get("P_a",  float("nan"))
        r_a  = row.get("R_a",  float("nan"))
        f1_a = row.get("F1_a", float("nan"))
        p_b  = row.get("P_b",  float("nan"))
        r_b  = row.get("R_b",  float("nan"))
        f1_b = row.get("F1_b", float("nan"))
        dpct = row["delta_f1"] * 100
        flag = "*" if samples < 10 else " "

        print(
            f"{str(row['Class']):<15}{flag}"
            f"{samples:>5d}  "
            f"{p_a:>6.3f}  "
            f"{r_a:>6.3f}  "
            f"{f1_a:>6.3f}  "
            f"{p_b:>6.3f}  "
            f"{r_b:>6.3f}  "
            f"{f1_b:>6.3f}  "
            f"{dpct:>+6.1f}  "
            f"{row['Winner']}"
        )

    print(f"\n  Class wins  —  {name_a}: {wins_a}   {name_b}: {wins_b}   ties: {ties}")
    print("  * fewer than 10 test samples — treat metric with caution")

    return wins_a, wins_b, ties


# ---------- Main ----------

def compare_runs(specs):
    """Print the terminal comparison for exactly two runs.

    Importable, non-interactive entry point shared by the CLI (`main`) and the
    automated pipeline (`compare_uie.py`).

    specs: list of two dicts, each {"name", "results_path", "metrics_path"}.
    """
    if len(specs) != 2:
        raise ValueError("compare_runs expects exactly two runs")

    dataset_infos = []
    for spec in specs:
        res_summary = summarize_results(spec["results_path"])
        met_summary = summarize_metrics(spec["metrics_path"])
        dataset_infos.append({
            "name": spec["name"],
            **res_summary,
            "macro_f1": met_summary["macro_f1"],
            "macro_balanced_accuracy": met_summary["macro_balanced_accuracy"],
            "per_class": met_summary["per_class"],
            "label_col": met_summary["label_col"],
        })

    info_a, info_b = dataset_infos[0], dataset_infos[1]

    # ---- Aggregate comparison table ----
    print("\n================= Comparison Summary =================")
    print("  (Macro metrics exclude zero-sample classes; weighted F1 omitted")
    print("   as it is biased toward majority classes in imbalanced datasets)\n")

    header = (
        f"{'Dataset':<20}"
        f"{'Best epoch':>11}  "
        f"{'Top-1':>8}  "
        f"{'Top-5':>8}  "
        f"{'Val loss':>10}  "
        f"{'Macro F1':>10}  "
        f"{'Macro BalAcc':>13}"
    )
    print(header)
    print("-" * len(header))

    for info in dataset_infos:
        bal_str = (
            f"{info['macro_balanced_accuracy']:>13.4f}"
            if info["macro_balanced_accuracy"] is not None
            else f"{'N/A':>13}"
        )
        print(
            f"{info['name']:<20}"
            f"{info['epoch']:>11d}  "
            f"{info['best_top1']:>8.4f}  "
            f"{info['best_top5']:>8.4f}  "
            f"{info['val_loss_at_best']:>10.4f}  "
            f"{info['macro_f1']:>10.4f}  "
            f"{bal_str}"
        )

    # ---- Overall verdict ----
    winner_name, reason = overall_winner(info_a, info_b)
    print(f"\n  >> OVERALL BETTER MODEL: {winner_name}  ({reason})")

    # ---- Per-class comparison ----
    wins_a, wins_b, _ = print_comparison_table(info_a, info_b)

    # Reinforce verdict with class-win tally
    if wins_a > wins_b:
        class_winner = info_a["name"]
    elif wins_b > wins_a:
        class_winner = info_b["name"]
    else:
        class_winner = "neither (tied)"
    print(f"  >> BETTER MODEL BY CLASS COUNT: {class_winner}")

    print("\nDone.")


def main():
    print("Compare two CoralNet-Toolbox training runs (e.g., Hand-edited vs UIE).")
    print("You'll be prompted for the results.csv and metrics_report.csv for each.\n")

    specs = []
    for label in ["Dataset 1", "Dataset 2"]:
        name = input(f"\nFriendly name for {label} (e.g., 'Hand edited', 'UIE'): ").strip()
        if not name:
            name = label
        results_path, metrics_path = prompt_dataset(label)
        specs.append({"name": name, "results_path": results_path, "metrics_path": metrics_path})

    compare_runs(specs)


if __name__ == "__main__":
    main()
