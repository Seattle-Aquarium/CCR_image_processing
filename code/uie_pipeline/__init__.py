"""
uie_pipeline
============

Automation of the "Comparing UIE Output to Hand-Edited Images" SOP
(see documents/comparison_UIE.md).

The package turns the manual CoralNet-Toolbox workflow into a single,
config-driven Python run:

    annotations JSON + image folders   -->   model_comparison.xlsx

Modules
-------
config       Load & validate the YAML run config.
annotations  Parse the Toolbox annotation JSON and crop patches.
dataset      Build matched classification datasets with a shared split.
training     Train YOLO*-cls models via the Ultralytics Python API (SOP step 11).
evaluation   Score models on the test split and write metrics_report.csv (step 12).
"""

__all__ = ["config", "annotations", "dataset", "training", "evaluation"]
