"""
dataset.py
==========

Build matched YOLO classification datasets from a single annotation set.

This collapses SOP steps 4 and 7-10 into one deterministic operation:

  * The train/val/test split is computed **once** from the annotation list,
    stratified by class and seeded, so it is identical for every dataset.
  * Each dataset (the hand-edited baseline and every UIE variant) is then
    cropped from its own image folder using the *same* annotations and the
    *same* split assignment.

The result is N datasets containing the exact same patches in the exact same
splits -- the "identical image patches in both datasets" guarantee the SOP
asks for -- without the manual export/remap/match-split round-trip.

Output layout per dataset (Ultralytics ImageFolder classify format)::

    <output_dir>/datasets/<name>/
        train/<label>/<label>_<id>.jpg
        val/<label>/<label>_<id>.jpg
        test/<label>/<label>_<id>.jpg
"""

from __future__ import annotations

import os
import random
from collections import defaultdict

from PIL import Image

from . import annotations as anno


SPLITS = ("train", "val", "test")


def assign_splits(annos: list, split, ) -> dict:
    """
    Stratified split of annotation ids by class label.

    Returns {ann_id: "train"|"val"|"test"}.  Within each class the ids are
    shuffled with the configured seed and partitioned by the train/val/test
    fractions.  Tiny classes degrade gracefully: the first sample always goes
    to train, the next prefers val, the next test.
    """
    rng = random.Random(split.seed)
    by_label = defaultdict(list)
    for a in annos:
        by_label[a.label].append(a.ann_id)

    assignment: dict = {}
    for label in sorted(by_label):
        ids = sorted(by_label[label])      # deterministic base order
        rng.shuffle(ids)
        n = len(ids)

        n_train = int(round(n * split.train))
        n_val = int(round(n * split.val))
        # Guarantee everything is assigned; test gets the remainder.
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        # Make sure train is never empty for a non-empty class.
        if n_train == 0 and n > 0:
            n_train = 1
            n_val = min(n_val, n - n_train)

        for i, ann_id in enumerate(ids):
            if i < n_train:
                assignment[ann_id] = "train"
            elif i < n_train + n_val:
                assignment[ann_id] = "val"
            else:
                assignment[ann_id] = "test"
    return assignment


def split_counts(assignment: dict) -> dict:
    counts = {s: 0 for s in SPLITS}
    for s in assignment.values():
        counts[s] += 1
    return counts


def build_dataset(
    dataset_spec,
    annos: list,
    assignment: dict,
    output_root: str,
    patch_format: str = "jpg",
    patch_quality: int = 100,
) -> dict:
    """
    Crop every annotation from ``dataset_spec.images_dir`` and write it into the
    split given by ``assignment``.

    Returns a report dict with the dataset root, per-split counts, the set of
    labels, and any source images that could not be found.
    """
    root = os.path.join(output_root, "datasets", dataset_spec.name)
    labels = sorted({a.label for a in annos})
    ext = "png" if patch_format.lower() == "png" else "jpg"

    # Pre-create every label folder in every split so all splits share an
    # identical, consistently-ordered class list (keeps class indices aligned).
    for s in SPLITS:
        for label in labels:
            os.makedirs(os.path.join(root, s, label), exist_ok=True)

    report = {
        "name": dataset_spec.name,
        "root": root,
        "labels": labels,
        "counts": {s: 0 for s in SPLITS},
        "written": 0,
        "missing_images": set(),
        "missing_annotations": 0,
    }

    # Group annotations by source image so each raster is opened once.
    by_image = defaultdict(list)
    for a in annos:
        by_image[(a.image_name, a.image_stem)].append(a)

    for (image_name, image_stem), group in by_image.items():
        src = anno.resolve_source_image(image_name, image_stem, dataset_spec.images_dir)
        if src is None:
            report["missing_images"].add(image_name)
            report["missing_annotations"] += len(group)
            continue

        with Image.open(src) as im:
            im = im.convert("RGB")
            for a in group:
                split = assignment.get(a.ann_id)
                if split is None:
                    continue
                patch = anno.crop_patch(im, a.box)
                dest_dir = os.path.join(root, split, a.label)
                dest = os.path.join(dest_dir, f"{a.patch_filename}.{ext}")
                if ext == "jpg":
                    patch.save(dest, "JPEG", quality=patch_quality)
                else:
                    patch.save(dest, "PNG")
                report["counts"][split] += 1
                report["written"] += 1

    report["missing_images"] = sorted(report["missing_images"])
    return report
