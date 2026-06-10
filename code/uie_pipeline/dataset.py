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

import numpy as np
from PIL import Image

from . import annotations as anno


SPLITS = ("train", "val", "test")

# Candidate geometric relationships between a variant image and the reference
# (baseline) image whose pixel space the annotations live in.  See
# annotations.map_box for the meaning of each mode.
ALIGN_MODES = ("center", "scale")


def _patch_gray(image: Image.Image, box: tuple, size: int = 48) -> np.ndarray:
    """Small grayscale crop for alignment scoring (structure, not colour)."""
    g = anno.crop_patch(image, box).convert("L").resize((size, size))
    return np.asarray(g, dtype=np.float32)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom else 0.0


def detect_alignment(by_image: dict, reference_images_dir: str,
                     variant_images_dir: str, samples: int = 12) -> dict:
    """Pick how a variant's pixels relate to the reference's, empirically.

    For a handful of images whose variant resolution differs from the
    reference, crop the same annotation under each candidate mode and keep the
    one whose patches best correlate (in grayscale structure) with the
    reference patch.  Returns {mode, score, n_samples} -- ``mode`` is None when
    no differently-sized images exist (no remapping needed).
    """
    scores = {m: [] for m in ALIGN_MODES}
    used = 0
    for (image_name, image_stem), group in by_image.items():
        ref = anno.resolve_source_image(image_name, image_stem, reference_images_dir)
        var = anno.resolve_source_image(image_name, image_stem, variant_images_dir)
        if ref is None or var is None:
            continue
        with Image.open(ref) as rim, Image.open(var) as vim:
            rim = rim.convert("RGB")
            vim = vim.convert("RGB")
            if rim.size == vim.size:
                continue  # same size -> no remapping to disambiguate
            for a in group[:2]:
                ref_patch = _patch_gray(rim, a.box)
                for mode in ALIGN_MODES:
                    vbox = anno.map_box(a.box, rim.size, vim.size, mode)
                    scores[mode].append(_corr(ref_patch, _patch_gray(vim, vbox)))
        used += 1
        if used >= samples:
            break

    if used == 0:
        return {"mode": None, "score": None, "n_samples": 0}
    means = {m: (sum(v) / len(v) if v else -1.0) for m, v in scores.items()}
    best = max(means, key=means.get)
    return {"mode": best, "score": means[best], "n_samples": used,
            "all_scores": means}


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
    reference_images_dir: str | None = None,
) -> dict:
    """
    Crop every annotation from ``dataset_spec.images_dir`` and write it into the
    split given by ``assignment``.

    Annotation coordinates live in the pixel space of the reference (baseline)
    images.  If ``reference_images_dir`` is given and a source image has a
    different resolution than its reference counterpart (e.g. an enhanced variant
    exported at native sensor size while the baseline was downscaled), the crop
    box is scaled by the per-image ratio so every dataset crops the *same* field
    of view -- preserving the SOP's "identical patches" guarantee even when SOP
    step 5 (matching dimensions) was not satisfied upstream.

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
        "remapped_images": 0,
        "alignment": None,
    }

    # Group annotations by source image so each raster is opened once.
    by_image = defaultdict(list)
    for a in annos:
        by_image[(a.image_name, a.image_stem)].append(a)

    # Decide how this dataset's images relate to the reference coordinate space
    # (only when a separate reference is given, e.g. for enhanced variants).
    align_mode = None
    if reference_images_dir and os.path.abspath(reference_images_dir) != os.path.abspath(dataset_spec.images_dir):
        align = detect_alignment(by_image, reference_images_dir, dataset_spec.images_dir)
        align_mode = align["mode"]
        report["alignment"] = align

    for (image_name, image_stem), group in by_image.items():
        src = anno.resolve_source_image(image_name, image_stem, dataset_spec.images_dir)
        if src is None:
            report["missing_images"].add(image_name)
            report["missing_annotations"] += len(group)
            continue

        with Image.open(src) as im:
            im = im.convert("RGB")
            var_size = im.size

            # Remap annotation boxes into this image's pixel space when it
            # differs in resolution from the reference (annotation) image.
            ref_size = None
            if align_mode:
                ref = anno.resolve_source_image(
                    image_name, image_stem, reference_images_dir
                )
                if ref is not None:
                    with Image.open(ref) as rim:
                        ref_size = rim.size
                    if ref_size != var_size:
                        report["remapped_images"] += 1

            for a in group:
                split = assignment.get(a.ann_id)
                if split is None:
                    continue
                box = a.box
                if ref_size and ref_size != var_size:
                    box = anno.map_box(a.box, ref_size, var_size, align_mode)
                patch = anno.crop_patch(im, box)
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
