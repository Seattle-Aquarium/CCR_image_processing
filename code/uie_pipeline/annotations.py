"""
annotations.py
==============

Parse a CoralNet-Toolbox annotation JSON export and crop image patches from a
source raster folder.

The Toolbox annotation JSON (File -> Export -> Annotations -> JSON) is a dict
mapping an image-path string to a list of annotation dicts::

    {
      "C:/.../2024_10_08_11-40-27.jpg": [
        {
          "id": "a1b2c3d4-...",
          "type": "PatchAnnotation",
          "label_short_code": "SU_sand",
          "label_long_code": "Substrate: sand",
          "image_path": "C:/.../2024_10_08_11-40-27.jpg",
          "center_xy": [1234.0, 567.0],
          "annotation_size": 224
        },
        ...
      ],
      ...
    }

Supported annotation ``type`` values and the bounding box each yields:

    PatchAnnotation         square of side ``annotation_size`` centred on ``center_xy``
    RectangleAnnotation     axis-aligned box ``top_left`` -> ``bottom_right``
    PolygonAnnotation       axis-aligned bounding box of ``points``
    MultiPolygonAnnotation  bounding box covering every sub-polygon

This mirrors the crop behaviour of the Toolbox "Export -> Dataset -> Classify"
step so that datasets produced here line up with the SOP.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from PIL import Image


@dataclass
class Annotation:
    """One annotation, reduced to what the classify export needs."""

    ann_id: str          # stable unique id (drives patch filenames + split assignment)
    label: str           # class label (folder name in the exported dataset)
    image_name: str      # basename of the source image (e.g. "2024_10_08_11-40-27.jpg")
    image_stem: str      # source image name without extension
    box: tuple           # (left, top, right, bottom) in source-pixel coordinates (floats)
    ann_type: str        # original Toolbox annotation type, for diagnostics

    @property
    def patch_filename(self) -> str:
        """Toolbox-style patch filename: <label>_<id>.jpg (extension added by caller)."""
        return f"{self.label}_{self.ann_id}"


# --------------------------------------------------------------------------- #
# Coordinate helpers
# --------------------------------------------------------------------------- #
def _xy(value):
    """Coerce a [x, y] pair (list/tuple) into (float, float)."""
    return float(value[0]), float(value[1])


def _box_from_annotation(ann: dict) -> tuple | None:
    """Return (left, top, right, bottom) for a single annotation dict, or None."""
    ann_type = ann.get("type", "")

    if ann_type == "PatchAnnotation":
        cx, cy = _xy(ann["center_xy"])
        size = float(ann.get("annotation_size", 0))
        half = size / 2.0
        return (cx - half, cy - half, cx + half, cy + half)

    if ann_type == "RectangleAnnotation":
        x1, y1 = _xy(ann["top_left"])
        x2, y2 = _xy(ann["bottom_right"])
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    if ann_type in ("PolygonAnnotation",):
        pts = [_xy(p) for p in ann.get("points", [])]
        if not pts:
            return None
        xs, ys = zip(*pts)
        return (min(xs), min(ys), max(xs), max(ys))

    if ann_type in ("MultiPolygonAnnotation",):
        xs, ys = [], []
        for poly in ann.get("polygons", []):
            for p in poly.get("points", []):
                x, y = _xy(p)
                xs.append(x)
                ys.append(y)
        if not xs:
            return None
        return (min(xs), min(ys), max(xs), max(ys))

    # Unknown / unsupported type (e.g. MaskAnnotation) -- caller reports it.
    return None


def _label_of(ann: dict) -> str:
    """Pick the class label. Toolbox uses label_short_code for folder names."""
    for key in ("label_short_code", "label_long_code", "label", "category"):
        val = ann.get(key)
        if val:
            return str(val)
    return "unlabeled"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_annotations(json_path: str) -> tuple[list[Annotation], dict]:
    """
    Parse a Toolbox annotation JSON file.

    Returns (annotations, stats) where stats reports counts of skipped /
    unsupported / id-less entries so the caller can warn the user.
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            "Annotation JSON must be a dict mapping image paths to annotation lists."
        )

    annotations: list[Annotation] = []
    stats = {"total": 0, "unsupported_type": 0, "no_box": 0, "generated_ids": 0}
    seen_types: set = set()

    for image_key, ann_list in data.items():
        if not isinstance(ann_list, list):
            continue
        for idx, ann in enumerate(ann_list):
            if not isinstance(ann, dict):
                continue
            stats["total"] += 1
            seen_types.add(ann.get("type", "<none>"))

            box = _box_from_annotation(ann)
            if box is None:
                if ann.get("type", "") in ("MaskAnnotation",) or "type" not in ann:
                    stats["unsupported_type"] += 1
                else:
                    stats["no_box"] += 1
                continue

            # Source image: prefer the per-annotation image_path, fall back to the key.
            src = ann.get("image_path") or image_key
            image_name = os.path.basename(str(src).replace("\\", "/"))
            image_stem = os.path.splitext(image_name)[0]

            ann_id = ann.get("id")
            if not ann_id:
                # Build a deterministic id so reruns and variants stay aligned.
                ann_id = f"{image_stem}_{idx}"
                stats["generated_ids"] += 1

            annotations.append(
                Annotation(
                    ann_id=str(ann_id),
                    label=_label_of(ann),
                    image_name=image_name,
                    image_stem=image_stem,
                    box=tuple(float(v) for v in box),
                    ann_type=ann.get("type", "<none>"),
                )
            )

    stats["seen_types"] = sorted(seen_types)
    return annotations, stats


# --------------------------------------------------------------------------- #
# Cropping
# --------------------------------------------------------------------------- #
def resolve_source_image(image_name: str, image_stem: str, images_dir: str) -> str | None:
    """
    Find the source raster for an annotation inside ``images_dir``.

    Matches by exact basename first (the SOP requires identical filenames),
    then falls back to the filename stem with any extension -- this tolerates
    enhanced variants saved as e.g. ``<stem>_enhanced.jpg`` is *not* matched
    (different stem), but ``<stem>.tif`` vs ``<stem>.jpg`` is.
    """
    exact = os.path.join(images_dir, image_name)
    if os.path.isfile(exact):
        return exact

    # Stem fallback: same stem, any extension.
    try:
        for entry in os.listdir(images_dir):
            if os.path.splitext(entry)[0] == image_stem and \
                    os.path.isfile(os.path.join(images_dir, entry)):
                return os.path.join(images_dir, entry)
    except FileNotFoundError:
        return None
    return None


def crop_patch(image: Image.Image, box: tuple) -> Image.Image:
    """Crop ``box`` (left, top, right, bottom) from ``image``, clamped to bounds."""
    w, h = image.size
    left, top, right, bottom = box
    left = max(0, min(int(round(left)), w - 1))
    top = max(0, min(int(round(top)), h - 1))
    right = max(left + 1, min(int(round(right)), w))
    bottom = max(top + 1, min(int(round(bottom)), h))
    return image.crop((left, top, right, bottom))
