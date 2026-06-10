"""
config.py
=========

Load and validate the YAML run config for ``compare_uie.py``.

Schema (see code/compare_uie.example.yaml for a worked example)::

    annotations: /path/to/hand_edited_annotations.json   # required
    output_dir:  /path/to/runs/2026-06-02_comparison      # required

    patch:
      format: jpg          # patch image extension (jpg|png)
      quality: 100         # JPEG quality (ignored for png)

    split:
      train: 0.7
      val:   0.2
      test:  0.1
      seed:  42

    baseline:                                   # the hand-edited reference
      name: hand_edited
      images_dir: /path/to/hand_edited_images

    variants:                                   # one or more UIE methods
      - name: UIE_v1
        images_dir: /path/to/uie_v1_images
      - name: UIE_v2
        images_dir: /path/to/uie_v2_images

    training:
      model:    yolo11s-cls.pt   # Ultralytics classify weights
      epochs:   100
      imgsz:    224
      batch:    64
      device:   ""               # "", "cpu", "mps", or a CUDA index like "0"
      patience: 100
      workers:  8
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class DatasetSpec:
    name: str
    images_dir: str
    is_baseline: bool = False


@dataclass
class SplitSpec:
    train: float = 0.7
    val: float = 0.2
    test: float = 0.1
    seed: int = 42

    def validate(self):
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"split fractions must sum to 1.0 (got {total:.4f}: "
                f"train={self.train}, val={self.val}, test={self.test})"
            )
        for name, v in (("train", self.train), ("val", self.val), ("test", self.test)):
            if v < 0:
                raise ValueError(f"split.{name} must be >= 0 (got {v})")


@dataclass
class TrainingSpec:
    model: str = "yolo11s-cls.pt"
    epochs: int = 100
    imgsz: int = 224
    batch: int = 64
    device: str = ""
    patience: int = 100
    workers: int = 8


@dataclass
class PatchSpec:
    format: str = "jpg"
    quality: int = 100

    def validate(self):
        fmt = self.format.lower().lstrip(".")
        if fmt not in ("jpg", "jpeg", "png"):
            raise ValueError(f"patch.format must be jpg or png (got '{self.format}')")
        self.format = fmt


@dataclass
class RunConfig:
    annotations: str
    output_dir: str
    baseline: DatasetSpec
    variants: list[DatasetSpec] = field(default_factory=list)
    split: SplitSpec = field(default_factory=SplitSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)
    patch: PatchSpec = field(default_factory=PatchSpec)

    @property
    def datasets(self) -> list[DatasetSpec]:
        """Baseline first, then variants -- the full set of models to build."""
        return [self.baseline, *self.variants]

    def validate(self):
        if not os.path.isfile(self.annotations):
            raise FileNotFoundError(f"annotations file not found: {self.annotations}")
        self.split.validate()
        self.patch.validate()

        names = [d.name for d in self.datasets]
        if len(names) != len(set(names)):
            raise ValueError(f"dataset names must be unique, got: {names}")
        if len(self.datasets) < 2:
            raise ValueError("need at least a baseline plus one variant to compare")

        for d in self.datasets:
            if not d.name:
                raise ValueError("every dataset needs a non-empty name")
            if not os.path.isdir(d.images_dir):
                raise FileNotFoundError(
                    f"images_dir for '{d.name}' not found: {d.images_dir}"
                )


def _require(d: dict, key: str, where: str):
    if key not in d or d[key] in (None, ""):
        raise ValueError(f"config: missing required '{key}' in {where}")
    return d[key]


def load_config(path: str) -> RunConfig:
    """Parse a YAML config file into a validated RunConfig."""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError("config file must contain a top-level mapping")

    base_raw = _require(raw, "baseline", "config root")
    baseline = DatasetSpec(
        name=_require(base_raw, "name", "baseline"),
        images_dir=_require(base_raw, "images_dir", "baseline"),
        is_baseline=True,
    )

    variants_raw = raw.get("variants") or []
    if not isinstance(variants_raw, list):
        raise ValueError("config: 'variants' must be a list")
    variants = [
        DatasetSpec(
            name=_require(v, "name", "variant"),
            images_dir=_require(v, "images_dir", "variant"),
        )
        for v in variants_raw
    ]

    split_raw = raw.get("split") or {}
    split = SplitSpec(
        train=float(split_raw.get("train", 0.7)),
        val=float(split_raw.get("val", 0.2)),
        test=float(split_raw.get("test", 0.1)),
        seed=int(split_raw.get("seed", 42)),
    )

    tr_raw = raw.get("training") or {}
    training = TrainingSpec(
        model=str(tr_raw.get("model", "yolo11s-cls.pt")),
        epochs=int(tr_raw.get("epochs", 100)),
        imgsz=int(tr_raw.get("imgsz", 224)),
        batch=int(tr_raw.get("batch", 64)),
        device=str(tr_raw.get("device", "")),
        patience=int(tr_raw.get("patience", 100)),
        workers=int(tr_raw.get("workers", 8)),
    )

    patch_raw = raw.get("patch") or {}
    patch = PatchSpec(
        format=str(patch_raw.get("format", "jpg")),
        quality=int(patch_raw.get("quality", 100)),
    )

    cfg = RunConfig(
        annotations=_require(raw, "annotations", "config root"),
        output_dir=_require(raw, "output_dir", "config root"),
        baseline=baseline,
        variants=variants,
        split=split,
        training=training,
        patch=patch,
    )
    cfg.validate()
    return cfg
