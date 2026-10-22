"""
train/dataset/split.py: image-level train/validation split.

Purpose
    Assigns whole source images (never individual patches) to training or
    validation, so no image contributes patches to both sets.

Usage
    from train.dataset.split import build_image_manifest, split_images, split_summary
    manifest = build_image_manifest(images_dir)
    train, val, mode = split_images(manifest, val_fraction, seed)

Inputs
    images_dir     folder of source images (.tif, .tiff, .png)
    val_fraction   fraction of images held out for validation (recipe: 0.33)
    seed           random seed (recipe default: 1)

Outputs
    train / val manifests: lists of {"stem", "path", "phenotype"}
    mode: "stratified" or "random_image"

Split rules
    Stratified (normal path): every image name starts with ctrl_ or regen_.
        Each phenotype is split separately, so validation contains both
        phenotypes in the same proportion as the dataset.
    Random image-level (fallback): one or more image names have neither
        prefix. Images are shuffled and split as a single group, with a
        warning, because validation may then under-represent a phenotype.

    Validation size per group: round(n * val_fraction), at least 1.

Locked (utils/recipe.py)
    val_fraction, default seed

Depends on
    numpy
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import math
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np

# ── Project ───────────────────────────────────────────────────────────────────
from utils.filesystem import is_stray_file

# Filename prefixes that identify phenotype. Stratification uses them; images
# without either prefix can still be trained on (random image-level split).
CTRL_PREFIX  = "ctrl_"
REGEN_PREFIX = "regen_"
IMAGE_EXTENSIONS = (".tif", ".tiff", ".png")


def get_phenotype(stem: str) -> str | None:
    """'ctrl', 'regen', or None when the name has neither prefix."""
    if stem.startswith(CTRL_PREFIX):
        return "ctrl"
    if stem.startswith(REGEN_PREFIX):
        return "regen"
    return None


def build_image_manifest(images_dir: str | Path) -> list[dict]:
    """One entry per source image, in sorted path order."""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    # Sorted order matters: the split below indexes into this list, so a
    # different order would give a different split for the same seed.
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and not is_stray_file(p))
    if not files:
        raise FileNotFoundError(f"No .tif/.tiff/.png images found in {images_dir}")

    return [{"stem": p.stem, "path": p, "phenotype": get_phenotype(p.stem)} for p in files]


def _n_val(n: int, val_fraction: float) -> int:
    # Round half up (floor(x + 0.5)), not Python's round(), which rounds
    # half to even and would change the split size for some n.
    return max(1, math.floor(n * val_fraction + 0.5))


def split_images(
    manifest: list[dict],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict], list[dict], str]:
    """
    Split images into (train, val, mode). See the module header for the rules.

    Raises ValueError if any group would end up with no training images.
    """
    if len(manifest) < 2:
        raise ValueError(f"At least 2 images are needed for a train/validation split; found {len(manifest)}.")

    rng = np.random.default_rng(seed)

    unlabeled = [m["stem"] for m in manifest if m["phenotype"] is None]
    if unlabeled:
        n_val = _n_val(len(manifest), val_fraction)
        if n_val >= len(manifest):
            raise ValueError(
                f"val_fraction={val_fraction} leaves no training images out of {len(manifest)}."
            )
        order = rng.permutation(len(manifest)).tolist()
        val   = [manifest[i] for i in order[:n_val]]
        train = [manifest[i] for i in order[n_val:]]
        return train, val, "random_image"

    ctrl  = [m for m in manifest if m["phenotype"] == "ctrl"]
    regen = [m for m in manifest if m["phenotype"] == "regen"]

    # The two rng.choice calls select every image of each phenotype (the
    # sorted result is the original order). They are kept because they
    # advance the generator exactly as the training code used for the
    # released models did, so seed 1 on the same 30 images reproduces the
    # released models' validation set.
    ctrl  = [ctrl[i]  for i in sorted(rng.choice(len(ctrl),  size=len(ctrl),  replace=False).tolist())] if ctrl else []
    regen = [regen[i] for i in sorted(rng.choice(len(regen), size=len(regen), replace=False).tolist())] if regen else []

    train, val = [], []
    for group_name, group in (("ctrl", ctrl), ("regen", regen)):
        if not group:
            continue
        n_val = _n_val(len(group), val_fraction)
        if n_val >= len(group):
            raise ValueError(
                f"Only {len(group)} {group_name}_ image(s): at least {n_val + 1} are needed "
                f"so that {group_name} has both training and validation images."
            )
        order = rng.permutation(len(group)).tolist()
        val   += [group[i] for i in order[:n_val]]
        train += [group[i] for i in order[n_val:]]

    return train, val, "stratified"


def split_summary(train: list[dict], val: list[dict]) -> dict:
    """Counts and stem lists for logging and checkpoint metadata."""
    def count(items, phenotype):
        return sum(1 for m in items if m["phenotype"] == phenotype)

    return {
        "train_images": len(train),
        "train_ctrl":   count(train, "ctrl"),
        "train_regen":  count(train, "regen"),
        "val_images":   len(val),
        "val_ctrl":     count(val, "ctrl"),
        "val_regen":    count(val, "regen"),
        "train_stems":  [m["stem"] for m in train],
        "val_stems":    [m["stem"] for m in val],
        "unlabeled":    [m["stem"] for m in train + val if m["phenotype"] is None],
    }
