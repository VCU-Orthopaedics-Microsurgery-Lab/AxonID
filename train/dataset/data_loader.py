"""
train/dataset/data_loader.py: load training and validation patches.

Purpose
    Reads the patches written by train/dataset/preprocess.py, assigns them
    to training or validation by source image, converts masks to class
    indices and applies the training input normalization.

Usage
    from train.dataset.data_loader import load_all_patches
    X_train, Y_train, X_val, Y_val, val_patch_stems = load_all_patches(
        images_dir, masks_dir, train_stems, val_stems, log)

Inputs
    images_dir/cropped/patches/   image patches  {stem}_{RRCC}.png
    masks_dir/cropped/patches/    mask patches, same file names
    train_stems, val_stems        source image names from train/dataset/split.py

Outputs
    X arrays  (N, 256, 256, 1) float32, L2-normalized per patch column
    Y arrays  (N, 256, 256, 1) float32 class indices: 0 background, 1 myelin, 2 axon
    val_patch_stems  source image of each validation patch, in array order

Locked (utils/recipe.py)
    Training normalization (NORMALIZATION_TRAINING). It differs from
    inference on purpose; see the note in utils/recipe.py before changing it.

Depends on
    utils/mask_contract.py (snap_to_bgw), utils/filesystem.py
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np

# ── Local ─────────────────────────────────────────────────────────────────────
from utils.filesystem import list_files
from utils.mask_contract import BGW_AXON, BGW_MYELIN_CANONICAL, snap_to_bgw

PATCH_EXTENSIONS = (".png", ".tif", ".tiff")


# ─── Mask encoding ────────────────────────────────────────────────────────────
def base_label(arr: np.ndarray) -> np.ndarray:
    """BGW mask values -> class indices. Off-contract values are snapped first."""
    snapped = snap_to_bgw(arr)
    out = np.zeros(arr.shape, dtype=np.float32)
    out[snapped == BGW_MYELIN_CANONICAL] = 1
    out[snapped == BGW_AXON] = 2
    return out


# ─── Patch reading ────────────────────────────────────────────────────────────
def _load_file_list(files: list, is_mask: bool) -> np.ndarray:
    """Read patches in sorted order into a (N, H, W, 1) float32 array."""
    patches = []
    for f in sorted(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Never skip: dropping one patch shifts every later image/mask pair.
            raise ValueError(f"Could not read patch (refusing to skip, it would misalign image/mask pairs): {f}")
        patches.append(img)

    arr = np.expand_dims(np.array(patches, dtype=np.float32), axis=-1)
    if is_mask:
        return base_label(arr)

    # axis=1 of (N, H, W, 1) is the patch height: each column is divided by
    # its own norm. Inference normalizes rows instead; see utils/recipe.py.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return arr / norms


def _patch_stem(f: Path) -> str:
    """Source image name of a patch: ctrl_animal1_0304.png -> ctrl_animal1."""
    return "_".join(Path(f).stem.split("_")[:-1])


# ─── Main loader ──────────────────────────────────────────────────────────────
def load_all_patches(
    images_dir: str | Path,
    masks_dir: str | Path,
    train_stems: list[str],
    val_stems: list[str],
    log=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load patches for the given split. See the module header."""
    patches_img  = Path(images_dir) / "cropped" / "patches"
    patches_mask = Path(masks_dir)  / "cropped" / "patches"

    img_files  = list_files(str(patches_img),  extensions=PATCH_EXTENSIONS)
    mask_files = list_files(str(patches_mask), extensions=PATCH_EXTENSIONS)
    if not img_files:
        raise FileNotFoundError(f"No image patches found in {patches_img}")

    # Image and mask patches are paired by sorted file name, so the two
    # folders must hold exactly the same names.
    img_names, mask_names = {Path(f).name for f in img_files}, {Path(f).name for f in mask_files}
    if img_names != mask_names:
        only_img, only_mask = sorted(img_names - mask_names), sorted(mask_names - img_names)
        raise ValueError(
            f"Image and mask patches do not match: {len(only_img)} image patch(es) without a mask "
            f"(e.g. {only_img[:3]}), {len(only_mask)} mask patch(es) without an image "
            f"(e.g. {only_mask[:3]}). Delete images/cropped and masks/cropped to regenerate them."
        )

    # Every image in the split must have patches. A missing one means the
    # patch folders are older than the image folder.
    patch_stems = {_patch_stem(f) for f in img_files}
    missing = sorted(set(train_stems + val_stems) - patch_stems)
    if missing:
        raise ValueError(
            f"{len(missing)} image(s) have no patches (e.g. {missing[:3]}). The patch folders are out "
            f"of date: delete images/cropped and masks/cropped and run training again to regenerate them."
        )
    extra = sorted(patch_stems - set(train_stems + val_stems))
    if extra and log:
        log.warn(f"Ignoring patches from {len(extra)} image(s) no longer in the images folder: {extra[:5]}")

    train_set, val_set = set(train_stems), set(val_stems)
    train_img  = [f for f in img_files  if _patch_stem(f) in train_set]
    train_mask = [f for f in mask_files if _patch_stem(f) in train_set]
    val_img    = [f for f in img_files  if _patch_stem(f) in val_set]
    val_mask   = [f for f in mask_files if _patch_stem(f) in val_set]

    X_train = _load_file_list(train_img,  is_mask=False)
    Y_train = _load_file_list(train_mask, is_mask=True)
    X_val   = _load_file_list(val_img,    is_mask=False)
    Y_val   = _load_file_list(val_mask,   is_mask=True)
    val_patch_stems = [_patch_stem(f) for f in sorted(val_img)]   # same order as X_val

    if log:
        n_total = len(train_stems) + len(val_stems)
        log.info(f"Train: {len(train_stems)} images -> {len(X_train)} patches")
        log.info(f"Val:   {len(val_stems)} images -> {len(X_val)} patches "
                 f"({round(len(val_stems) / n_total * 100, 1)}% of images)")
        log.info(f"Val images: {', '.join(val_stems)}")

    return X_train, Y_train, X_val, Y_val, val_patch_stems
