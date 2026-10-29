"""
train/dataset/augment.py: training-patch augmentation (when augmentation is on).

Purpose
    Applies the AxonID augmentation recipe to training patches. Geometric
    transforms move image and mask together; intensity transforms change the
    image only.

Usage
    from train.dataset.augment import augment_dataset_np
    X, Y, n_augmented, counts = augment_dataset_np(X_train, Y_train, rng=rng)

Inputs
    images  (N, H, W, 1) float32, already L2-normalized by data_loader.py
    masks   (N, H, W, 1) class indices
    rng     numpy Generator (seeded by train.py for reproducibility)

Outputs
    Augmented copies of images and masks (same shapes), the number of
    patches changed, and per-transform counts.

Behaviour
    Each patch is augmented once, before training starts (not re-drawn
    every epoch). This is how the released augmented model was trained.
    With probability p_clean a patch is left untouched; otherwise each
    transform is applied independently with its own probability.
    Gamma and noise act on the patch's own intensity range, because patches
    are L2-normalized (values are small, not 0-1). After any change the
    patch is re-normalized exactly as data_loader.py does, so augmented and
    clean patches share one input scale.

Locked (utils/recipe.py)
    AUGMENTATION: every probability and strength.

Depends on
    opencv, scipy, utils/recipe.py
"""

from __future__ import annotations

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# ── Local ─────────────────────────────────────────────────────────────────────
from utils.recipe import AUGMENTATION, AugmentationRecipe


# ─── One patch ────────────────────────────────────────────────────────────────
def _augment_pair_core(
    image: np.ndarray,
    mask: np.ndarray,
    p: AugmentationRecipe,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list]:
    applied = []

    if rng.random() < p.p_clean:
        return image.astype(np.float32), mask.astype(np.float32), applied

    # ── Geometric: image and mask ─────────────────────────────────────────────
    if rng.random() < p.hflip_prob:
        image, mask = np.fliplr(image), np.fliplr(mask)
        applied.append("hflip")

    if rng.random() < p.vflip_prob:
        image, mask = np.flipud(image), np.flipud(mask)
        applied.append("vflip")

    if rng.random() < p.rotation_prob:
        angle = rng.uniform(-p.rotation_deg, p.rotation_deg)
        h, w  = image.shape[:2]
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)[..., np.newaxis]
        # Nearest-neighbour keeps mask values as class indices.
        mask  = cv2.warpAffine(mask.squeeze(-1), M, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_REFLECT_101)[..., np.newaxis]
        applied.append("rotation")

    if rng.random() < p.elastic_prob:
        image, mask = _elastic_deform(image, mask, p.elastic_alpha, p.elastic_sigma, rng)
        applied.append("elastic")

    # ── Intensity: image only ─────────────────────────────────────────────────
    # Reserved draw (brightness, not in the 1.0 recipe). The unused draws in
    # this function keep the random sequence identical to the code that
    # trained the released augmented model, so a given seed reproduces it.
    rng.random()

    if rng.random() < p.gamma_prob:
        gamma  = rng.uniform(p.gamma_range[0], p.gamma_range[1])
        lo, hi = float(image.min()), float(image.max())
        if hi > lo:
            image = np.power((image - lo) / (hi - lo), gamma) * (hi - lo) + lo
        applied.append("gamma")

    if rng.random() < p.noise_prob:
        # sigma is a fraction of this patch's own intensity range.
        dyn_range = float(image.max() - image.min())
        noise = rng.normal(0, p.noise_sigma * dyn_range, image.shape).astype(np.float32)
        image = np.maximum(image + noise, 0.0)
        applied.append("noise")

    if rng.random() < p.blur_prob:
        image = cv2.GaussianBlur(image.squeeze(-1), (0, 0),
                                 sigmaX=p.blur_sigma, sigmaY=p.blur_sigma)[..., np.newaxis]
        applied.append("blur")

    # Reserved draws (CLAHE, contrast stretch, random erase; not in the recipe).
    rng.random(); rng.random(); rng.random()

    return image.astype(np.float32), mask.astype(np.float32), applied


def _elastic_deform(image, mask, alpha: float, sigma: float, rng: np.random.Generator):
    """Smooth random displacement field (sigma), scaled by alpha; same field for image and mask."""
    h, w = image.shape[:2]
    dx = gaussian_filter(rng.standard_normal((h, w)), sigma) * alpha
    dy = gaussian_filter(rng.standard_normal((h, w)), sigma) * alpha
    x, y  = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
    image_d = cv2.remap(image.squeeze(-1), map_x, map_y, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101)[..., np.newaxis]
    mask_d  = cv2.remap(mask.squeeze(-1), map_x, map_y, interpolation=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_REFLECT_101)[..., np.newaxis]
    return image_d, mask_d


def augment_pair(image, mask, rng: np.random.Generator, p: AugmentationRecipe = AUGMENTATION):
    """Augment one patch, then restore the training normalization if anything changed."""
    image, mask, applied = _augment_pair_core(image, mask, p, rng)
    if applied:
        # (H, W, 1): axis 0 is the patch height, i.e. the same per-column
        # L2 normalization data_loader.py applies to (N, H, W, 1) on axis 1.
        norms = np.linalg.norm(image, axis=0, keepdims=True)
        image = image / np.where(norms == 0, 1.0, norms)
    return image.astype(np.float32), mask, applied


# ─── Whole training set ───────────────────────────────────────────────────────
def augment_dataset_np(
    images: np.ndarray,
    masks: np.ndarray,
    rng: np.random.Generator,
    p: AugmentationRecipe = AUGMENTATION,
) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """Replace each training patch with its augmented version (once)."""
    aug_images, aug_masks = images.copy(), masks.copy()
    counts = {k: 0 for k in ("hflip", "vflip", "rotation", "elastic", "gamma", "noise", "blur")}
    n_augmented = 0
    for i in range(len(images)):
        aug_images[i], aug_masks[i], applied = augment_pair(images[i], masks[i], rng, p)
        if applied:
            n_augmented += 1
            for a in applied:
                counts[a] += 1
    return aug_images, aug_masks, n_augmented, counts
