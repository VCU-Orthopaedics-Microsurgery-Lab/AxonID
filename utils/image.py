"""
utils/image.py: centre-crop an image to the patch grid.

Usage
    from utils.image import center_crop_to_patch_grid
    crop = center_crop_to_patch_grid(img, 256)

Inputs / Outputs
    Image or mask array -> centred crop whose height and width are
    multiples of patch_size. Raises ValueError when either side is smaller
    than patch_size. Used identically by training and segmentation.

Depends on
    numpy
"""

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np


def center_crop_to_patch_grid(
    img: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    """
    Center-crop an image to dimensions divisible by patch_size.

    Args:
        img:
            Image or mask array with spatial dimensions in the first
            two axes.
        patch_size:
            Patch width and height used by the patch-based pipeline.

    Returns:
        Center-cropped array whose height and width are divisible by
        patch_size.

    Raises:
        ValueError: if either spatial dimension of img is smaller than
            patch_size — such an image can't produce a non-empty
            patch-grid-aligned crop.
    """
    h, w = img.shape[:2]

    if h < patch_size or w < patch_size:
        raise ValueError(
            f"Image dimensions ({h}, {w}) are smaller than patch_size "
            f"({patch_size}) in at least one axis — cannot produce a "
            f"non-empty patch-grid-aligned crop."
        )

    crop_h = (h // patch_size) * patch_size
    crop_w = (w // patch_size) * patch_size

    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2

    return img[
        start_h:start_h + crop_h,
        start_w:start_w + crop_w,
    ]