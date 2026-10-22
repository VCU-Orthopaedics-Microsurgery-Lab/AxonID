"""
utils/resize.py: downsample large images to the working resolution.

Usage
    from utils.resize import resize_img, resize_img_with_scale, get_image_resolution
    img, scale, (orig_w, orig_h) = resize_img_with_scale(path)

Inputs
    Grayscale image or BGW mask file.

Outputs
    The image, unchanged if it already fits within the working resolution
    (1440 x 1024), otherwise scaled down by one factor on both axes so it
    fits with its aspect ratio intact. resize_img_with_scale() also returns
    that factor (1.0 when unchanged) and the original size, which
    segmentation records so morphometrics can correct the pixel size
    (utils/calibration.py).

Rules
    Never upscales (that would invent detail) and never changes the aspect
    ratio. Images: LANCZOS4. Masks: nearest-neighbour, so class values
    survive. Training and segmentation both resize through this module.

Locked (utils/recipe.py)
    target_resolution

Depends on
    opencv, numpy, pillow
"""

# ── Third-party ──────────────────────────────────────────────────────────────
import cv2
import numpy as np
from PIL import Image

# ── Local ─────────────────────────────────────────────────────────────────────
from utils.config import get_target_resolution

# (width, height) for cv2 — sourced from config.toml, see
# utils.config.get_target_resolution() for the fallback default and
# why this isn't a hardware invariant.
TARGET_SIZE = get_target_resolution()


def resize_img_with_scale(img_path: str, is_mask: bool = False) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Same as resize_img(), but also return the scale factor that was applied
    and the original (width, height).

    The scale factor is what morphometrics needs to convert the user's
    calibration (measured on the original image) to the downsampled image:
    um_per_px_downsampled = um_per_px_original / scale. It is recorded
    exactly here rather than reconstructed later from rounded widths.
    """
    img = cv2.imread(img_path, 0)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    h, w = img.shape[:2]
    if w <= TARGET_SIZE[0] and h <= TARGET_SIZE[1]:
        return img, 1.0, (w, h)
    scale = min(TARGET_SIZE[0] / w, TARGET_SIZE[1] / h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LANCZOS4
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation), scale, (w, h)


def resize_img(img_path: str, is_mask: bool = False) -> np.ndarray:
    """
    Load an image and downsample it to fit within TARGET_SIZE, preserving
    its original aspect ratio exactly.

    If both dimensions are already at or below TARGET_SIZE, the image is
    returned unchanged. Otherwise both axes are scaled by the SAME factor —
    whichever axis needs the larger reduction determines the scale — so the
    result fits within TARGET_SIZE with at least one dimension exactly at
    the target and the other at or below it.

    Upscaling is never performed, and aspect ratio is never changed. For a
    source image whose aspect ratio already matches TARGET_SIZE's, this
    produces the exact same output as forcing both dimensions to
    TARGET_SIZE would — the two differ only for mismatched aspect ratios,
    which previously could silently upscale one axis while distorting the
    image's shape.

    Args:
        img_path: path to the image file
        is_mask:  if True, use nearest-neighbour interpolation

    Returns:
        np.ndarray of shape (H, W) dtype uint8 — same aspect ratio as the
        source image, fit within TARGET_SIZE. NOT guaranteed to be exactly
        TARGET_SIZE unless the source's aspect ratio already matches it —
        callers that assume a single universal output size (e.g. a fixed
        patches-per-image count) must account for this.

    Raises:
        ValueError: if the image cannot be read
    """
    # One implementation for both functions, so the scale recorded in
    # segmentation provenance is always the scale actually applied.
    return resize_img_with_scale(img_path, is_mask=is_mask)[0]


def get_image_resolution(img_path: str) -> tuple[int, int]:
    """
    Return (width, height) of an image without fully loading it.
    Uses PIL for speed — does not decode pixel data.
    """
    with Image.open(img_path) as img:
        return img.size  # (width, height)