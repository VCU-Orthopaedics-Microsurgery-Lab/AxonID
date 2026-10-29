"""
utils/image.py: read a source image as 8-bit grayscale, and centre-crop to the patch grid.

Usage
    from utils.image import read_gray_u8, center_crop_to_patch_grid
    img  = read_gray_u8("nerve_40X_001.tif", log=log)   # uint8, 2D
    crop = center_crop_to_patch_grid(img, 256)

read_gray_u8()
    8-bit gray or RGB             read exactly as before (OpenCV grayscale read).
    16-bit, maximum above 16383   divided by 256 (OpenCV), as before.
    16-bit, maximum 16383 or less treated as 14-, 12- or 8-bit data stored unscaled
                                  and rescaled to 0-255, with a warning. Without
                                  this only a few gray levels would survive.
    32-bit float and other types  ValueError (the intended range is unknowable).
    More than 3 channels          ValueError (brightfield images are expected).
    Multi-page / multi-series     the first series (full resolution) is used.
    Not a TIFF, or unreadable by tifffile: plain OpenCV read, as before.

center_crop_to_patch_grid()
    Image or mask array -> centred crop whose height and width are
    multiples of patch_size. Raises ValueError when either side is smaller
    than patch_size. Used identically by training and segmentation.

Depends on
    numpy, opencv, tifffile
"""

# ── Standard library ──────────────────────────────────────────────────────────
import warnings
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import tifffile

# Full-scale values of the bit depths a camera may store in a 16-bit file.
# A 16-bit image whose maximum is at or below one of these is treated as data
# of that depth, stored unscaled. Brightfield backgrounds sit near saturation,
# so the image maximum is a reliable cue.
_FULL_SCALES = (255, 4095, 16383)


def _warn(msg: str, log=None) -> None:
    if log is not None:
        log.warn(msg)
    else:
        warnings.warn(msg, UserWarning, stacklevel=3)


def _tiff_layout(path: str):
    """(dtype, squeezed axes) of the first series, or None if tifffile cannot say."""
    try:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            keep = [(a, n) for a, n in zip(series.axes, series.shape) if n > 1]
            return np.dtype(series.dtype), "".join(a for a, _ in keep), [n for _, n in keep]
    except Exception:
        return None


def read_gray_u8(path, log=None) -> np.ndarray:
    """
    Read a source image as a 2D uint8 array. See the module header for the rules.

    Raises:
        ValueError: unreadable file, floating-point or otherwise unsupported
            data, or an image with more than 3 channels.
    """
    path = str(path)
    layout = _tiff_layout(path) if Path(path).suffix.lower() in (".tif", ".tiff") else None


    def plain_read():
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return img

    if layout is None:
        return plain_read()

    dtype, axes, sizes = layout
    name = Path(path).name

    if dtype.kind == "f":
        raise ValueError(
            f"{name}: floating-point images are not supported ({dtype}). "
            "Export the image as 8-bit."
        )
    if dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError(f"{name}: unsupported pixel type {dtype}. Export the image as 8-bit.")

    rgb = False
    if axes == "YX":
        pass
    elif len(axes) == 3 and set(axes) - {"Y", "X"} <= {"S", "C"} and 3 in sizes and sizes.count(3) == 1:
        rgb = True
    else:
        raise ValueError(
            f"{name}: unsupported layout (axes '{axes}', shape {tuple(sizes)}). "
            "AxonID expects a single grayscale or RGB brightfield image."
        )

    if dtype == np.dtype("uint8"):
        return plain_read()

    # 16-bit: decide the nominal depth from the maximum of the grayscale image.
    with tifffile.TiffFile(path) as tif:
        arr = np.squeeze(tif.series[0].asarray())
    if rgb:
        if axes[0] in ("S", "C"):          # planar (3, Y, X) -> (Y, X, 3)
            arr = np.moveaxis(arr, 0, -1)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mx = int(arr.max())
    if mx > _FULL_SCALES[-1]:
        return plain_read()                 # full 16-bit range: OpenCV divides by 256

    full = next(f for f in _FULL_SCALES if mx <= f)
    bits = {255: 8, 4095: 12, 16383: 14}[full]
    _warn(
        f"{name}: 16-bit file with maximum value {mx}; treated as {bits}-bit data stored "
        f"unscaled and rescaled to 8-bit. If this is wrong, export the image as 8-bit.",
        log,
    )
    return np.clip(np.rint(arr.astype(np.float32) * (255.0 / full)), 0, 255).astype(np.uint8)


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