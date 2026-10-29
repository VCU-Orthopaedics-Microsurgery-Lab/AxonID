"""
utils/calibration.py: pixel size of a segmented image, in µm per pixel.

Purpose
    Segmentation downsamples images larger than the working resolution
    (1440 x 1024) before predicting, so a segmented image's pixels can be
    larger than the microscope's. This module turns the user's microscope
    calibration (config.toml pixel_size_um, keyed by the ORIGINAL image
    width) into the pixel size of the segmented image.

Usage
    from utils.calibration import read_tiff_provenance, resolve_pixel_size
    prov = read_tiff_provenance(seg_path)
    cal  = resolve_pixel_size(mag, seg_width, seg_height, prov)
    cal.px_um   # µm per segmented pixel, or None if not calibrated

Inputs
    mag                    magnification string, e.g. "40X"
    seg_width, seg_height  size of the segmented image in pixels
    provenance             JSON written by segmentation into the TIFF
                           description (original_width, original_height,
                           resize_scale)

Outputs
    Calibration(px_um, original_width, original_height, resize_scale, source, note)
    source: "provenance" (resize recorded by segmentation) or
            "legacy" (no record; image assumed not downsampled)

Rule
    px_um = pixel_size_um[mag][original_width] / resize_scale
    resize_scale < 1 when the image was downsampled, so each segmented
    pixel covers 1 / resize_scale microscope pixels in each direction.

Depends on
    tifffile, utils/config.py (get_pixel_size)
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
from dataclasses import dataclass
import json
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import tifffile

# ── Local ─────────────────────────────────────────────────────────────────────
from utils.config import get_pixel_size


@dataclass(frozen=True)
class Calibration:
    px_um: float | None
    original_width: int
    original_height: int
    resize_scale: float
    source: str
    note: str = ""


def read_tiff_provenance(path: str | Path) -> dict:
    """Segmentation provenance stored as JSON in the TIFF description; {} if absent."""
    try:
        with tifffile.TiffFile(str(path)) as tif:
            desc = tif.pages[0].description
        return json.loads(desc) if desc else {}
    except Exception:
        # Segmented TIFFs from older versions, or edited in another program,
        # may have no (or non-JSON) description.
        return {}


def resolve_pixel_size(mag: str, seg_width: int, seg_height: int, provenance: dict) -> Calibration:
    """See the module header for the rule."""
    ow, oh, scale = (provenance.get(k) for k in ("original_width", "original_height", "resize_scale"))
    if ow and oh and scale:
        ow, oh, scale = int(ow), int(oh), float(scale)
        base = get_pixel_size(mag, ow)
        if base is None:
            return Calibration(None, ow, oh, scale, "provenance",
                               f"no pixel_size_um entry for {mag} at original width {ow}px")
        return Calibration(base / scale, ow, oh, scale, "provenance")

    base = get_pixel_size(mag, seg_width)
    note = ("no resize record in the segmented TIFF (segmented by an older version); "
            "assuming it was not downsampled")
    if base is None:
        note += f"; no pixel_size_um entry for {mag} at width {seg_width}px"
    return Calibration(base, seg_width, seg_height, 1.0, "legacy", note)
