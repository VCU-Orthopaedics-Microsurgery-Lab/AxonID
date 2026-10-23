"""
morphometrics/tracing.py: keep only the axons inside a CSA tracing.

Purpose
    When an image has a QuPath CSA tracing, only the axons whose centre lies
    inside it (union of "+" regions minus union of "-" regions) are measured.
    The others are not lost: they are counted, listed on the image's
    "Outside tracing" sheet and pooled per nerve in
    <nerve>_outside_tracing.xlsx, with their position in the original image
    (the pixels QuPath uses) so each one can be found again.

Usage
    from morphometrics.tracing import (exclude_image, get_geojson_exclusion,
                                       save_outside_tracing, split_by_tracing, tracing_path)
    geojson = tracing_path(nerve_dir, image_stem, config)
    reason = get_geojson_exclusion(geojson, log)          # whole image tagged EXCLUDE?
    if reason:
        kept, kept_px, outside, info = exclude_image(df, pixel_df, resize_scale, reason)
    else:
        kept, kept_px, outside, info = split_by_tracing(df, pixel_df, geojson, resize_scale, log)
    save_outside_tracing([outside, ...], morph_dir, nerve_name)

Inputs
    df, pixel_df   accepted axons from morphometrics.get_morphometrics()
                   (columns x = row, y = column, in segmented-image pixels;
                   label, shared by both frames)
    geojson_path   CSA/{image}_CSA.geojson, traced on the ORIGINAL image
    resize_scale   segmented width / original width (< 1 when downsampled)

Outputs
    kept, kept_px  the axons inside the tracing
    outside        the axons outside it (a DataFrame, possibly empty), with
                   qupath_x (column) and qupath_y (row) in original-image pixels
    info           {"n_all": accepted axons, "n_outside": outside the tracing,
                   "region_px": the tracing in segmented-image pixels, for QC
                   drawing}, or None when there is no usable tracing (nothing
                   is removed). For an excluded image: n_outside = n_all,
                   region_px None and "excluded": the reason.

Depends on
    batch_axon/overlay/process_overlay_qupath.py, morphometrics/analyze_nerve.py
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from shapely.affinity import affine_transform

# ── Local ─────────────────────────────────────────────────────────────────────
from batch_axon.overlay.process_overlay_qupath import (
    get_geojson_region,
    points_in_region,
)
from morphometrics.analyze_nerve import _strip_ome


def tracing_path(nerve_path, image_stem: str, config: dict) -> Path:
    """
    Path of an image's CSA tracing, named as batch_axon looks it up.

    Args:
        nerve_path: nerve folder (contains CSA/)
        image_stem: image name; an .ome suffix is ignored
        config:     loaded config (csa_folder, csa_suffix_geojson)

    Returns:
        CSA/{image}_CSA.geojson as a Path; the file may not exist.
    """
    csa_dir = Path(nerve_path) / config.get("csa_folder", "CSA")
    suffix  = config.get("csa_suffix_geojson", "_CSA.geojson")
    return csa_dir / f"{_strip_ome(image_stem)}{suffix}"


def split_by_tracing(df, pixel_df, geojson_path, resize_scale, log):
    """
    Split accepted axons into those inside and outside the CSA tracing.

    An axon's centre is (y, x) = (column, row) in segmented pixels, with the
    pixel centre at +0.5. Dividing by resize_scale maps it to the original
    image, the grid QuPath traced on.

    Args:
        df:           accepted axons (x = row, y = column, segmented px)
        pixel_df:     the same axons in pixel units, shared "label" column
        geojson_path: CSA/{image}_CSA.geojson
        resize_scale: segmented width / original width
        log:          logger instance

    Returns:
        (kept, kept_pixel_df, outside, info)
            kept, kept_pixel_df: axons inside the tracing
            outside:             axons outside it, plus qupath_x (column) and
                                 qupath_y (row) in original-image pixels
            info:                {"n_all", "n_outside", "region_px"}
        When the GeoJSON cannot be used (unreadable, no "+" region) nothing is
        removed and (df, pixel_df, None, None) is returned; the reason is logged.
    """
    region = get_geojson_region(str(geojson_path), log)
    if region is None:
        return df, pixel_df, None, None

    scale = float(resize_scale) if resize_scale and resize_scale > 0 else 1.0
    # The tracing in segmented-image pixels, with pixel centres at integers
    # (matplotlib's convention), for drawing on the QC figure.
    region_px = affine_transform(region, [scale, 0, 0, scale, -0.5, -0.5])

    if df is None or df.empty:
        return df, pixel_df, pd.DataFrame(), {"n_all": 0, "n_outside": 0, "region_px": region_px}

    cols   = (df["y"].to_numpy(float) + 0.5) / scale
    rows   = (df["x"].to_numpy(float) + 0.5) / scale
    inside = points_in_region(region, cols, rows)

    kept    = df[inside].reset_index(drop=True)
    outside = df[~inside].reset_index(drop=True)
    outside["excluded_because"] = "outside tracing"
    outside["qupath_x"] = np.round(cols[~inside], 1)      # column in the original image
    outside["qupath_y"] = np.round(rows[~inside], 1)      # row in the original image
    if pixel_df is not None and not pixel_df.empty and "label" in pixel_df.columns:
        pixel_df = pixel_df[pixel_df["label"].isin(kept["label"])].reset_index(drop=True)
    info = {"n_all": len(df), "n_outside": len(outside), "region_px": region_px}
    return kept, pixel_df, outside, info


def exclude_image(df, pixel_df, resize_scale, reason):
    """
    Exclude a whole image: none of its axons are measured.

    The accepted axons are not lost: they are returned as "outside" (with
    their QuPath position and the reason) for the Outside tracing sheet and
    the nerve's pooled file.

    Args:
        df:           accepted axons (x = row, y = column, segmented px)
        pixel_df:     the same axons in pixel units
        resize_scale: segmented width / original width
        reason:       why the image is excluded (from get_geojson_exclusion)

    Returns:
        (kept, kept_pixel_df, outside, info): kept and kept_pixel_df are
        empty; info is {"n_all", "n_outside", "region_px": None, "excluded": reason}.
   """
    scale = float(resize_scale) if resize_scale and resize_scale > 0 else 1.0
    if df is None or df.empty:
        return df, pixel_df, pd.DataFrame(), {"n_all": 0, "n_outside": 0, "region_px": None, "excluded": reason}
    outside = df.reset_index(drop=True).copy()
    outside["qupath_x"] = np.round((outside["y"].to_numpy(float) + 0.5) / scale, 1)
    outside["qupath_y"] = np.round((outside["x"].to_numpy(float) + 0.5) / scale, 1)
    outside["excluded_because"] = f"image excluded: {reason}"
    kept_pixel = pixel_df.iloc[0:0] if pixel_df is not None else pixel_df
    info = {"n_all": len(df), "n_outside": len(df), "region_px": None, "excluded": reason}
    return df.iloc[0:0], kept_pixel, outside, info


def save_outside_tracing(frames, morph_dir, nerve_name):
    """
    Pool the axons outside the tracing, across a nerve's images, into one workbook.

    Args:
        frames:     "outside" DataFrames from split_by_tracing(), one per image
                    (None or empty entries are skipped)
        morph_dir:  nerve's Morphometrics folder
        nerve_name: nerve name, used in the file name

    Returns:
        Path of <morph_dir>/<nerve_name>_outside_tracing.xlsx, or None when no
        axon was outside a tracing (no file is written).
    """
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return None
    pooled = pd.concat(frames, ignore_index=True)
    first  = ["image", "label", "excluded_because", "qupath_x", "qupath_y", "x", "y"]
    extra  = [c for c in pooled.columns
              if c.startswith(("axon_diam", "gratio", "myelin_thickness", "fiber_equiv"))]
    pooled = pooled[[c for c in first + extra if c in pooled.columns]]

    about = pd.DataFrame({"About this file": [
        "Axons accepted by the watershed whose centre lies outside the CSA tracing "
        "(outside every + region, or inside a - region).",
        "They are excluded from every count, mean, distribution and density.",
        "excluded_because: 'outside tracing', or 'image excluded: <reason>' when the whole image was tagged EXCLUDE.",
        "qupath_x, qupath_y: position in the original image, in the pixels QuPath uses (x = column, y = row).",
        "x, y: row and column in the segmented image (the QC figure marks these with a cyan x).",
    ]})
    path = Path(morph_dir) / f"{nerve_name}_outside_tracing.xlsx"
    Path(morph_dir).mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pooled.to_excel(writer, sheet_name="Outside tracing", index=False)
        about.to_excel(writer, sheet_name="About", index=False)
        writer.sheets["About"].column_dimensions["A"].width = 110
    return path