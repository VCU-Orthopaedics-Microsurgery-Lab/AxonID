"""
batch_axon/overlay/process_overlay_qupath.py: CSA from a QuPath GeoJSON tracing.

Purpose
    Returns the traced cross-sectional area of one image, in pixels of the
    image QuPath traced on.

Usage
    from batch_axon.overlay.process_overlay_qupath import get_geojson_area
    area_px = get_geojson_area("CSA/image_CSA.geojson", log)

Inputs
    GeoJSON exported from QuPath (export_csa_geojson.groovy). Annotations
    named "+" are included regions, "-" are excluded regions.

Outputs
    get_geojson_exclusion: the reason an image is excluded, from a feature
    classified "EXCLUDE" (the reason is the annotation Name), or None.
    get_geojson_region: the traced region as a shapely geometry (union of
    "+" regions minus union of "-" regions), in pixels of the traced image.
    None if no "+" region exists.
    get_geojson_area: the area of that region in px² (float), or None.
    points_in_region: which points lie inside the region.

Depends on
    shapely, numpy
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import json
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────

import numpy as np
import shapely
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.validation import make_valid

EXCLUDE_CLASS = "exclude"      # classification (case-insensitive) that excludes a whole image


def get_geojson_exclusion(geojson_path: str, log=None, quiet: bool = False) -> str | None:
    """
    The reason a whole image is excluded, or None when it is not.

    An image is excluded when its GeoJSON has a feature classified
    "EXCLUDE" (any capitalisation). The reason is that feature's name; the
    QuPath export script copies it there from the annotation's Name or
    Description. A tag without a reason is refused: it is reported and the
    image is NOT excluded. Several tags give their reasons joined by "; ".

    Args:
        geojson_path: CSA/{image}_CSA.geojson
        log:          logger (warnings go to print when None)
        quiet:        suppress the warning (for a second read of the same file)

    Returns:
        The reason text, or None (no tag, a tag without reason, or an
        unreadable file).
    """
    def warn(msg):
        if quiet:
            return
        if log is not None:
            log.warn(msg)
        else:
            print(f"[WARN] {msg}")

    path = Path(geojson_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    features = data if isinstance(data, list) else data.get("features", [])

    reasons, refused = [], 0
    for feat in features:
        props = (feat.get("properties") or {}) if isinstance(feat, dict) else {}
        cls_name = ((props.get("classification") or {}).get("name") or "").strip()
        if cls_name.lower() != EXCLUDE_CLASS:
            continue
        reason = (props.get("name") or props.get("description") or "").strip()
        if reason and reason.lower() != EXCLUDE_CLASS:
            if reason not in reasons:
                reasons.append(reason)
        else:
            refused += 1
    if refused:
        warn(f"{refused} EXCLUDE tag(s) in {path.name} have no reason (the annotation Name or "
             f"Description) and are ignored: the image is NOT excluded")
    return "; ".join(reasons) if reasons else None


def get_geojson_region(geojson_path: str, log=None, quiet: bool = False):
    """
    The net traced region (a shapely geometry, in pixels of the traced
    image) from a QuPath-exported GeoJSON.

    Unions all "+"-named features together, unions all "-"-named features
    together, and returns (positive - negative). Multiple
    regions of either are handled, not just one of each. quiet=True
    suppresses the warnings (for a second read of a file already read).

    Checks both properties.name and properties.classification.name for the
    "+"/"-" tag — confirmed against a real QuPath 0.7.0 export that "+"/"-"
    set via the annotation Name field land in properties.name, not
    properties.classification.name (the latter is what QuPath's separate
    Classification system uses instead, if that's used in the future).

    The area is analytic (polygon geometry), not a count of rasterized
    pixels.

    Invalid geometries (e.g. a self-intersecting polygon from a sloppy
    trace) are auto-repaired via make_valid() rather than silently producing
    a wrong area or raising.

    Returns None (and logs why) if no "+" region is found — that's treated
    as a real problem, not an empty-but-valid result, since it usually means
    the tracer forgot to tag their include region.
    """
    def warn(msg):
        if quiet:
            return
        if log is not None:
            log.warn(msg)
        else:
            print(f"[WARN] {msg}")

    path = Path(geojson_path)
    if not path.exists():
        warn(f"GeoJSON not found: {path}")
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        warn(f"Could not read/parse {path.name}: {e}")
        return None

    if isinstance(data, list):
        features = data
    else:
        features = data.get("features", [])

    if not features:
        warn(f"No features found in {path.name}")
        return None

    positive_geoms, negative_geoms, unclassified = [], [], 0

    for feat in features:
        try:
            geom = shape(feat["geometry"])
        except (KeyError, ValueError) as e:
            warn(f"Skipping unparseable feature in {path.name}: {e}")
            continue

        if not geom.is_valid:
            geom = make_valid(geom)
            warn(f"Auto-repaired an invalid (self-intersecting?) geometry in {path.name} — verify visually")

        props = feat.get("properties", {}) or {}
        cls = props.get("classification", {}) or {}
        # Real QuPath 0.7.0 exports put "+"/"-" in properties.name when set via
        # the annotation Name field, not properties.classification.name — check
        # both so this works regardless of which mechanism was used.
        cls_name = (cls.get("name") or "").strip()
        if not cls_name:
            cls_name = (props.get("name") or "").strip()

        if cls_name == "+":
            positive_geoms.append(geom)
        elif cls_name == "-":
            negative_geoms.append(geom)
        elif cls_name.lower() == EXCLUDE_CLASS:
            continue                      # whole-image exclusion tag: see get_geojson_exclusion()
        else:
            unclassified += 1

    if unclassified:
        warn(f"{unclassified} feature(s) in {path.name} have no '+'/'-' tag — ignored")

    if not positive_geoms:
        warn(f"No '+' (include) region found in {path.name} — nothing to measure")
        return None

    positive_union = unary_union(positive_geoms)
    final_geom = positive_union
    if negative_geoms:
        negative_union = unary_union(negative_geoms)
        final_geom = positive_union.difference(negative_union)

    return final_geom


def get_geojson_area(geojson_path: str, log=None) -> float | None:
    """Net traced area in px² (union of "+" minus union of "-"), or None."""
    region = get_geojson_region(geojson_path, log)
    return None if region is None else float(region.area)


def points_in_region(region, cols, rows) -> np.ndarray:
    """
    Boolean array: which points lie inside the region. cols is x (column)
    and rows is y (row), in pixels of the image QuPath traced, with the
    pixel corner as the origin (QuPath's convention). A point exactly on
    the boundary counts as outside.
    """
    return shapely.contains_xy(region, np.asarray(cols, float), np.asarray(rows, float))