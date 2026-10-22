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
    Area in px² (float): union of "+" regions minus union of "-" regions,
    computed analytically from the polygons. None if no "+" region exists.

Depends on
    shapely
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import json
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────

from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.validation import make_valid


def get_geojson_area(geojson_path: str, log=None) -> float | None:
    """
    Compute the net traced area (in px²) from a QuPath-exported GeoJSON.

    Unions all "+"-named features together, unions all "-"-named features
    together, and returns the area of (positive - negative). Multiple
    regions of either are handled, not just one of each.

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

    return float(final_geom.area)