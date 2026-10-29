"""
morphometrics/morphometrics.py: per-axon measurements from one segmented image.

Purpose
    Separates touching axons and fibers (watershed), pairs each axon with
    its fiber, computes size, shape and g-ratio for every pair, applies the
    acceptance rules, and writes the per-image workbook.

Usage
    from morphometrics.morphometrics import get_morphometrics, save_morphometrics
    df, pixel_df = get_morphometrics(seg_path, mag, log, calibration=cal)
    save_morphometrics(df, pixel_df, out_dir, stem, provenance, calibration=cal)

Inputs
    Segmented BGW TIFF (0 background, 128 myelin, 255 axon; see
    utils/mask_contract.py), its magnification, and its calibration
    (utils/calibration.py; µm per segmented pixel, corrected for any
    downsampling done by segmentation).

Outputs
    get_morphometrics(): accepted pairs in µm (or pixel units if the image
        is not calibrated) plus the same pairs in pixel units; optionally
        the watershed label images and rejected centroids for QC.
    save_morphometrics(): {stem}_morphometrics.xlsx with Summary, Axon,
        Myelin+G-ratio, Fiber and Pixel Measurements sheets, plus an
        Outside tracing sheet when a CSA tracing excluded axons.
    save_watershed_qc(): one QC figure per nerve.

G-ratio
    gratio_equiv_diam  axon equivalent diameter / fiber equivalent diameter
    gratio_mean_axes   mean axon axis length / mean fiber axis length
    gratio             mean of the two
    config.toml primary_gratio_method picks which one nerve-level summaries
    report; all three are always stored.

Locked (utils/recipe.py)
    MORPHOMETRICS: watershed seeding (distance threshold 0.08, disk 5,
    minimum seed area 2 px) and the acceptance rules. In 1.0.0 the
    configurable morphology filters are off; two hard checks always apply
    (axon equivalent diameter >= 1.01 µm when calibrated, and both g-ratios
    strictly between 0 and 1).

Depends on
    scikit-image, scipy, opencv, openpyxl, utils/calibration.py, utils/recipe.py
"""

import gc
from pathlib import Path

import cv2
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import dilation, disk, h_maxima
from skimage.segmentation import watershed

from utils.calibration import Calibration, read_tiff_provenance, resolve_pixel_size
from utils.config import load_config
from utils.logger import AxonIDLogger
from utils.recipe import MORPHOMETRICS, MORPHOMETRICS_ID
from utils.version import __version__

# Per-filter rejection counts are logged whenever the configurable filters
# are on (they are off in 1.0.0, see utils/recipe.py).
LOG_FILTER_BREAKDOWN = True


# ─── Watershed labelling ──────────────────────────────────────────────────────

def get_labels_axons(img: np.ndarray) -> np.ndarray:
    """
    Split touching axons with a distance-transform watershed.

    Seeds: distance > distance_threshold * max distance (or h-maxima of
    depth h_minima_px, if set), seeds smaller than min_seed_area_px dropped.
    Background marker: outside the mask dilated by dilation_disk.
    All values from MORPHOMETRICS (utils/recipe.py).
    """
    threshold     = MORPHOMETRICS.distance_threshold
    disk_r        = MORPHOMETRICS.dilation_disk
    h_minima_px   = MORPHOMETRICS.h_minima_px
    min_seed_area = MORPHOMETRICS.min_seed_area_px

    distance = ndi.distance_transform_edt(img)
    if h_minima_px is not None:
        seeds        = h_maxima(distance, h_minima_px)
        sure_fg_mask = seeds > 0
    else:
        sure_fg_mask = distance > threshold * distance.max()
    markers      = label(sure_fg_mask)
    if min_seed_area > 0:
        for region in regionprops(markers):
            if region.area < min_seed_area:
                markers[markers == region.label] = 0
        markers = label(markers > 0)
    sure_bg_mask = dilation(img, disk(disk_r))
    markers[sure_bg_mask == 0] = markers.max() + 1
    segmented = watershed(-distance, markers, mask=img)
    del distance, sure_fg_mask, sure_bg_mask, markers
    return segmented


def get_labels_fiber(
    fiber_mask: np.ndarray,
    axon_label: np.ndarray,
) -> np.ndarray:
    """
    Watershed segmentation of fiber_mask seeded from axon labels.

    Instead of deriving markers from fiber_mask's own distance transform
    (which fails when myelin sheaths touch), we transfer one seed per
    axon region from the already-labelled axon_mask.

    Foreground seeds: the maximum distance-transform point within each
    axon-label region. Each seed inherits the corresponding axon label.

    Background marker: dilation of fiber_mask by MORPHOMETRICS.dilation_disk.
    """
    disk_r = MORPHOMETRICS.dilation_disk

    axon_distance = ndi.distance_transform_edt(axon_label > 0)
    markers       = np.zeros_like(fiber_mask, dtype=int)

    for region in regionprops(axon_label):
        region_dist = np.where(axon_label == region.label, axon_distance, 0)
        peak        = np.unravel_index(np.argmax(region_dist), region_dist.shape)
        markers[peak] = region.label

    lost = (markers > 0) & (fiber_mask == 0)
    if np.any(lost):
        fg_coords   = np.array(np.where(fiber_mask > 0)).T
        lost_coords = np.array(np.where(lost)).T
        tree        = cKDTree(fg_coords)
        for coord in lost_coords:
            _, idx  = tree.query(coord)
            nearest = tuple(fg_coords[idx])
            markers[nearest]      = markers[tuple(coord)]
            markers[tuple(coord)] = 0

    sure_bg               = dilation(fiber_mask, disk(disk_r))
    markers[sure_bg == 0] = markers.max() + 1

    fiber_distance = ndi.distance_transform_edt(fiber_mask)
    segmented      = watershed(-fiber_distance, markers, mask=fiber_mask)

    del axon_distance, fiber_distance, sure_bg, markers
    return segmented


# ─── Matching helpers ─────────────────────────────────────────────────────────

def get_axon_row(axon_df: pd.DataFrame, left, right, top, bottom) -> pd.DataFrame:
    """Find the largest axon whose centroid falls within the fiber bounding box."""
    left, right, top, bottom = int(left), int(right), int(top), int(bottom)
    axons_in_box = axon_df[
        (axon_df['centroid-0'] >= left) & (axon_df['centroid-0'] <= right) &
        (axon_df['centroid-1'] >= top)  & (axon_df['centroid-1'] <= bottom)
    ]
    if axons_in_box.empty:
        return axons_in_box
    return axons_in_box[axons_in_box['area'] == axons_in_box['area'].max()]


# ─── QC colormap helper ───────────────────────────────────────────────────────

def make_random_cmap(n_labels: int, seed: int = 42) -> mcolors.ListedColormap:
    """Randomized colormap. Label 0 (background) always black."""
    rng    = np.random.default_rng(seed)
    colors = rng.random((max(n_labels + 1, 2), 3))
    colors[0] = [0, 0, 0]
    return mcolors.ListedColormap(colors)


# ─── Main analysis ────────────────────────────────────────────────────────────

def get_morphometrics(
    seg_path: str,
    mag: str,
    log: AxonIDLogger,
    return_labels: bool = False,
    calibration: Calibration | None = None,
) -> tuple:
    """
    Extract morphometric measurements from a single segmented BGW image.

    Args:
        seg_path:      path to segmented .tif (BGW: black=bg, grey=myelin, white=axon)
        mag:           magnification string e.g. '40X'
        log:           logger instance
        return_labels: if True, also return the label images and rejected
                       centroids. Used by save_watershed_qc().
        calibration:   from utils.calibration.resolve_pixel_size(); read
                       from the TIFF itself when None.

    Returns:
        Without return_labels:
            (df, pixel_df)

        With return_labels:
            (df, pixel_df, axon_label, fiber_label, filtered_centroids)

        Returns (None, None) on image-read failure.

    Acceptance rules: MORPHOMETRICS (utils/recipe.py), see the module header.
    """
    img = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        log.error(f"Could not read: {seg_path}")
        return (None, None, None, None, []) if return_labels else (None, None)
    
    h, w = img.shape
    if calibration is None:
        calibration = resolve_pixel_size(mag, w, h, read_tiff_provenance(seg_path))
    px_size = calibration.px_um

    if px_size is None:
        log.warn(f"{Path(seg_path).name}: {calibration.note}; outputting pixel units only")

    # ── Acceptance rules (utils/recipe.py) ────────────────────────────────────
    m = MORPHOMETRICS
    qf_enabled            = m.quality_filters_enabled
    min_axon_area_um2     = m.min_axon_area_um2
    max_axon_area_um2     = m.max_axon_area_um2
    min_axon_solidity     = m.min_axon_solidity
    max_axon_eccentricity = m.max_axon_eccentricity
    min_myelin_thick_um   = m.min_myelin_thickness_um
    gratio_min            = m.gratio_min
    gratio_max            = m.gratio_max
    min_fiber_area_um2    = m.min_fiber_area_um2
    max_fiber_area_um2    = m.max_fiber_area_um2

    # ── Extract masks from BGW image ─────────────────────────────────────────
    # BGW contract: white(255)=axon, grey(128)=myelin
    # See segment/segment.py recolor() — this mapping must stay in sync.
    axon_mask  = cv2.inRange(img, 200, 255)
    fiber_mask = cv2.inRange(img, 1, 255)

    if not np.any(axon_mask):
        log.warn(f"No axons detected in {Path(seg_path).name}")
        return (pd.DataFrame(), pd.DataFrame(), None, None, []) if return_labels else (pd.DataFrame(), pd.DataFrame())

    axon_label  = get_labels_axons(axon_mask.astype(bool))
    fiber_label = get_labels_fiber(fiber_mask.astype(bool), axon_label)

    axon_props = regionprops_table(axon_label, properties=(
        'label', 'centroid', 'area',
        'axis_minor_length', 'axis_major_length',
        'eccentricity', 'orientation', 'perimeter', 'solidity'
    ))
    fiber_props = regionprops_table(fiber_label, properties=(
        'label', 'bbox', 'area',
        'axis_minor_length', 'axis_major_length',
        'eccentricity', 'orientation', 'perimeter'
    ))

    axon_df  = pd.DataFrame.from_dict(axon_props)
    fiber_df = pd.DataFrame.from_dict(fiber_props)

    # Keep label arrays alive if return_labels requested
    axon_label_out  = axon_label  if return_labels else None
    fiber_label_out = fiber_label if return_labels else None

    if not return_labels:
        del axon_label, fiber_label

    del axon_mask, fiber_mask, img
    gc.collect()

    if axon_df.empty or fiber_df.empty:
        log.warn(f"No regions found in {Path(seg_path).name}")
        return (pd.DataFrame(), pd.DataFrame(), axon_label_out, fiber_label_out, []) if return_labels else (pd.DataFrame(), pd.DataFrame())

    rows                 = []
    pixel_rows           = []
    filtered_centroids   = []  # (x, y, reason) of every rejected pair — used by the watershed QC
    assigned_axon_labels = set()

    # ── Per-filter counters (always tracked, logged when qf_enabled) ──────────
    # Configurable-filter counters exclude the two unconditional hard-validity
    # checks below. filtered_centroids includes failures from both groups so
    # the QC image shows every rejected matched pair.
    n_filtered                = 0
    n_filtered_gratio_invalid = 0
    n_filtered_gratio_range   = 0
    n_filtered_solidity       = 0
    n_filtered_eccentricity   = 0
    n_filtered_area           = 0
    n_filtered_myelin         = 0


    def _filtered(x_coord: float, y_coord: float, reason: str = "filter") -> None:
        filtered_centroids.append((x_coord, y_coord, reason))
            
    for _, row in fiber_df.iterrows():
        left   = row['bbox-0']
        right  = row['bbox-2']
        top    = row['bbox-1']
        bottom = row['bbox-3']

        axon_row = get_axon_row(axon_df, left, right, top, bottom)
        if axon_row.empty:
            continue

        axon_label_id = int(axon_row['label'].iloc[0])
        if axon_label_id in assigned_axon_labels:
            continue
        assigned_axon_labels.add(axon_label_id)

        # ── Axon morphology ───────────────────────────────────────────────────
        axon_area         = float(axon_row['area'].iloc[0])
        axon_perimeter    = float(axon_row['perimeter'].iloc[0])
        axon_major        = float(axon_row['axis_major_length'].iloc[0])
        axon_minor        = float(axon_row['axis_minor_length'].iloc[0])
        axon_eccentricity = float(axon_row['eccentricity'].iloc[0])
        axon_orientation  = float(axon_row['orientation'].iloc[0])
        axon_solidity     = float(axon_row['solidity'].iloc[0])
        # Image coordinates follow NumPy/skimage indexing:
        # x = row coordinate, y = column coordinate.
        x                 = float(axon_row['centroid-0'].iloc[0])
        y                 = float(axon_row['centroid-1'].iloc[0])

        axon_equiv_diam  = np.sqrt(4 * axon_area / np.pi)
        axon_deformation = axon_major / axon_equiv_diam if axon_equiv_diam > 0 else np.nan

        # ── Fiber / Myelin morphology ─────────────────────────────────────────
        fiber_area         = float(row['area'])
        fiber_major        = float(row['axis_major_length'])
        fiber_minor        = float(row['axis_minor_length'])
        fiber_perimeter    = float(row['perimeter'])
        fiber_eccentricity = float(row['eccentricity'])
        fiber_orientation  = float(row['orientation'])

        fiber_equiv_diam  = np.sqrt(fiber_area / np.pi) * 2
        fiber_deformation = fiber_major / fiber_minor if fiber_minor > 0 else np.nan
        myelin_area       = fiber_area - axon_area
        myelin_thickness = (fiber_equiv_diam - axon_equiv_diam) / 2

        # ── G-ratios ──────────────────────────────────────────────────────────
        gratio_area = axon_equiv_diam / fiber_equiv_diam if fiber_equiv_diam > 0 else np.nan
        axon_mean_axis  = (axon_major + axon_minor) / 2
        fiber_mean_axis = (fiber_major + fiber_minor) / 2
        gratio_axes     = axon_mean_axis / fiber_mean_axis if fiber_mean_axis > 0 else np.nan

        if np.isnan(gratio_area) or np.isnan(gratio_axes):
            continue
        gratio = (gratio_area + gratio_axes) / 2

        # ── Configurable morphology filters (off in 1.0.0) ────────────────────
        if qf_enabled:
            if not (0 < gratio_area < 1 and 0 < gratio_axes < 1 and 0 < gratio < 1):
                n_filtered += 1; n_filtered_gratio_invalid += 1; _filtered(x, y, "filter:gratio_invalid"); continue
            if not (gratio_min < gratio < gratio_max):
                n_filtered += 1; n_filtered_gratio_range += 1; _filtered(x, y, "filter:gratio_range"); continue
            if axon_solidity < min_axon_solidity:
                n_filtered += 1; n_filtered_solidity += 1; _filtered(x, y, "filter:solidity"); continue
            if axon_eccentricity > max_axon_eccentricity:
                n_filtered += 1; n_filtered_eccentricity += 1; _filtered(x, y, "filter:eccentricity"); continue

            if px_size is not None:
                axon_area_um2   = axon_area   * px_size ** 2
                fiber_area_um2  = fiber_area  * px_size ** 2
                myelin_thick_um = myelin_thickness * px_size

                if not (min_axon_area_um2 <= axon_area_um2 <= max_axon_area_um2):
                    n_filtered += 1; n_filtered_area += 1; _filtered(x, y, "filter:axon_area"); continue
                if not (min_fiber_area_um2 <= fiber_area_um2 <= max_fiber_area_um2):
                    n_filtered += 1; n_filtered_area += 1; _filtered(x, y, "filter:fiber_area"); continue
                if myelin_thick_um < min_myelin_thick_um:
                    n_filtered += 1; n_filtered_myelin += 1; _filtered(x, y, "filter:myelin_thickness"); continue

        # ── Hard validity checks (always applied) ─────────────────────────────
        if px_size is not None:
            min_axon_diam_um = m.min_axon_diam_um
            if (
                min_axon_diam_um > 0
                and (axon_equiv_diam * px_size) < min_axon_diam_um
            ):
                _filtered(x, y, "hard:min_axon_diam")
                continue

        if not (0 < gratio_area < 1 and 0 < gratio_axes < 1):
            _filtered(x, y, "hard:gratio")
            continue

        # ── Build output row — µm units (primary) ─────────────────────────────
        if px_size is not None:
            new_row = {
                'label':               int(row['label']),
                'x':                   round(x, 1),
                'y':                   round(y, 1),
                'axon_area_um2':       round(axon_area        * px_size ** 2, 3),
                'axon_diam_um':        round(axon_equiv_diam  * px_size, 3),
                'axon_major_um':       round(axon_major       * px_size, 3),
                'axon_minor_um':       round(axon_minor       * px_size, 3),
                'axon_solidity':       round(axon_solidity, 4),
                'axon_deformation':    round(axon_deformation, 4) if not np.isnan(axon_deformation) else np.nan,
                'axon_eccentricity':   round(axon_eccentricity, 4),
                'axon_orientation':    round(axon_orientation, 4),
                'myelin_area_um2':     round(myelin_area      * px_size ** 2, 3),
                'myelin_thickness_um': round(myelin_thickness * px_size, 3),
                'myelin_perimeter_um': round(fiber_perimeter  * px_size, 3),
                'fiber_area_um2':      round(fiber_area       * px_size ** 2, 3),
                'fiber_equiv_diam_um': round(fiber_equiv_diam * px_size, 3),
                'fiber_major_um':      round(fiber_major      * px_size, 3),
                'fiber_minor_um':      round(fiber_minor      * px_size, 3),
                'fiber_deformation':   round(fiber_deformation, 4) if not np.isnan(fiber_deformation) else np.nan,
                'fiber_eccentricity':  round(fiber_eccentricity, 4),
                'fiber_orientation':   round(fiber_orientation, 4),
                'gratio_equiv_diam':   round(gratio_area, 4),
                'gratio_mean_axes':    round(gratio_axes, 4),
                'gratio':              round(gratio, 4),
            }
        else:
            # Uncalibrated fallback — pixel units with _px suffix
            new_row = {
                'label':               int(row['label']),
                'x':                   round(x, 1),
                'y':                   round(y, 1),
                'axon_area_px':        round(axon_area, 2),
                'axon_diam_px':        round(axon_equiv_diam, 3),
                'axon_major_px':       round(axon_major, 3),
                'axon_minor_px':       round(axon_minor, 3),
                'axon_solidity':       round(axon_solidity, 4),
                'axon_deformation':    round(axon_deformation, 4) if not np.isnan(axon_deformation) else np.nan,
                'axon_eccentricity':   round(axon_eccentricity, 4),
                'axon_orientation':    round(axon_orientation, 4),
                'myelin_area_px':      round(myelin_area, 2),
                'myelin_thickness_px': round(myelin_thickness, 3),
                'myelin_perimeter_px': round(fiber_perimeter, 3),
                'fiber_area_px':       round(fiber_area, 2),
                'fiber_equiv_diam_px': round(fiber_equiv_diam, 3),
                'fiber_major_px':      round(fiber_major, 3),
                'fiber_minor_px':      round(fiber_minor, 3),
                'fiber_deformation':   round(fiber_deformation, 4) if not np.isnan(fiber_deformation) else np.nan,
                'fiber_eccentricity':  round(fiber_eccentricity, 4),
                'fiber_orientation':   round(fiber_orientation, 4),
                'gratio_equiv_diam':   round(gratio_area, 4),
                'gratio_mean_axes':    round(gratio_axes, 4),
                'gratio':              round(gratio, 4),
            }

        # ── Accepted pixel-unit row (Sheet 5 calibration/sanity check) ────────
        pixel_row = {
            'label':              int(row['label']),
            'x':                  round(x, 1),
            'y':                  round(y, 1),
            'axon_area':          round(axon_area, 2),
            'axon_perimeter':     round(axon_perimeter, 3),
            'axon_diam':          round(axon_equiv_diam, 3),
            'axon_major':         round(axon_major, 3),
            'axon_minor':         round(axon_minor, 3),
            'axon_solidity':      round(axon_solidity, 4),
            'axon_deformation':   round(axon_deformation, 4) if not np.isnan(axon_deformation) else np.nan,
            'axon_eccentricity':  round(axon_eccentricity, 4),
            'axon_orientation':   round(axon_orientation, 4),
            'myelin_area':        round(myelin_area, 2),
            'myelin_thickness':   round(myelin_thickness, 3),
            'myelin_perimeter':   round(fiber_perimeter, 3),
            'fiber_area':         round(fiber_area, 2),
            'fiber_equiv_diam':   round(fiber_equiv_diam, 3),
            'fiber_major':        round(fiber_major, 3),
            'fiber_minor':        round(fiber_minor, 3),
            'fiber_deformation':  round(fiber_deformation, 4) if not np.isnan(fiber_deformation) else np.nan,
            'fiber_eccentricity': round(fiber_eccentricity, 4),
            'fiber_orientation':  round(fiber_orientation, 4),
            'gratio_equiv_diam':  round(gratio_area, 4),
            'gratio_mean_axes':   round(gratio_axes, 4),
            'gratio':             round(gratio, 4),
        }
        rows.append(new_row)
        pixel_rows.append(pixel_row)                                      

    del axon_df, fiber_df
    gc.collect()

    if not rows:
        log.warn(f"No valid axon-fiber pairs in {Path(seg_path).name}")
        return (pd.DataFrame(), pd.DataFrame(), axon_label_out, fiber_label_out, filtered_centroids) if return_labels else (pd.DataFrame(), pd.DataFrame())

    if qf_enabled and n_filtered > 0:
        log.info(f"  Quality filters removed {n_filtered} pair(s)")
        if LOG_FILTER_BREAKDOWN:
            log.info(f"    g-ratio invalid: {n_filtered_gratio_invalid} | "
                    f"g-ratio range: {n_filtered_gratio_range} | "
                    f"solidity: {n_filtered_solidity} | "
                    f"eccentricity: {n_filtered_eccentricity} | "
                    f"area: {n_filtered_area} | "
                    f"myelin: {n_filtered_myelin}")

    # Calibration columns travel with every row so nerve-level aggregation
    # (analyze_nerve.py) can recover the original image size for CSA.
    id_cols = {
        'image':               Path(seg_path).stem,
        'resolution':          f"{w}x{h}",
        'magnification':       mag,
        'original_resolution': f"{calibration.original_width}x{calibration.original_height}",
        'resize_scale':        round(calibration.resize_scale, 6),
        'px_size_um':          round(px_size, 6) if px_size is not None else None,
    }
    df = pd.DataFrame(rows)
    pixel_df = pd.DataFrame(pixel_rows)
    for i, (col, val) in enumerate(id_cols.items()):
        df.insert(i, col, val)
        pixel_df.insert(i, col, val)

    if return_labels:
        return df, pixel_df, axon_label_out, fiber_label_out, filtered_centroids  
    return df, pixel_df                                                             


# ─── Watershed QC sheet ───────────────────────────────────────────────────────

def _get_watershed_seeding_text() -> str:
    """Human-readable description of the watershed seeding (utils/recipe.py)."""
    if MORPHOMETRICS.h_minima_px is not None:
        return f"h-minima={float(MORPHOMETRICS.h_minima_px):g}px"
    return f"distance-threshold={MORPHOMETRICS.distance_threshold:g}"


def _draw_region(ax, region) -> None:
    """
    Outline a shapely region on a QC axis.

    Args:
        ax:     matplotlib axis (pixel coordinates, pixel centres at integers)
        region: shapely Polygon or MultiPolygon; outer edges are drawn solid,
                holes (excluded "-" areas) dashed
    """
    for poly in getattr(region, 'geoms', [region]):
        if poly.is_empty:
            continue
        x, y = poly.exterior.xy
        ax.plot(x, y, color='yellow', linewidth=1.4)
        for ring in poly.interiors:
            hx, hy = ring.xy
            ax.plot(hx, hy, color='yellow', linewidth=1.2, linestyle='--')


# ─── Watershed QC sheet ───────────────────────────────────────────────────
# Everything you might want to tune is in this first block.

WSQC_WRITE_MORPH      = True    # sheet 2: axon watershed | fiber watershed | tagged overlay on the tissue
WSQC_WRITE_COMBINED   = True    # sheet 3: tissue | segmentation overlay | watershed overlay
WSQC_CROPS            = True    # sheet 2, 3rd panel: 6 zoomed crops of the worst cases (False = full tagged overlay)
WSQC_ZOOM             = 4       # zoom of those crops
WSQC_LOST_FRAC        = 1 / 3   # an unmeasured axon at least this fraction of the measured median area counts as a possible lost axon
WSQC_PNG_COMPRESS     = 3       # 3 = close to default size, noticeably faster
WSQC_MAX_PX           = 100000  # sheet is split into _partN above this height

# Outline colors on the tissue (axon boundaries, 1 px)
WSQC_KEPT_RGB    = (0, 255, 110)     # measured axon
WSQC_RULE_RGB    = (255, 59, 48)     # rejected by a quality filter (config.toml)
WSQC_HARD_RGB    = (255, 0, 190)     # rejected by a hard rule (always on)
WSQC_DROP_RGB    = (80, 170, 255)    # axon region that was never measured
WSQC_MERGE_RGB   = (255, 214, 0)     # measured axon that may be two merged axons
WSQC_TRACE_RGB   = (170, 255, 60)    # CSA tracing
WSQC_OUTSIDE_RGB = (0, 229, 255)     # axon outside the tracing (x)
WSQC_TRACE_PX    = 2
WSQC_OUTSIDE_COMBINED_RGB = (0, 229, 255)   # outside-tracing x on the combined sheet
WSQC_OUTSIDE_PX  = 1                  # line width of that x
WSQC_FIBER_EDGE  = 0.35              # opacity of the faint fiber-territory lines
WSQC_BG          = (38, 38, 38)      # ground of the combined watershed panel
WSQC_CLASSIC_KEPT_RGB = (235, 40, 40)   # measured axon outline in the combined panel
WSQC_CLASSIC_RULE_RGB = (255, 165, 0)   # quality-filter reject there (red would look like a measured axon)
WSQC_DOT_RGB     = (255, 40, 40)     # dot at the centroid of every measured axon (fiber panel)
WSQC_DOT_R       = 2

# Merge suspects: the lowest-solidity axons of the nerve (a dumbbell shape),
# never above the ceiling. None disables the mark.
WSQC_MERGE_PCT   = 3
WSQC_MERGE_CEIL  = 0.75

_WSQC_KEPT, _WSQC_RULE, _WSQC_HARD, _WSQC_DROP, _WSQC_MERGE = 1, 2, 3, 4, 5


def _wsqc_font(size: int, bold: bool = False):
    """PIL TrueType font: system font first, then matplotlib's bundled DejaVu."""
    from PIL import ImageFont
    names = (["DejaVuSans-Bold.ttf", "arialbd.ttf"] if bold
             else ["DejaVuSans.ttf", "arial.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        import matplotlib
        ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / (
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        return ImageFont.truetype(str(ttf), size)
    except Exception:
        return ImageFont.load_default()


def _wsqc_text(draw, xy, text, font, fill, anchor):
    try:
        draw.text(xy, text, font=font, fill=fill, anchor=anchor)
    except (ValueError, TypeError):   # bitmap fallback font has no anchors
        draw.text(xy, text, font=font, fill=fill)


def _wsqc_edges(lbl: np.ndarray) -> np.ndarray:
    """Pixels of a labeled region that touch a different label or the background."""
    e = np.zeros(lbl.shape, dtype=bool)
    d = lbl[:, :-1] != lbl[:, 1:]
    e[:, :-1] |= d
    e[:, 1:] |= d
    d = lbl[:-1, :] != lbl[1:, :]
    e[:-1, :] |= d
    e[1:, :] |= d
    return e & (lbl > 0)


WSQC_MULTI_FRAC = 0.5    # a second axon counts in a fiber only if this share of it lies inside


def _wsqc_multi(axon_lbl: np.ndarray, fiber_lbl: np.ndarray) -> np.ndarray:
    """True for every fiber label whose territory holds more than one axon
    (an axon counts when at least WSQC_MULTI_FRAC of its area is inside)."""
    return _wsqc_multi_counts(axon_lbl, fiber_lbl) > 1


def _wsqc_multi_counts(axon_lbl: np.ndarray, fiber_lbl: np.ndarray) -> np.ndarray:
    """Number of axons (see _wsqc_multi) inside every fiber label."""
    m = axon_lbl > 0
    n_ax = int(axon_lbl.max()) + 1
    key, cnt = np.unique(fiber_lbl[m].astype(np.int64) * n_ax + axon_lbl[m].astype(np.int64),
                         return_counts=True)
    area = np.bincount(axon_lbl[m].astype(np.int64), minlength=n_ax)
    keep = cnt >= WSQC_MULTI_FRAC * area[key % n_ax]
    counts = np.bincount(key[keep] // n_ax, minlength=int(fiber_lbl.max()) + 1)
    counts[0] = 0
    return counts


def _wsqc_audit(axon_lbl, fiber_lbl, cls):
    """
    Why was each unmeasured axon region (class 4) never measured? Rows of
    (label, area_px, row, col, fiber_at_centroid, cause), cause being:
      no_own_fiber    no fiber territory carries this axon's label (its seed was lost)
      inside_other    its centroid lies in another axon's fiber territory (swallowed)
      own_fiber       it has its own fiber territory; lost later, in the axon match
      outside_fiber   its centroid lies outside every fiber territory
    """
    rows = []
    lab = np.flatnonzero(cls == _WSQC_DROP)
    lab = lab[lab > 0]
    if lab.size == 0:
        return rows
    pr = regionprops_table(axon_lbl, properties=("label", "area", "centroid"))
    at = {int(l): i for i, l in enumerate(pr["label"])}
    f_area = np.bincount(fiber_lbl.ravel().astype(np.int64), minlength=int(axon_lbl.max()) + 1)
    h, w = axon_lbl.shape
    for l in lab:
        i = at.get(int(l))
        if i is None:
            continue
        r = min(max(int(round(pr["centroid-0"][i])), 0), h - 1)
        c = min(max(int(round(pr["centroid-1"][i])), 0), w - 1)
        fl = int(fiber_lbl[r, c])
        if fl == int(l):
            cause = "own_fiber"
        elif l < len(f_area) and f_area[l] == 0:
            cause = "no_own_fiber" if fl > 0 else "outside_fiber"
        elif fl > 0:
            cause = "inside_other"
        else:
            cause = "outside_fiber"
        rows.append((int(l), int(pr["area"][i]), r, c, fl, cause))
    return rows


def _wsqc_classify(axon_lbl, df, filtered_xy):
    """
    Class of every axon region, by label id.

    1 measured, 2 rejected by a quality filter, 3 rejected by a hard rule,
    4 never measured (e.g. a second axon inside one fiber). Also returns each
    measured axon's solidity by label id (NaN elsewhere).
    Regions are matched to the rows of df / filtered_xy by centroid.
    """
    n = int(axon_lbl.max())
    cls = np.zeros(n + 1, dtype=np.uint8)
    sol = np.full(n + 1, np.nan)
    if n == 0:
        return cls, sol
    props = regionprops_table(axon_lbl, properties=("label", "centroid"))
    lab = np.asarray(props["label"])
    tree = cKDTree(np.column_stack([props["centroid-0"], props["centroid-1"]]))
    cls[lab] = _WSQC_DROP

    if df is not None and not df.empty and "x" in df.columns:
        dist, idx = tree.query(df[["x", "y"]].to_numpy(float), distance_upper_bound=0.5)
        ok = np.isfinite(dist)
        cls[lab[idx[ok]]] = _WSQC_KEPT
        if "axon_solidity" in df.columns:
            sol[lab[idx[ok]]] = df["axon_solidity"].to_numpy(float)[ok]
    for item in filtered_xy or []:
        fx, fy = item[0], item[1]
        reason = item[2] if len(item) > 2 else "filter"
        dist, i = tree.query([fx, fy], distance_upper_bound=0.5)
        if np.isfinite(dist):
            cls[lab[i]] = _WSQC_HARD if str(reason).startswith("hard") else _WSQC_RULE
    return cls, sol


def _wsqc_crop_grid(h, w, tissue, axon_lbl, fiber_lbl, cls, sol, df, bg, edge_rgb_lut,
                    font, zoom, text_fn, fiber_rgb):
    """
    Panel (h x w) of 2 rows x 3 columns of crops at `zoom`x around the worst cases:
    largest unmeasured axon | lowest-solidity merge suspect | fiber with most axons.
    Top row: tissue with tagged outlines. Bottom row: fiber watershed with dots.
    """
    from PIL import Image, ImageDraw
    g = 6
    tw, th = (w - 2 * g) // 3, (h - g) // 2
    ww, wh = tw // zoom, th // zoom
    panel = np.full((h, w, 3), bg, dtype=np.uint8)

    cases = []   # (title, (row, col)) per column; None centre = nothing to show
    # 1. largest unmeasured axon
    aud = _wsqc_audit(axon_lbl, fiber_lbl, cls)
    if aud:
        l, area, r, c, fl, cause = max(aud, key=lambda t: t[1])
        cases.append((f"largest unmeasured · {area}px · {cause}", (r, c)))
    else:
        cases.append(("no unmeasured axons", None))
    # 2. lowest-solidity merge suspect
    sus = np.flatnonzero(cls == _WSQC_MERGE)
    sus = sus[sus > 0]
    if sus.size:
        l = int(sus[np.nanargmin(sol[sus])])
        rr, cc = np.nonzero(axon_lbl == l)
        cases.append((f"lowest solidity {sol[l]:.2f}", (int(rr.mean()), int(cc.mean()))))
    else:
        cases.append(("no merge suspects", None))
    # 3. fiber holding the most axons
    cnt = _wsqc_multi_counts(axon_lbl, fiber_lbl)
    if cnt.size and cnt.max() > 1:
        l = int(np.argmax(cnt))
        rr, cc = np.nonzero(fiber_lbl == l)
        cases.append((f"most axons in a fiber · {int(cnt[l])}", (int(rr.mean()), int(cc.mean()))))
    else:
        cases.append(("no multi-axon fibers", None))

    multi = cnt > 1
    for j, (title, ctr) in enumerate(cases):
        x0 = j * (tw + g)
        for i in (0, 1):
            y0 = i * (th + g)
            tile = np.full((th, tw, 3), (30, 30, 30), dtype=np.uint8)
            if ctr is not None:
                r0 = min(max(ctr[0] - wh // 2, 0), max(h - wh, 0))
                c0 = min(max(ctr[1] - ww // 2, 0), max(w - ww, 0))
                sl = (slice(r0, r0 + wh), slice(c0, c0 + ww))
                big = lambda a: np.repeat(np.repeat(a[sl], zoom, axis=0), zoom, axis=1)
                al, fl_ = big(axon_lbl), big(fiber_lbl)
                if i == 0:
                    base = tissue[sl] if tissue is not None else np.zeros((wh, ww, 3), np.uint8)
                    img = cv2.resize(np.ascontiguousarray(base), (ww * zoom, wh * zoom),
                                     interpolation=cv2.INTER_CUBIC)
                    fe = _wsqc_edges(fl_)
                    img[fe] = (img[fe] * (1 - WSQC_FIBER_EDGE) + 205 * WSQC_FIBER_EDGE).astype(np.uint8)
                    img[fe & multi[fl_]] = WSQC_MERGE_RGB
                else:
                    img = np.ascontiguousarray(fiber_rgb[sl].repeat(zoom, 0).repeat(zoom, 1))
                    if df is not None and not df.empty and "x" in df.columns:
                        for cx, cy in zip(df["x"].to_numpy(float), df["y"].to_numpy(float)):
                            if sl[0].start <= cx < sl[0].stop and sl[1].start <= cy < sl[1].stop:
                                cv2.circle(img, (int((cy - sl[1].start + 0.5) * zoom),
                                                 int((cx - sl[0].start + 0.5) * zoom)),
                                           WSQC_DOT_R + 2, WSQC_DOT_RGB, -1)
                # class-coloured outline, 2 px, with a thin dark casing so it reads on any ground
                E = _wsqc_edges(al).astype(np.uint8)
                thick = (cv2.dilate(E, np.ones((2, 2), np.uint8)) > 0) & (al > 0)
                case = (cv2.dilate(E, np.ones((4, 4), np.uint8)) > 0) & ~thick
                img[case] = (img[case] * 0.35).astype(np.uint8)
                col = edge_rgb_lut[cls[al]]
                img[thick] = col[thick]
                tile[:img.shape[0], :img.shape[1]] = img[:th, :tw]
            if i == 0:
                pil = Image.fromarray(tile)
                d = ImageDraw.Draw(pil)
                tl = d.textlength(title, font=font) if hasattr(d, "textlength") else 200
                d.rectangle([0, 0, int(tl) + 14, 30], fill=(20, 20, 20))
                _wsqc_text(d, (7, 15), title, font, (225, 225, 230), "lm")
                tile = np.asarray(pil)
            panel[y0:y0 + th, x0:x0 + tw] = tile
    return panel


def _wsqc_risk(axon_lbl, fiber_lbl, cls, multi, region):
    """
    Per-image numbers for the banner, all restricted to the CSA tracing when there is one.
      measured, filtered, hard, merge   counts of axon regions by class
      n_total       axon regions considered (measured + rejected + unmeasured)
      lost          unmeasured axons whose area is >= WSQC_LOST_FRAC x the measured median
      n_unmeasured  all unmeasured axons;  med_unmeasured / med_measured  median areas (px)
      multi, n_fiber  fibers holding 2+ axons / fibers
    """
    h, w = axon_lbl.shape
    n = int(axon_lbl.max())
    rm = None
    if region is not None:
        rm = np.zeros((h, w), np.uint8)
        for g in getattr(region, "geoms", [region]):
            if getattr(g, "geom_type", "") != "Polygon" or g.is_empty:
                continue
            cv2.fillPoly(rm, [np.round(np.asarray(g.exterior.coords)).astype(np.int32)], 1)
            for ring in g.interiors:
                cv2.fillPoly(rm, [np.round(np.asarray(ring.coords)).astype(np.int32)], 0)

    def inside_ids(lbl, k):
        """Boolean per label id: centroid inside the tracing (all True without one)."""
        if rm is None:
            return np.ones(k + 1, dtype=bool)
        ids = lbl.ravel().astype(np.int64)
        cnt = np.maximum(np.bincount(ids, minlength=k + 1), 1)
        rr = np.repeat(np.arange(h), w)
        cc = np.tile(np.arange(w), h)
        r = (np.bincount(ids, weights=rr, minlength=k + 1) / cnt).round().astype(int).clip(0, h - 1)
        c = (np.bincount(ids, weights=cc, minlength=k + 1) / cnt).round().astype(int).clip(0, w - 1)
        return rm[r, c] > 0

    ia = inside_ids(axon_lbl, n)
    cls_in = np.where(ia, cls[:n + 1], 0)
    cls_in[0] = 0
    c = np.bincount(cls_in, minlength=6)
    area = np.bincount(axon_lbl.ravel().astype(np.int64), minlength=n + 1)
    meas = (cls_in == _WSQC_KEPT) | (cls_in == _WSQC_MERGE)
    med_m = float(np.median(area[meas])) if meas.any() else 0.0
    drop = cls_in == _WSQC_DROP
    lost = int((drop & (area >= WSQC_LOST_FRAC * med_m)).sum()) if med_m else 0
    nf = int(fiber_lbl.max())
    fi = inside_ids(fiber_lbl, nf)
    fa = np.bincount(fiber_lbl.ravel().astype(np.int64), minlength=nf + 1) > 0
    fa[0] = False
    fmask = fi & fa
    return dict(
        measured=int(c[_WSQC_KEPT] + c[_WSQC_MERGE]), merge=int(c[_WSQC_MERGE]),
        filtered=int(c[_WSQC_RULE]), hard=int(c[_WSQC_HARD]),
        n_unmeasured=int(c[_WSQC_DROP]), lost=lost,
        n_total=int(c[_WSQC_KEPT] + c[_WSQC_MERGE] + c[_WSQC_RULE] + c[_WSQC_HARD] + c[_WSQC_DROP]),
        med_unmeasured=float(np.median(area[drop])) if drop.any() else 0.0, med_measured=med_m,
        multi=int((multi[:nf + 1] & fmask).sum()), n_fiber=int(fmask.sum()))


def _wsqc_tiles(data: dict, qc_dir: Path, stem: str, cache: dict):
    """
    The segmentation sheet's panels of one image as {role: array} (roles: tissue,
    overlay, stack) and its name there, or (None, None).
    """
    from PIL import Image
    best = None
    for key in data.get("images", {}):
        if stem.startswith(key) and (len(stem) == len(key) or not stem[len(key)].isalnum()):
            if best is None or len(key) > len(best):
                best = key
    if best is None:
        return None, None
    ent = data["images"][best]
    fname = data["files"][ent["part"] - 1]
    if fname not in cache:
        cache.clear()                       # one sheet in memory at a time
        cache[fname] = np.asarray(Image.open(qc_dir / fname).convert("RGB"))
    sheet = cache[fname]
    boxes = ent["boxes"]
    roles = ent.get("roles") or (["tissue", "overlay"] if len(boxes) == 2 else ["overlay"])
    return {r: sheet[y:y + h, x:x + w] for r, (x, y, w, h) in zip(roles, boxes)}, best


def save_watershed_qc(
    seg_images: list,
    all_data: list,
    morph_dir: Path,
    nerve_name: str,
    suffix: str,
    mag: str,
    log: AxonIDLogger = None,
    tracing_overlays: dict | None = None,
) -> None:
    """
    Generate the morphometrics QC sheets for all segmented images in a nerve.
    Two sheets, same equal tiles as the segmentation sheet, in the QC/ folder:

    {nerve_name}_morphometrics_qc{suffix}.png
        axon watershed | fiber watershed | tagged overlay on the tissue
    {nerve_name}_combined_qc{suffix}.png
        tissue | segmentation overlay | watershed overlay
    (each split into _partN when very tall)

    The tissue and the segmentation overlay are copied from
    {nerve_name}_segmentation_qc.png using its .layout.json; without them the
    tagged overlay is drawn on black and the combined sheet has only its
    watershed panel.

    Tagged overlay outlines: lilac = measured; red = rejected by a quality
    filter; white = rejected by a hard rule (always on); blue = never measured;
    yellow = measured but possibly two merged axons (lowest solidity), or a
    fiber territory holding more than one axon region.
    Combined watershed panel: opaque fibers on a dark ground, a red outline and
    a dot on every measured axon, exceptions in the colors above.
    CSA tracing in green, axons outside it marked with a cyan x. An excluded
    image is named in its bar and has every axon marked.

    Args:
        seg_images: Segmented image paths, in the same order as all_data.
        all_data: Per-image tuples of
            (df, axon_label, fiber_label, filtered), where filtered is a list
            of (x, y, reason) for every rejected axon; reasons starting with
            "hard" are the always-on rules.
        morph_dir: Morphometrics directory; the sheets go to its sibling QC dir.
        nerve_name / suffix / mag: used in the file names and the title.
        log: Optional AxonIDLogger.
        tracing_overlays: Optional {segmented image path: (region, outside_df)}
            (a third item is the reason of an excluded image).
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        if log:
            log.warn("Pillow not available — skipping watershed QC sheet")
        return

    config = load_config()
    qc_dir = morph_dir.parent / config.get("qc_folder", "QC")
    qc_dir.mkdir(parents=True, exist_ok=True)
    n_images = len(seg_images)
    if n_images == 0:
        return

    BG, BAR   = (20, 20, 20), (52, 56, 66)
    GAP, BLOCK_GAP, BAR_H, BANNER_H = 12, 28, 56, 64
    TAG_RGB   = (255, 170, 60)

    # ── Pass 1: classify every axon region, set the merge-suspect cut-off ───
    info, sols = [], []
    for data in all_data:
        df, axon_lbl, fiber_lbl, filtered_xy = data
        if axon_lbl is None or fiber_lbl is None:
            info.append(None)
            continue
        info.append(list(_wsqc_classify(axon_lbl, df, filtered_xy)) + [_wsqc_multi(axon_lbl, fiber_lbl)])
        if df is not None and not df.empty and "axon_solidity" in df.columns:
            sols.append(df["axon_solidity"].to_numpy(float))
    # Audit of the unmeasured axons: always logged and written to a CSV next to the sheets.
    audit_rows = []
    for k, entry in enumerate(info):
        if entry is None:
            continue
        for row in _wsqc_audit(all_data[k][1], all_data[k][2], entry[0]):
            audit_rows.append((seg_images[k].name,) + row)
    if audit_rows:
        import csv
        from collections import Counter
        with open(qc_dir / f"{nerve_name}_unmeasured_axons{suffix}.csv", "w", newline="",
                  encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["image", "axon_label", "area_px", "centroid_row", "centroid_col",
                         "fiber_label_at_centroid", "cause"])
            wr.writerows(audit_rows)
        if log:
            for k, entry in enumerate(info):
                if entry is None:
                    continue
                cnt = Counter(r[-1] for r in audit_rows if r[0] == seg_images[k].name)
                if cnt:
                    log.info(f"Unmeasured axons ({seg_images[k].name}): "
                             + ", ".join(f"{n} {c}" for c, n in sorted(cnt.items())))
    merge_thr = None
    if WSQC_MERGE_PCT is not None and sols:
        s = np.concatenate(sols)
        s = s[~np.isnan(s)]
        if s.size:
            merge_thr = min(float(np.percentile(s, WSQC_MERGE_PCT)), WSQC_MERGE_CEIL)
    totals = np.zeros(6, dtype=int)
    n_multi = 0
    for entry in info:
        if entry is None:
            continue
        cls, sol, multi = entry
        n_multi += int(multi.sum())
        if merge_thr is not None:
            with np.errstate(invalid="ignore"):
                cls[(cls == _WSQC_KEPT) & (sol < merge_thr)] = _WSQC_MERGE
        totals += np.bincount(cls[1:], minlength=6)

    # Tracked every run (log + CSV), whether or not the sheets are written.
    summ = []
    for k, entry in enumerate(info):
        if entry is None:
            continue
        ov0 = (tracing_overlays or {}).get(seg_images[k], ())
        rk = _wsqc_risk(all_data[k][1], all_data[k][2], entry[0], entry[2], ov0[0] if ov0 else None)
        summ.append((seg_images[k].name, rk))
        if log:
            log.info(f"Axon accounting ({seg_images[k].name}): {rk['measured']} measured, "
                     f"{rk['hard']} hard-rejected, {rk['filtered']} filtered, "
                     f"{rk['n_unmeasured']} unmeasured (median {rk['med_unmeasured']:.0f}px vs "
                     f"measured {rk['med_measured']:.0f}px; {rk['lost']} at least "
                     f"{WSQC_LOST_FRAC:.2f}x the measured median), "
                     f"{rk['multi']}/{rk['n_fiber']} multi-axon fibers")
    if summ:
        import csv
        with open(qc_dir / f"{nerve_name}_axon_accounting{suffix}.csv", "w", newline="",
                  encoding="utf-8") as fh:
            wr = csv.writer(fh)
            keys = list(summ[0][1].keys())
            wr.writerow(["image"] + keys)
            for name_, rk in summ:
                wr.writerow([name_] + [rk[k_] for k_ in keys])

    # ── Geometry and the segmentation sheet's tissue ────────────────────────
    layout, cache = None, {}
    side = qc_dir / f"{nerve_name}_segmentation_qc.layout.json"
    if side.is_file():
        try:
            import json
            layout = json.loads(side.read_text(encoding="utf-8"))
        except Exception:
            layout = None
    if layout is None and log:
        log.warn("Morphometrics QC: no segmentation QC layout found, so no tissue panels. "
                 "Re-run segmentation (current version) to create it, then run morphometrics again.")

    max_w = max((d[1].shape[1] for d in all_data if d[1] is not None), default=0)

    def block_h(k):
        lbl = all_data[k][1]
        return BAR_H + (lbl.shape[0] if lbl is not None else 0) + BLOCK_GAP

    parts, cur, cur_h = [], [], BANNER_H
    for k in range(n_images):
        if cur and cur_h + block_h(k) > WSQC_MAX_PX:
            parts.append((cur, cur_h)); cur, cur_h = [], BANNER_H
        cur.append(k); cur_h += block_h(k)
    parts.append((cur, cur_h))

    font_title = _wsqc_font(30, bold=True)
    font_small = _wsqc_font(26)
    col_lut = np.array([(0, 0, 0), WSQC_KEPT_RGB, WSQC_RULE_RGB, WSQC_HARD_RGB,
                        WSQC_DROP_RGB, WSQC_MERGE_RGB], dtype=np.uint8)
    any_trace   = any(v and v[0] is not None for v in (tracing_overlays or {}).values())
    any_outside = any(v and v[1] is not None and len(v[1]) for v in (tracing_overlays or {}).values())

    def legend_for(mode):
        """Legend (left to right): only what is in this nerve."""
        kept = WSQC_CLASSIC_KEPT_RGB if mode == "combined" else WSQC_KEPT_RGB
        legend = [("Measured", kept, "line")]
        if totals[_WSQC_RULE]:
            legend.append(("Filtered", WSQC_CLASSIC_RULE_RGB if mode == "combined" else WSQC_RULE_RGB, "line"))
        if totals[_WSQC_HARD]:
            legend.append(("Hard-rejected", WSQC_HARD_RGB, "line"))
        if totals[_WSQC_DROP]:
            legend.append(("Not measured", WSQC_DROP_RGB, "line"))
        if totals[_WSQC_MERGE] or (n_multi and mode == "morph"):
            legend.append(("Merge suspect", WSQC_MERGE_RGB, "line"))
        if any_trace and (mode == "combined" or not WSQC_CROPS):
            legend.append(("CSA tracing", WSQC_TRACE_RGB, "line"))
        if any_outside and (mode == "combined" or not WSQC_CROPS):
            legend.append(("Outside tracing", WSQC_OUTSIDE_COMBINED_RGB if mode == "combined" else WSQC_OUTSIDE_RGB, "x"))
        return legend

    saved = []

    def fit(img, h, w):
        if img.shape[:2] != (h, w):
            return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(img)

    def build_row(k, mode):
        """Panels, bar text, tag and name for image k."""
        img_path = seg_images[k]
        df, axon_lbl, fiber_lbl, filtered_xy = all_data[k]
        cls, _, multi = info[k]
        h, w = axon_lbl.shape

        tissue = overlay = None
        key = None
        if layout is not None:
            try:
                tiles, key = _wsqc_tiles(layout, qc_dir, img_path.stem, cache)
            except Exception:
                tiles = None
            if tiles:
                if tiles.get("tissue") is not None:
                    tissue = fit(tiles["tissue"], h, w)
                if tiles.get("overlay") is not None:
                    overlay = fit(tiles["overlay"], h, w)

        ov = tuple((tracing_overlays or {}).get(img_path, ())) + (None, None, None)
        region, outside, reason = ov[:3]

        # The random-color watersheds (fixed seeds, neighbors differ)
        n_fiber = int(fiber_lbl.max())
        fcols = (np.asarray(make_random_cmap(n_fiber, seed=99).colors) * 255).astype(np.uint8)
        fiber_rgb = np.ascontiguousarray(fcols[fiber_lbl])

        def dots(img):
            if df is not None and not df.empty and "x" in df.columns:
                for cx, cy in zip(df["x"].to_numpy(float), df["y"].to_numpy(float)):
                    cv2.circle(img, (int(round(cy)), int(round(cx))), WSQC_DOT_R,
                               WSQC_DOT_RGB, -1)

        lut = col_lut
        if mode == "combined":
            # Opaque fibers on a dark ground, red axon outlines and dots; only the
            # exceptions get another outline color.
            inside = (fiber_lbl > 0)[..., None]
            out = np.where(inside, fiber_rgb, np.array(WSQC_BG, dtype=np.uint8)).astype(np.uint8)
            lut = col_lut.copy()
            lut[_WSQC_KEPT] = WSQC_CLASSIC_KEPT_RGB
            lut[_WSQC_RULE] = WSQC_CLASSIC_RULE_RGB
        else:
            # Tagged overlay: outlines on the tissue
            out = tissue.copy() if tissue is not None else (
                overlay.copy() if overlay is not None else np.zeros((h, w, 3), np.uint8))
            fe = _wsqc_edges(fiber_lbl)
            out[fe] = (out[fe] * (1 - WSQC_FIBER_EDGE) + 205 * WSQC_FIBER_EDGE).astype(np.uint8)
            out[fe & multi[fiber_lbl]] = WSQC_MERGE_RGB   # fiber territory holding 2+ axon regions
        ys, xs = np.nonzero(_wsqc_edges(axon_lbl))
        out[ys, xs] = lut[cls[axon_lbl[ys, xs]]]
        if mode == "combined":
            dots(out)
        if region is not None:
            for poly in getattr(region, "geoms", [region]):
                if poly.is_empty:
                    continue
                for ring, px in [(poly.exterior, WSQC_TRACE_PX)] + [(r, 1) for r in poly.interiors]:
                    pts = np.round(np.asarray(ring.coords)).astype(np.int32)
                    cv2.polylines(out, [pts], True, WSQC_TRACE_RGB, px, cv2.LINE_AA)
        if outside is not None and len(outside):
            for ox, oy in zip(outside["x"].to_numpy(float), outside["y"].to_numpy(float)):
                cv2.drawMarker(out, (int(round(oy)), int(round(ox))),
                               WSQC_OUTSIDE_COMBINED_RGB if mode == "combined" else WSQC_OUTSIDE_RGB,
                               cv2.MARKER_TILTED_CROSS, 14,
                               WSQC_OUTSIDE_PX if mode == "combined" else 2, cv2.LINE_AA)

        if mode == "combined":
            panels = [p for p in (tissue, overlay) if p is not None] + [out]
        else:
            n_axon = int(axon_lbl.max())
            acols = (np.asarray(make_random_cmap(n_axon).colors) * 255).astype(np.uint8)
            panel_axon = np.ascontiguousarray(acols[axon_lbl])
            panel_fiber = fiber_rgb.copy()
            dots(panel_fiber)
            if WSQC_CROPS:
                crops = _wsqc_crop_grid(h, w, tissue if tissue is not None else overlay,
                                        axon_lbl, fiber_lbl, cls, info[k][1], df, BG, lut,
                                        _wsqc_font(20), WSQC_ZOOM, None, fiber_rgb)
                panels = [panel_axon, panel_fiber, crops]
            else:
                panels = [panel_axon, panel_fiber, out]

        n_out = int(len(outside)) if outside is not None else 0
        rk = _wsqc_risk(axon_lbl, fiber_lbl, cls, multi, region)
        pc = lambda a_, b_: f"{100 * a_ / max(b_, 1):.0f}%"
        parts_txt = [f"{rk['measured']} measured"]
        if rk["filtered"]:
            parts_txt.append(f"filtered {pc(rk['filtered'], rk['n_total'])}")
        parts_txt.append(f"hard-rejected {pc(rk['hard'], rk['n_total'])}")
        if rk["merge"]:
            parts_txt.append(f"merge suspect {pc(rk['merge'], rk['measured'])}")
        parts_txt.append(f"lost axons {pc(rk['lost'], rk['n_total'])}")
        parts_txt.append(f"multi-axon fibers {pc(rk['multi'], rk['n_fiber'])}")
        if region is not None and n_out:
            parts_txt.append(f"{n_out} outside tracing")
        tag = f"EXCLUDED — {reason}" if reason else None
        shown = key or img_path.stem
        if shown.lower().endswith(".ome"):
            shown = shown[:-4]
        return panels, "  ·  ".join(parts_txt), tag, shown

    def render(mode):
        if mode == "combined":
            n_cols = (2 if layout else 0) + 1
            title, file_tag = "combined QC", "combined_qc"
            hint = ("tissue  |  segmentation  |  " if layout else "") + "watershed (dot = measured axon)"
        else:
            n_cols = 3
            title, file_tag = "morphometrics QC", "morphometrics_qc"
            hint = ("axon watershed  |  fiber watershed (dot = measured axon)  |  worst cases "
                    f"{WSQC_ZOOM}x: tissue above, fibers below" if WSQC_CROPS else
                    "axon watershed  |  fiber watershed (dot = measured axon)  |  tagged overlay")
        width  = max(n_cols * max_w + (n_cols - 1) * GAP, 1200)
        legend = legend_for(mode)

        for p_idx, (rows, total_h) in enumerate(parts, start=1):
            canvas = np.full((total_h, width, 3), BG, dtype=np.uint8)

            banner = Image.new("RGB", (width, BANNER_H), BG)
            d = ImageDraw.Draw(banner)
            label = f"{nerve_name}{suffix}  |  {title}  |  {mag}"
            if len(parts) > 1:
                label += f"  |  part {p_idx}/{len(parts)}"
            _wsqc_text(d, (16, BANNER_H // 2), label, font_title, (255, 255, 255), "lm")
            hint_x = 16 + int(d.textlength(label, font=font_title) if hasattr(d, "textlength") else 700) + 40
            _wsqc_text(d, (hint_x, BANNER_H // 2), hint, font_small, (150, 150, 160), "lm")
            x = width - 16
            for text, color, style in reversed(legend):
                tw = d.textlength(text, font=font_small) if hasattr(d, "textlength") else 70
                _wsqc_text(d, (x, BANNER_H // 2), text, font_small, (210, 210, 215), "rm")
                x -= int(tw) + 10
                box = [x - 22, BANNER_H // 2 - 11, x, BANNER_H // 2 + 11]
                if style == "x":
                    d.line([box[0], box[1], box[2], box[3]], fill=color, width=2)
                    d.line([box[0], box[3], box[2], box[1]], fill=color, width=2)
                else:
                    d.rectangle(box, outline=color, width=3)
                x -= 22 + 28
            canvas[:BANNER_H] = np.asarray(banner)

            y = BANNER_H
            for k in rows:
                name = seg_images[k].name
                for sfx in (".tif", ".tiff"):
                    if name.lower().endswith(sfx):
                        name = name[: -len(sfx)]
                bar = Image.new("RGB", (width, BAR_H), BAR)
                d = ImageDraw.Draw(bar)
                if info[k] is None:
                    _wsqc_text(d, (16, BAR_H // 2), f"{name}  —  no data", font_title,
                               (255, 120, 120), "lm")
                    canvas[y:y + BAR_H] = np.asarray(bar)
                    y += BAR_H + BLOCK_GAP
                    continue
                panels, counts, tag, name = build_row(k, mode)
                _wsqc_text(d, (16, BAR_H // 2), name, font_title, (255, 255, 255), "lm")
                tx = 16 + int(d.textlength(name, font=font_title) if hasattr(d, "textlength") else 400) + 40
                _wsqc_text(d, (tx, BAR_H // 2), counts, font_small, (190, 190, 200), "lm")
                if tag:
                    cw = d.textlength(counts, font=font_small) if hasattr(d, "textlength") else 500
                    _wsqc_text(d, (tx + int(cw) + 40, BAR_H // 2), tag, font_title, TAG_RGB, "lm")
                _wsqc_text(d, (width - 16, BAR_H // 2), f"{k + 1}/{n_images}", font_small,
                           (190, 190, 200), "rm")
                canvas[y:y + BAR_H] = np.asarray(bar)
                y += BAR_H
                xx = 0
                for panel in panels:
                    ph, pw = panel.shape[:2]
                    canvas[y:y + ph, xx:xx + pw] = panel
                    xx += pw + GAP
                y += panels[0].shape[0] + BLOCK_GAP
                del panels

            sfx = "" if len(parts) == 1 else f"_part{p_idx}"
            out_path = qc_dir / f"{nerve_name}_{file_tag}{suffix}{sfx}.png"
            Image.fromarray(canvas).save(str(out_path), "PNG", compress_level=WSQC_PNG_COMPRESS)
            saved.append(out_path)
            del canvas

    if WSQC_WRITE_MORPH:
        render("morph")
    if WSQC_WRITE_COMBINED:
        render("combined")

    if log:
        for out_path in saved:
            log.success(f"Morphometrics QC saved → QC/{out_path.name}")


# ─── Save output ──────────────────────────────────────────────────────────────

def _autofit_columns(ws, min_width: int = 12, max_width: int = 40) -> None:
    """Auto-fit column widths based on max content length."""
    for col in ws.columns:
        max_len = max(
            len(str(cell.value)) if cell.value is not None else 0
            for cell in col
        )
        ws.column_dimensions[col[0].column_letter].width = min(
            max(max_len + 4, min_width), max_width
        )


def save_morphometrics(
    df:         pd.DataFrame,
    pixel_df:     pd.DataFrame,
    output_dir: str,
    stem:       str,
    provenance: dict = None,
    calibration: Calibration | None = None,
    tracing: dict | None = None,
    outside_df: pd.DataFrame | None = None,
) -> str:
    """
    Save morphometrics to a 5-sheet Excel file.

    Sheet 1 — Summary              :provenance block + one headline row
    Sheet 2 — Axon                 :per-axon axon measurements (µm)
    Sheet 3 — Myelin+G             :per-axon myelin and g-ratio measurements (µm)
    Sheet 4 — Fiber                :per-axon fiber measurements (µm)
    Sheet 5 — Pixel Measurements   :accepted measurements in pixel units
                                     for calibration/sanity checking
    Sheet 6 — Outside tracing      :only when a CSA tracing was applied and
                                     axons fell outside it; those axons are
                                     excluded from sheets 1-5

    tracing is None when no CSA tracing was applied; otherwise a dict with
    n_all (accepted axons before the tracing) and n_outside. df and pixel_df
    must then hold only the axons inside the tracing (morphometrics/tracing.py).

    The pixel-unit sheet contains the same accepted axon-fiber pairs as
    the primary morphometric output. It is not a pre-filter or raw
    segmentation dataset and must not be interpreted as one.

    Returns output path.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment

    config     = load_config()
    suffix     = config.get("morphometrics_suffix", "_morphometrics")
    provenance = provenance or {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}{suffix}.xlsx"

    wb = Workbook()

    # ── Shared styles ─────────────────────────────────────────────────────────
    hdr_font   = Font(bold=True)
    hdr_fill   = PatternFill('solid', fgColor='D9E1F2')
    prov_font  = Font(italic=True, color='666666')

    def write_header_row(ws, columns: list[str]) -> None:
        """Write a bold blue header row and freeze it."""
        for col_idx, col_name in enumerate(columns, 1):
            cell            = ws.cell(row=1, column=col_idx, value=col_name)
            cell.font       = hdr_font
            cell.fill       = hdr_fill
            cell.alignment  = Alignment(horizontal='center')
        ws.freeze_panes = ws['A2']

    def write_df_to_sheet(
        ws,
        frame: pd.DataFrame,
        start_row: int = 1,
    ) -> None:
        """Write DataFrame to sheet with header row."""
        cols = list(frame.columns)
        write_header_row(ws, cols)
        for r_idx, row_data in enumerate(frame.itertuples(index=False), start_row + 1):
            for c_idx, value in enumerate(row_data, 1):
                ws.cell(row=r_idx, column=c_idx, value=value)

    # ── Sheet 1 — Summary ─────────────────────────────────────────────────────
    ws1 = wb.active
    ws1.title = 'Summary'

    axonid_version = (
        provenance.get('AxonID_version')
        or provenance.get('DeepAxon_version')
        or __version__
        or 'unknown'
    )
    
    if calibration is not None:
        px_text = f"{calibration.px_um:.6g}" if calibration.px_um is not None else 'not calibrated (pixel units)'
        cal_rows = [
            ('Pixel size (µm/px)',  px_text),
            ('Original image size', f"{calibration.original_width}x{calibration.original_height}"),
            ('Resize scale',        round(calibration.resize_scale, 6)),
            ('Calibration source',  calibration.source + (f" ({calibration.note})" if calibration.note else "")),
        ]
    else:
        cal_rows = []

    if tracing is None:
        tracing_rows = [('CSA tracing applied', 'no (all accepted axons are included)')]
    else:
        if tracing.get('excluded'):
            tracing_rows = [
                ('CSA tracing applied',            'no (image excluded)'),
                ('Image excluded',                 f"yes: {tracing['excluded']}"),
                ('Accepted axons in excluded image', tracing['n_all']),
            ]
        else:
            tracing_rows = [
                ('CSA tracing applied',            'yes (axons outside the tracing are excluded)'),
                ('Accepted axons (all)',           tracing['n_all']),
                ('Accepted axons outside tracing', tracing['n_outside']),
            ]

    prov_rows = [
        ('AxonID version',     axonid_version),
        ('Model',              provenance.get('model_name',       'unknown')),
        ('Recipe',             provenance.get('recipe_id',        'unknown')),
        ('Architecture',       provenance.get('architecture',     'unknown')),
        ('Encoder',            provenance.get('encoder',          'unknown')),
        ('Segmented date',     provenance.get('segmented_date',   'unknown')),
        ('Patch size',         provenance.get('patch_size',       'unknown')),
        ('Magnification',      provenance.get('magnification',    'unknown')),
        *cal_rows,
        *tracing_rows,
        ('Morphometrics rules', MORPHOMETRICS_ID),
        ('Watershed seeding',  _get_watershed_seeding_text()),
        ('Watershed disk',     MORPHOMETRICS.dilation_disk),
        ('Morphometrics date', pd.Timestamp.now().strftime('%Y-%m-%d')),
    ]

    for r_idx, (row_label, value) in enumerate(prov_rows, 1):
        lbl_cell       = ws1.cell(row=r_idx, column=1, value=row_label)
        lbl_cell.font  = prov_font
        ws1.cell(row=r_idx, column=2, value=value)

    # Blank separator row
    sep_row = len(prov_rows) + 2

    # Headline summary row
    if not df.empty:
        # Same g-ratio as the Data workbooks and _binned.xlsx (primary method).
        gratio_method = config.get('primary_gratio_method', 'equiv_diam')
        summary_headers = [
            'Image', 'Axon Count',
            'Mean Axon Diam (µm)', 'SD Axon Diam (µm)',
            f'Mean G-ratio ({gratio_method})', f'SD G-ratio ({gratio_method})',
            'Mean Myelin Thick (µm)', 'SD Myelin Thick (µm)',
        ]
        for c_idx, h in enumerate(summary_headers, 1):
            cell       = ws1.cell(row=sep_row, column=c_idx, value=h)
            cell.font  = hdr_font
            cell.fill  = hdr_fill
        ws1.freeze_panes = ws1.cell(row=sep_row + 1, column=1)

        if 'axon_diam_um' in df.columns:
            diam_col = 'axon_diam_um'
        elif 'axon_diam_px' in df.columns:
            diam_col = 'axon_diam_px'
        else:
            diam_col = None

        gratio_col = 'gratio_equiv_diam' if gratio_method == 'equiv_diam' else 'gratio_mean_axes'

        if 'myelin_thickness_um' in df.columns:
            myelin_col = 'myelin_thickness_um'
        elif 'myelin_thickness_px' in df.columns:
            myelin_col = 'myelin_thickness_px'
        else:
            myelin_col = None

        summary_values = [
            stem,
            len(df),
            round(df[diam_col].mean(),   3) if diam_col  else 'N/A',
            round(df[diam_col].std(),    3) if diam_col  else 'N/A',
            round(df[gratio_col].mean(), 4) if gratio_col in df.columns else 'N/A',
            round(df[gratio_col].std(),  4) if gratio_col in df.columns else 'N/A',
            round(df[myelin_col].mean(), 3) if myelin_col else 'N/A',
            round(df[myelin_col].std(),  3) if myelin_col else 'N/A',
        ]
        for c_idx, val in enumerate(summary_values, 1):
            ws1.cell(row=sep_row + 1, column=c_idx, value=val)

    _autofit_columns(ws1)

    # ── Sheet 2 — Axon ────────────────────────────────────────────────────────
    ws2 = wb.create_sheet('Axon')
    axon_cols = [c for c in df.columns if any(c.startswith(p) for p in
        ['image', 'resolution', 'magnification', 'original_resolution', 'resize_scale', 'px_size_um',
         'label', 'x', 'y',
         'axon_area', 'axon_diam', 'axon_major', 'axon_minor',
         'axon_solidity', 'axon_deformation', 'axon_eccentricity', 'axon_orientation'])]
    write_df_to_sheet(ws2, df[axon_cols])
    _autofit_columns(ws2)

    # ── Sheet 3 — Myelin + G-ratio ────────────────────────────────────────────
    ws3 = wb.create_sheet('Myelin+G-ratio')
    myelin_cols = [c for c in df.columns if any(c.startswith(p) for p in
        ['image', 'resolution', 'magnification', 'original_resolution', 'resize_scale', 'px_size_um',
         'label', 'myelin', 'fiber', 'gratio'])]
    write_df_to_sheet(ws3, df[myelin_cols])
    _autofit_columns(ws3)

    # ── Sheet 4 — Fiber ───────────────────────────────────────────────────────
    ws4 = wb.create_sheet('Fiber')
    fiber_cols = [c for c in df.columns if any(c.startswith(p) for p in
        ['image', 'resolution', 'magnification', 'original_resolution', 'resize_scale', 'px_size_um',
         'label', 'fiber_area', 'fiber_equiv', 'fiber_major', 'fiber_minor',
         'fiber_deformation', 'fiber_eccentricity', 'fiber_orientation'])]
    write_df_to_sheet(ws4, df[fiber_cols])
    _autofit_columns(ws4)

    # ── Sheet 5 — Pixel Measurements ──────────────────────────────────────────
    # This is the accepted morphometric dataset expressed in pixel units.
    # It is provided for calibration/sanity checking only; it is not raw
    # or pre-filter data.
    ws5 = wb.create_sheet('Pixel Measurements')
    ws5.cell(
        row=1,
        column=1,
        value=(
            'Accepted measurements in pixel units — for calibration/sanity '
            'checking only. Do not use as the primary analysis output.'
        ),
    ).font = prov_font
    if not pixel_df.empty:
        write_df_to_sheet(ws5, pixel_df)
        # Shift data down one row to leave space for the warning label
        ws5.insert_rows(1)
        ws5.cell(
            row=1,
            column=1,
            value=(
                'Accepted measurements in pixel units — for calibration/sanity '
                'checking only. Do not use as the primary analysis output.'
            ),
        ).font = prov_font
    _autofit_columns(ws5)

    # ── Sheet 6 — Outside tracing ─────────────────────────────────────────────
    # Accepted axons that lie outside the CSA tracing. They are excluded from
    # every sheet above and listed here, so aberrant segmentation outside the
    # traced tissue stays visible.
    if outside_df is not None and not outside_df.empty:
        ws6 = wb.create_sheet('Outside tracing')
        outside_cols = [c for c in outside_df.columns if any(c.startswith(p) for p in
            ['image', 'label', 'excluded', 'qupath', 'x', 'y', 'axon_area', 'axon_diam', 'myelin', 'fiber_equiv', 'gratio'])]
        write_df_to_sheet(ws6, outside_df[outside_cols])
        _autofit_columns(ws6)

    wb.save(str(out_path))
    return str(out_path)