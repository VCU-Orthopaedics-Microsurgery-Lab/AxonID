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
        Myelin+G-ratio, Fiber and Pixel Measurements sheets.
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
    filtered_centroids   = []  # (x, y) of filtered pairs — used by QC sheet col 4
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

    def _filtered(x_coord: float, y_coord: float) -> None:
        filtered_centroids.append((x_coord, y_coord))
            
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
                n_filtered += 1; n_filtered_gratio_invalid += 1; _filtered(x, y); continue
            if not (gratio_min < gratio < gratio_max):
                n_filtered += 1; n_filtered_gratio_range += 1; _filtered(x, y); continue
            if axon_solidity < min_axon_solidity:
                n_filtered += 1; n_filtered_solidity += 1; _filtered(x, y); continue
            if axon_eccentricity > max_axon_eccentricity:
                n_filtered += 1; n_filtered_eccentricity += 1; _filtered(x, y); continue

            if px_size is not None:
                axon_area_um2   = axon_area   * px_size ** 2
                fiber_area_um2  = fiber_area  * px_size ** 2
                myelin_thick_um = myelin_thickness * px_size

                if not (min_axon_area_um2 <= axon_area_um2 <= max_axon_area_um2):
                    n_filtered += 1; n_filtered_area += 1; _filtered(x, y); continue
                if not (min_fiber_area_um2 <= fiber_area_um2 <= max_fiber_area_um2):
                    n_filtered += 1; n_filtered_area += 1; _filtered(x, y); continue
                if myelin_thick_um < min_myelin_thick_um:
                    n_filtered += 1; n_filtered_myelin += 1; _filtered(x, y); continue

        # ── Hard validity checks (always applied) ─────────────────────────────
        if px_size is not None:
            min_axon_diam_um = m.min_axon_diam_um
            if (
                min_axon_diam_um > 0
                and (axon_equiv_diam * px_size) < min_axon_diam_um
            ):
                _filtered(x, y)
                continue

        if not (0 < gratio_area < 1 and 0 < gratio_axes < 1):
            _filtered(x, y)
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


def save_watershed_qc(
    seg_images: list,
    all_data: list,
    morph_dir: Path,
    nerve_name: str,
    suffix: str,
    mag: str,
    log: AxonIDLogger = None,
) -> None:
    """
    Generate a watershed QC image for all segmented images in a nerve.

    Saved as {nerve_name}_watershed_qc{suffix}.png in the configured
    QC/ folder.

    Layout: one row per image.
        Always:
            Col 1: Axon watershed with randomized region colors
            Col 2: Fiber watershed with randomized region colors
            Col 3: Fiber regions with axon boundaries and matched centroids

        Quality filters ON:
            Col 4: Filter map — green=kept, red=filtered,
                   black=background

        Quality filters OFF:
            Col 4 is omitted.

    Args:
        seg_images: Segmented image paths, in the same order as all_data.
        all_data: Per-image tuples of
            (df, axon_label, fiber_label, filtered_centroids).
        morph_dir: Path to the corresponding Morphometrics directory.
                   The QC image is saved to its sibling QC directory.
        nerve_name: Nerve name used in the output filename and title.
        suffix: Model/output suffix, e.g. '_M1'.
        mag: Magnification string.
        log: Optional AxonIDLogger.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        if log:
            log.warn("matplotlib not available — skipping watershed QC sheet")
        return

    config    = load_config()
    qc_folder = config.get("qc_folder", "QC")
    qc_dir    = morph_dir.parent / qc_folder
    qc_dir.mkdir(parents=True, exist_ok=True)

    n_images = len(seg_images)
    if n_images == 0:
        return

    qf_on  = MORPHOMETRICS.quality_filters_enabled
    n_cols = 4 if qf_on else 3
    fig_w  = 24
    row_h  = fig_w / n_cols

    fig, axes = plt.subplots(
        n_images, n_cols,
        figsize=(fig_w, row_h * n_images),
        squeeze=False
    )

    col_titles = [
        "Axon watershed",
        "Fiber watershed",
        "Matched axons (grey overlay + red dots)",
    ]
    if qf_on:
        col_titles.append("Filter map (green=kept  red=filtered)")

    for col, title in enumerate(col_titles):
        axes[0][col].set_title(
            title,
            fontsize=8,
            fontweight='bold',
            pad=6,
        )

    seeding_text = _get_watershed_seeding_text()
    disk_r = MORPHOMETRICS.dilation_disk

    for row_idx, (img_path, data) in enumerate(zip(seg_images, all_data)):
        df, axon_lbl, fiber_lbl, filtered_xy = data

        ax_axon  = axes[row_idx][0]
        ax_fiber = axes[row_idx][1]
        ax_over  = axes[row_idx][2]
        ax_filt  = axes[row_idx][3] if qf_on else None

        n_kept     = len(df) if df is not None and not df.empty else 0
        n_filtered = len(filtered_xy)
        row_label  = (
            f"{img_path.name}  —  {n_kept} kept  |  {n_filtered} filtered"
            if qf_on else
            f"{img_path.name}  —  {n_kept} axons"
        )
        ax_over.set_title(row_label, fontsize=7, fontweight='normal', pad=4) 

        if axon_lbl is None or fiber_lbl is None:
            no_data_axes = [ax_axon, ax_fiber, ax_over]
            if ax_filt is not None:
                no_data_axes.append(ax_filt)
            for ax in no_data_axes:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10)
                ax.axis('off')
            continue

        # ── Col 1: Axon watershed ─────────────────────────────────────────────
        n_axon = axon_lbl.max()
        ax_axon.imshow(axon_lbl, cmap=make_random_cmap(n_axon),
                       interpolation='nearest')
        ax_axon.set_title(f"Axon watershed  ({n_axon} regions)", fontsize=7)
        ax_axon.axis('off')

        # ── Col 2: Fiber watershed ────────────────────────────────────────────
        n_fiber = fiber_lbl.max()
        ax_fiber.imshow(fiber_lbl, cmap=make_random_cmap(n_fiber, seed=99),
                        interpolation='nearest')
        ax_fiber.set_title(f"Fiber watershed  ({n_fiber} regions)", fontsize=7)
        ax_fiber.axis('off')

        # ── Col 3: Fiber colors + red axon outlines + red matched centroids (image label above) ──────────
        ax_over.imshow(fiber_lbl, cmap=make_random_cmap(n_fiber, seed=99),
                       interpolation='nearest', alpha=0.85)

        axon_boundary = np.zeros((*axon_lbl.shape, 4), dtype=np.float32)        
        binary_axon   = (axon_lbl > 0).astype(np.uint8)                            
        contours, _   = cv2.findContours(                                           
            binary_axon, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE                
        )                                                                           
        contour_mask  = np.zeros(axon_lbl.shape, dtype=np.uint8)                   
        cv2.drawContours(contour_mask, contours, -1, 1, thickness=1)               
        axon_boundary[contour_mask == 1] = [0.9, 0.15, 0.15, 0.9]                 
        ax_over.imshow(axon_boundary, interpolation='nearest')   

        if df is not None and not df.empty and 'x' in df.columns and 'y' in df.columns:
            ax_over.scatter(df['y'].values, df['x'].values,
                            s=6, c='red', marker='.', linewidths=0)
        ax_over.axis('off')

        # ── Col 4: Filter map (qf ON only) ────────────────────────────────────
        if qf_on and ax_filt is not None:
            filter_rgb = np.zeros((*axon_lbl.shape, 3), dtype=np.float32)

            if df is not None and not df.empty and 'x' in df.columns and 'y' in df.columns:
                kept_mask = np.zeros(axon_lbl.shape, dtype=bool)
                for _, r in df.iterrows():
                    xi, yi = int(round(r['x'])), int(round(r['y']))
                    if 0 <= xi < axon_lbl.shape[0] and 0 <= yi < axon_lbl.shape[1]:
                        lbl_id = axon_lbl[xi, yi]
                        if lbl_id > 0:
                            kept_mask |= (axon_lbl == lbl_id)
                filter_rgb[kept_mask] = [0.2, 0.8, 0.2]

            if filtered_xy:
                for (fx, fy) in filtered_xy:
                    xi, yi = int(round(fx)), int(round(fy))
                    if 0 <= xi < axon_lbl.shape[0] and 0 <= yi < axon_lbl.shape[1]:
                        lbl_id = axon_lbl[xi, yi]
                        if lbl_id > 0:
                            filter_rgb[axon_lbl == lbl_id] = [0.9, 0.15, 0.15]

            ax_filt.imshow(filter_rgb, interpolation='nearest')
            ax_filt.axis('off')

    qf_label = "filters ON" if qf_on else "filters OFF"

    fig.suptitle(
        f"Watershed QC — {nerve_name}{suffix}  |  {mag}  |  "
        f"seeding={seeding_text}  disk={disk_r}  |  {qf_label}",
        fontsize=10,
        y=1.001,
    )
    plt.tight_layout()

    out_path = qc_dir / f"{nerve_name}_watershed_qc{suffix}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)

    if log:
        log.success(f"Watershed QC saved → QC/{nerve_name}_watershed_qc{suffix}.png")


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
) -> str:
    """
    Save morphometrics to a 5-sheet Excel file.

    Sheet 1 — Summary              : provenance block + one headline row
    Sheet 2 — Axon                 : per-axon axon measurements (µm)
    Sheet 3 — Myelin+G            : per-axon myelin and g-ratio measurements (µm)
    Sheet 4 — Fiber               : per-axon fiber measurements (µm)
    Sheet 5 — Pixel Measurements  : accepted measurements in pixel units
                                    for calibration/sanity checking

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

    wb.save(str(out_path))
    return str(out_path)