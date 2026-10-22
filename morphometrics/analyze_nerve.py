"""
morphometrics/analyze_nerve.py: nerve-level summary for batch_axon.

Purpose
    Combines a nerve's per-image morphometric workbooks with its QuPath CSA
    tracings into image rows and nerve totals: CSA, axon count, density,
    g-ratio, axon diameter and estimated whole-nerve axon count.

Usage
    from morphometrics.analyze_nerve import get_nerve_data
    image_rows, aggregate, bins = get_nerve_data(nerve_dir, mag, log)

Inputs
    Morphometrics/{image}_morphometrics.xlsx   (morphometrics/morphometrics.py)
    CSA/{image}_CSA.geojson                     QuPath tracing of each 40X image
    4X_TIFF/{nerve}_4X_*.tif (any case, .ome optional) + CSA/{image}_CSA.geojson
                                                whole-nerve tracing(s) at 4X

Outputs
    image rows: name, resolution, CSA (µm²), axon count, g-ratio, axon
                diameter, density
    aggregate:  4X whole-nerve CSA, totals and means, estimated total axons
                (= sampled axons / sampled CSA x whole-nerve CSA, 40X only,
                and only when every image has a CSA)

Calibration
    QuPath traces the ORIGINAL image, so CSA is converted with
    pixel_size_um at the original width (from the workbook's
    original_resolution column). Workbooks from older versions have no such
    column; their segmented width is used instead.

Names
    Workbooks derived from .ome.tif images carry a stray ".ome" (Path.stem
    strips one extension); _strip_ome() removes it for CSA lookup only.

Settings (config.toml)
    primary_gratio_method, pixel_size_um, csa_folder, csa_suffix_geojson

Depends on
    batch_axon/overlay/process_overlay_qupath.py, morphometrics/distributions.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import pandas as pd
import tifffile

# ── Local ─────────────────────────────────────────────────────────────────────
from batch_axon.overlay.process_overlay_qupath import get_geojson_area
from morphometrics.distributions import bin_nerve_diameters
from utils.config import get_pixel_size, load_config
from utils.logger import AxonIDLogger
from utils.study import _strip_suffix


def make_aggregate_df(
    morph_dir: Path,
    morph_suffix: str,
    log: AxonIDLogger,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Load every per-image morphometric workbook in morph_dir and merge
    each one's Axon and Myelin+G-ratio sheets into a single per-image
    DataFrame — gratio and fiber columns live on Myelin+G-ratio, not
    Axon (see morphometrics.py save_morphometrics()), so both sheets
    are needed together.

    Columns already present on the Axon sheet are dropped from
    Myelin+G-ratio before merging (except "label", the join key) to
    avoid pandas creating duplicate "_x"/"_y" suffixed columns.
    validate="one_to_one" raises if a label isn't unique on either
    side — caught by the caller's try/except, which skips that one
    file with a warning rather than silently merging corrupted rows.

    Returns:
        (concatenated DataFrame across all images, {file_stem: per-image DataFrame})
        Both empty if no workbooks were found or all failed to read.
        The dict preserves sorted-glob insertion order — callers iterate it
        directly rather than re-globbing, so this order is relied upon, not
        incidental.
    """
    dfs = []
    per_file = {}

    for f in sorted(morph_dir.glob(f"*{morph_suffix}.xlsx")):
        try:
            df = pd.read_excel(
                str(f),
                sheet_name="Axon",
            )
            if df.empty:
                # No axons in this image (empty Axon sheet). Keep it as a
                # measured 0 rather than skipping it: skipping drops its CSA
                # from the nerve total and inflates density. Its image size
                # comes from the Summary sheet (the Axon rows normally carry it).
                df = pd.DataFrame()
                df.attrs["summary"] = _read_summary(f)
                per_file[f.stem] = df
                continue
            # Axon sheet has no gratio/fiber columns — those live on
            # Myelin+G-ratio (see morphometrics.py save_morphometrics()).
            # Merge them in by label, same approach as
            # distributions.bin_nerve_diameters(), so gratio_col and
            # fiber_equiv_diam_um are available to the aggregation below.
            df_myelin = pd.read_excel(
                str(f),
                sheet_name="Myelin+G-ratio",
            )
            df = df.merge(
                df_myelin.drop(columns=[
                    c for c in df_myelin.columns
                    if c in df.columns and c != "label"
                ]),
                on="label", how="left", validate="one_to_one",
            )
            dfs.append(df)
            per_file[f.stem] = df

        except Exception as e:
            log.warn(
                f"Could not read {f.name}: {e}"
            )

    if not per_file:
        return pd.DataFrame(), {}

    if not dfs:                       # every image had zero axons
        return pd.DataFrame(), per_file

    return pd.concat(
        dfs,
        ignore_index=True,
    ), per_file


def _read_summary(path: Path) -> dict:
    """Key/value block at the top of a morphometrics workbook's Summary sheet."""
    try:
        s = pd.read_excel(str(path), sheet_name="Summary", header=None, usecols=[0, 1])
    except Exception:
        return {}
    out = {}
    for key, value in zip(s[0], s[1]):
        if pd.isna(key):              # blank row ends the block
            break
        out[str(key)] = value
    return out


def _widths_from_summary(summary: dict) -> tuple[int | None, int | None]:
    """(segmented width, original width) from Summary 'Original image size' and 'Resize scale'."""
    try:
        orig = int(str(summary["Original image size"]).split("x")[0])
    except (KeyError, ValueError, TypeError):
        return None, None
    try:
        scale = float(summary.get("Resize scale", 1))
    except (TypeError, ValueError):
        scale = 1.0
    return int(round(orig * scale)), orig


def get_image_csa(
    img_name: str,
    csa_dir: Path,
    mag: str,
    img_width: int,
    log: AxonIDLogger,
    quiet: bool = False,
) -> float | None:
    """
    Cross-sectional area (µm²) of one image from its QuPath GeoJSON
    ({img_name}_CSA.geojson). img_width is the ORIGINAL image width, the
    pixel grid QuPath traced on, so pixel_size_um applies directly.

    For 100X: per-image CSA not applicable, returns None.
    For unknown mag: returns None.
    
    quiet: suppress the per-image warning. get_nerve_data() sets this
    and aggregates missing-CSA images into one summary line instead —
    N identical per-image warnings make a systematic problem (missing
    export, naming mismatch) look like routine per-image noise.
    """
    config         = load_config()
    geojson_suffix = config.get("csa_suffix_geojson", "_CSA.geojson")
    px_size        = get_pixel_size(mag, img_width)

    if mag == "40X":
        geojson_file = csa_dir / f"{img_name}{geojson_suffix}"
        if not geojson_file.exists():
            if not quiet:
                log.warn(f"No CSA file found ({geojson_file.name}): {img_name}")
            return None
        area_px = get_geojson_area(str(geojson_file), log=log)

        if area_px is None:
            if not quiet:
                log.warn(f"Could not extract overlay area for {img_name}")
            return None

        if px_size is not None:
            return area_px * (px_size ** 2)

        if not quiet:
            log.warn(
                f"No pixel-size calibration for {mag} at width "
                f"{img_width}px — cannot compute CSA in µm²: {img_name}"
            )
        return None
    # 100X fields do not have per-image CSA GeoJSON files. They are treated
    # as sampled endoneurial/axonal tissue, so image-level CSA is not derived
    # from an overlay. The eventual 100X aggregation workflow should instead
    # use the 4X whole-nerve CSA for scaling the sampled 100X axon count.
    elif mag == "100X":
        return None

    else:
        log.warn(f"Unknown magnification '{mag}' — cannot compute CSA")
        return None


def get_nerve_fourx_csa(
    nerve_name: str,
    nerve_dir: Path,
    csa_dir: Path,
    log: AxonIDLogger
) -> float | None:
    """
    Get the whole-nerve CSA from all 4X images.

    Each 4X image is matched to its QuPath GeoJSON by image stem.

    The actual width of each 4X image is used to obtain its configured
    pixel-size calibration.

    Handles single or multiple 4X images for large nerves.
    """
    config = load_config()
    geojson_suffix = config.get("csa_suffix_geojson", "_CSA.geojson")

    fourx_dir = _find_fourx_dir(nerve_dir, log)
    if fourx_dir is None:
        log.warn(f"No 4X image folder (e.g. 4X_TIFF) found for nerve: {nerve_name}")
        return None

    # <nerve>_4X_<anything>.tif / .tiff / .ome.tif / .ome.tiff, any case.
    # Each file on disk is visited once, so nothing can be counted twice.
    prefix = f"{nerve_name}_4x_".lower()
    fourx_images = sorted(
        p for p in fourx_dir.iterdir()
        if p.is_file()
        and p.name.lower().startswith(prefix)
        and p.suffix.lower() in (".tif", ".tiff")
    )

    if not fourx_images:
        log.warn(f"No 4X image(s) named {nerve_name}_4X_*.tif found in {fourx_dir.name} for nerve: {nerve_name}")
        return None

    total_csa_um2 = 0.0
    valid_csa_count = 0

    for image_path in fourx_images:
        image_stem = _strip_ome(image_path.stem)   # "X.ome.tif" and "X.tif" both give "X"

        csa_file = csa_dir / f"{image_stem}{geojson_suffix}"
        if not csa_file.exists():
            log.warn(f"No CSA file found for 4X image: {image_path.name} (expected {csa_file.name})")
            continue

        with tifffile.TiffFile(str(image_path)) as tif:
            series      = tif.series[0]
            image_shape = series.shape
            image_axes  = getattr(series, "axes", "")

        if len(image_shape) < 2:
            log.warn(
                f"Could not determine 2D dimensions for 4X image: "
                f"{image_path.name}"
            )
            continue

        # shape[-1] is only the width for a plain (Y, X) array. A
        # channel-last (Y, X, C) OME-TIFF has shape[-1] == channel count
        # — which is exactly what produced "image width 3" here. Use
        # tifffile's own axes string to locate X when available; fall
        # back to the old assumption only if axes info is missing.
        if "X" in image_axes:
            image_width = image_shape[image_axes.index("X")]
        else:
            image_width = image_shape[-1]

        px_size = get_pixel_size("4X", image_width)

        if px_size is None:
            log.warn(
                f"No 4X pixel-size calibration configured for image width "
                f"{image_width}: {image_path.name}"
            )
            continue

        area_px = get_geojson_area(str(csa_file), log=log)

        if area_px is None:
            log.warn(
                f"Could not extract area from {csa_file.name} — skipping"
            )
            continue

        csa_um2 = area_px * (px_size ** 2)

        log.info(
            f"  {image_path.name}: "
            f"{image_width}px wide, "
            f"{px_size:.3f} µm/px, "
            f"{csa_file.name}: "
            f"{area_px:,.0f} px² → {csa_um2:,.1f} µm²"
        )

        total_csa_um2 += csa_um2
        valid_csa_count += 1

    if valid_csa_count == 0:
        return None

    if valid_csa_count != len(fourx_images):
        log.warn(
            f"Incomplete 4X CSA coverage for {nerve_name}: "
            f"{valid_csa_count} of {len(fourx_images)} images have valid CSA. "
            f"Whole-nerve CSA will not be reported."
        )
        return None

    return total_csa_um2


def _find_fourx_dir(nerve_dir: Path, log: AxonIDLogger) -> Path | None:
    """
    The nerve's 4X image folder: a subfolder named 4X<suffix> for any
    config.toml tiff_suffixes entry, compared case-insensitively (the same
    rule utils/study.py uses for the 40X folders).
    """
    suffixes = [s.upper() for s in load_config().get("tiff_suffixes", ["_TIFF"])]
    matches = sorted(
        d for d in nerve_dir.iterdir()
        if d.is_dir() and any(
            d.name.upper().endswith(sfx) and _strip_suffix(d.name.upper(), sfx) == "4X"
            for sfx in suffixes
        )
    )
    if len(matches) > 1:
        log.warn(f"Several 4X image folders in {nerve_dir.name} ({[d.name for d in matches]}); using {matches[0].name}")
    return matches[0] if matches else None


def _strip_ome(name: str) -> str:
    """
    Path.stem only strips one extension level, so an OME-TIFF's compound
    .ome.tif/.ome.tiff extension leaves a stray ".ome" on every name
    derived from it downstream (segmented TIFF stem, morphometrics
    workbook stem, and therefore img_name below). CSA overlay files
    (from QuPath, read by process_overlay_qupath.py) are named
    from the image's true base name, without ".ome" — this mismatch is
    what makes get_image_csa() report every 40X image as having no CSA
    file, even when one exists. Normalizes the name used for lookup and
    display only; does not rename anything on disk.
    """
    return name[:-4] if name.lower().endswith(".ome") else name


def _width_from(img_df: pd.DataFrame, column: str) -> int | None:
    """Width from a 'WxH' column of a workbook sheet; None if absent or unreadable."""
    if column not in img_df.columns or img_df.empty:
        return None
    try:
        return int(str(img_df[column].iloc[0]).split('x')[0])
    except (ValueError, TypeError, IndexError):
        return None


def _px_size_from(img_df: pd.DataFrame, mag: str, img_width: int | None) -> float | None:
    """µm per segmented pixel: the workbook's own px_size_um, else a lookup (older workbooks)."""
    if 'px_size_um' in img_df.columns and not img_df.empty and pd.notna(img_df['px_size_um'].iloc[0]):
        return float(img_df['px_size_um'].iloc[0])
    return get_pixel_size(mag, img_width) if img_width is not None else None


def get_nerve_data(
    nerve_dir: Path,
    mag: str,
    log: AxonIDLogger
) -> tuple[list[dict], dict, pd.DataFrame | None]:
    """
    Compile per-image and aggregate data for a single nerve.

    Returns:
        (image_rows, aggregate, bins_df)
    """
    config        = load_config()
    morph_folder  = config.get("morphometrics_folder", "Morphometrics")
    morph_suffix  = config.get("morphometrics_suffix", "_morphometrics")

    gratio_method = config.get('primary_gratio_method', 'equiv_diam')
    gratio_col = (
        'gratio_equiv_diam'
        if gratio_method == 'equiv_diam'
        else 'gratio_mean_axes'
    )
    morph_dir     = nerve_dir / morph_folder
    csa_dir       = nerve_dir / config.get("csa_folder", "CSA")

    if not morph_dir.exists():
        log.warn(f"No morphometrics folder found: {morph_dir}")
        return [], {}, None

    agg_df, per_file = make_aggregate_df(
        morph_dir,
        morph_suffix,
        log,
    )
    if not per_file:
        log.warn(
            f"No morphometrics workbooks found in {morph_dir}"
        )
        return [], {}, None

    bins_df = bin_nerve_diameters(morph_dir, nerve_dir.name, mag, log)

    # Whole-nerve 4X CSA is collected independently of image-level CSA.
    # For 40X, it is used to scale the sampled 40X tissue to an estimated
    # whole-nerve axon count. The future 100X workflow should use this same
    # whole-nerve CSA for scaling 100X axon counts, even though 100X images
    # do not have individual CSA GeoJSON files and are assumed to represent
    # fully endoneurial/axonal tissue.
    fourx_csa = None
    if csa_dir.exists():
        fourx_csa = get_nerve_fourx_csa(
            nerve_dir.name,
            nerve_dir,
            csa_dir,
            log
        )

    image_rows = []
    img_count  = 0
    images_missing_csa = []

    for stem, img_df in per_file.items():
        img_name = _strip_ome(stem.removesuffix(morph_suffix))

        if img_df.empty and "summary" in img_df.attrs:              # image with no axons
            img_width, orig_width = _widths_from_summary(img_df.attrs["summary"])
        else:
            img_width = _width_from(img_df, 'resolution')           # segmented image
            orig_width = _width_from(img_df, 'original_resolution')  # image QuPath traced
        if img_width is None:
            log.warn(f"Could not parse image resolution: {stem}.xlsx")
        if orig_width is None:
            orig_width = img_width   # workbook from an older version: no resize record

        img_csa = None
        if csa_dir.exists() and orig_width is not None:
            img_csa = get_image_csa(
                img_name, csa_dir, mag, orig_width, log, quiet=True
            )
            if img_csa is None:
                images_missing_csa.append(img_name)

        px_size = _px_size_from(img_df, mag, img_width)

        row = {
            'name':        img_name,
            'resolution': (
                f"{img_width}px" if img_width is not None else None
            ),
            'csa_um2':     img_csa,
            'total_axons': len(img_df),
            'gratio':      (
                img_df[gratio_col].mean() if gratio_col in img_df.columns
                else img_df['gratio'].mean() if 'gratio' in img_df.columns
                else None
            ),
            'axon_diam_um': (
                img_df['axon_diam_um'].mean() if 'axon_diam_um' in img_df.columns else
                img_df['axon_diam_px'].mean() * px_size if px_size and 'axon_diam_px' in img_df.columns else
                None
            ),
        }

        # Image-level density currently applies only to 40X because 40X
        # images have explicit tissue CSA overlays. 100X images intentionally
        # leave image-level CSA and density unset; their future total-axon
        # calculation will use the 4X whole-nerve CSA instead.
        row['axon_density_per_um2'] = (
            len(img_df) / img_csa
            if mag == "40X" and img_csa is not None and img_csa > 0
            else None
        )

        image_rows.append(row)
        img_count += 1
        
    if images_missing_csa:
        if len(images_missing_csa) == img_count:
            log.warn(
                f"CSA missing for all {img_count}/{img_count} images — "
                f"check that CSA overlays exist for this nerve"
            )
        else:
            shown = ', '.join(images_missing_csa[:5])
            more  = (f", +{len(images_missing_csa) - 5} more"
                     if len(images_missing_csa) > 5 else "")
            log.warn(
                f"CSA missing for {len(images_missing_csa)}/{img_count} "
                f"image(s): {shown}{more}"
            )

    aggregate = {
        'fourx_csa_um2': fourx_csa,
        'total_images':  img_count,
        'total_axons':   len(agg_df),
        'mean_gratio':   (
            agg_df[gratio_col].mean() if gratio_col in agg_df.columns
            else agg_df['gratio'].mean() if 'gratio' in agg_df.columns
            else None
        ),
    }

    if 'axon_diam_um' in agg_df.columns:
        aggregate['mean_axon_diam_um'] = agg_df['axon_diam_um'].mean()
    elif 'axon_diam_px' in agg_df.columns:
        diameter_values_um = []

        for stem, img_df in per_file.items():
            if 'axon_diam_px' not in img_df.columns:
                continue

            img_width = _width_from(img_df, 'resolution')
            if img_width is None:
                log.warn(f"No image resolution for pixel-unit diameters: {stem}.xlsx")
                continue

            px_size = _px_size_from(img_df, mag, img_width)
            if px_size is None:
                log.warn(
                    f"No pixel-size calibration for {mag} at width "
                    f"{img_width}px: {stem}.xlsx"
                )
                continue

            diameter_values_um.extend(
                img_df['axon_diam_px'].dropna() * px_size
            )

        aggregate['mean_axon_diam_um'] = (
            sum(diameter_values_um) / len(diameter_values_um)
            if diameter_values_um
            else None
        )
    else:
        aggregate['mean_axon_diam_um'] = None

    valid_csas = [
        r['csa_um2']
        for r in image_rows
        if r.get('csa_um2') is not None and r['csa_um2'] > 0
    ]
    aggregate['total_sample_csa_um2'] = (
        sum(valid_csas) if valid_csas else None
    )

    csa_complete = (
        len(image_rows) > 0
        and len(valid_csas) == len(image_rows)
    )

    sample_csa = aggregate.get('total_sample_csa_um2')
    
    # Current implementation estimates whole-nerve axon count only from
    # 40X fields, where each sampled image has an explicit CSA overlay.
    # Future 100X support should use the 4X whole-nerve CSA as the scaling
    # reference while treating the 100X sampled field as endoneurial/axonal
    # tissue rather than requiring a per-image CSA overlay.
    if (
        mag == "40X"
        and csa_complete
        and fourx_csa is not None
        and fourx_csa > 0
        and sample_csa is not None
        and sample_csa > 0
    ):
        aggregate['estimated_total_axons'] = int(
            aggregate['total_axons'] / sample_csa * fourx_csa
        )
    else:
        aggregate['estimated_total_axons'] = None

    return image_rows, aggregate, bins_df