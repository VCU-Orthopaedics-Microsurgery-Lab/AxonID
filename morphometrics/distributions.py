"""
morphometrics/distributions.py: nerve-level fiber-diameter distributions.

Purpose
    Pools the per-image morphometric workbooks of one nerve and bins fiber
    diameters at three resolutions, with g-ratio mean and SD per bin.

Usage
    from morphometrics.distributions import bin_nerve_diameters, save_distributions
    data = bin_nerve_diameters(morph_dir, nerve_name, mag, log, model_name)
    save_distributions(data, morph_dir, nerve_name)

Inputs
    {image}_morphometrics.xlsx workbooks (Axon and Myelin+G-ratio sheets,
    joined one-to-one by label). Calibrated fiber_equiv_diam_um is
    required; uncalibrated nerves are skipped with a warning.

Outputs
    {nerve_name}_binned.xlsx: Summary (nerve metadata and per-image table),
    Granular, Mid and Coarse distribution sheets.

Behaviour
    Bins are [left, right); the last bin is open-ended (999 in config.toml
    means "and above"). The g-ratio column follows config.toml
    primary_gratio_method, as in analyze_nerve.py.

Settings (config.toml)
    morphometrics_bins (reporting only; defaults in utils/config.py),
    primary_gratio_method

Depends on
    pandas, openpyxl, utils/recipe.py (watershed values for the summary)
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Local ─────────────────────────────────────────────────────────────────────
from utils.config import (
     DEFAULT_GRANULAR_BINS_UM,
     DEFAULT_MID_BINS_UM,
     DEFAULT_COARSE_BINS_UM,
    load_config,
 )
from utils.logger import AxonIDLogger
from utils.recipe import MORPHOMETRICS, MORPHOMETRICS_ID
from utils.version import __version__


def image_exclusion_reason(workbook_path) -> str | None:
    """
    Why an image was excluded, from its workbook's Summary sheet ("Image
    excluded: yes: <reason>"), or None when it was not (or the workbook has
    no such row, as in older versions).

    Args:
        workbook_path: {image}_morphometrics.xlsx
    """
    try:
        s = pd.read_excel(str(workbook_path), sheet_name="Summary", header=None, usecols=[0, 1])
    except Exception:
        return None
    for key, value in zip(s[0], s[1]):
        if pd.isna(key):                    # blank row ends the key/value block
            break
        if str(key) == "Image excluded" and str(value).startswith("yes"):
            return str(value).partition(":")[2].strip() or "no reason recorded"
    return None


def bin_nerve_diameters(
    morph_dir: Path,
    nerve_name: str,
    mag: str,
    log: AxonIDLogger,
    model_name: str = '',
) -> dict | None:
    """
    Pool axon data across all per-image morphometrics files for a nerve.
    Computes global summary, per-image summary, and diameter distribution.

    Reads bin edges from config.toml morphometrics_bins.
    Requires fiber_equiv_diam_um column — returns None if uncalibrated.

    Args:
        morph_dir:   Path to nerve Morphometrics/ folder
        nerve_name:  Name of the nerve (for metadata)
        mag:         Magnification string e.g. '40X'
        log:         Logger instance

    Returns:
        dict with keys: 'global_df', 'per_image_df', 'bins_granular_df', 'bins_mid_df', 'bins_coarse_df'
        Or None if no data found or pixel size uncalibrated.
    """
    config   = load_config()
    bins_cfg = config.get("morphometrics_bins", {})

    morph_dir = Path(morph_dir)
    if not morph_dir.exists():
        log.warn(f"Morphometrics folder not found: {morph_dir}")
        return None

    # ── Load and pool all morphometrics files ─────────────────────────────────
    morph_suffix = config.get("morphometrics_suffix", "_morphometrics")
    dfs          = {}
    for f in sorted(morph_dir.glob(f"*{morph_suffix}.xlsx")):
        if image_exclusion_reason(f):       # image tagged EXCLUDE: in no distribution
            continue
        try:
            df = pd.read_excel(str(f), sheet_name='Axon')
            if df.empty:
                # No axons in this image: keep it in the per-image table as 0.
                dfs[f.stem.removesuffix(morph_suffix)] = pd.DataFrame()
                continue
            # Also read Myelin+G-ratio sheet for fiber_equiv_diam_um and gratio
            df_myelin = pd.read_excel(str(f), sheet_name='Myelin+G-ratio')
            # Join Myelin+G-ratio measurements to the Axon rows by label.
            # A one-to-one merge is required so each axon receives at most one
            # corresponding fiber-diameter / g-ratio record.
            df = df.merge(
                df_myelin.drop(columns=[c for c in df_myelin.columns if c in df.columns and c != 'label']),
                on='label', how='left', validate='one_to_one',
            )
            dfs[f.stem.removesuffix(morph_suffix)] = df
        except Exception as e:
            log.warn(f"Could not read {f.name}: {e}")

    if not dfs:
        log.warn(f"No morphometrics files found in {morph_dir}")
        return None

    measured = [d for d in dfs.values() if not d.empty]
    if not measured:
        log.warn(f"  No axons in any image in {morph_dir}")
        return None
    agg_df = pd.concat(measured, ignore_index=True)
    log.info(f"  Pooled {len(agg_df)} axons from {len(dfs)} image(s)")

    # ── Require physical unit columns ─────────────────────────────────────────
    if 'fiber_equiv_diam_um' not in agg_df.columns or not agg_df['fiber_equiv_diam_um'].notna().any():
        log.warn(
            f"  fiber_equiv_diam_um not found — pixel size may not be calibrated for {mag}. "
            f"Distributions require physical unit calibration in config.toml."
        )
        return None

    config_gratio = config.get('primary_gratio_method', 'equiv_diam')
    gratio_col = (
        'gratio_equiv_diam'
        if config_gratio == 'equiv_diam'
        else 'gratio_mean_axes'
    )
    
    if gratio_col not in agg_df.columns:
        log.warn(
            f"  Primary g-ratio column '{gratio_col}' not found — "
            f"g-ratio statistics will be unavailable."
        )

    # ── Global summary ────────────────────────────────────────────────────────
    total_axons      = len(agg_df)
    mean_fiber_diam = (
        round(agg_df['fiber_equiv_diam_um'].mean(), 3)
        if 'fiber_equiv_diam_um' in agg_df.columns
        else None
    )
    mean_gratio = (
        round(agg_df[gratio_col].mean(), 6)
        if gratio_col in agg_df.columns
        else None
    )
    mean_axon_diam = (
        round(agg_df['axon_diam_um'].mean(), 3)
        if 'axon_diam_um' in agg_df.columns
        else None
    )

    global_df = pd.DataFrame([
        ['Nerve',                    nerve_name],
        ['Magnification',            mag],
        ['AxonID version',           __version__],
        ['Model',                    model_name or 'unknown'],
        ['Morphometrics rules',      MORPHOMETRICS_ID],
        ['Watershed threshold',      MORPHOMETRICS.distance_threshold],
        ['Watershed disk',           MORPHOMETRICS.dilation_disk],
        ['Primary g-ratio method',   config_gratio],
        ['Total axons',              total_axons],
        ['Mean fiber diameter (µm)', mean_fiber_diam if mean_fiber_diam is not None else 'N/A'],
        ['Mean axon diameter (µm)',  mean_axon_diam  if mean_axon_diam  is not None else 'N/A'],
        [f'Mean g-ratio ({config_gratio})',
                                    mean_gratio     if mean_gratio     is not None else 'N/A'],
    ], columns=['Metric', 'Value'])

    # ── Per-image summary ─────────────────────────────────────────────────────
    per_image_rows = []
    for img_name, img_df in dfs.items():
        if gratio_col in img_df.columns:
            img_df[gratio_col] = pd.to_numeric(
                img_df[gratio_col],
                errors='coerce'
            )

        per_image_rows.append({
            'Image': img_name,
            'Axon Count': len(img_df),
            f'Mean G-ratio ({config_gratio})': (
                round(img_df[gratio_col].dropna().mean(), 6)
                if gratio_col in img_df.columns
                else None
            ),
            'Mean Fiber Diam (µm)': (
                round(img_df['fiber_equiv_diam_um'].dropna().mean(), 3)
                if 'fiber_equiv_diam_um' in img_df.columns
                else None
            ),
        })
    per_image_df = pd.DataFrame(per_image_rows)

    # ── Diameter distribution — three-tier ───────────────────────────────────
    diameters   = agg_df['fiber_equiv_diam_um'].dropna()
    total_distribution_axons = len(diameters)

    if total_distribution_axons == 0:
        log.warn("  No valid fiber diameters found")
        return None

    granular_edges = bins_cfg.get("granular_um", DEFAULT_GRANULAR_BINS_UM)
    mid_edges      = bins_cfg.get("mid_um",      DEFAULT_MID_BINS_UM)
    coarse_edges   = bins_cfg.get("coarse_um",   DEFAULT_COARSE_BINS_UM)
    
    
    def make_bins_df(edges: list) -> pd.DataFrame:
        """
        Bin diameters by edges, compute count/percent/cumulative and
        g-ratio mean+SD per bin. Last bin is open-ended catch (≥ lower bound).
        right=False: ≥ left, < right. 
        """
        # Align the configured primary g-ratio method to the same index as
        # diameters for per-bin lookup.
        diam_series = agg_df['fiber_equiv_diam_um'].dropna()
        gratio_series = (
            agg_df.loc[diam_series.index, gratio_col]
            if gratio_col in agg_df.columns
            else None
        )

        # Treat the final configured edge as an open-ended catch bin.
        # config.toml uses 999 as a practical sentinel for "and above"
        # (JSON has no literal for infinity); convert it to a real
        # unbounded edge here so pd.cut()'s last bin has no upper limit.
        bin_edges = list(edges)
        if bin_edges and bin_edges[-1] == 999:
            bin_edges[-1] = np.inf

        cut = pd.cut(diam_series, bins=bin_edges, right=False)
        groups = diam_series.groupby(cut, observed=False)
        counts   = groups.count()
        percent    = (counts.values / total_distribution_axons * 100).round(2)
        cumulative = np.cumsum(percent).round(2)

        # G-ratio per bin — use mask-based lookup to avoid KeyError on empty intervals
        gratio_mean = []
        gratio_sd   = []
        if gratio_series is not None:
            for interval in counts.index:
                mask = (cut == interval)
                g    = gratio_series[mask].dropna()
                gratio_mean.append(round(g.mean(), 6) if len(g) > 0 else None)
                gratio_sd.append(round(g.std(),  6)   if len(g) > 1 else None)
        else:
            gratio_mean = [None] * len(counts)
            gratio_sd   = [None] * len(counts)

        # Build labels — last bin is catch box
        labels = []
        for idx, b in enumerate(counts.index):
            if idx == len(counts.index) - 1:
                labels.append(f"≥{b.left} µm")
            else:
                labels.append(f"≥{b.left} and <{b.right} µm")

        return pd.DataFrame({
            'Diameter Range':  labels,
            'Count':           counts.values.astype(int),
            'Percent (%)':     percent,
            'Cumulative (%)':  cumulative,
            'Mean G-ratio':    gratio_mean,
            'SD G-ratio':      gratio_sd,
        })

    bins_granular_df = make_bins_df(granular_edges)
    bins_mid_df      = make_bins_df(mid_edges)
    bins_coarse_df   = make_bins_df(coarse_edges)

    log.success(
        f"  Distribution complete: granular={len(bins_granular_df)} bins | "
        f"mid={len(bins_mid_df)} bins | coarse={len(bins_coarse_df)} bins | "
        f"{total_distribution_axons} axons | mean g-ratio={mean_gratio}"
    )

    return {
        'global_df':         global_df,
        'per_image_df':      per_image_df,
        'bins_granular_df':  bins_granular_df,
        'bins_mid_df':       bins_mid_df,
        'bins_coarse_df':    bins_coarse_df,
    }


def save_distributions(
    data: dict,
    output_dir: str,
    nerve_name: str
) -> str:
    """
    Save distribution data to a multi-sheet Excel file.

    Uses the configured granular, mid, and coarse diameter bins.
    Sheet 1 — Summary    : global nerve metadata + per-image summary
    Sheet 2 — Granular   : 0.5µm bins, 0–8µm + catch
    Sheet 3 — Mid        : 2µm bins, 0–16µm + catch
    Sheet 4 — Coarse     : 5µm bins, full range + catch

    File named: {nerve_name}_binned.xlsx
    Returns output path.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{nerve_name}_binned.xlsx"

    global_df        = data['global_df']
    per_image_df     = data['per_image_df']
    bins_granular_df = data['bins_granular_df']
    bins_mid_df      = data['bins_mid_df']
    bins_coarse_df   = data['bins_coarse_df']

    hdr_font = Font(bold=True)
    hdr_fill = PatternFill('solid', fgColor='D9E1F2')
    prov_font = Font(italic=True, color='666666')

    def write_header_row(ws, columns: list, start_row: int = 1):
        for c_idx, col_name in enumerate(columns, 1):
            cell           = ws.cell(row=start_row, column=c_idx, value=col_name)
            cell.font      = hdr_font
            cell.fill      = hdr_fill
            cell.alignment = Alignment(horizontal='center')
        ws.freeze_panes = ws.cell(row=start_row + 1, column=1)

    def write_df(ws, frame: pd.DataFrame, start_row: int = 1):
        write_header_row(ws, list(frame.columns), start_row)
        for r_idx, row_data in enumerate(frame.itertuples(index=False), start_row + 1):
            for c_idx, value in enumerate(row_data, 1):
                ws.cell(row=r_idx, column=c_idx, value=value)

    def autofit(ws, min_width: int = 12, max_width: int = 40):
        for col in ws.columns:
            max_len = max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in col
            )
            ws.column_dimensions[col[0].column_letter].width = min(
                max(max_len + 4, min_width), max_width
            )

    wb = Workbook()

    # ── Sheet 1 — Summary ─────────────────────────────────────────────────────
    ws1 = wb.active
    ws1.title = 'Summary'

    # Global metadata block
    for r_idx, row_data in enumerate(global_df.itertuples(index=False), 1):
        lbl_cell      = ws1.cell(row=r_idx, column=1, value=row_data[0])
        lbl_cell.font = prov_font
        ws1.cell(row=r_idx, column=2, value=row_data[1])

    # Per-image table below with blank separator
    sep = len(global_df) + 2
    write_df(ws1, per_image_df, start_row=sep)
    autofit(ws1)

    # ── Sheet 2 — Granular ────────────────────────────────────────────────────
    ws2 = wb.create_sheet('Granular (0.5µm)')
    write_df(ws2, bins_granular_df)
    autofit(ws2)

    # ── Sheet 3 — Mid ─────────────────────────────────────────────────────────
    ws3 = wb.create_sheet('Mid (2µm)')
    write_df(ws3, bins_mid_df)
    autofit(ws3)

    # ── Sheet 4 — Coarse ──────────────────────────────────────────────────────
    ws4 = wb.create_sheet('Coarse (5µm)')
    write_df(ws4, bins_coarse_df)
    autofit(ws4)

    wb.save(str(out_path))
    return str(out_path)