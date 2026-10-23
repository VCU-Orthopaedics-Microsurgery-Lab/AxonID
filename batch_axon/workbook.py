"""
batch_axon/workbook.py: Excel layout for the animal and study workbooks.

Purpose
    Writes one nerve block (per-image rows plus a Totals row), the study
    Provenance sheet, and the static copy used by --compile-only. No
    prompts, no stage running: everything here takes data in and writes
    cells, so it can be tested on its own.

Usage
    from batch_axon.workbook import add_formats, write_nerve_block
    formats = add_formats(workbook)
    next_row = write_nerve_block(ws, row, nerve, animal, image_rows, aggregate, formats)

Inputs
    image_rows, aggregate   from morphometrics.analyze_nerve.get_nerve_data()

Outputs
    Cells in an xlsxwriter worksheet. Derived cells (density, totals,
    estimated axon count) are live Excel formulas; "missing" is red, a
    measured 0 is yellow (conditional formatting, so it updates live).
    Images tagged EXCLUDE in QuPath are listed under the Totals row with their
    reason and are in none of the numbers.
    The last two columns report the axons morphometrics excluded because they
    lay outside the CSA tracing, as a count and as a share of all accepted
    axons; they are "n/a" for an image without an applied tracing.

Depends on
    xlsxwriter (writing), openpyxl (reading for --compile-only)
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import math
from datetime import datetime
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import openpyxl

# ── Local ─────────────────────────────────────────────────────────────────────
from utils.logger import AxonIDLogger

NUM_FORMATS = {
    'csa':    '#,##0.00',
    'count':  '#,##0',
    'gratio': '0.00',
    'diam':   '0.00',
    'est':    '#,##0',
    'dens':   '0.0000',
    'pct':    '0.0%',
}

NA = "n/a"      # no tracing was applied to this image: nothing was excluded


def add_formats(workbook):
    fmts = {
        'header':       workbook.add_format({'border': 2, 'bold': True, 'bg_color': '#D9E1F2'}),
        'bold':         workbook.add_format({'bold': True}),
        'grey':         workbook.add_format({'bg_color': '#8a8a8a', 'pattern': 1}),
        'normal':       workbook.add_format({}),
        'italic':       workbook.add_format({'italic': True, 'font_color': '#666666'}),
        'total':        workbook.add_format({'bold': True, 'top': 2}),
        'missing':      workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'}),  # Excel "Bad" style — no measurement at all
        'suspect_zero': workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'}),  # Excel "Neutral" style — real 0, likely a segmentation error, worth a look
    }
    # Display precision per column (stored values keep full precision).
    for key, nf in NUM_FORMATS.items():
        fmts[key]            = workbook.add_format({'num_format': nf})
        fmts[f'{key}_bold']  = workbook.add_format({'bold': True, 'num_format': nf})
        fmts[f'{key}_total'] = workbook.add_format({'bold': True, 'top': 2, 'num_format': nf})
    return fmts

def _is_num(v) -> bool:
    """True for a value Excel's ISNUMBER() would accept (not text, None, NaN or bool)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)


def _weighted_mean(weights: list, values: list):
    """Mirror of SUMPRODUCT(w, v) / SUMPRODUCT(w, --ISNUMBER(v)): non-numbers count as 0."""
    pairs = [(w, v) for w, v in zip(weights, values) if _is_num(w) and _is_num(v)]
    den = sum(w for w, _ in pairs)
    return sum(w * v for w, v in pairs) / den if den > 0 else None


HEADERS = [
    'Slide / Nerve', 'Sample', 'CSA (µm²)',
    'Total Axons', 'G-ratio', 'Axon Diameter (µm)',
    'Est. Full Axon Count', 'Axons / µm²',
    'Axons Outside Tracing', 'Outside Tracing (%)'
]


def write_nerve_block(worksheet, row, nerve_name, animal_name, image_rows, aggregate, formats):
    """Write a nerve's data block to the worksheet. Returns next available row.

    Base data (per-image CSA, axon count, g-ratio, diameter) is written as
    static values — it's raw pipeline output, nothing to formula-ize it
    against. Everything DERIVED from those cells (per-image density, and
    the whole Totals row) is written as a live Excel formula instead, so
    manually correcting a base cell after a QC pass recalculates every
    downstream value automatically instead of silently going stale.

    Two distinct "something's off" states, not one:
      - MISSING (no measurement at all, e.g. no CSA file found): written as
        the literal text "missing", flagged red. Excel's SUM/AVERAGE skip
        text cells automatically, so a missing G-ratio correctly drops out
        of the average rather than dragging it toward 0.
      - TRUE ZERO (a real measured 0, e.g. an image with no axons found):
        written as the actual number 0, flagged yellow — plausible but
        unlikely, worth a look, probably a segmentation error. Only the
        Total Axons column realistically ever hits a genuine 0; CSA/
        G-ratio/Diameter essentially never do in real data, but the same
        yellow rule is applied there too as a cheap safety net.

    Both flags are CONDITIONAL FORMATTING, not a static format applied at
    write time — they re-evaluate live along with the formulas, so filling
    in a previously-missing value later clears the red flag automatically.

    100X images do not have per-image CSA overlays and are therefore written
    with missing image-level CSA/density values. They are treated as sampled
    endoneurial/axonal tissue; the future 100X whole-nerve axon-count
    calculation will use the 4X whole-nerve CSA as its scaling reference.
    """
    MISSING = "missing"

    # Header row
    for col, h in enumerate(HEADERS):
        worksheet.write(row, col, h, formats['header'])
    row += 1

    # Whole-nerve 4X CSA is stored once at the nerve level and is independent
    # of the per-image CSA values below. It is the scaling reference for
    # estimated whole-nerve axon count. This same 4X CSA will also be used
    # for the future 100X total-axon calculation, because 100X fields do not
    # have individual CSA overlays and are treated as sampled
    # endoneurial/axonal tissue.
    #
    # The 4X CSA itself is computed upstream by get_nerve_fourx_csa().
    # This function only writes the returned value into the workbook.
    
    # Nerve label row (4X CSA)
    label     = f"{animal_name} - {nerve_name}"
    fourx_csa = aggregate.get('fourx_csa_um2')
    worksheet.write(row, 0, label, formats['bold'])
    worksheet.write(row, 1, f"{animal_name}_4X_{nerve_name}")
    worksheet.write(row, 2, fourx_csa if fourx_csa is not None else MISSING, formats['csa_bold'])
    for col in range(3, 10):
        worksheet.write(row, col, '', formats['grey'])
    fourx_csa_row = row  # 0-indexed row of this label row, col C holds fourx_csa
    row += 1

    # Per-image rows
    first_data_row = row
    col_csa, col_axons, col_gratio, col_diam, col_outside = [], [], [], [], []
    for entry in image_rows:
        worksheet.write(row, 1, entry.get('name', ''))

        # 100X image-level CSA is intentionally missing: 100X fields do not
        # have CSA GeoJSON overlays and are treated as sampled
        # endoneurial/axonal tissue. The absence of image-level CSA does not
        # prevent the future nerve-level 100X axon-count calculation, which
        # will use the separate whole-nerve 4X CSA.
        csa = entry.get('csa_um2')
        csa = csa if _is_num(csa) else MISSING
        worksheet.write(row, 2, csa, formats['csa'])

        total_axons = entry.get('total_axons')
        total_axons = total_axons if _is_num(total_axons) else MISSING
        worksheet.write(row, 3, total_axons, formats['count'])

        gratio = entry.get('gratio')
        gratio = round(gratio, 4) if _is_num(gratio) else MISSING
        worksheet.write(row, 4, gratio, formats['gratio'])

        diam = entry.get('axon_diam_um')
        diam = round(diam, 3) if _is_num(diam) else MISSING
        worksheet.write(row, 5, diam, formats['diam'])

        # The values written above, for the Totals row's stored values.
        col_csa.append(csa); col_axons.append(total_axons)
        col_gratio.append(gratio); col_diam.append(diam)

        worksheet.write(row, 6, '', formats['grey'])

        # Density requires both image-level CSA and total axons. ISNUMBER
        # prevents a missing CSA or axon count from becoming an Excel error.
        # 100X image-level density is intentionally left missing because
        # 100X fields do not have individual CSA measurements; the future
        # 100X whole-nerve axon-count calculation uses the separate 4X CSA.
        #
        # Every formula below is also given the value it evaluates to, so a
        # program that reads stored values without recalculating (pandas,
        # openpyxl, --compile-only) sees exactly what Excel shows.
        xl_row = row + 1  # Excel is 1-indexed
        if not (_is_num(csa) and _is_num(total_axons)):
            dens = MISSING
        else:
            dens = 0 if csa == 0 else total_axons / csa
        worksheet.write_formula(
            row, 7,
            f'=IF(OR(NOT(ISNUMBER(C{xl_row})),NOT(ISNUMBER(D{xl_row}))),"{MISSING}",'
            f'IF(C{xl_row}=0,0,D{xl_row}/C{xl_row}))',
            formats['dens'], value=dens
        )
        
        # Axons outside the CSA tracing: excluded by morphometrics from every
        # number in this row (the count above is already inside the tracing).
        # The share is outside / (outside + inside). "n/a" when no tracing
        # was applied to the image.
        outside = entry.get('axons_outside_tracing')
        outside = outside if _is_num(outside) else NA
        worksheet.write(row, 8, outside, formats['count'])
        if _is_num(outside) and _is_num(total_axons):
            pct = outside / (outside + total_axons) if outside + total_axons > 0 else 0
        else:
            pct = NA
        worksheet.write_formula(
            row, 9,
            f'=IF(AND(ISNUMBER(I{xl_row}),ISNUMBER(D{xl_row})),'
            f'IF(I{xl_row}+D{xl_row}=0,0,I{xl_row}/(I{xl_row}+D{xl_row})),"{NA}")',
            formats['pct'], value=pct
        )
        col_outside.append(outside)
        row += 1

    last_data_row = row - 1

    # Conditional formatting — the 2-color-scale gradients (unchanged), plus
    # two separate rules per column: text "missing" -> red, numeric 0 -> yellow
    if last_data_row >= first_data_row:
        worksheet.conditional_format(first_data_row, 3, last_data_row, 3, {
            'type': '2_color_scale',
            'min_color': '#ffffff', 'max_color': '#d53c3c'
        })
        worksheet.conditional_format(first_data_row, 4, last_data_row, 4, {
            'type': '2_color_scale',
            'min_color': '#ffffff', 'max_color': '#2DB133'
        })
        for col in (2, 3, 4, 5, 7):  # CSA, Total Axons, G-ratio, Diameter, Axons/um2
            worksheet.conditional_format(first_data_row, col, last_data_row, col, {
                'type': 'text', 'criteria': 'containing', 'value': MISSING, 'format': formats['missing']
            })
            worksheet.conditional_format(first_data_row, col, last_data_row, col, {
                'type': 'cell', 'criteria': '==', 'value': 0, 'format': formats['suspect_zero']
            })
        for col in (8, 9):      # axons outside the tracing: anything above 0 is worth a look
            worksheet.conditional_format(first_data_row, col, last_data_row, col, {
                'type': 'cell', 'criteria': '>', 'value': 0, 'format': formats['suspect_zero']
            })

    # Totals row — derived values remain live Excel formulas.
    #
    # CSA is different from ordinary SUM-able data: a missing per-image CSA
    # makes the sample CSA incomplete, so the total must be "missing" rather
    # than the partial SUM of the available images.
    #
    # Each formula's stored value is computed here from the cells written
    # above, exactly as the formula computes it, so the file reads the same
    # with or without recalculation.
    n = len(image_rows)

    xl_first      = first_data_row + 1
    xl_last       = last_data_row + 1
    xl_totals_row = row + 1
    xl_fourx_row  = fourx_csa_row + 1

    worksheet.write(row, 1, f"Totals (n={n})", formats['total'])

    # Total sample CSA is valid only when every image has a numeric CSA.
    # COUNT() therefore must equal the number of image rows; otherwise the
    # total is explicitly marked missing rather than silently using a partial
    # denominator.
    total_csa = sum(col_csa) if all(_is_num(v) for v in col_csa) else MISSING
    worksheet.write_formula(
        row, 2,
        f'=IF(COUNT(C{xl_first}:C{xl_last})={n},'
        f'SUM(C{xl_first}:C{xl_last}),"{MISSING}")',
        formats['csa_total'],
        value=total_csa
    )

    # Require a numeric axon count for every image before reporting
    # the total. This prevents SUM() from silently producing a partial
    # total when any per-image axon count is missing.
    total_axons = sum(col_axons) if all(_is_num(v) for v in col_axons) else MISSING
    worksheet.write_formula(
        row, 3,
        f'=IF(COUNT(D{xl_first}:D{xl_last})={n},'
        f'SUM(D{xl_first}:D{xl_last}),"{MISSING}")',
        formats['total'],
        value=total_axons
    )

    # G-ratio and axon diameter: per-image means weighted by each image's
    # axon count, which equals the mean over all axons pooled (every axon
    # counts once, as in Total Axons and the _binned distributions). An
    # unweighted AVERAGE would let an image with 20 axons count as much as
    # one with 900. Images without a numeric value drop out of both the
    # numerator and the denominator.
    for col, letter, values, fmt in ((4, 'E', col_gratio, 'gratio_total'),
                                     (5, 'F', col_diam,   'diam_total')):
        rng_d = f'D{xl_first}:D{xl_last}'
        rng_v = f'{letter}{xl_first}:{letter}{xl_last}'
        mean = _weighted_mean(col_axons, values)
        worksheet.write_formula(
            row, col,
            f'=IF(SUMPRODUCT({rng_d},--ISNUMBER({rng_v}))>0,'
            f'SUMPRODUCT({rng_d},{rng_v})/SUMPRODUCT({rng_d},--ISNUMBER({rng_v})),'
            f'"{MISSING}")',
            formats[fmt],
            value=mean if mean is not None else MISSING
        )

    # Current whole-nerve axon-count estimate:
    # total sampled axons / total sampled CSA × whole-nerve 4X CSA.
    #
    # This is currently implemented for 40X, where each sampled image has
    # an explicit CSA measurement. Future 100X support will use the same
    # whole-nerve 4X CSA reference, but will not require per-image CSA
    # overlays because 100X fields are treated as sampled
    # endoneurial/axonal tissue.
    if _is_num(total_csa) and _is_num(fourx_csa) and _is_num(total_axons) and total_csa != 0:
        est = total_axons / total_csa * fourx_csa
    else:
        est = MISSING
    worksheet.write_formula(
        row, 6,
        f'=IF(OR(NOT(ISNUMBER(C{xl_totals_row})),'
        f'NOT(ISNUMBER(C{xl_fourx_row}))),'
        f'"{MISSING}",'
        f'IFERROR(D{xl_totals_row}/C{xl_totals_row}*C{xl_fourx_row},'
        f'"{MISSING}"))',
        formats['est_total'],
        value=est
    )

    # Axons/µm² = total_axons / total_csa.
    # A missing total CSA must propagate as missing rather than producing a
    # density from an incomplete sample denominator.
    if _is_num(total_csa) and _is_num(total_axons) and total_csa != 0:
        density_val = total_axons / total_csa
    else:
        density_val = MISSING
    worksheet.write_formula(
        row, 7,
        f'=IF(NOT(ISNUMBER(C{xl_totals_row})),"{MISSING}",'
        f'IFERROR(D{xl_totals_row}/C{xl_totals_row},"{MISSING}"))',
        formats['dens_total'],
        value=density_val
    )
    
    # Axons outside the tracing: total, and the share among the images that
    # had a tracing applied (images without one hold "n/a" and are left out of
    # both the numerator and the denominator).
    rng_i = f'I{xl_first}:I{xl_last}'
    rng_d = f'D{xl_first}:D{xl_last}'
    outside_nums = [v for v in col_outside if _is_num(v)]
    total_outside = sum(outside_nums) if outside_nums else NA
    worksheet.write_formula(
        row, 8,
        f'=IF(COUNT({rng_i})=0,"{NA}",SUM({rng_i}))',
        formats['count_total'], value=total_outside
    )
    if outside_nums:
        inside_in_traced = sum(a for a, o in zip(col_axons, col_outside) if _is_num(o) and _is_num(a))
        denom = total_outside + inside_in_traced
        share = total_outside / denom if denom > 0 else 0
    else:
        share = NA
    worksheet.write_formula(
        row, 9,
        f'=IF(COUNT({rng_i})=0,"{NA}",'
        f'IF(SUM({rng_i})+SUMPRODUCT({rng_d},--ISNUMBER({rng_i}))=0,0,'
        f'SUM({rng_i})/(SUM({rng_i})+SUMPRODUCT({rng_d},--ISNUMBER({rng_i})))))',
        formats['pct_total'], value=share
    )
    for col in (8, 9):
        worksheet.conditional_format(row, col, row, col, {
            'type': 'cell', 'criteria': '>', 'value': 0, 'format': formats['suspect_zero']
        })

    # fourx_csa cell: "missing" text -> red; a real 0 -> yellow
    worksheet.conditional_format(fourx_csa_row, 2, fourx_csa_row, 2, {
        'type': 'text', 'criteria': 'containing', 'value': MISSING, 'format': formats['missing']
    })
    worksheet.conditional_format(fourx_csa_row, 2, fourx_csa_row, 2, {
        'type': 'cell', 'criteria': '==', 'value': 0, 'format': formats['suspect_zero']
    })
    # Totals row: missing values are red; genuine numeric zeroes are yellow.
    for col in (2, 3, 4, 5, 6, 7):
        worksheet.conditional_format(row, col, row, col, {
            'type': 'text',
            'criteria': 'containing',
            'value': MISSING,
            'format': formats['missing']
        })
        worksheet.conditional_format(row, col, row, col, {
            'type': 'cell',
            'criteria': '==',
            'value': 0,
            'format': formats['suspect_zero']
        })

    # Images tagged EXCLUDE: in no number above; listed here with their reason.
    excluded = aggregate.get('excluded_images') or []
    if excluded:
        row += 1
        worksheet.write(row, 1, f"Excluded images (n={len(excluded)}), left out of every number above", formats['italic'])
        for entry in excluded:
            row += 1
            worksheet.write(row, 1, entry.get('name', ''))
            worksheet.write(row, 2, f"EXCLUDED: {entry.get('reason', '')}", formats['italic'])


    row += 4  # Gap between nerves
    return row


def get_animal_xlsx_path(animal_path: Path, animal_name: str) -> Path:
    """Each animal's individual workbook lives inside that animal's own
    folder, colocated with its data — same principle as CSA/Morphometrics
    already living inside each nerve folder."""
    return animal_path / f"{animal_name}_Data.xlsx"


def find_animal_path(nerves: dict) -> Path | None:
    """An animal's own folder is any of its nerves' tiff_dir's grandparent
    (tiff_dir -> Nerve folder -> Animal folder). Any nerve works."""
    for info in nerves.values():
        if info.get('tiff_dir') is not None:
            return info['tiff_dir'].parent.parent
    return None


def write_provenance_sheet(workbook, study_path, mag, config):
    from utils.recipe import MORPHOMETRICS, MORPHOMETRICS_ID
    from utils.version import __version__
    prov_sheet = workbook.add_worksheet('Provenance')
    prov_fmt   = workbook.add_format({'italic': True, 'font_color': '#666666'})
    prov_rows  = [
        ('Study',                 str(study_path)),
        ('Generated',             datetime.now().strftime('%Y-%m-%d')),
        ('AxonID version',        __version__),
        ('Magnification',         mag),
        ('Primary g-ratio method',config.get('primary_gratio_method', 'equiv_diam')),
        ('Morphometrics rules',   MORPHOMETRICS_ID),
        ('Watershed threshold',   MORPHOMETRICS.distance_threshold),
        ('Watershed disk',        MORPHOMETRICS.dilation_disk),
    ]
    for r_idx, (label, value) in enumerate(prov_rows):
        prov_sheet.write(r_idx, 0, label, prov_fmt)
        prov_sheet.write(r_idx, 1, str(value))
    prov_sheet.set_column(0, 0, 28)
    prov_sheet.set_column(1, 1, 50)


def copy_animal_sheet_static(src_xlsx: Path, dst_workbook, animal_name: str, log: AxonIDLogger) -> bool:
    """
    COMPILE-ONLY mode: reads an existing animal .xlsx's CURRENT cell values
    (data_only=True picks up whatever Excel last calculated/saved, including
    any manual edits made since the last full run) and writes them as a
    static snapshot into the combined workbook. Preserves fill color so the
    missing/suspect_zero highlighting stays visible, but does NOT try to
    reconstruct live formulas or conditional formatting rules across
    workbooks — this combined sheet is a compiled report, not something
    meant to be edited further itself.
    """
    try:
        # Compile-only intentionally snapshots saved cell values rather than
        # copying formulas. This preserves manual QC edits made in the existing
        # animal workbook, but the resulting study workbook is no longer
        # formula-linked to that source workbook.
        src_wb = openpyxl.load_workbook(str(src_xlsx), data_only=True)
    except Exception as e:
        log.error(f"  → Could not read {src_xlsx.name}: {e}")
        return False

    src_ws = src_wb.worksheets[0]
    dst_ws = dst_workbook.add_worksheet(animal_name[:31])

    fmt_cache = {}
    for row in src_ws.iter_rows():
        for cell in row:
            if cell.value is None or cell.value == '':
                continue
            fill_rgb = None
            if cell.fill and cell.fill.fgColor and cell.fill.fgColor.rgb and cell.fill.fgColor.rgb != '00000000':
                fill_rgb = cell.fill.fgColor.rgb[-6:]
            nf = cell.number_format if cell.number_format != 'General' else None
            key = (fill_rgb, bool(cell.font and cell.font.bold), nf)
            if key not in fmt_cache:
                props = {}
                if fill_rgb:
                    props['bg_color'] = f'#{fill_rgb}'
                if key[1]:
                    props['bold'] = True
                if key[2]:
                    props['num_format'] = key[2]
                fmt_cache[key] = dst_workbook.add_format(props) if props else None
            dst_ws.write(cell.row - 1, cell.column - 1, cell.value, fmt_cache[key])

    dst_ws.set_column(0, 1, 28)
    dst_ws.set_column(2, 9, 18)
    return True
