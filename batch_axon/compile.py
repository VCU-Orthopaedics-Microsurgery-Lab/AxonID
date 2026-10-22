"""
batch_axon/compile.py: build the animal and study workbooks from morphometrics output.

Purpose
    The compile step of Batch_Axon, without prompts or stage running, so it
    can be called from code or tests on a study that has already been
    segmented and measured.

Usage
    from batch_axon.compile import compile_study, compile_from_animal_workbooks
    from utils.study import detect_study_mag, resolve_scan
    study, study_dir = resolve_scan(study_folder)
    result = compile_study(study, study_dir, detect_study_mag(study), load_config(), log)

Inputs
    study       the scan from utils.study.resolve_scan(): {animal: {nerve: info}}
    study_dir   the study folder
    mag         study magnification ("40X")
    config      loaded config (for the Provenance sheet)
    log         AxonIDLogger

Outputs
    compile_study                   <Animal>/<animal>_Data.xlsx for every animal
                                    with data, and <Study>/<study>_Data.xlsx;
                                    both written from the same in-memory data,
                                    with live formulas.
    compile_from_animal_workbooks   <Study>/<study>_Data.xlsx only, copied as
                                    values from the existing animal workbooks
                                    (keeps manual edits; no formulas).
    Both return a CompileResult with the counts for the run summary.

Depends on
    morphometrics.analyze_nerve, batch_axon.workbook, xlsxwriter
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import time
from dataclasses import dataclass
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import xlsxwriter

# ── Local ─────────────────────────────────────────────────────────────────────
from batch_axon.workbook import (
    add_formats,
    copy_animal_sheet_static,
    find_animal_path,
    get_animal_xlsx_path,
    write_nerve_block,
    write_provenance_sheet,
)
from morphometrics.analyze_nerve import get_nerve_data
from utils.logger import AxonIDLogger


@dataclass
class CompileResult:
    combined_path:     Path
    animals_compiled:  int = 0
    animals_skipped:   int = 0
    nerves_processed:  int = 0
    nerves_failed:     int = 0
    nerves_incomplete: int = 0     # 40X nerves with no whole-nerve estimate (CSA incomplete)
    elapsed_seconds:   float = 0.0


def _open_study_workbook(study_dir, mag, config):
    study_path    = Path(study_dir)
    combined_path = study_path / f"{study_path.name}_Data.xlsx"
    workbook      = xlsxwriter.Workbook(str(combined_path))
    write_provenance_sheet(workbook, study_path, mag, config)
    return workbook, combined_path


# ─── --compile-only: from existing animal workbooks ───────────────────────────
def compile_from_animal_workbooks(study: dict, study_dir, mag, config, log: AxonIDLogger) -> CompileResult:
    """Rebuild the study workbook from each animal's existing workbook (values, not formulas)."""
    combined_workbook, combined_path = _open_study_workbook(study_dir, mag, config)
    result = CompileResult(combined_path=combined_path)

    log.rule("COMPILING FROM EXISTING ANIMAL WORKBOOKS")
    for animal_name, nerves in study.items():
        animal_path = find_animal_path(nerves)
        if animal_path is None:
            log.warn(f"Could not resolve a folder for {animal_name}, skipping")
            result.animals_skipped += 1
            continue
        xlsx_path = get_animal_xlsx_path(animal_path, animal_name)
        if not xlsx_path.exists():
            log.warn(f"No existing workbook for {animal_name} ({xlsx_path.name}) — "
                     f"run a full pass first. Skipping.")
            result.animals_skipped += 1
            continue
        if copy_animal_sheet_static(xlsx_path, combined_workbook, animal_name, log):
            log.success(f"  → Compiled {animal_name} from {xlsx_path.name}")
            result.animals_compiled += 1
        else:
            result.animals_skipped += 1

    combined_workbook.close()
    log.success(f"Combined workbook saved: {combined_path}")
    return result


# ─── Full compile: from morphometrics output ──────────────────────────────────
def compile_study(study: dict, study_dir, mag, config, log: AxonIDLogger) -> CompileResult:
    """
    Walk the study once and write BOTH the per-animal workbook and the study
    workbook from the same in-memory data (live formulas in both, no
    recomputation, no cross-workbook copying).
    """
    combined_workbook, combined_path = _open_study_workbook(study_dir, mag, config)
    result = CompileResult(combined_path=combined_path)

    log.rule("SCANNING STUDY")
    cohort_start = time.time()
    total_nerves = sum(len(nerves) for nerves in study.values())
    log.info(f"Detected magnification: [bold]{mag}[/bold]")
    log.info(f"Animals: {len(study)} | Nerves: {total_nerves}")

    for animal_name, nerves in study.items():
        if not nerves:
            log.warn(f"No nerves found for {animal_name}, skipping")
            result.animals_skipped += 1
            continue

        animal_path = find_animal_path(nerves)
        if animal_path is None:
            log.warn(f"Could not resolve a folder for {animal_name}, skipping")
            result.animals_skipped += 1
            continue

        log.rule(f"ANIMAL: {animal_name}")
        animal_start = time.time()

        # Per-animal individual workbook — always overwritten on a full run.
        animal_xlsx_path = get_animal_xlsx_path(animal_path, animal_name)
        animal_workbook   = xlsxwriter.Workbook(str(animal_xlsx_path))
        animal_formats    = add_formats(animal_workbook)
        animal_worksheet  = animal_workbook.add_worksheet(animal_name[:31])
        animal_row        = 0

        # Combined workbook's sheet for this animal — same data, own formats
        # instance (xlsxwriter formats are workbook-specific, can't share).
        combined_formats   = add_formats(combined_workbook)
        combined_worksheet = combined_workbook.add_worksheet(animal_name[:31])
        combined_row       = 0

        any_nerve_written = False

        for nerve_name, info in nerves.items():
            log.info(f"  Processing nerve: {nerve_name}")
            nerve_path = info['tiff_dir'].parent
            nerve_mag = info.get("mag") or mag

            if info['morphometrics_dir'] is None:
                log.warn(
                    f"  → No morphometrics folder found for "
                    f"{nerve_name}, skipping"
                )
                result.nerves_failed += 1
                continue

            nerve_start = time.time()

            try:
                image_rows, aggregate, _bins_df = get_nerve_data(
                    nerve_path,
                    nerve_mag,
                    log,
                )

                if not image_rows:
                    nerve_elapsed = time.time() - nerve_start
                    log.warn(
                        f"  → No data returned for {nerve_name} "
                        f"({nerve_elapsed:.1f}s)"
                    )
                    result.nerves_failed += 1
                    continue

                animal_row = write_nerve_block(
                    animal_worksheet, animal_row, nerve_name, animal_name,
                    image_rows, aggregate, animal_formats
                )
                combined_row = write_nerve_block(
                    combined_worksheet, combined_row, nerve_name, animal_name,
                    image_rows, aggregate, combined_formats
                )
                any_nerve_written = True

                # Distributions (_binned.xlsx) are written by the morphometrics
                # stage with the real model name. get_nerve_data() also returns
                # bins_df, but it is deliberately not re-saved here: doing so
                # overwrote the correct _binned.xlsx with wrong defaults
                # (Model: unknown) on every nerve. If morphometrics was skipped
                # and no prior _binned.xlsx exists, the nerve simply has none
                # this run; rerun without --skip-morphometrics.

                nerve_elapsed = time.time() - nerve_start
                estimate_note = ""
                if (
                    nerve_mag == "40X"
                    and aggregate.get('estimated_total_axons') is None
                ):
                    estimate_note = (
                        " (⚠ whole-nerve estimate unavailable — "
                        "CSA incomplete)"
                    )
                    result.nerves_incomplete += 1
                log.success(
                    f"  → {nerve_name}: "
                    f"{aggregate.get('total_axons', 0)} axons, "
                    f"{aggregate.get('total_images', 0)} images"
                    f"{estimate_note} ({nerve_elapsed:.1f}s)"
                )
                result.nerves_processed += 1

            except Exception as e:
                nerve_elapsed = time.time() - nerve_start
                log.error(
                    f"  → FAILED for {nerve_name}: {e} "
                    f"({nerve_elapsed:.1f}s)"
                )
                result.nerves_failed += 1

        animal_worksheet.set_column(0, 1, 28)
        animal_worksheet.set_column(2, 7, 18)
        combined_worksheet.set_column(0, 1, 28)
        combined_worksheet.set_column(2, 7, 18)
        animal_workbook.close()

        if any_nerve_written:
            animal_elapsed = time.time() - animal_start
            log.success(
                f"  → Animal workbook saved: {animal_xlsx_path} "
                f"({animal_elapsed:.1f}s)"
            )
            result.animals_compiled += 1
        else:
            if animal_xlsx_path.exists():
                animal_xlsx_path.unlink()

            animal_elapsed = time.time() - animal_start
            log.warn(
                f"  → No nerve data for {animal_name}; "
                f"no animal workbook retained ({animal_elapsed:.1f}s)"
            )
            result.animals_skipped += 1

    result.elapsed_seconds = time.time() - cohort_start

    combined_workbook.close()
    log.success(f"Combined workbook saved: {combined_path}")
    return result
