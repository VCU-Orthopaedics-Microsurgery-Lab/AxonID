"""
segment/__main__.py: interactive entry point for segmentation.

Purpose
    Ask for a study, animal or nerve folder, pick a model, and segment every
    nerve that does not have complete output yet.

Usage
    python -m segment                                   prompts for the folder and the model
    python -m segment --input FOLDER --model MODEL      no prompts (cluster / scripted use)
    from segment.__main__ import run                    (used by batch_axon)

    MODEL is a number from the menu, or a model file name, file stem or label
    (for example axonid_v1_unetpp_dn169_aug).

Inputs
    A folder in the Study/Animal/Nerve/<mag>_TIFF layout (see README).
    A model: a released model in models/ (listed in models/models.json) or
    one you trained (models/user/). The recommended released model is offered
    first.
Outputs
    <nerve>/Segmented/*.tif (BGW masks with provenance), optional timing.csv
    and QC sheet per nerve. run() returns a StageResult (utils/study.py).

Resume behaviour
    A nerve is skipped when every input TIFF already has a segmented file.
    Output folders are not named by model, so to re-segment a nerve with a
    different model, delete its Segmented/ folder first. Each segmented TIFF
    records the model that made it.

Checks
    The model's magnification (from its metadata) must match the study's
    magnification; a mismatch stops the run. Released model files are checked
    against the MD5 listed in models/models.json.

Depends on
    segment/segment.py       load_model(), segment_dir()
    utils/model_registry.py  model menu
    utils/study.py           folder scanning, magnification detection
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import torch
from rich.align import Align
from rich.box import DOUBLE
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Local ─────────────────────────────────────────────────────────────────────
# Local packages (utils, segment) are imported as top-level packages —
# the repo root must be on sys.path before any of the imports below
# can resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from segment.segment import _fmt4, load_model, segment_dir
from utils.config import (
    format_scan_config_summary,
    format_segment_config_summary,
    load_config,
)
from utils.console import print_panel
from utils.filesystem import list_files
from utils.model_registry import available_models, missing_released_models, verify_file
from utils.gpu import setup_gpu_console
from utils.logger import AxonIDLogger
from utils.study import StageResult, detect_study_mag, resolve_scan

console = Console()


def select_model(models: list):
    """Interactive model menu. Returns the chosen ModelEntry."""
    console.print()
    console.rule("MODEL SELECTION", style="bright_cyan")
    table = Table(
        box=DOUBLE,
        show_header=True,
        header_style="bold bright_cyan",
        border_style="bright_cyan",
        expand=True,
        padding=(0, 2)
    )
    table.add_column("#",      style="bold cyan",  width=4, justify="center")
    table.add_column("Model",  style="bold white",          justify="left")
    table.add_column("Source", style="white",      width=10, justify="left")
    table.add_column("Notes",  style="dim",                 justify="left")

    for i, m in enumerate(models, 1):
        label = f"{m.label}  [green](recommended)[/green]" if m.recommended else m.label
        source = "released" if m.source == "released" else "your model"
        table.add_row(str(i), label, source, m.notes)

    console.print(table)

    while True:
        raw = input(f"Select model [1-{len(models)}]: ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(models):
                return models[idx]
        except ValueError:
            pass
        console.print(f"[red]Please enter a number between 1 and {len(models)}[/red]")


def find_model(models: list, choice: str):
    """Match a --model value: menu number, file name, file stem or label.

    Returns the ModelEntry, or None when nothing matches exactly.
    """
    c = choice.strip().strip('"').lower()
    if c.isdigit():
        idx = int(c) - 1
        return models[idx] if 0 <= idx < len(models) else None
    for m in models:
        if c in (m.path.name.lower(), m.path.stem.lower(), m.label.lower()):
            return m
    return None


def run(input_dir: str | Path, quiet_banner: bool = False,
        model: str | None = None) -> StageResult:
    """Run segmentation for a study, animal, or nerve folder.

    Parameters
    ----------
    input_dir
        Study, animal, or nerve folder accepted by resolve_scan().

    quiet_banner
        When True, skip the initial "Folder: <path>" line — used when a
        caller (e.g. batch_axon) has already printed the study path once
        and doesn't need it repeated for every stage it invokes. Does
        not affect this stage's own scan — it still runs in full.

    model
        Optional model to use without the menu: a menu number, or a model
        file name, file stem or label (see find_model). When None, the
        menu is shown and you choose interactively.

    Returns
    -------
    StageResult
        success is True when the segmentation stage completes
        successfully, False when a required input, model, or processing
        step prevents completion. elapsed_seconds and images_processed
        cover the full run() call, including scanning and model loading.
        segmentation_seconds is the sum of actual per-image inference
        time across all nerves (excludes scanning, model loading, and
        QC generation). qc_seconds is the sum of QC-sheet generation
        time, reported separately and deliberately excluded from any
        per-image average, since QC is config-toggleable and will
        almost always be left on for review.

    Notes
    -----
    Without the model argument, model selection is interactive. Batch_Axon
    calls this function after resolving and validating the study-level input.
    """
    run_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config    = load_config()

    logging_cfg = config.get("logging", {})
    logging_on = (
        logging_cfg.get("segment", False)
        if isinstance(logging_cfg, dict)
        else bool(logging_cfg)
    )

    # Resolve the supplied path into the standard study representation.
    try:
        study, study_dir = resolve_scan(str(input_dir))
    except ValueError as e:
        console.print(f"[red]✗  {e}[/red]")
        return StageResult(False, time.time() - run_start, 0)

    log_path = (
        str(Path(input_dir) / f"segment_log_{timestamp}.txt")   # the folder the user entered
        if logging_on
        else None
    )
    log = AxonIDLogger(
        log_path=log_path,
        program="AxonID Segment"
    )

    if not quiet_banner:
        log.info(f"Folder: {input_dir}")
        for line in format_scan_config_summary():
            log.info(line)

    # ── Scan study ─────────────────────────────────────────────────────────
    log.rule("SCANNING STUDY")

    total_nerves = sum(len(nerves) for nerves in study.values())
    if total_nerves == 0:
        log.error("No nerve folders with TIFF directories found.")
        log.finalize({
            'Folder': str(input_dir),
            'Nerves processed': 0,
            'Nerves skipped': 0,
        }, title="SEGMENTATION COMPLETE", elapsed_label="Elapsed time")
        return StageResult(False, time.time() - run_start, 0)

    mag = detect_study_mag(study)
    log.info(f"Detected magnification: [bold]{mag}[/bold]")
    for line in format_segment_config_summary(mag):
        log.info(line)
    log.info(f"Animals found: {len(study)}")
    log.info(f"Nerves found:  {total_nerves}")

    # ── Model selection ────────────────────────────────────────────────────
    models = available_models()
    missing = missing_released_models()
    if missing:
        log.warn(
            "Released model file(s) not found in models/: "
            + ", ".join(missing)
            + " (run: python -m utils.fetch_models)."
        )
    if not models:
        log.error("No models available: run python -m utils.fetch_models to download "
                  "the released models, or train one (python -m train).")
        log.finalize({
            'Folder': str(input_dir),
            'Nerves processed': 0,
            'Nerves skipped': 0,
        }, title="SEGMENTATION COMPLETE", elapsed_label="Elapsed time")
        return StageResult(False, time.time() - run_start, 0)

    if model is not None:
        selected = find_model(models, model)
        if selected is None:
            log.error(
                f"Model '{model}' not found. Available: "
                + ", ".join(m.path.stem for m in models)
            )
            log.finalize({'Folder': str(input_dir),
                          'Nerves processed': 0, 'Nerves skipped': 0},
                         title="SEGMENTATION COMPLETE", elapsed_label="Elapsed time")
            return StageResult(False, time.time() - run_start, 0)
    else:
        selected = select_model(models)
    selected_model_path = selected.path
    checksum_ok = verify_file(selected_model_path, selected)
    if checksum_ok is False:
        log.error(
            f"{selected_model_path.name} does not match the checksum in "
            f"models/models.json. Delete it and run python -m utils.fetch_models."
        )
        log.finalize({'Folder': str(input_dir), 'Model': selected_model_path.stem,
                      'Nerves processed': 0, 'Nerves skipped': 0},
                     title="SEGMENTATION COMPLETE", elapsed_label="Elapsed time")
        return StageResult(False, time.time() - run_start, 0)
    # Model identity is no longer encoded in the output folder name — one
    # flat Segmented/ folder is used regardless of which model produced
    # it. Model name/architecture/encoder are embedded in each segmented
    # TIFF's provenance and surface in the morphometrics workbook's
    # Summary sheet instead of the folder path.
    seg_folder = config.get("segmented_folder", "Segmented")
    seg_suffix = config.get("segmented_suffix", "_segmented")
    expected_seg_name = seg_folder

    to_process = []
    to_skip    = []

    for animal, nerves in study.items():
        for nerve, info in nerves.items():
            tiff_dir = info['tiff_dir']
            expected_seg_dir = tiff_dir.parent / expected_seg_name

            input_images = list_files(
                str(tiff_dir),
                extensions=('.tif', '.tiff')
            )

            expected_outputs = {
                f"{img_path.stem}{seg_suffix}.tif"
                for img_path in input_images
            }

            existing_outputs = {
                path.name
                for path in expected_seg_dir.glob("*.tif")
            } if expected_seg_dir.is_dir() else set()

            has_expected_output = (
                bool(expected_outputs)
                and expected_outputs.issubset(existing_outputs)
            )

            if has_expected_output:
                to_skip.append(f"{animal}/{nerve}")
            else:
                to_process.append((animal, nerve, info))

    # Skipping is by output files only; it cannot tell which model made them.
    # To re-segment with another model, delete the nerve's Segmented/ folder.
    if to_skip:
        log.warn(
            f"Skipping {len(to_skip)} nerve(s) with existing segmentation "
            f"output:"
        )
        for s in to_skip:
            log.warn(f"  {s}")

    if not to_process:
        log.success(
            "All nerves already have segmentation output. Nothing to do."
        )
        log.finalize({
            'Folder': str(input_dir),
            'Model': selected_model_path.stem,
            'Nerves processed': 0,
            'Nerves skipped': len(to_skip),
        }, title="SEGMENTATION COMPLETE", elapsed_label="Elapsed time")
        return StageResult(True, time.time() - run_start, 0)

    log.info(f"To process: {len(to_process)} nerve(s)")

    # ── GPU setup ──────────────────────────────────────────────────────────
    setup_gpu_console(activity="Segmentation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──────────────────────────────────────────────────────────
    meta = {}
    try:
        model, meta = load_model(
            str(selected_model_path),
            device=device,
            log=log
        )
        log.success(f"Model loaded: {selected_model_path.stem}")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        log.finalize({
            'Folder': str(input_dir),
            'Model': selected_model_path.stem,
            'Nerves processed': 0,
            'Nerves skipped': len(to_skip),
        }, title="SEGMENTATION COMPLETE", elapsed_label="Elapsed time")
        return StageResult(False, time.time() - run_start, 0)

    # The study folders define the image magnification; the model's metadata
    # says which magnification it was trained on. They must agree: a model
    # trained at 40X sees structures at the wrong pixel size on other images.
    model_mag = meta.get('magnification')

    if model_mag:
        if model_mag != mag:
            log.error(
                f"Selected model expects {model_mag}, but the study was "
                f"detected as {mag}. Select a model matching the study "
                f"magnification."
            )
            log.finalize({
                'Folder':            str(input_dir),
                'Model':            selected_model_path.stem,
                'Model magnification': model_mag,
                'Study magnification': mag,
                'Nerves processed': 0,
                'Nerves skipped': len(to_skip),
            }, title="SEGMENTATION COMPLETE", elapsed_label="Elapsed time")
            return StageResult(False, time.time() - run_start, 0)

        log.info(
            f"Model magnification confirmed: {model_mag}"
        )
    else:
        log.warn(
            "Selected model has no magnification metadata; "
            f"using study-detected magnification {mag}"
        )

    # ── Process nerves ─────────────────────────────────────────────────────
    current_animal = None
    nerves_processed = 0
    total_images_segmented = 0
    total_seg_core_seconds = 0.0
    total_qc_seconds = 0.0

    for animal, nerve, info in to_process:
        if animal != current_animal:
            console.rule(animal, characters="═", style="white")
            current_animal = animal

        tiff_dir   = info['tiff_dir']
        nerve_path = tiff_dir.parent
        output_dir = nerve_path / seg_folder
        timing_csv = (
            str(output_dir / "timing.csv")
            if config.get("timing", False)
            else None
        )
        nerve_image_count = len(
            list_files(str(tiff_dir), extensions=('.tif', '.tiff'))
        )
 

        try:
            nerve_result = segment_dir(
                tiff_dir=str(tiff_dir),
                output_dir=str(output_dir),
                model=model,
                mag=mag,
                log=log,
                timing_csv=timing_csv,
                model_name=meta.get('model_name') or selected_model_path.stem,
                meta=meta,
            )

            if nerve_result.success:
                nerves_processed += 1
                total_images_segmented += nerve_image_count
                total_seg_core_seconds += nerve_result.seg_seconds
                total_qc_seconds += nerve_result.qc_seconds
            else:
                log.error(
                    f"Segmentation incomplete for {animal}/{nerve}: "
                    "one or more images failed."
                )

        except Exception as e:
            log.error(f"Failed to segment {animal}/{nerve}: {e}")

    run_elapsed = time.time() - run_start
    log.finalize(summary={
        'Folder':            str(input_dir),
        'Model':            selected_model_path.stem,
        'Trained':          meta.get('trained_date', '?'),
        'Magnification':    mag,
        'Best axon dice':   _fmt4(meta.get('best_axon_dice')),
        'Best myelin dice': _fmt4(meta.get('best_myelin_dice')),
        'Nerves processed': nerves_processed,
        'Nerves skipped':   len(to_skip),
        'Segmentation core time': f"{int(total_seg_core_seconds // 60)}m {int(total_seg_core_seconds % 60)}s",
        'QC time (excluded from avg)': f"{int(total_qc_seconds // 60)}m {int(total_qc_seconds % 60)}s",
        'Avg time per image (core, excl. QC)': (
            f"{total_seg_core_seconds / total_images_segmented:.1f}s"
            if total_images_segmented > 0 else "n/a"
        ),
    }, title="SEGMENTATION COMPLETE", elapsed_label="Elapsed time")

    if log_path:
        console.print(f"\n[dim]Log saved to: {log_path}[/dim]")

    return StageResult(
        success=(nerves_processed + len(to_skip) == total_nerves),
        elapsed_seconds=run_elapsed,
        images_processed=total_images_segmented,
        segmentation_seconds=total_seg_core_seconds,
        qc_seconds=total_qc_seconds,
    )


def main():
    parser = argparse.ArgumentParser(description="AxonID segmentation.")
    parser.add_argument("--input", metavar="FOLDER",
                        help="study, animal or nerve folder (skips the folder prompt)")
    parser.add_argument("--model", metavar="MODEL",
                        help="menu number, or model file name, file stem or label "
                             "(skips the model menu)")
    args = parser.parse_args()

    print_panel(console, Panel(
        Align.center(
            "[bold white]Automated Axon-Myelin Brightfield Image Segmentation[/bold white]"
        ),
        title="[bold cyan]AxonID — SEGMENTATION[/bold cyan]",
        border_style="bright_cyan",
        box=DOUBLE,
        expand=True,
        padding=(1, 4)
    ))

    # ── Study-folder input: --input, or interactive prompt ────────────────
    while True:
        if args.input is not None:
            input_dir = args.input.strip().strip('"')
        else:
            input_dir = input(
                "\nInput the path to the study, animal, or nerve folder: "
            ).strip().strip('"')

        if not os.path.isdir(input_dir):
            console.print(f"[red]✗  Folder not found: {input_dir}[/red]")
            if args.input is not None:
                sys.exit(1)
            continue

        try:
            resolve_scan(input_dir)
            break
        except ValueError as e:
            console.print(f"[red]✗  {e}[/red]")
            if args.input is not None:
                sys.exit(1)

    if not run(input_dir, model=args.model).success:
        sys.exit(1)


if __name__ == "__main__":
    main()