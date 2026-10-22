"""
batch_axon/__main__.py: whole-study run: segmentation, morphometrics, then Excel summaries.

Usage
    python -m batch_axon                       segment + morphometrics + compile
    python -m batch_axon --skip-segment        morphometrics + compile
    python -m batch_axon --skip-morphometrics  segment + compile
    python -m batch_axon --compile-only        rebuild the study workbook from the
                                               existing animal workbooks (keeps manual edits)
    The study folder is entered at the prompt; segmentation prompts for the model.

Inputs
    A study folder in the AxonID layout: Study/Animal/Nerve/40X_TIFF, plus
    CSA/ (QuPath GeoJSONs) and 4X_TIFF/ for whole-nerve CSA.

Outputs
    <Animal>/<animal>_Data.xlsx     one block per nerve: per-image CSA, axon
                                    count, g-ratio, diameter, density, totals
    <Study>/<study>_Data.xlsx       every animal, plus a Provenance sheet
    Derived cells (density, totals, estimated axon count) are live Excel
    formulas, so correcting a value after QC updates them. Missing values
    are written as "missing" (red); a measured 0 is flagged yellow.

Behaviour
    Distributions (_binned.xlsx) are written by the morphometrics stage,
    not here. The estimated whole-nerve axon count needs a CSA for every
    40X image and the 4X whole-nerve CSA; otherwise it is "missing".

Code layout
    __main__.py   prompts, flags, running the stages, run summary
    compile.py    building the workbooks (callable without prompts)
    workbook.py   Excel layout: nerve blocks, formulas, flags, Provenance

Depends on
    segment, morphometrics, batch_axon.compile
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
from rich.box import DOUBLE
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# ── Local ─────────────────────────────────────────────────────────────────────
# Local packages (utils, segment) are imported as top-level packages —
# the repo root must be on sys.path before any of the imports below
# can resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from batch_axon.compile import compile_from_animal_workbooks, compile_study
from morphometrics.__main__ import run as run_morphometrics
from segment.__main__ import run as run_segment
from utils.config import (
    format_morphometrics_config_summary,
    format_scan_config_summary,
    format_segment_config_summary,
    load_config,
)
from utils.console import print_panel
from utils.logger import AxonIDLogger
from utils.study import detect_input_level, detect_study_mag, resolve_scan

console = Console()




def main():
    run_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-only", action="store_true",
                         help="Skip the expensive walk/compute step — just re-read whatever's "
                              "currently in each animal's existing individual .xlsx file "
                              "(including manual edits) and rebuild the combined workbook from that.")
    parser.add_argument("--skip-segment", action="store_true",
                         help="Skip the segmentation stage (assume it's already been run) and "
                              "go straight to morphometrics + compilation.")
    parser.add_argument("--skip-morphometrics", action="store_true",
                         help="Skip the morphometrics stage (assume it's already been run) and "
                              "go straight to compilation. Segmentation still runs unless "
                              "--skip-segment is also passed.")
    args = parser.parse_args()
    
    if args.compile_only and (args.skip_segment or args.skip_morphometrics):
        parser.error(
            "--compile-only already skips segmentation and morphometrics "
            "entirely (it reads existing animal workbooks, not morphometrics "
            "output) — combining it with --skip-segment/--skip-morphometrics "
            "doesn't mean anything additional."
        )

    mode_tags = []
    if args.compile_only:
        mode_tags.append("COMPILE-ONLY")
    if args.skip_segment:
        mode_tags.append("SKIP-SEGMENT")
    if args.skip_morphometrics:
        mode_tags.append("SKIP-MORPHOMETRICS")
    mode_suffix = "".join(f" [{tag}]" for tag in mode_tags)


    print_panel(console, Panel(
        Text.from_markup(
            "[bold white]Study-Level Morphometric\nCompilation[/bold white]",
            justify="center"
        ),
        title="[bold cyan]AxonID — BATCH AXON[/bold cyan]" + mode_suffix,
        border_style="bright_cyan",
        box=DOUBLE,
        expand=True,
        padding=(1, 4)
    ))

    while True:
        input_dir = input("\nInput the path to the study folder: ").strip().strip('"')
        if not os.path.isdir(input_dir):
            console.print(f"[red]✗  Folder not found: {input_dir}[/red]")
            continue

        try:
            level = detect_input_level(input_dir)
            if level != "study":
                raise ValueError(
                    "Batch_Axon requires a study folder. "
                    "Please select the study folder containing the animal folders."
                )

            study, study_dir = resolve_scan(input_dir)
            break

        except ValueError as e:
            console.print(f"[red]✗  {e}[/red]")
            continue

    study_path  = Path(study_dir)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    config      = load_config()
    logging_cfg = config.get("logging", {})
    logging_on  = logging_cfg.get("batch_axon", False) if isinstance(logging_cfg, dict) else bool(logging_cfg)
    log_path    = str(study_path / f"batch_axon_log_{timestamp}.txt") if logging_on else None
    log         = AxonIDLogger(log_path=log_path, program="AxonID Batch_Axon")

    log.info(f"Study: {study_dir}")
    for line in format_scan_config_summary():
        log.info(line)
    mag = detect_study_mag(study)

    if args.compile_only:
        # ── Compile-only: read existing per-animal files, no recomputation ──
        result = compile_from_animal_workbooks(study, study_dir, mag, config, log)
        log.finalize(summary={
            'Mode':               'Compile-only',
            'Study':              study_dir,
            'Combined workbook':  str(result.combined_path),
            'Animals compiled':   result.animals_compiled,
            'Animals skipped':    result.animals_skipped,
        }, title="BATCH AXON COMPLETE")

        if log_path:
            console.print(f"\n[dim]Log saved to: {log_path}[/dim]")
        return

    # ── Upstream stages: segmentation + morphometrics ────────────────────────
    # Each run() walks the WHOLE study internally — call once each, not per
    # nerve. run_segment() will prompt for model selection interactively;
    # that's expected. Neither call is expected to raise — both already wrap
    # their own per-nerve work in try/except and return a StageResult — but an
    # outer guard here means a genuinely unexpected failure in one stage
    # can't take down the whole Batch_Axon run.
    if not args.skip_segment:
        log.rule("SEGMENTATION")
        try:
            seg_result = run_segment(str(study_dir), quiet_banner=True)
            if not seg_result.success:
                log.warn(
                    "Segmentation did not complete for every nerve — "
                    "continuing with whatever was produced."
                )
            segment_elapsed = seg_result.elapsed_seconds
        except Exception as e:
            log.error(f"Segmentation stage failed: {e}")
            segment_elapsed = 0.0
    else:
        segment_elapsed = 0.0
    if args.skip_segment:
        log.info("Skipping segmentation stage (--skip-segment)")
        for line in format_segment_config_summary(mag):
            log.info(line)

    if not args.skip_morphometrics:
        log.rule("MORPHOMETRICS")
        try:
            morph_result = run_morphometrics(str(study_dir), quiet_banner=True)
            if not morph_result.success:
                log.warn(
                    "Morphometrics did not complete for every nerve — "
                    "continuing with whatever was produced."
                )
            morph_elapsed = morph_result.elapsed_seconds
            morph_images = morph_result.images_processed
            morph_core_seconds = (
                morph_result.morphometrics_seconds
                + morph_result.distributions_seconds
            )
        except Exception as e:
            log.error(f"Morphometrics stage failed: {e}")
            morph_elapsed = 0.0
            morph_images = 0
            morph_core_seconds = 0.0
    else:
        morph_elapsed = 0.0
        morph_images = 0
        morph_core_seconds = 0.0
    if args.skip_morphometrics:
        log.info("Skipping morphometrics stage (--skip-morphometrics)")
        for level, line in format_morphometrics_config_summary(mag):
            (log.warn if level == "warn" else log.info)(line)

    # Re-scan: the study dict built at the top of main() predates the
    # Segmented_M<n>/Morphometrics folders the two stages above may have
    # just created. Without this, info['morphometrics_dir'] below is stale
    # and every nerve gets skipped exactly as in the original run.
    # Cheap and correct to run unconditionally even when both stages above
    # were skipped — it's just a re-read of whatever's already on disk.
    study, study_dir = resolve_scan(str(study_dir))
    mag = detect_study_mag(study)
    
            
    # ── Compile: animal workbooks + study workbook ──────────────────────────
    result = compile_study(study, study_dir, mag, config, log)

    mode_label = "Full run" + (
        " (" + ", ".join(
            t for t, flag in [("skip-segment", args.skip_segment),
                               ("skip-morphometrics", args.skip_morphometrics)]
            if flag
        ) + ")"
        if (args.skip_segment or args.skip_morphometrics) else ""
    )

    def _fmt(seconds: float) -> str:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"

    run_elapsed = time.time() - run_start
    study_avg_per_image = (
        f"{morph_core_seconds / morph_images:.1f}s"
        if morph_images > 0 else "n/a"
    )

    log.finalize(summary={
        'Mode':                             mode_label,
        'Study':                            study_dir,
        'Magnification':                    mag,
        'Combined workbook':                str(result.combined_path),
        'Animals compiled':                 result.animals_compiled,
        'Animals skipped':                  result.animals_skipped,
        'Nerves processed':                 result.nerves_processed,
        'Nerves failed/skipped':            result.nerves_failed,
        'Nerves w/ incomplete CSA (40X)':   result.nerves_incomplete,
        'Segmentation stage time':          _fmt(segment_elapsed) if not args.skip_segment else 'skipped',
        'Morphometrics stage time':         _fmt(morph_elapsed) if not args.skip_morphometrics else 'skipped',
        'Cohort processing time':           _fmt(result.elapsed_seconds),
        'Total time (call to end)':         _fmt(run_elapsed),
        'Avg time per image (study, core)': study_avg_per_image,
    }, title="BATCH AXON COMPLETE")
    
    if log_path:
        console.print(f"\n[dim]Log saved to: {log_path}[/dim]")

    if result.animals_compiled == 0:
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())