"""
morphometrics/__main__.py: entry point for morphometrics on segmented images.

Purpose
    Finds every segmented dataset under a study, animal or nerve folder and
    produces per-image workbooks, an optional watershed QC figure and a
    nerve-level diameter distribution for each.

Usage
    python -m morphometrics            prompts for the folder
    from morphometrics.__main__ import run; run(folder)   (used by batch_axon)

Inputs
    Study / animal / nerve folders in the AxonID layout (see README), with
    segmented TIFFs in Segmented*/ folders.

Outputs
    Morphometrics*/{image}_morphometrics.xlsx   per image
    Morphometrics*/{nerve}_binned.xlsx          per nerve
    QC/{nerve}_watershed_qc*.png                when config.toml qc.watershed is true

Behaviour
    A dataset is skipped when every segmented TIFF already has its
    workbook; partial output is redone as a unit. Pixel size comes from
    config.toml pixel_size_um at the ORIGINAL image width, corrected for
    any downsampling recorded by segmentation (utils/calibration.py).
    Segmented TIFFs without that record (older versions) are assumed not
    downsampled, with a warning.

Locked (utils/recipe.py)
    MORPHOMETRICS (watershed and acceptance rules)

Depends on
    morphometrics/morphometrics.py, morphometrics/distributions.py,
    utils/calibration.py, utils/study.py
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
from rich.align import Align
from rich.box import DOUBLE
from rich.console import Console
from rich.panel import Panel

# ── Local ─────────────────────────────────────────────────────────────────────
# Local packages (utils, segment) are imported as top-level packages —
# the repo root must be on sys.path before any of the imports below
# can resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from morphometrics.distributions import bin_nerve_diameters, save_distributions
from morphometrics.morphometrics import (
    get_morphometrics,
    save_morphometrics,
    save_watershed_qc,
)
from utils.calibration import read_tiff_provenance, resolve_pixel_size
from utils.config import (
    format_morphometrics_config_summary,
    format_scan_config_summary,
    load_config,
)
from utils.console import print_panel
from utils.filesystem import list_files
from utils.logger import AxonIDLogger
from utils.recipe import SUPPORTED_MAGS
from utils.resize import get_image_resolution
from utils.study import StageResult, _strip_suffix, detect_study_mag, resolve_scan

console = Console()


def run(input_dir: str | Path, quiet_banner: bool = False) -> StageResult:

    """
    Run the AxonID morphometrics pipeline for a study, animal, or nerve.

    Args:
        input_dir: Study, animal, or nerve folder containing segmented images.
        quiet_banner: When True, skip the initial "Folder: <path>" line and
            the scan-level config summary (tiff_suffixes) — used when a
            caller (batch_axon) has already printed both once and doesn't
            need them repeated for every stage it invokes. Does not affect
            the per-magnification morphometrics config summary below,
            which is specific to this stage and always prints.

    Returns:
        StageResult. success is True when the pipeline completes its
        run, False when the input cannot be processed or an
        unrecoverable setup error occurs. elapsed_seconds and
        images_processed cover the full run() call. morphometrics_seconds
        is per-image extraction time only; distributions_seconds is
        binning/distribution time; qc_seconds is watershed-QC generation
        time, reported separately and excluded from any per-image
        average since QC is config-toggleable (qc.watershed) and will
        almost always be left on for review.

    RESUME CONTRACT
        Morphometrics output is tracked per segmented-directory/model.
        A segmented directory is considered complete only when every
        segmented TIFF has its corresponding per-image morphometric
        workbook. Complete datasets are skipped on subsequent runs;
        incomplete datasets are reprocessed as a unit so that the
        in-memory image ordering required by watershed QC remains
        aligned.

    BATCH CONTRACT
        This function contains the actual processing workflow so that
        batch_axon can invoke morphometrics directly without launching
        an interactive subprocess.
    """
    run_start = time.time()

    input_dir = Path(input_dir).expanduser()

    if not input_dir.is_dir():
        console.print(f"[red]✗  Folder not found: {input_dir}[/red]")
        return StageResult(False, time.time() - run_start, 0)

    try:
        study, study_dir = resolve_scan(str(input_dir))
    except ValueError as e:
        console.print(f"[red]✗  {e}[/red]")
        return StageResult(False, time.time() - run_start, 0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = load_config()

    # qc.watershed only controls the QC figure; it does not change which
    # axons are accepted (that is MORPHOMETRICS in utils/recipe.py).
    qc_cfg = config.get("qc", {})
    watershed_qc_on = qc_cfg.get("watershed", True)

    logging_cfg = config.get("logging", {})
    logging_on = (
        logging_cfg.get("morphometrics", False)
        if isinstance(logging_cfg, dict)
        else bool(logging_cfg)
    )

    log_path = (
        str(Path(input_dir) / f"morphometrics_log_{timestamp}.txt")   # the folder the user entered
        if logging_on
        else None
    )
    log = AxonIDLogger(
        log_path=log_path,
        program="AxonID Morphometrics",
    )

    if not quiet_banner:
        log.info(f"Folder: {input_dir}")
        for line in format_scan_config_summary():
            log.info(line)

    # ── Scan study ────────────────────────────────────────────────────────────
    log.rule("SCANNING STUDY")

    mag = detect_study_mag(study)
    morph_folder = config.get("morphometrics_folder", "Morphometrics")
    morph_suffix = config.get("morphometrics_suffix", "_morphometrics")
    seg_suffix = config.get("segmented_suffix", "_segmented")
    seg_folder = config.get("segmented_folder", "Segmented")

    if mag is None:
        log.error(
            "No magnification found: no nerve folder contains an image folder ending in one of "
            f"config.toml tiff_suffixes {config.get('tiff_suffixes')} (e.g. 40X_TIFF). Check the "
            "folder names or tiff_suffixes."
        )
        return StageResult(False, time.time() - run_start, 0)
    log.info(f"Detected magnification: [bold]{mag}[/bold]")
    if mag not in SUPPORTED_MAGS:
        log.warn(f"AxonID's released models and calibration workflow support {list(SUPPORTED_MAGS)}; "
                 f"{mag} results are not validated.")
    for level, line in format_morphometrics_config_summary(mag):
        (log.warn if level == "warn" else log.info)(line)

    no_seg = []
    to_process = []
    to_skip = []

    for animal, nerves in study.items():
        for nerve, info in nerves.items():
            seg_dirs = info["segmented_dir"]

            if not seg_dirs:
                no_seg.append(f"{animal}/{nerve}")
                continue

            for seg_dir in seg_dirs:
                seg_dir = Path(seg_dir)

                suffix = seg_dir.name.removeprefix(seg_folder)   # e.g. "Segmented_M1" -> "_M1"
                morph_dir = seg_dir.parent / f"{morph_folder}{suffix}"

                seg_images = list_files(
                    str(seg_dir),
                    extensions=(".tif", ".tiff"),
                )

                if not seg_images:
                    log.warn(f"No segmented images found in {seg_dir}")
                    continue

                expected_outputs = [
                    morph_dir
                    / f"{_strip_suffix(img_path.stem, seg_suffix)}{morph_suffix}.xlsx"
                    for img_path in seg_images
                ]

                # Complete only when every segmented TIFF has its own
                # workbook; a partial run is redone as a unit.
                has_morph_output = (
                    morph_dir.is_dir()
                    and all(path.is_file() for path in expected_outputs)
                )

                item = (
                    animal,
                    nerve,
                    info,
                    seg_dir,
                    morph_dir,
                    suffix,
                    seg_images,
                )

                if has_morph_output:
                    to_skip.append(item)
                else:
                    to_process.append(item)

    if no_seg:
        log.warn(
            f"No segmentation found — {len(no_seg)} nerve(s) skipped"
        )

    if to_skip:
        log.warn(
            f"Skipping {len(to_skip)} already-processed "
            f"segmented dataset(s):"
        )
        for item in to_skip:
            animal, nerve, _, seg_dir, _, _, _ = item
            log.warn(f"  {animal}/{nerve} — {seg_dir.name}")

    if not to_process:
        if to_skip:
            log.success(
                "All segmented datasets already have complete morphometrics. "
                "Nothing to do."
            )
        else:
            log.warn(
                "No segmented datasets found, so nothing was processed. Each nerve "
                "needs a Segmented folder containing <image name>_segmented.tif files "
                "(README, Folder layout and file names)."
            )

        log.finalize(summary={
            "Folder": str(input_dir),
            "Magnification": mag,
            "Nerves — no segmentation": len(no_seg),
            "Datasets — already processed": len(to_skip),
            "Datasets — processed this run": 0,
            "Images processed": 0,
            "Images failed": 0,
            "Elapsed time": "0m 0s",
            "Avg time per image": "0.0s",
        }, title="MORPHOMETRICS COMPLETE")

        if log_path:
            console.print(
                f"\n[dim]Log saved to: {log_path}[/dim]"
            )

        return StageResult(True, time.time() - run_start, 0)

    log.info(f"To process: {len(to_process)} segmented dataset(s)")

    # ── Process each segmented dataset ────────────────────────────────────────
    total_images = 0
    total_failed = 0
    morph_extraction_elapsed = 0.0
    qc_elapsed = 0.0
    distribution_elapsed = 0.0

    current_animal = None

    for (
        animal,
        nerve,
        info,
        seg_dir,
        morph_dir,
        suffix,
        seg_images,
    ) in to_process:

        if animal != current_animal:
            console.rule(
                animal,
                characters="═",
                style="white",
            )
            current_animal = animal

        nerve_path = seg_dir.parent
        nerve_mag = info.get("mag") or mag

        log.rule(
            f"{nerve}" if not suffix else f"{nerve} — {seg_dir.name}"
        )

        log.info(f"Found {len(seg_images)} segmented image(s)")

        # ── Resolution check ─────────────────────────────────────────────────
        resolutions = {}

        for img_path in seg_images:
            try:
                w, h = get_image_resolution(str(img_path))
                resolutions[img_path.name] = (w, h)
            except Exception as e:
                log.warn(
                    f"Could not read resolution for "
                    f"{img_path.name}: {e}"
                )

        unique_res = set(resolutions.values())

        if len(unique_res) > 1:
            log.warn("Resolution mismatch detected:")

            for name, res in resolutions.items():
                log.warn(
                    f"  {name}: {res[0]}×{res[1]}"
                )

        nerve_success = 0

        # Collect:
        #   (df, axon_labels, fiber_labels, filtered_xy)
        # for each image for optional watershed QC.
        all_data = []

        # Model names recovered from segmentation TIFF provenance.
        nerve_models = set()
        n_legacy_calibration = 0

        # ── Per-image morphometrics ───────────────────────────────────────────
        for img_path in seg_images:
            stem = _strip_suffix(img_path.stem, seg_suffix)

            res = resolutions.get(
                img_path.name,
                ("?", "?"),
            )

            log.info(
                f"  Processing {img_path.name} "
                f"[{res[0]}×{res[1]}]..."
            )
            img_start = time.time()
            try:
                # Segmentation provenance: model name for the workbook, and
                # the original size / resize scale for calibration.
                provenance = read_tiff_provenance(img_path)
                if res[0] == "?":
                    res = get_image_resolution(str(img_path))
                calibration = resolve_pixel_size(nerve_mag, res[0], res[1], provenance)
                if calibration.source == "legacy":
                    n_legacy_calibration += 1

                if watershed_qc_on:
                    (
                        df,
                        raw_df,
                        axon_lbl,
                        fiber_lbl,
                        filtered_xy,
                    ) = get_morphometrics(
                        str(img_path),
                        nerve_mag,
                        log,
                        return_labels=True,
                        calibration=calibration,
                    )

                    all_data.append(
                        (
                            df,
                            axon_lbl,
                            fiber_lbl,
                            filtered_xy,
                        )
                    )

                else:
                    df, raw_df = get_morphometrics(
                        str(img_path),
                        nerve_mag,
                        log,
                        calibration=calibration,
                    )

                    all_data.append(
                        (
                            df,
                            None,
                            None,
                            [],
                        )
                    )

                if df is None:
                    raise RuntimeError(
                        f"Morphometric calculation returned no DataFrame "
                        f"for {img_path.name}"
                    )

                if provenance.get("model_name"):
                    nerve_models.add(
                        provenance["model_name"]
                    )

                out = save_morphometrics(
                    df,
                    raw_df,
                    str(morph_dir),
                    stem,
                    provenance=provenance,
                    calibration=calibration,
                )

                log.success(
                    f"  → {len(df)} axons → "
                    f"{Path(out).name}"
                )

                total_images += 1
                nerve_success += 1
                morph_extraction_elapsed += time.time() - img_start

            except Exception as e:
                log.error(
                    f"  → FAILED: {e}"
                )

                all_data.append(
                    (
                        None,
                        None,
                        None,
                        [],
                    )
                )

                total_failed += 1
                morph_extraction_elapsed += time.time() - img_start

        if n_legacy_calibration:
            log.warn(
                f"{n_legacy_calibration} segmented image(s) have no resize record (segmented by an older "
                f"version): pixel size looked up at the segmented width, assuming no downsampling."
            )

        # ── Watershed QC sheet ────────────────────────────────────────────────
        if watershed_qc_on and nerve_success > 0:
            qc_start = time.time()
            try:
                save_watershed_qc(
                    seg_images=seg_images,
                    all_data=all_data,
                    morph_dir=morph_dir,
                    nerve_name=nerve_path.name,
                    suffix=suffix,
                    mag=nerve_mag,
                    log=log,
                )

            except Exception as e:
                log.error(
                    f"Watershed QC failed: {e}"
                )
                total_failed += 1
            qc_elapsed += time.time() - qc_start
                

        # ── Nerve-level diameter distributions ───────────────────────────────
        if nerve_success > 0:
            log.rule(
                f"DISTRIBUTIONS — {nerve}{suffix}"
            )
            
            dist_start = time.time()
            try:
                data = bin_nerve_diameters(
                    morph_dir,
                    nerve_path.name,
                    nerve_mag,
                    log,
                    model_name=(
                        ", ".join(sorted(nerve_models))
                        or "unknown"
                    ),
                )

                if data is not None:
                    dist_out = save_distributions(
                        data,
                        str(morph_dir),
                        nerve_path.name,
                    )

                    log.success(
                        f"  → {Path(dist_out).name}"
                    )

                else:
                    log.warn(
                        f"  → No distribution data for {nerve}"
                    )

            except Exception as e:
                log.error(
                    f"  → Distribution FAILED: {e}"
                )
                total_failed += 1
            distribution_elapsed += time.time() - dist_start

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - run_start
    core_elapsed = morph_extraction_elapsed + distribution_elapsed
    avg_per_img = (
        core_elapsed / total_images
        if total_images > 0
        else 0
    )

    log.finalize(summary={
        "Folder": str(input_dir),
        "Magnification": mag,
        "Nerves — no segmentation": len(no_seg),
        "Datasets — already processed": len(to_skip),
        "Datasets — processed this run": len(to_process),
        "Images processed": total_images,
        "Images failed": total_failed,
        "Morphometrics extraction time": f"{int(morph_extraction_elapsed // 60)}m {int(morph_extraction_elapsed % 60)}s",
        "Distribution/binning time":     f"{int(distribution_elapsed // 60)}m {int(distribution_elapsed % 60)}s",
        "QC time (excluded from avg)":   f"{int(qc_elapsed // 60)}m {int(qc_elapsed % 60)}s",
        "Elapsed time": (
            f"{int(elapsed // 60)}m "
            f"{int(elapsed % 60)}s"
        ),
        "Avg time per image (core, excl. QC)": f"{avg_per_img:.1f}s",
    }, title="MORPHOMETRICS COMPLETE")

    if log_path:
        console.print(
            f"\n[dim]Log saved to: {log_path}[/dim]"
        )

    return StageResult(
        success=(total_failed == 0),
        elapsed_seconds=elapsed,
        images_processed=total_images,
        morphometrics_seconds=morph_extraction_elapsed,
        distributions_seconds=distribution_elapsed,
        qc_seconds=qc_elapsed,
    )
    

def main():
    print_panel(
        console,
        Panel(
            Align.center(
                "[bold white]"
                "Automated Axon-Myelin Histomorphometry "
                "of Brightfield Images"
                "[/bold white]"
            ),
            title="[bold cyan]AxonID — MORPHOMETRICS[/bold cyan]",
            border_style="bright_cyan",
            box=DOUBLE,
            expand=True,
            padding=(1, 4),
        ),
    )

    while True:
        input_dir = input(
            "\nInput the path to the study, animal, or nerve folder "
            "containing the segmented images: "
        ).strip().strip('"')

        if not os.path.isdir(input_dir):
            console.print(
                f"[red]✗  Folder not found: {input_dir}[/red]"
            )
            continue

        try:
            resolve_scan(input_dir)
            break

        except ValueError as e:
            console.print(
                f"[red]✗  {e}[/red]"
            )
            continue

    if not run(input_dir).success:
        sys.exit(1)
        
if __name__ == "__main__":
    main()