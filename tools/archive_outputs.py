"""
tools/archive_outputs.py: move a study's outputs aside so a stage can be redone safely.

Purpose
    AxonID skips work that is already done, so redoing a stage (for example
    morphometrics after drawing a CSA tracing) means clearing that stage's
    outputs and everything built on them. This tool does that the same way
    every time and never deletes anything: outputs are moved into an archive
    folder next to the study, from where they can be restored.

Usage
    python -m tools.archive_outputs                                   prompts for everything
    python -m tools.archive_outputs --input STUDY --from morphometrics --dry-run
    python -m tools.archive_outputs --input STUDY --from morphometrics --yes
    python -m tools.archive_outputs --restore STUDY_archive/20261023_143000

    --from says where to redo from; that stage and every later stage are archived:
        segment         Segmented*/, Morphometrics*/, QC/, the Data workbooks
        morphometrics   Morphometrics*/, QC/, the Data workbooks
        compile         the Data workbooks only (<animal>_Data.xlsx, <study>_Data.xlsx)
    Without --yes the plan is shown and you are asked to confirm; --dry-run
    shows the plan and stops.

Inputs
    A study, animal or nerve folder in the AxonID layout (see README).

Outputs
    <study folder name>_archive/<YYYYmmdd_HHMMSS>/ next to the study folder (not
    inside it, so it is never mistaken for an animal), holding the moved items
    under their original relative paths, plus manifest.json for --restore.
    Never touched: the image folders (<mag>_TIFF), CSA/ tracings, models, and
    anything not named in the table above.

Depends on
    utils/study.py, utils/config.py, utils/input.py
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ── Local ─────────────────────────────────────────────────────────────────────
from utils.config import load_config
from utils.input import get_path_input, get_yes_no
from utils.study import resolve_scan

STAGES = ("segment", "morphometrics", "compile")      # in pipeline order
MANIFEST = "manifest.json"


def find_outputs(nerve_paths: list[Path], study_dir: Path, from_stage: str, config: dict) -> list[Path]:
    """
    List the output folders and files to archive for a stage and everything after it.

    Args:
        nerve_paths: nerve folders in scope
        study_dir:   study folder (holds <study>_Data.xlsx)
        from_stage:  "segment", "morphometrics" or "compile"
        config:      loaded config (folder names)

    Returns:
        Existing paths, sorted. Image folders, CSA folders and anything not
        recognised as an output are never listed.
    """
    seg_name  = config.get("segmented_folder", "Segmented").lower()
    morph_name = config.get("morphometrics_folder", "Morphometrics").lower()
    qc_name   = config.get("qc_folder", "QC").lower()
    tiff_sfx  = tuple(s.upper() for s in config.get("tiff_suffixes", ["_TIFF"]))
    start     = STAGES.index(from_stage)

    found: list[Path] = []
    animals: set[Path] = set()
    for nerve in nerve_paths:
        animals.add(nerve.parent)
        if start > 1:
            continue
        for d in sorted(nerve.iterdir()):
            name = d.name.lower()
            if not d.is_dir() or d.name.upper().endswith(tiff_sfx):
                continue
            if (start == 0 and name.startswith(seg_name)) or name.startswith(morph_name) or name == qc_name:
                found.append(d)

    # Data workbooks are built from the nerves' results, so they go at every stage.
    for folder in sorted(animals | {study_dir}):
        found.extend(sorted(p for p in folder.glob("*_Data.xlsx") if p.is_file()))
    return found


def archive(items: list[Path], study_dir: Path, from_stage: str, now: datetime | None = None) -> Path:
    """
    Move items into <study>_archive/<timestamp>/ keeping their relative paths.

    Args:
        items:      paths from find_outputs()
        study_dir:  study folder
        from_stage: recorded in the manifest
        now:        timestamp (for tests)

    Returns:
        The archive run folder. Raises FileExistsError rather than overwrite.
    """
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    run = study_dir.parent / f"{study_dir.name}_archive" / stamp
    if run.exists():
        raise FileExistsError(f"{run} already exists; wait a second and run again.")
    run.mkdir(parents=True)

    moves = []
    for src in items:
        rel = src.relative_to(study_dir)
        dst = run / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moves.append(rel.as_posix())

    (run / MANIFEST).write_text(json.dumps(
        {"study": str(study_dir), "from": from_stage, "created": stamp, "moves": moves}, indent=2),
        encoding="utf-8")
    return run


def restore(run: Path) -> tuple[list[str], list[str]]:
    """
    Move an archive run's items back to where they came from.

    Args:
        run: an archive run folder (contains manifest.json)

    Returns:
        (restored, skipped): relative paths. An item is skipped, not
        overwritten, when something already exists at its original place.
    """
    manifest = json.loads((run / MANIFEST).read_text(encoding="utf-8"))
    study = Path(manifest["study"])
    restored, skipped = [], []
    for rel in manifest["moves"]:
        src, dst = run / rel, study / rel
        if dst.exists() or not src.exists():
            skipped.append(rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        restored.append(rel)
    return restored, skipped


def _choose_stage() -> str:
    print("\nRedo from which stage? That stage and everything after it is archived.")
    print("  1  segment        Segmented*, Morphometrics*, QC, Data workbooks")
    print("  2  morphometrics  Morphometrics*, QC, Data workbooks")
    print("  3  compile        Data workbooks only")
    while True:
        raw = input("Choose 1-3: ").strip().lower()
        if raw in ("1", "2", "3"):
            return STAGES[int(raw) - 1]
        if raw in STAGES:
            return raw
        print("  Please enter 1, 2 or 3.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Move a study's outputs aside so a stage can be redone.")
    parser.add_argument("--input", metavar="FOLDER", help="study, animal or nerve folder")
    parser.add_argument("--from", dest="from_stage", choices=STAGES, help="stage to redo from")
    parser.add_argument("--dry-run", action="store_true", help="show what would move and stop")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    parser.add_argument("--restore", metavar="ARCHIVE_RUN", help="move an archive run back")
    args = parser.parse_args()

    if args.restore:
        run = Path(args.restore.strip().strip('"'))
        if not (run / MANIFEST).is_file():
            print(f"Not an archive run (no {MANIFEST}): {run}")
            sys.exit(1)
        restored, skipped = restore(run)
        print(f"Restored {len(restored)} item(s).")
        for rel in skipped:
            print(f"  not restored (something already exists there, or it is gone): {rel}")
        sys.exit(1 if skipped else 0)

    folder = Path(args.input.strip().strip('"')) if args.input else get_path_input("Path to the study, animal or nerve folder: ")
    if not folder.is_dir():
        print(f"Folder not found: {folder}")
        sys.exit(1)
    stage = args.from_stage or _choose_stage()

    try:
        study, study_dir = resolve_scan(str(folder))
    except ValueError as e:
        print(e)
        sys.exit(1)
    study_dir = Path(study_dir)
    nerves = [Path(info["tiff_dir"]).parent for animal in study.values() for info in animal.values()]
    items = find_outputs(nerves, study_dir, stage, load_config())

    if not items:
        print("Nothing to archive: no outputs found for that stage.")
        return
    print(f"\nWill move {len(items)} item(s) (nothing is deleted):")
    for p in items:
        print(f"  {p.relative_to(study_dir)}")
    print(f"into {study_dir.parent / (study_dir.name + '_archive')}{Path('/')}<timestamp>/")
    if args.dry_run:
        return
    if not args.yes and not get_yes_no("Move these now?", default=False):
        print("Nothing moved.")
        return
    run = archive(items, study_dir, stage)
    print(f"Moved. To undo: python -m tools.archive_outputs --restore \"{run}\"")


if __name__ == "__main__":
    main()