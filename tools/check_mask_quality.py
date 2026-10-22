"""
tools/check_mask_quality.py: check a folder of BGW masks before training or evaluation.

Usage
    python -m tools.check_mask_quality --path "C:/path/to/masks"
    python -m tools.check_mask_quality --path "C:/path/to/masks" --images-path "C:/path/to/images"

Inputs
    Folder of BGW masks (.png/.tif/.tiff); optionally the matching images
    folder (a sibling images/ folder is found automatically).

Checks
    Readability, grayscale mode, pixel values (0 / 127 / 128 / 255),
    resolution consistency, class balance, classes with no pixels, and
    matching image names and sizes.

Outputs
    Results on the console and a timestamped text report next to the masks
    folder. Values outside the contract are reported with the class
    training will snap them to (utils/mask_contract.snap_to_bgw). Masks
    and images are never modified.

Depends on
    numpy, pillow, utils/mask_contract.py
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console

from utils.input import get_path_input
from utils.mask_contract import BGW_AXON, BGW_BG, BGW_MYELIN, BGW_MYELIN_CANONICAL, BGW_VALID, snap_to_bgw

console = Console()

VALID_EXTENSIONS = (".tif", ".tiff", ".png")
# Pillow modes accepted as grayscale input. This is a mode check only;
# BGW pixel validity is checked separately against BGW_VALID.
GRAYSCALE_MODES = ("L", "1", "I", "I;16")

# Below this fraction of the dominant image size, a resolution is flagged as
# an outlier rather than just "different" — catches a handful of stray
# small crops in an otherwise consistent full-resolution folder.
DEGENERATE_CLASS_THRESHOLD_PCT = 0.0  # a class at exactly 0% is flagged as degenerate
# Same threshold train.py uses to warn about heavy snapping.
SNAP_WARN_PCT = 5.0


def find_mask_files(folder: Path) -> list[Path]:
    return sorted(
        [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS],
        key=lambda f: f.name,
    )


def find_sibling_images_dir(masks_dir: Path) -> Path | None:
    """Looks for a conventional sibling 'images' folder next to 'masks'
    (e.g. .../Final Hold Out Set/masks -> .../Final Hold Out Set/images)."""
    for candidate_name in ("images", "Images", "IMAGES", "imgs"):
        candidate = masks_dir.parent / candidate_name
        if candidate.is_dir():
            return candidate
    return None


def check_one_mask(path: Path, images_dir: Path | None) -> dict:
    result = {
        "file": path.name,
        "error": None,
        "size": None,
        "mode": None,
        "off_contract_values": [],
        "snap": None,          # off-contract pixels and the class training snaps them to
        "class_pct": None,
        "degenerate_classes": [],
        "image_match": None,   # None = not checked, True/False = checked
        "image_size_match": None,
    }

    try:
        img_pil = Image.open(path)
        img_pil.load()  # force-read pixel data now, not lazily later
    except Exception as e:  # noqa: BLE001 - report any failure, don't crash the run
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    result["size"] = img_pil.size  # (W, H)
    result["mode"] = img_pil.mode

    arr = np.array(img_pil.convert("L") if img_pil.mode not in GRAYSCALE_MODES else img_pil)
    unique_vals = np.unique(arr)
    off_contract = [int(v) for v in unique_vals if v not in BGW_VALID]
    result["off_contract_values"] = off_contract
    if off_contract:
        off_mask = ~np.isin(arr, BGW_VALID)
        snapped  = snap_to_bgw(arr[off_mask])
        result["snap"] = {
            "off_contract_pct": round(float(off_mask.sum()) / arr.size * 100, 3),
            "to_bg":     int((snapped == BGW_BG).sum()),
            "to_myelin": int((snapped == BGW_MYELIN_CANONICAL).sum()),
            "to_axon":   int((snapped == BGW_AXON).sum()),
        }

    total = arr.size
    bg_n = (arr == BGW_BG).sum()
    myelin_n = np.isin(arr, BGW_MYELIN).sum()
    axon_n = (arr == BGW_AXON).sum()
    class_pct = {
        "bg_pct": round(float(bg_n) / total * 100, 3),
        "myelin_pct": round(float(myelin_n) / total * 100, 3),
        "axon_pct": round(float(axon_n) / total * 100, 3),
        "unaccounted_pct": round(float(total - (bg_n + myelin_n + axon_n)) / total * 100, 3),
    }
    result["class_pct"] = class_pct
    degenerate = [
    name
    for name, pct in (
        ("myelin", class_pct["myelin_pct"]),
        ("axon", class_pct["axon_pct"]),
    )
    if pct <= DEGENERATE_CLASS_THRESHOLD_PCT
]
    result["degenerate_classes"] = degenerate

    if images_dir is not None:
        match = None
        for ext in (".tif", ".tiff", ".png", ".TIF", ".TIFF", ".PNG"):
            candidate = images_dir / f"{path.stem}{ext}"
            if candidate.exists():
                match = candidate
                break
        result["image_match"] = match is not None
        if match is not None:
            try:
                with Image.open(match) as im:
                    result["image_size_match"] = (im.size == result["size"])
            except Exception:  # noqa: BLE001
                result["image_size_match"] = None

    return result


def run(masks_dir: Path, images_dir: Path | None) -> list[dict]:
    files = find_mask_files(masks_dir)
    if not files:
        console.print(f"[red]No .tif/.tiff/.png files found in {masks_dir}[/red]")
        return []

    console.print(f"\nChecking {len(files)} masks in: {masks_dir}")
    if images_dir is not None:
        console.print(f"Cross-checking against sibling images folder: {images_dir}")
    else:
        console.print("[yellow]No sibling images/ folder found — skipping image cross-check.[/yellow]")

    results = [check_one_mask(f, images_dir) for f in files]

    size_counts: dict[tuple, int] = {}
    for r in results:
        if r["size"] is not None:
            size_counts[r["size"]] = size_counts.get(r["size"], 0) + 1
    dominant_size = max(size_counts, key=size_counts.get) if size_counts else None

    console.print()
    for r in results:
        flags = []
        if r["error"]:
            console.print(f"[red]✗ {r['file']}: FAILED TO LOAD — {r['error']}[/red]")
            continue

        if r["mode"] not in GRAYSCALE_MODES:
            flags.append(f"non-grayscale mode '{r['mode']}'")
        if r["off_contract_values"]:
            sn = r["snap"]
            flags.append(
                f"off-contract pixel values {r['off_contract_values'][:10]}"
                f"{' ...' if len(r['off_contract_values']) > 10 else ''}: {sn['off_contract_pct']}% of pixels, "
                f"snapped in training to bg {sn['to_bg']} / myelin {sn['to_myelin']} / axon {sn['to_axon']}"
                f"{'  <-- more than ' + str(SNAP_WARN_PCT) + '%, check the mask colours' if sn['off_contract_pct'] > SNAP_WARN_PCT else ''}"
            )
        if dominant_size is not None and r["size"] != dominant_size:
            flags.append(f"resolution {r['size']} differs from folder-dominant {dominant_size}")
        if r["degenerate_classes"]:
            flags.append(f"zero-pixel class(es): {', '.join(r['degenerate_classes'])}")
        if r["image_match"] is False:
            flags.append("no matching image found in images/ folder")
        if r["image_size_match"] is False:
            flags.append("mask/image resolution mismatch")

        cp = r["class_pct"]
        line = (
            f"{r['file']}  size={r['size']} mode={r['mode']}  "
            f"bg={cp['bg_pct']}% myelin={cp['myelin_pct']}% "
            f"axon={cp['axon_pct']}%"
        )
        if cp["unaccounted_pct"] > 0:
            line += f" unaccounted={cp['unaccounted_pct']}%"

        if flags:
            console.print(f"[yellow]⚠ {line}[/yellow]")
            for f in flags:
                console.print(f"    - {f}")
        else:
            console.print(f"[green]✓[/green] {line}")

    console.print(f"\n--- Summary ({len(results)} masks) ---")
    n_errors = sum(1 for r in results if r["error"])
    n_off_contract = sum(1 for r in results if r["off_contract_values"])
    n_non_grayscale = sum(1 for r in results if r["mode"] and r["mode"] not in GRAYSCALE_MODES)
    n_size_outliers = sum(1 for r in results if r["size"] is not None and r["size"] != dominant_size)
    n_degenerate = sum(1 for r in results if r["degenerate_classes"])
    n_no_image_match = sum(1 for r in results if r["image_match"] is False)
    n_image_size_mismatch = sum(1 for r in results if r["image_size_match"] is False)

    console.print(f"  Failed to load:              {n_errors}")
    console.print(f"  Off-contract pixel values:   {n_off_contract}")
    console.print(f"  Non-grayscale mode:          {n_non_grayscale}")
    console.print(f"  Resolution outliers:         {n_size_outliers}  (dominant size: {dominant_size})")
    console.print(f"  Degenerate (zero-pixel) class: {n_degenerate}")
    if images_dir is not None:
        console.print(f"  No matching image found:     {n_no_image_match}")
        console.print(f"  Mask/image size mismatch:    {n_image_size_mismatch}")

    return results


def write_report(masks_dir: Path, results: list[dict], images_dir: Path | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = masks_dir.parent / f"mask_qc_report_{masks_dir.name}_{timestamp}.txt"    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Mask QC report — {masks_dir}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("BGW contract: 0=background, 127/128=myelin, 255=axon\n")
        f.write(f"Images cross-check dir: {images_dir}\n")
        f.write(f"n masks: {len(results)}\n\n")
        for r in results:
            f.write(f"{r['file']}\n")
            if r["error"]:
                f.write(f"  ERROR: {r['error']}\n\n")
                continue
            f.write(f"  size={r['size']} mode={r['mode']}\n")
            f.write(f"  off_contract_values={r['off_contract_values']}\n")
            f.write(f"  snap={r['snap']}\n")
            f.write(f"  class_pct={r['class_pct']}\n")
            f.write(f"  degenerate_classes={r['degenerate_classes']}\n")
            f.write(f"  image_match={r['image_match']} image_size_match={r['image_size_match']}\n\n")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="QA check for BGW ground-truth masks before training/holdout.")
    parser.add_argument("--path", type=str, default=None, help="Path to the masks folder (skips the interactive prompt).")
    parser.add_argument("--images-path", type=str, default=None, help="Optional explicit path to the matching images folder.")
    args = parser.parse_args()

    masks_dir = Path(args.path) if args.path else get_path_input("Enter path to the masks folder: ")
    if not masks_dir.is_dir():
        console.print(f"[red]Not a folder: {masks_dir}[/red]")
        return

    if args.images_path:
        images_dir = Path(args.images_path)
        if not images_dir.is_dir():
            console.print(f"[yellow]--images-path given but not found: {images_dir} — skipping cross-check[/yellow]")
            images_dir = None
    else:
        images_dir = find_sibling_images_dir(masks_dir)

    results = run(masks_dir, images_dir)
    if results:
        report_path = write_report(masks_dir, results, images_dir)
        console.print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()