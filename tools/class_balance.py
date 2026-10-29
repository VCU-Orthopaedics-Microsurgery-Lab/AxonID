"""
tools/class_balance.py: class balance of a folder of BGW masks.

Usage
    python -m tools.class_balance

Inputs
    Folder of BGW masks (.png/.tif/.tiff): 0 background, 127/128 myelin,
    255 axon.

Outputs
    Per-image and overall background / myelin / axon percentages.
    Informational only: class weights are fixed by the recipe.

Depends on
    numpy, pillow, utils/mask_contract.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from utils.input import get_path_input
from utils.mask_contract import BGW_BG, BGW_MYELIN, BGW_AXON


def _fractions(img_path: Path) -> dict:
    img = np.array(Image.open(img_path).convert("L"))
    total = img.size

    return {
        "file": img_path.name,
        "bg_pct": round((img == BGW_BG).sum() / total * 100, 2),
        "myelin_pct": round(np.isin(img, BGW_MYELIN).sum() / total * 100, 2),
        "axon_pct": round((img == BGW_AXON).sum() / total * 100, 2),
    }


def run(images_dir: str) -> None:
    p = Path(images_dir)
    images = sorted(
        [
            f
            for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in (".png", ".tif", ".tiff")
        ],
        key=lambda f: f.name,
    )

    if not images:
        print(f"No image files found in {images_dir}")
        return

    rows = [_fractions(f) for f in images]

    header = f"{'File':<45} {'BG%':>7} {'Myelin%':>9} {'Axon%':>7}"
    print()
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['file']:<45} "
            f"{r['bg_pct']:>7.2f} "
            f"{r['myelin_pct']:>9.2f} "
            f"{r['axon_pct']:>7.2f}"
        )

    bg_vals = [r["bg_pct"] for r in rows]
    my_vals = [r["myelin_pct"] for r in rows]
    ax_vals = [r["axon_pct"] for r in rows]

    print()
    print(f"{'':45} {'BG%':>7} {'Myelin%':>9} {'Axon%':>7}")
    print(
        f"{'Mean':<45} "
        f"{np.mean(bg_vals):>7.2f} "
        f"{np.mean(my_vals):>9.2f} "
        f"{np.mean(ax_vals):>7.2f}"
    )
    print(
        f"{'SD':<45}  "
        f"{np.std(bg_vals):>7.2f}  "
        f"{np.std(my_vals):>9.2f} "
        f"{np.std(ax_vals):>7.2f}"
    )
    print(
        f"{'Min':<45}  "
        f"{np.min(bg_vals):>7.2f}  "
        f"{np.min(my_vals):>9.2f} "
        f"{np.min(ax_vals):>7.2f}"
    )
    print(
        f"{'Max':<45}  "
        f"{np.max(bg_vals):>7.2f}  "
        f"{np.max(my_vals):>9.2f} "
        f"{np.max(ax_vals):>7.2f}"
    )
    print(f"\nn = {len(rows)} images")


def main() -> None:
    images_dir = get_path_input("Enter path to directory: ")
    run(str(images_dir))


if __name__ == "__main__":
    main()
