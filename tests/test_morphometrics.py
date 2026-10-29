"""Watershed separation, axon/fiber pairing and g-ratio on synthetic fibers of known size."""
import numpy as np
import pytest
import tifffile

from conftest import draw_fibers
from morphometrics.morphometrics import get_morphometrics
from utils.calibration import Calibration

PX = 0.2     # µm per pixel for these tests


def run(tmp_path, img, log):
    p = tmp_path / "seg.tif"
    tifffile.imwrite(p, img)
    cal = Calibration(PX, img.shape[1], img.shape[0], 1.0, "provenance")
    return get_morphometrics(str(p), "40X", log, calibration=cal)


def grid(n, spacing, offset):
    return [(offset + r * spacing, offset + c * spacing) for r in range(n) for c in range(n)]


def test_isolated_fibers_count_and_size(tmp_path, log):
    r_fiber, r_axon = 15, 9
    centres = grid(6, 50, 40)
    df, _ = run(tmp_path, draw_fibers((340, 340), centres, r_fiber, r_axon), log)
    assert len(df) == len(centres)
    # Equivalent diameters of rasterised discs are within ~3% of 2r.
    assert df["axon_diam_um"].median() == pytest.approx(2 * r_axon * PX, rel=0.03)
    assert df["fiber_equiv_diam_um"].median() == pytest.approx(2 * r_fiber * PX, rel=0.03)
    assert df["gratio_equiv_diam"].median() == pytest.approx(r_axon / r_fiber, abs=0.02)
    assert df["myelin_thickness_um"].median() == pytest.approx((r_fiber - r_axon) * PX, rel=0.06)


def test_touching_fibers_are_separated(tmp_path, log):
    # Pairs of fibers whose myelin sheaths touch (centres 2 x r_fiber apart).
    r_fiber, r_axon = 15, 9
    centres = []
    for row in range(3):
        y = 40 + row * 60
        centres += [(y, 40), (y, 40 + 2 * r_fiber)]
    df, _ = run(tmp_path, draw_fibers((200, 140), centres, r_fiber, r_axon), log)
    assert len(df) == len(centres)


def test_hard_floor_rejects_tiny_axons(tmp_path, log):
    # Axon equivalent diameter 2 * 2 px * 0.2 = 0.8 µm < 1.01 µm floor.
    df, _ = run(tmp_path, draw_fibers((120, 120), grid(2, 50, 35), 6, 2), log)
    assert len(df) == 0


def test_output_columns_and_units(tmp_path, log):
    df, pixel_df = run(tmp_path, draw_fibers((120, 120), grid(2, 50, 35), 15, 9), log)
    for col in ("original_resolution", "resize_scale", "px_size_um", "gratio", "gratio_mean_axes"):
        assert col in df.columns
    assert df["px_size_um"].iloc[0] == pytest.approx(PX)
    # Pixel sheet holds the same pairs, in pixels.
    assert len(pixel_df) == len(df)
    assert pixel_df["axon_diam"].median() * PX == pytest.approx(df["axon_diam_um"].median(), rel=1e-3)


def test_empty_image(tmp_path, log):
    df, _ = run(tmp_path, np.zeros((64, 64), np.uint8), log)
    assert df.empty
