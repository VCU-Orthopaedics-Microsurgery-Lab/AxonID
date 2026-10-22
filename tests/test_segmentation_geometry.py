"""Patch geometry shared by training and segmentation (no model needed)."""
import numpy as np
import pytest

from segment.segment import get_pos, hann_window, recolor
from utils.image import center_crop_to_patch_grid
from utils.patching import get_hann_compatible_step


def test_step_is_half_patch():
    assert get_hann_compatible_step(256) == 128
    with pytest.raises(ValueError):
        get_hann_compatible_step(255)


def test_crop_to_patch_grid():
    img = np.zeros((1024, 1418), np.uint8)
    assert center_crop_to_patch_grid(img, 256).shape == (1024, 1280)
    with pytest.raises(ValueError):
        center_crop_to_patch_grid(np.zeros((100, 300), np.uint8), 256)


def test_hann_weights_cover_every_pixel():
    # Accumulate the position-aware windows over a 3 x 4 patch grid at 50%
    # overlap: every pixel must receive weight, or blending would divide by 0.
    rows, cols, p, step = 3, 4, 256, 128
    acc = np.zeros(((rows - 1) * step + p, (cols - 1) * step + p))
    for i in range(rows):
        for j in range(cols):
            acc[i * step:i * step + p, j * step:j * step + p] += hann_window(get_pos((rows, cols), i, j), p)
    assert acc.min() > 0.5 and acc.max() < 1.05


def test_recolor_bgw():
    out = recolor(np.array([[0, 1, 2]]))
    assert out[..., 0].tolist() == [[0, 128, 255]]
