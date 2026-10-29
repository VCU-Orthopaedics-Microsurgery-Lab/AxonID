"""Pixel size of segmented images, corrected for segmentation downsampling."""
import json

import numpy as np
import pytest
import tifffile

from utils.calibration import read_tiff_provenance, resolve_pixel_size

CONFIG = """
[pixel_size_um.40X]
1440 = 0.217
2880 = 0.1085
"""


def test_provenance_corrects_for_downsampling(temp_config):
    temp_config(CONFIG)
    prov = {"original_width": 2880, "original_height": 2048, "resize_scale": 0.5}
    cal = resolve_pixel_size("40X", 1440, 1024, prov)
    assert cal.source == "provenance"
    assert cal.px_um == pytest.approx(0.217)          # 0.1085 / 0.5
    assert (cal.original_width, cal.resize_scale) == (2880, 0.5)


def test_not_downsampled(temp_config):
    temp_config(CONFIG)
    prov = {"original_width": 1440, "original_height": 1024, "resize_scale": 1.0}
    assert resolve_pixel_size("40X", 1440, 1024, prov).px_um == pytest.approx(0.217)


def test_legacy_lookup_at_segmented_width(temp_config):
    temp_config(CONFIG)
    cal = resolve_pixel_size("40X", 1440, 1024, {})
    assert cal.source == "legacy" and cal.px_um == pytest.approx(0.217) and cal.note


def test_missing_calibration_gives_none(temp_config):
    temp_config(CONFIG)
    prov = {"original_width": 2000, "original_height": 1500, "resize_scale": 0.72}
    assert resolve_pixel_size("40X", 1440, 1080, prov).px_um is None
    assert resolve_pixel_size("100X", 1440, 1024, {}).px_um is None


def test_read_tiff_provenance(tmp_path):
    with_prov, without = tmp_path / "a.tif", tmp_path / "b.tif"
    img = np.zeros((8, 8), np.uint8)
    tifffile.imwrite(with_prov, img, description=json.dumps({"resize_scale": 0.5}))
    tifffile.imwrite(without, img)
    assert read_tiff_provenance(with_prov) == {"resize_scale": 0.5}
    # tifffile stores its own {"shape": ...} note when no description is given;
    # what matters is that no resize record is invented.
    assert "resize_scale" not in read_tiff_provenance(without)
    assert read_tiff_provenance(tmp_path / "missing.tif") == {}
