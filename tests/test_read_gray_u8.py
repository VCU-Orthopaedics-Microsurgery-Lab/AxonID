"""tests/test_read_gray_u8.py: bit depth and channel handling of the source image reader."""

import cv2
import numpy as np
import pytest
import tifffile

from utils.image import read_gray_u8
from utils.resize import resize_img_with_scale


def _scaled(ramp, full):
    """The 8-bit ramp stored unscaled at a lower bit depth (full = 4095 for 12-bit)."""
    return (ramp.astype(np.uint32) * full // 255).astype(np.uint16)


@pytest.fixture
def ramp():
    """A 64 x 64 8-bit image using the full 0-255 range."""
    return np.arange(64 * 64, dtype=np.uint32).reshape(64, 64).__mod__(256).astype(np.uint8)


def _write(path, arr, **kw):
    tifffile.imwrite(str(path), arr, **kw)
    return str(path)


def test_8bit_gray_is_read_unchanged(tmp_path, ramp):
    p = _write(tmp_path / "a.tif", ramp)
    assert np.array_equal(read_gray_u8(p), ramp)


def test_8bit_rgb_matches_opencv_grayscale(tmp_path, ramp):
    rgb = np.stack([ramp, ramp // 2, ramp // 3], axis=-1)
    p = _write(tmp_path / "a.tif", rgb, photometric="rgb")
    assert np.array_equal(read_gray_u8(p), cv2.imread(p, cv2.IMREAD_GRAYSCALE))


def test_16bit_full_range_is_divided_by_256(tmp_path, ramp):
    p = _write(tmp_path / "a.tif", ramp.astype(np.uint16) * 257)
    assert np.array_equal(read_gray_u8(p), ramp)


def test_12bit_stored_unscaled_is_rescaled_with_warning(tmp_path, ramp):
    data = _scaled(ramp, 4095)
    p = _write(tmp_path / "a.tif", data)
    with pytest.warns(UserWarning, match="12-bit"):
        out = read_gray_u8(p)
    assert np.abs(out.astype(int) - ramp.astype(int)).max() <= 1


def test_14bit_stored_unscaled_is_rescaled_with_warning(tmp_path, ramp):
    data = _scaled(ramp, 16383)
    p = _write(tmp_path / "a.tif", data)
    with pytest.warns(UserWarning, match="14-bit"):
        out = read_gray_u8(p)
    assert np.abs(out.astype(int) - ramp.astype(int)).max() <= 1


def test_12bit_rgb_is_rescaled_with_warning(tmp_path, ramp):
    a = _scaled(ramp, 4095)
    p = _write(tmp_path / "a.tif", np.stack([a, a, a], axis=-1), photometric="rgb")
    with pytest.warns(UserWarning, match="12-bit"):
        out = read_gray_u8(p)
    assert np.abs(out.astype(int) - ramp.astype(int)).max() <= 1


def test_warning_goes_to_the_logger_when_given(tmp_path, ramp):
    class Log:
        def __init__(self):
            self.msgs = []

        def warn(self, m):
            self.msgs.append(m)

    p = _write(tmp_path / "a.tif", _scaled(ramp, 4095))
    log = Log()
    read_gray_u8(p, log=log)
    assert len(log.msgs) == 1 and "rescaled" in log.msgs[0]


def test_float_image_is_rejected(tmp_path, ramp):
    p = _write(tmp_path / "a.tif", ramp.astype(np.float32) / 255)
    with pytest.raises(ValueError, match="floating-point"):
        read_gray_u8(p)


def test_four_channel_image_is_rejected(tmp_path, ramp):
    four = np.stack([ramp, ramp, ramp, ramp], axis=-1)
    p = _write(tmp_path / "a.tif", four, photometric="minisblack", planarconfig="contig")
    with pytest.raises(ValueError, match="unsupported layout"):
        read_gray_u8(p)


def test_missing_file_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Could not read"):
        read_gray_u8(str(tmp_path / "nope.tif"))


def test_resize_reads_images_through_the_checked_reader_but_masks_plainly(tmp_path, ramp):
    img = _write(tmp_path / "img.tif", _scaled(ramp, 4095))
    with pytest.warns(UserWarning):
        resized, scale, _ = resize_img_with_scale(img, is_mask=False)
    assert resized.dtype == np.uint8 and scale == 1.0
    mask = _write(tmp_path / "mask.tif", np.where(ramp > 128, 255, 0).astype(np.uint8))
    out, _, _ = resize_img_with_scale(mask, is_mask=True)   # no warning, no rescale
    assert set(np.unique(out)) <= {0, 255}