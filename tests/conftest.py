"""
Shared fixtures. Every test uses synthetic data made inside the test; no
study images, masks or models are needed.

Run from the AxonID folder:  python -m pytest
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def temp_config(tmp_path, monkeypatch):
    """Point AxonID at a config.toml written by the test; returns a writer function."""
    import utils.config as C

    def write(text: str) -> Path:
        path = tmp_path / "config.toml"
        path.write_text(text, encoding="utf-8")
        monkeypatch.setattr(C, "_CONFIG_PATH", path)
        monkeypatch.setattr(C, "_OLD_CONFIG_PATH", tmp_path / "config.json")
        monkeypatch.setattr(C, "_config_cache", None)
        return path

    yield write
    C._config_cache = None


@pytest.fixture
def log():
    from utils.logger import AxonIDLogger
    return AxonIDLogger(log_path=None, program="AxonID tests")


def draw_fibers(shape, centres, r_fiber, r_axon):
    """BGW image with one myelinated fiber (128 ring, 255 core) per centre."""
    import numpy as np
    img = np.zeros(shape, np.uint8)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    for cy, cx in centres:
        d = np.hypot(yy - cy, xx - cx)
        img[d <= r_fiber] = 128
    for cy, cx in centres:
        d = np.hypot(yy - cy, xx - cx)
        img[d <= r_axon] = 255
    return img
