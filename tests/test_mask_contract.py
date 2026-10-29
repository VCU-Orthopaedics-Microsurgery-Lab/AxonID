"""BGW mask values: snapping rule and class-index mapping used by training."""
import numpy as np

from utils.mask_contract import BGW_VALID, off_contract_count, snap_to_bgw
from train.dataset.data_loader import base_label


def test_snap_cut_points():
    ramp = np.arange(256, dtype=np.uint8)
    snapped = snap_to_bgw(ramp)
    assert set(np.unique(snapped)) == {0, 128, 255}
    assert (snapped[:64] == 0).all()
    assert (snapped[64:192] == 128).all()
    assert (snapped[192:] == 255).all()


def test_valid_values_unchanged():
    vals = np.array([0, 128, 255], dtype=np.uint8)
    assert (snap_to_bgw(vals) == vals).all()
    assert snap_to_bgw(np.array([127], np.uint8))[0] == 128     # 127 is myelin


def test_off_contract_count():
    arr = np.array([0, 127, 128, 255, 1, 200, 64], dtype=np.uint8)
    assert off_contract_count(arr) == 3
    assert set(BGW_VALID) == {0, 127, 128, 255}


def test_base_label_mapping():
    arr = np.array([0, 127, 128, 255, 30, 100, 220], dtype=np.float32)
    assert base_label(arr).tolist() == [0, 1, 1, 2, 0, 1, 2]
