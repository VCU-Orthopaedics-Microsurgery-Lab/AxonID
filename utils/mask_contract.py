"""
utils/mask_contract.py: the BGW mask pixel-value contract.

Purpose
    Defines the pixel values AxonID masks use, and the single rule for
    mapping any other grey value onto them.

Usage
    from utils.mask_contract import BGW_VALID, snap_to_bgw, off_contract_count

Inputs
    Grayscale mask arrays (any numeric dtype, values 0-255).

Outputs
    snap_to_bgw():        uint8 array containing only 0, 128 and 255
    off_contract_count(): number of pixels not in BGW_VALID

Contract
    0       background
    128     myelin (127 is accepted as myelin; it appears after anti-aliasing
            or rounding in some image editors)
    255     axon

    Any other value is snapped to the nearest class:
        0-63    -> 0   (background)
        64-191  -> 128 (myelin)
        192-255 -> 255 (axon)
    The cut points are the midpoints between the class values, so a pixel
    goes to whichever class value it is closest to.

Depends on
    numpy
"""

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np

# Part of the dataset format: changing these requires migrating every mask.
BGW_BG     = 0
BGW_MYELIN = (127, 128)   # both read as myelin; 128 is what AxonID writes
BGW_AXON   = 255
BGW_VALID  = (BGW_BG, *BGW_MYELIN, BGW_AXON)

BGW_MYELIN_CANONICAL = 128

# Snap cut points: midpoints between 0/128 and 128/255.
_SNAP_LOW  = 64
_SNAP_HIGH = 192


def snap_to_bgw(arr: np.ndarray) -> np.ndarray:
    """Map every pixel to 0, 128 or 255 (nearest class value)."""
    out = np.zeros(arr.shape, dtype=np.uint8)
    out[(arr >= _SNAP_LOW) & (arr < _SNAP_HIGH)] = BGW_MYELIN_CANONICAL
    out[arr >= _SNAP_HIGH] = BGW_AXON
    return out


def off_contract_count(arr: np.ndarray) -> int:
    """Number of pixels whose value is not in BGW_VALID (these get snapped)."""
    return int((~np.isin(arr, BGW_VALID)).sum())
