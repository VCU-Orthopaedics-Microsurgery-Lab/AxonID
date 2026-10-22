"""
utils/patching.py: patch step (overlap) shared by training and segmentation.

Usage
    from utils.patching import get_hann_compatible_step
    step = get_hann_compatible_step(256)   # 128

Rule
    step = patch_size // 2 (50% overlap). Segmentation's Hann-window
    blending (segment/segment.py) sums to a constant only at exactly 50%
    overlap, so this must not change on its own. Odd or non-positive patch
    sizes raise ValueError.

Locked (utils/recipe.py)
    patch_size
"""

def get_hann_compatible_step(patch_size: int) -> int:
    """
    Return the patch step size used by Hann-window reconstruction.

    The current reconstruction convention uses 50% overlap
    (step = patch_size // 2). The position-aware Hann weighting in
    segment/segment.py is defined around this overlap convention.

    Changing the overlap requires corresponding changes to the
    reconstruction geometry and should not be done independently.

    Raises:
        ValueError: if patch_size is not a positive, even integer — an
            odd value would silently break the exact-50%-overlap
            guarantee by half a pixel.
    """
    if patch_size <= 0 or patch_size % 2 != 0:
        raise ValueError(
            f"patch_size must be a positive, even integer for exact "
            f"50% overlap; got {patch_size}."
        )
    return patch_size // 2