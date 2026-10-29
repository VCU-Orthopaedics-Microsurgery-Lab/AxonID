"""
train/dataset/preprocess.py: turn training images and masks into patches.

Purpose
    Applies the same geometry segmentation uses (downsample to fit the
    working resolution, centre-crop to the patch grid, 50%-overlap patches)
    to every training image and its mask.

Usage
    from train.dataset.preprocess import batch_process
    n_img, n_mask = batch_process(images_dir, masks_dir, patches_img_dir,
                                  patches_mask_dir, cropped_img_dir, cropped_mask_dir, log)

Inputs
    images_dir   source images (.tif, .tiff, .png)
    masks_dir    BGW masks with the same file names (see utils/mask_contract.py)

Outputs
    {images,masks}/cropped/{stem}_cropped.(tif|png)     cropped full image
    {images,masks}/cropped/patches/{stem}_{RRCC}.png    256 x 256 patches

Behaviour
    Images larger than the working resolution are downsampled (LANCZOS4 for
    images, nearest-neighbour for masks so class values survive); smaller
    images are used at their own size. Mask values are not changed here:
    off-contract values are reported by train.py and snapped when patches
    are loaded (train/dataset/data_loader.py).

Locked (utils/recipe.py)
    patch_size, target_resolution; overlap is fixed at 50% (utils/patching.py)

Depends on
    patchify, utils/resize.py, utils/image.py, utils/patching.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
from pathlib import Path

# cv2 reads OPENCV_LOG_LEVEL at import time — must be set BEFORE "import cv2"
# below, or the suppression has no effect. Deliberately not grouped with the
# rest of the standard-library imports above for that reason.
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2
from patchify import patchify

# ── Local ─────────────────────────────────────────────────────────────────────
from utils.filesystem import natural_sort_key, list_files
from utils.image import center_crop_to_patch_grid
from utils.patching import get_hann_compatible_step
from utils.recipe import RECIPE
from utils.resize import get_image_resolution, resize_img, TARGET_SIZE


def process_single_image(
    img_path,
    patches_dir,
    cropped_dir,
    patch_size=256,
    is_mask=False,
    log=None,
    max_stem=0
):
    """
    Resize, centre-crop and patch one image or mask.

    Resize interpolation:
      Images: INTER_LANCZOS4 (via resize_img is_mask=False)
      Masks:  INTER_NEAREST  (via resize_img is_mask=True) — preserves label values

    Returns number of patches saved.
    """
    step = get_hann_compatible_step(patch_size)  # 50% overlap, same as segmentation

    img      = resize_img(img_path, is_mask=is_mask)
    img_crop = center_crop_to_patch_grid(img, patch_size)
    crop_h, crop_w = img_crop.shape[:2]

    # Save cropped image
    stem     = Path(img_path).stem
    crop_ext = '.tif' if not is_mask else '.png'
    Path(cropped_dir).mkdir(parents=True, exist_ok=True)
    crop_out = Path(cropped_dir) / f"{stem}_cropped{crop_ext}"
    cv2.imwrite(str(crop_out), img_crop)

    # Patchify with 50% overlap
    Path(patches_dir).mkdir(parents=True, exist_ok=True)
    patches = patchify(img_crop, (patch_size, patch_size), step=step)
    n_rows, n_cols = patches.shape[:2]

    count = 0
    for i in range(n_rows):
        for j in range(n_cols):
            patch     = patches[i, j]
            out_path  = Path(patches_dir) / f"{stem}_{i:02d}{j:02d}.png"
            cv2.imwrite(str(out_path), patch)
            count += 1

    if log:
        h_orig, w_orig = img.shape[:2]
        log.info(
            f"  {'[MASK]' if is_mask else '[IMG] '} {stem:<{max_stem}} | "
            f"Resized: ({h_orig}x{w_orig}) -> Cropped: ({crop_h}x{crop_w}) -> {count} patches"
        )

    return count


def batch_process(images_dir, masks_dir, patches_img_dir, patches_mask_dir,
                  cropped_img_dir, cropped_mask_dir, log):
    """
    Patch every image and its mask. Every image must have a mask with the
    same name and vice versa. Returns (n_img_patches, n_mask_patches).
    """
    patch_size  = RECIPE.patch_size
    step        = get_hann_compatible_step(patch_size)
    overlap_pct = 100 - int(step / patch_size * 100)

    images = sorted(list_files(images_dir, extensions=('.tif', '.tiff', '.png')), key=natural_sort_key)
    masks  = sorted(list_files(masks_dir,  extensions=('.tif', '.tiff', '.png')), key=natural_sort_key)

    # Checked in both directions: an orphan mask would be patched with no
    # image counterpart and break the image/mask pairing the loader relies on.
    img_stems      = {Path(p).stem for p in images}
    mask_stems     = {Path(p).stem for p in masks}
    missing_masks  = img_stems - mask_stems
    missing_images = mask_stems - img_stems
    if missing_masks or missing_images:
        problems = []
        if missing_masks:
            problems.append(f"{len(missing_masks)} image(s) with no mask: {sorted(missing_masks)}")
        if missing_images:
            problems.append(f"{len(missing_images)} mask(s) with no image: {sorted(missing_images)}")
        raise ValueError(" | ".join(problems))

    # Images smaller than the working resolution are used at their own size,
    # as in segmentation, so they give fewer patches than full-size images.
    small = []
    for img_path in images:
        w, h = get_image_resolution(str(img_path))
        if w < TARGET_SIZE[0] or h < TARGET_SIZE[1]:
            small.append(f"{Path(img_path).name} ({w}x{h})")
    if small:
        log.warn(
            f"{len(small)} image(s) are smaller than the working resolution "
            f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]} and are used at their own size: {small[:5]}"
        )

    max_stem = max(len(Path(p).stem) for p in images) if images else 0

    log.rule("IMAGE PREPROCESSING")
    log.info(f"Patch size: {patch_size}px | Interpolation: INTER_LANCZOS4 (image quality)")
    img_counts = []
    for img_path in images:
        img_counts.append(process_single_image(
            str(img_path), patches_img_dir, cropped_img_dir,
            patch_size=patch_size, is_mask=False, log=log, max_stem=max_stem
        ))
    total_img = sum(img_counts)

    log.rule("MASK PREPROCESSING")
    log.info(f"Patch size: {patch_size}px | Interpolation: INTER_NEAREST (mask values unchanged)")
    total_mask = 0
    for mask_path in masks:
        total_mask += process_single_image(
            str(mask_path), patches_mask_dir, cropped_mask_dir,
            patch_size=patch_size, is_mask=True, log=log, max_stem=max_stem
        )

    # ── Preprocessing summary ─────────────────────────────────────────────────
    log.rule("PREPROCESSING SUMMARY")
    if img_counts and min(img_counts) == max(img_counts):
        per_image = str(img_counts[0])
    else:
        per_image = f"{min(img_counts, default=0)}-{max(img_counts, default=0)}"
    log.info(f"Images: {len(images)} | Masks: {len(masks)} | Patches per image: {per_image}")
    log.info(f"Patch size: {patch_size}px | Overlap: {overlap_pct}% | Step: {step}px")
    log.info(f"Total patches: {total_img} image | {total_mask} mask")
    return total_img, total_mask
