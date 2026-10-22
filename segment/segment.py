"""
segment/segment.py: patch-based inference that turns micrographs into BGW masks.

Purpose
    Downsample each image to the working resolution, cut it into 256 px
    patches with 50% overlap, predict class probabilities per patch, blend
    the patches back with a position-aware Hann window, and write a BGW TIFF
    with provenance metadata.

Usage
    python -m segment                          (entry point, interactive)
    from segment.segment import load_model, segment_dir, segment_image

Inputs
    TIFF micrographs (Toluidine Blue, brightfield, 40X); a checkpoint built
    with the AxonID recipe (models/models.json or models/user/).
Outputs
    <nerve>/Segmented/<image>_segmented.tif: uint8 BGW (0 background,
    128 myelin, 255 axon) at the downsampled resolution, with JSON provenance
    in the TIFF description (model, version, original size, resize scale).
    <nerve>/QC/<nerve>_segmentation_qc.png when QC is on.

Processing
    downsample to fit 1440x1024 (aspect kept; never upscaled)
    -> reflect-pad bottom/right to complete the patch grid
    -> per-patch L2 normalization (inference convention, see below)
    -> model -> softmax probabilities
    -> Hann-weighted probability blending, normalized by accumulated weight
    -> argmax -> crop back to the downsampled size -> BGW recolor

Locked (utils/recipe.py, not configurable)
    Architecture, encoder, patch size, overlap, working resolution,
    normalization. Checkpoints built any other way are refused.

Depends on
    utils/recipe.py         build_model(), check_checkpoint_compatible(), RECIPE
    utils/patching.py       patch step (exactly 50% overlap)
    utils/resize.py         downsampling and the recorded scale factor
    utils/mask_contract.py  BGW values (via recolor(); morphometrics reads them)
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import tifffile
import torch
from patchify import patchify
from torch.serialization import add_safe_globals
from torch.torch_version import TorchVersion

# ── Local ─────────────────────────────────────────────────────────────────────
from utils.config import load_config
from utils.filesystem import list_files
from utils.logger import AxonIDLogger
from utils.mask_contract import BGW_AXON
from utils.model_io import load_model_file
from utils.patching import get_hann_compatible_step
from utils.recipe import (
    NORMALIZATION_INFERENCE,
    RECIPE,
    RECIPE_ID,
    build_model,
    check_checkpoint_compatible,
)
from utils.resize import get_image_resolution, resize_img, resize_img_with_scale
from utils.version import __version__

add_safe_globals([TorchVersion])


@dataclass
class NerveSegmentResult:
    """
    Return type for segment_dir(). Local to segment.py/segment.__main__ —
    unlike StageResult, nothing outside these two files needs this shape,
    so it doesn't belong in utils/study.py.
    """
    success: bool
    seg_seconds: float
    qc_seconds: float


# ─── Hann window (position-aware) ────────────────────────────────────────────
# 9-position grid: corners (0,2,6,8), edges (1,3,5,7), center (4)
# Each position gets an asymmetric taper so that overlap is blended
# toward the image interior while the outer image boundary remains
# fully weighted.
# _hann_fn maps pixel index → weight in [0, 1] using a cosine taper.
def _hann_fn(x, patch_size=256):
    return (1 - np.cos(2 * np.pi * x / (patch_size - 1))) / 2


def get_pos(shape, i, j):
    """
    Return the 3x3 position code for a patch in the reconstruction grid.

    POSITION MAP
        0  1  2
        3  4  5
        6  7  8

    The position-aware Hann weighting assumes the normal AxonID
    reconstruction geometry with multiple patch rows and columns.
    Degenerate 1-row or 1-column grids are handled by hann_window()'s
    multi_row/multi_col flags, which neutralize the taper in whichever
    dimension has no second patch to blend against.
    """
    i_max = shape[0] - 1
    j_max = shape[1] - 1
    if   i == 0     and j == 0:     return 0
    elif i == 0     and j == j_max: return 2
    elif i == i_max and j == 0:     return 6
    elif i == i_max and j == j_max: return 8
    elif i == 0:                    return 1
    elif i == i_max:                return 7
    elif j == 0:                    return 3
    elif j == j_max:                return 5
    else:                           return 4


@lru_cache(maxsize=16)
def hann_window(pos, patch_size=256, multi_row=True, multi_col=True):
    """
    Build the position-aware Hann weight map for one patch.

    POSITION MAP
        0  1  2
        3  4  5
        6  7  8

    Outer image boundaries remain fully weighted; boundaries shared with
    overlapping patches taper toward zero. Regions adjacent to two
    overlapping boundaries intentionally multiply vertical and horizontal
    tapers.

    The patch is partitioned into four non-overlapping midpoint regions
    (c1–c4) so each pixel receives exactly one weighting rule.
    
    multi_row / multi_col: whether the reconstruction grid has more than
    one patch row / column. A degenerate dimension has no neighboring
    patch to blend toward, so its taper must be neutralized to 1.0 —
    otherwise the far corner/edge of that patch gets a near-zero weight.
    """
    half = patch_size // 2
    # Matrix indexing: i is the row (vertical) coordinate and j is the
    # column (horizontal) coordinate. Therefore hi is the vertical taper
    # and hj is the horizontal taper.
    i, j = np.meshgrid(
        np.arange(patch_size),
        np.arange(patch_size),
        indexing='ij'
    ) 
    
    # Four non-overlapping regions around the patch midpoint.
    #
    #     c1 | c3
    #     ---+---
    #     c2 | c4
    #
    # Midpoint inequalities are intentionally asymmetric; c4 contains the
    # remaining pixels so the four regions cover the patch exactly once.
    
    c1 = (i <= half) & (j <= half)
    c2 = (i > half)  & (j < half)
    c3 = (i < half)  & (j > half)
    c4 = ~c1 & ~c2 & ~c3

    s = np.zeros((patch_size, patch_size), dtype=float)

    # hi: vertical/row taper
    # hj: horizontal/column taper
    hi = _hann_fn(i.astype(float), patch_size)
    hj = _hann_fn(j.astype(float), patch_size)
    
    # No second patch to blend against in a degenerate dimension —
    # taper must be neutral (1.0), not applied.
    if not multi_row:
        hi = np.ones_like(hi)
    if not multi_col:
        hj = np.ones_like(hj)

    # Position-specific weighting; products apply both directional tapers
    # where a region borders two overlapping patch boundaries.
    if pos == 0:
        s[c1] = 1
        s[c2] = hi[c2]
        s[c3] = hj[c3]
        s[c4] = hi[c4] * hj[c4]

    elif pos == 1:
        s[c1] = hj[c1]
        s[c2] = hi[c2] * hj[c2]
        s[c3] = hj[c3]
        s[c4] = hi[c4] * hj[c4]

    elif pos == 2:
        s[c1] = hj[c1]
        s[c2] = hi[c2] * hj[c2]
        s[c3] = 1
        s[c4] = hi[c4]

    elif pos == 3:
        s[c1] = hi[c1]
        s[c2] = hi[c2]
        s[c3] = hi[c3] * hj[c3]
        s[c4] = hi[c4] * hj[c4]

    elif pos == 4:
        s[c1] = hi[c1] * hj[c1]
        s[c2] = hi[c2] * hj[c2]
        s[c3] = hi[c3] * hj[c3]
        s[c4] = hi[c4] * hj[c4]

    elif pos == 5:
        s[c1] = hi[c1] * hj[c1]
        s[c2] = hi[c2] * hj[c2]
        s[c3] = hi[c3]
        s[c4] = hi[c4]

    elif pos == 6:
        s[c1] = hi[c1]
        s[c2] = 1
        s[c3] = hi[c3] * hj[c3]
        s[c4] = hj[c4]

    elif pos == 7:
        s[c1] = hi[c1] * hj[c1]
        s[c2] = hj[c2]
        s[c3] = hi[c3] * hj[c3]
        s[c4] = hj[c4]

    elif pos == 8:
        s[c1] = hi[c1] * hj[c1]
        s[c2] = hj[c2]
        s[c3] = hi[c3]
        s[c4] = 1

    return s


# ─── Recolor ──────────────────────────────────────────────────────────────────
def recolor(pred):
    """
    Map class indices to BGW pixel values.

    BGW contract: 0=background(black), 1→128=myelin(grey), 2→255=axon(white)
    morphometrics/morphometrics.py inRange() thresholds depend on this mapping.
    Update both files if class assignments ever change.
    """
    out = np.zeros(pred.shape, dtype=np.uint8)
    out[pred == 1] = 128          # canonical myelin value (utils/mask_contract.py)
    out[pred == 2] = BGW_AXON     # 255
    return np.stack([out, out, out], axis=-1)


# ─── Model loader ─────────────────────────────────────────────────────────────
def load_model(
    model_path: str,
    device: torch.device,
    log: AxonIDLogger | None = None
) -> tuple:
    """
    Load an AxonID model file (.safetensors) and return (model, meta).

    The checkpoint must carry AxonID metadata with the locked architecture and
    encoder (utils/recipe.check_checkpoint_compatible). There is no fallback:
    a checkpoint that cannot be identified is refused, because loading it into
    the wrong network would produce wrong segmentations without an error.
    """
    model_path = Path(model_path).resolve()
    # .safetensors holds only tensors and text metadata: loading cannot run code.
    state_dict, meta = load_model_file(model_path)

    check_checkpoint_compatible(meta, model_path.name)

    model = build_model(encoder_weights=None)   # weights come from the checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    # eval(): BatchNorm uses its stored statistics and dropout is off, so a
    # patch's prediction does not depend on any other patch.
    model.eval()

    if log:
        log.rule("LOADING MODEL")
        log.info(f"Model      : {model_path.name}")
        log.info(f"Recipe     : {meta.get('recipe_id', RECIPE_ID + ' (pre-recipe checkpoint)')}")
        trained_where = [str(v) for v in (meta.get('gpu'), meta.get('hostname'))
                         if v and v != 'not recorded']
        log.info(
            f"Trained    : {meta.get('trained_date', '?')}"
            + (f" on {', '.join(trained_where)}" if trained_where else "")
        )
        log.info(
            f"Mag        : {meta.get('magnification', '?')} | "
            f"Patch: {meta.get('patch_size', RECIPE.patch_size)}px"
        )
        log.info(
            f"Best epoch : {meta.get('best_epoch', '?')} | "
            f"Axon dice: {_fmt4(meta.get('best_axon_dice'))} | "
            f"Myelin dice: {_fmt4(meta.get('best_myelin_dice'))}"
        )

    return model, meta


def _fmt4(value) -> str:
    """Format a metric with 4 decimals, or n/a when it is missing."""
    return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"


# ─── Segment single image ─────────────────────────────────────────────────────
def _pad_for_patchify(size, patch_size, step):             
    if size <= patch_size:                                  
        return patch_size - size                           
    n_steps = (size - patch_size + step - 1) // step       
    needed  = n_steps * step + patch_size                  
    return max(0, needed - size)                            


def segment_image(img_path, model, patch_size=RECIPE.patch_size, log=None):
    """
    Segment one image. Returns (bgw_rgb, seconds, height, width, scale,
    (original_width, original_height)), where height/width are the
    downsampled size the mask is written at and scale is the downsampling
    factor applied (1.0 if the image was not larger than 1440x1024).
    """
    t0   = time.time()
    step = get_hann_compatible_step(patch_size)

    # Downsample to the working resolution. This is the only step that changes
    # um per pixel; padding and the final crop do not.
    img, scale, original_size = resize_img_with_scale(img_path, is_mask=False)
    orig_h, orig_w = img.shape[:2]

    # ── Pad to complete the patch grid ────────────────────────────────────
    pad_h = _pad_for_patchify(orig_h, patch_size, step) 
    pad_w = _pad_for_patchify(orig_w, patch_size, step)      
    # Edge padding contract:
    # Reflect padding is added only as needed to complete the patch grid.
    # For images larger than one patch, the padding in each dimension is
    # guaranteed to be smaller than one patch step. With the current
    # get_hann_compatible_step() convention (step = patch_size // 2),
    # padding is therefore <50% of the patch size. Padding is applied only
    # to the bottom and right edges and is removed from the final prediction
    # by cropping back to the original image dimensions.
    img_proc = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')

    proc_h, proc_w = img_proc.shape[:2]

    # resize_img_with_scale() returns 2D uint8 grayscale, so patchify() uses a
    # 2D patch shape.
    patches        = patchify(img_proc, (patch_size, patch_size), step=step)
    n_rows, n_cols = patches.shape[:2]

    # S-04: accumulate one Hann-weighted probability map per class.
    # Shape = (class, height, width).
    pred_img   = np.zeros((3, proc_h, proc_w), dtype=np.float32)
    weight_img = np.zeros((proc_h, proc_w), dtype=np.float32) 

    for i in range(n_rows):
        for j in range(n_cols):
            patch = patches[i, j, :, :]
            # Inference normalization: one L2 norm per patch ROW (axis=1 of an
            # H x W patch). Training normalizes per patch COLUMN instead
            # (train/dataset/data_loader.py). The difference is deliberate:
            # this convention segmented better in a direct test (2026-08-14),
            # and every released result uses it. Do not change it to match
            # training without re-testing. See utils/recipe.py.
            norms = np.linalg.norm(patch, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            patch = patch / norms
            # Inference is currently one patch at a time for direct reconstruction.
            # Batching could improve throughput, but must preserve patch order,
            # per-patch normalization, and position-aware Hann weighting.
            patch = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
            patch = patch.to(next(model.parameters()).device)
            with torch.no_grad():
                logits = model(patch)

            # S-04: blend class probabilities, not categorical class indices.
            # Model output is raw logits because activation=None in load_model().
            # Softmax converts each patch prediction to probabilities for:
            # 0=background, 1=myelin, 2=axon.
            probabilities = torch.softmax(logits, dim=1)
            # Reconstruction uses NumPy accumulators, so probabilities return to CPU.
            # Batched inference can reduce transfer overhead without changing blending.
            probabilities = probabilities.squeeze(0).cpu().numpy()
            
            assert probabilities.shape[0] == 3, (
                f"Expected 3 classes, got {probabilities.shape[0]}"
            )

            # Position is based on the complete patch grid. Final patches may extend
            # into reflected padding, but their grid positions remain unchanged.
            pos = get_pos((n_rows, n_cols), i, j)
            hann = hann_window(
                pos, patch_size,
                multi_row=(n_rows > 1),
                multi_col=(n_cols > 1),
            )

            i_start = i * step
            j_start = j * step

            # Accumulate each class probability using the same spatial Hann weights.
            for class_idx in range(probabilities.shape[0]):
                pred_img[
                    class_idx,
                    i_start:i_start + patch_size,
                    j_start:j_start + patch_size
                ] += probabilities[class_idx] * hann

            weight_img[
                i_start:i_start + patch_size,
                j_start:j_start + patch_size
            ] += hann

    # Normalize each accumulated class-probability map by the same
    # accumulated Hann weights used during reconstruction.
    weight_img = np.maximum(weight_img, np.finfo(float).eps)

    pred_img = pred_img / weight_img[None, :, :]

    # Select the class with the highest blended probability.
    pred_img = np.argmax(pred_img, axis=0)

    # Crop back to original dimensions before recoloring.
    pred_img = pred_img[:orig_h, :orig_w]

    result  = recolor(pred_img)
    elapsed = time.time() - t0
    return result, elapsed, orig_h, orig_w, scale, original_size


# ─── Save Overlays ────────────────────────────────────────────────────────
def save_qc_sheet(
    tiff_dir: Path,
    output_dir: Path,
    images: list,
    seg_suffix: str,
    model_name: str,
    log: AxonIDLogger = None
):
    """
    Generate a QC sheet showing original / segmented / overlay for each image.
    Saved as {nerve_name}_segmentation_qc.png in a QC/ subfolder

    Layout: one row per image, three columns — original | segmented | overlay
    Overlay: axon=pink, myelin=purple, background=original grayscale image
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend — no display needed
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        if log:
            log.warn("matplotlib not available — skipping QC sheet")
        return

    config     = load_config()
    qc_folder  = config.get("qc_folder", "QC")
    nerve_name = tiff_dir.parent.name
    qc_dir     = tiff_dir.parent / qc_folder
    qc_dir.mkdir(parents=True, exist_ok=True)

    n_images = len(images)
    if n_images == 0:
        return

    fig, axes = plt.subplots(n_images, 3, figsize=(24, 8 * n_images))
    if n_images == 1:
        axes = [axes]  # ensure iterable rows

    fig.suptitle(
        f"{nerve_name}  |  model: {model_name}",
        fontsize=14, fontweight='bold', y=1.0
    )

    col_titles = ["Original", "Segmented", "Overlay"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=11, fontweight='bold', pad=8)

    for row, img_path in enumerate(images):
        seg_path = output_dir / f"{img_path.stem}{seg_suffix}.tif"

        # Load original
        orig = resize_img(str(img_path), is_mask=False)

        # Load segmented
        if not seg_path.exists():
            if log:
                log.warn(f"QC: segmented file not found: {seg_path}")
            continue
        seg = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
        if seg is None:
            continue

        # QC only: resize the display copy of the original if dimensions differ.
        # The saved segmentation is never resized or modified here.
        if orig.shape != seg.shape:
            orig = cv2.resize(orig, (seg.shape[1], seg.shape[0]))

        # Build overlay — RGB
        overlay = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB).astype(np.float32)
        alpha_axon   = 0.35                                                        
        alpha_myelin = 0.55  

        # Axon → pink (255, 105, 180)
        axon_mask = (seg == 255)[..., None]  # Shape: (H, W, 1)
        overlay = np.where(
            axon_mask,
            overlay * (1 - alpha_axon) + np.array([255, 105, 180], dtype=np.float32) * alpha_axon,
            overlay
        )

        # Myelin → purple (160, 80, 200)
        myelin_mask = (seg == 128)[..., None]  # Shape: (H, W, 1)
        overlay = np.where(
            myelin_mask,
            overlay * (1 - alpha_myelin) + np.array([160, 80, 200], dtype=np.float32) * alpha_myelin,
            overlay
        )
        overlay = overlay.astype(np.uint8)

        # Plot
        axes[row][0].imshow(orig, cmap='gray', vmin=0, vmax=255)  
        axes[row][0].axis('off')                                   

        axes[row][1].imshow(seg, cmap='gray', vmin=0, vmax=255)   
        axes[row][1].set_title(img_path.name, fontsize=8,          
                               fontweight='normal', pad=4)         
        axes[row][1].axis('off')                                   

        axes[row][2].imshow(overlay)
        axes[row][2].axis('off')

    # Legend on last row overlay panel
    legend = [
        mpatches.Patch(color=(1.0, 0.41, 0.71), label='Axon'),
        mpatches.Patch(color=(0.63, 0.31, 0.78), label='Myelin'),
    ]
    axes[-1][2].legend(handles=legend, loc='lower right', fontsize=9, framealpha=0.7)

    plt.tight_layout()

    out_path = qc_dir / f"{nerve_name}_segmentation_qc.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)

    if log:
        log.success(f"QC sheet saved → {out_path}")


# ─── Segment directory ────────────────────────────────────────────────────────
def segment_dir(tiff_dir, output_dir, model, mag, log, timing_csv=None, model_name='unknown', meta=None):
    """
    Segment every TIFF in tiff_dir, writing BGW output to output_dir.

    Returns:
        NerveSegmentResult — success is True only if every image
        segmented without error. seg_seconds and qc_seconds are summed
        across all images/this nerve's QC sheet; see module docstring's
        TIMING section.
    """
    meta = meta or {}
    config         = load_config()
    patch_size     = RECIPE.patch_size
    seg_suffix     = config.get("segmented_suffix", "_segmented")
    step           = get_hann_compatible_step(patch_size)
    overlap_pct    = 100 - int(step / patch_size * 100)

    tiff_dir   = Path(tiff_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_files(str(tiff_dir))   # default extensions: .tif, .tiff, .png (utils/filesystem.py)
    if not images:
        log.warn(f"No TIFF images found in {tiff_dir}")
        return NerveSegmentResult(
            success=False, seg_seconds=0.0, qc_seconds=0.0
        )

    log.rule(f"SEGMENTING {tiff_dir.parent.name} / {tiff_dir.name}")
    logging_cfg = config.get("logging", {})
    logging_on  = logging_cfg.get("segment", False) if isinstance(logging_cfg, dict) else bool(logging_cfg)

    log.info(
        f"Found {len(images)} image(s) | patch={patch_size}px | "
        f"step={step}px ({overlap_pct}% overlap) | "
        f"Logging={'ON' if logging_on else 'OFF'} | "
        f"Timing={'ON' if timing_csv else 'OFF'}"
    )

    # Resolution check
    resolutions = {}
    for img_path in images:
        try:
            w, h = get_image_resolution(str(img_path))
            resolutions[img_path.name] = (w, h)
        except Exception as e:
            log.warn(f"Could not read resolution for {img_path.name}: {e}")

    unique_res = set(resolutions.values())
    if len(unique_res) > 1:
        log.warn("Resolution mismatch detected:")
        for name, res in resolutions.items():
            log.warn(f"  {name}: {res[0]}x{res[1]}")
        log.warn("Continuing with mismatched resolutions.")

    # Timing is accumulated in memory and written after the batch completes;
    # interrupted runs do not retain unwritten timing records.
    timing_rows = []
    success     = 0
    failed      = 0
    seg_seconds_total = 0.0

    # Provenance shared by every image in this folder; per-image size and
    # scale are added below. Morphometrics reads original_width and
    # resize_scale to convert the user's calibration (measured on the original
    # image) to the downsampled mask.
    _provenance = {
        'AxonID_version':   __version__,
        'recipe_id':        meta.get('recipe_id', RECIPE_ID),
        'model_name':       model_name,
        'architecture':     meta.get('architecture', '?'),
        'encoder':          meta.get('encoder', '?'),
        'segmented_date':   datetime.now().strftime('%Y-%m-%d'),
        'patch_size':       patch_size,
        'normalization':    NORMALIZATION_INFERENCE,
        'magnification':    mag,
    }

    for img_path in images:
        stem     = img_path.stem
        out_path = output_dir / f"{stem}{seg_suffix}.tif"
        res_str  = "x".join(str(x) for x in resolutions.get(img_path.name, ('?', '?')))

        try:
            mask, elapsed, orig_h, orig_w, scale, (src_w, src_h) = segment_image(
                str(img_path), model,
                patch_size=patch_size,
                log=log,
            )
            scale_note = f" (downsampled from {src_w}x{src_h}, scale {scale:.4f})" if scale != 1.0 else ""
            log.success(f"{img_path.name} [{orig_w}x{orig_h}]{scale_note} -> {elapsed:.1f}s")
            seg_seconds_total += elapsed

            provenance = {
                **_provenance,
                'original_width':  src_w,
                'original_height': src_h,
                'resize_scale':    scale,
            }
            tifffile.imwrite(
                str(out_path),
                mask,
                # Provenance must remain JSON-serializable; do not silently stringify
                # unexpected objects, which could hide malformed metadata.
                description=json.dumps(provenance),
            )
            
            timing_rows.append({
                'image': img_path.name, 'resolution': res_str,
                'crop_size': f"{orig_w}x{orig_h}", 'patch_size': patch_size,
                'time_s': f"{elapsed:.2f}", 'status': 'ok'
            })
            success += 1
            
        except Exception as e:
            log.error(f"{img_path.name} - FAILED: {e}")
            timing_rows.append({
                'image': img_path.name, 'resolution': res_str,
                'crop_size': '', 'patch_size': patch_size,
                'time_s': '', 'status': f'FAILED: {e}'
            })
            failed += 1

    if timing_csv:
        with open(timing_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'image', 'resolution', 'crop_size', 'patch_size', 'time_s', 'status'
            ])
            writer.writeheader()
            writer.writerows(timing_rows)
            
    log.success(f"Done - {success} succeeded, {failed} failed")
    if success > 0:
        avg_time = sum(float(r['time_s']) for r in timing_rows if r['status'] == 'ok') / success
        log.info(f"Average segmentation time: {avg_time:.1f}s per image")
    
    qc_start = time.time()    
    save_qc_sheet(
        tiff_dir=tiff_dir,
        output_dir=output_dir,
        images=images,
        seg_suffix=seg_suffix,
        model_name=model_name,
        log=log
    )
    qc_elapsed = time.time() - qc_start

    return NerveSegmentResult(
        success=(failed == 0 and success == len(images)),
        seg_seconds=seg_seconds_total,
        qc_seconds=qc_elapsed,
    )