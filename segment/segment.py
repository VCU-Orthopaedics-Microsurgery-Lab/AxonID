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


def segment_image(img_path, model, patch_size=RECIPE.patch_size, log=None,
                  return_confidence=False):
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
    img, scale, original_size = resize_img_with_scale(img_path, is_mask=False, log=log)
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

    # QC: highest blended class probability per pixel, 0-255 (255 = certain).
    confidence = None
    if return_confidence:
        confidence = (pred_img.max(axis=0)[:orig_h, :orig_w] * 255).astype(np.uint8)

    # Select the class with the highest blended probability.
    pred_img = np.argmax(pred_img, axis=0)

    # Crop back to original dimensions before recoloring.
    pred_img = pred_img[:orig_h, :orig_w]

    result  = recolor(pred_img)
    elapsed = time.time() - t0
    if return_confidence:
        return result, elapsed, orig_h, orig_w, scale, original_size, confidence
    return result, elapsed, orig_h, orig_w, scale, original_size


# ─── Save Overlays ────────────────────────────────────────────────────────
# ─── QC sheet ─────────────────────────────────────────────────────────────
# Everything you might want to tune is in this first block.

QC_FORMAT          = "png"    # "png" = lossless (larger file); "jpg" = smaller, lossy
QC_SHOW_ORIGINAL   = True     # False → overlay only (smallest file)
QC_SHOW_PREDICTION = True     # third panel: segmented image (top) + prediction map (below)
QC_UNCERTAIN_BELOW  = 0.75     # a foreground pixel (myelin/axon) counts as uncertain below this confidence
QC_UNCERTAIN_WARN   = 0.15     # the per-image number turns amber at/above this uncertain share
QC_PRED_LO         = 0.50     # prediction map: certainty at/below this is drawn as fully uncertain
QC_PRED_HI         = 0.95     # ... and at/above this as fully certain

# Overlay colors and opacity (0 = invisible, 1 = solid). Myelin is dark in the
# tissue, so it needs a higher opacity than the bright axon lumens to read well.
QC_AXON_RGB = (235, 190, 255)
QC_MYELIN_RGB = (190, 60, 210)
QC_ALPHA_AXON = 0.50
QC_ALPHA_MYELIN = 0.60
QC_OVERLAY_STYLE   = "fill"   # "fill" = translucent color; "outline" = thin boundaries only
QC_EDGE_SOFTNESS   = 0.8      # px of blur on fill edges (0 = hard edges)
QC_NORMALIZE_OVERLAY_BG = True   # stretch contrast of the overlay's background only;
                                 # the original panel is never altered

# CSA tracing (inclusion)
QC_ROI_RGB = (170, 255, 60)
QC_ROI_PX          = 2
QC_ROI_ON_ORIGINAL = False    # also draw the tracing on the original panel

# Least-certain detections: two outline tiers just outside the fiber.
# Tier 1 (deep orange) = the least-certain QC_FLAG1_PCT % of detections in the nerve;
# tier 2 (yellow)      = the next band, up to the QC_FLAG2_PCT-th percentile.
QC_FLAG1_RGB       = (255, 87, 34)
QC_FLAG2_RGB       = (255, 214, 0)
QC_FLAG1_PCT       = 5
QC_FLAG2_PCT       = 15
QC_FLAG1_PX = 1
QC_FLAG2_PX = 1
QC_OUTLINE_GAP     = 2        # px between the fiber edge and the outline
QC_FLAG_CEILING    = 0.75     # tier 2 never reaches above this mean confidence
QC_FLAG_FLOOR      = 0.50     # anything below this is always tier 1
QC_MIN_OBJECT_PX   = 30       # specks smaller than this are not scored

_QC_JPEG_MAX_PX    = 65000    # JPEG hard limit is 65,535 px per side
_QC_PNG_MAX_PX     = 100000   # PNG has no hard limit; this just bounds memory


def _qc_font(size: int, bold: bool = False):
    """PIL TrueType font: system font first, then matplotlib's bundled DejaVu."""
    from PIL import ImageFont
    names = (["DejaVuSans-Bold.ttf", "arialbd.ttf"] if bold
             else ["DejaVuSans.ttf", "arial.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        import matplotlib
        ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / (
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        return ImageFont.truetype(str(ttf), size)
    except Exception:
        return ImageFont.load_default()


def _qc_text(draw, xy, text, font, fill, anchor):
    try:
        draw.text(xy, text, font=font, fill=fill, anchor=anchor)
    except (ValueError, TypeError):   # bitmap fallback font has no anchors
        draw.text(xy, text, font=font, fill=fill)


def _qc_side_stack(seg, conf, gap, bg):
    """
    Third panel: two small panels stacked in one column, the segmented image on
    top and the prediction map below. Same total height as the other panels.
    The map is dark where the model is certain and bright where it is not.
    """
    h, w = seg.shape
    sh = max(1, (h - gap) // 2)
    sw = max(1, int(round(w * sh / h)))
    mask = cv2.cvtColor(seg, cv2.COLOR_GRAY2RGB)      # the segmentation exactly as saved (BGW)
    stack = np.full((h, sw, 3), bg, dtype=np.uint8)
    stack[:sh] = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_AREA)
    if conf is not None:
        t = np.clip((conf.astype(np.float32) / 255.0 - QC_PRED_LO) / (QC_PRED_HI - QC_PRED_LO), 0, 1)
        unc = ((1.0 - t) * 255).astype(np.uint8)
        heat = cv2.cvtColor(cv2.applyColorMap(unc, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        stack[sh + gap:sh + gap + sh] = cv2.resize(heat, (sw, sh), interpolation=cv2.INTER_AREA)
    return stack


def _qc_display_name(name: str) -> str:
    """Filename without its .tif / .ome.tif extension, for the bar."""
    low = name.lower()
    for ext in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if low.endswith(ext):
            return name[:-len(ext)]
    return name


def _qc_tracing(tiff_dir, img_path, scale, magnification, config, log):
    """
    CSA tracing for one image, mapped onto the segmented image.

    Returns (region_px, reason, status):
      status None         tracing does not apply here (not 40X, no CSA folder,
                          or the tracing modules cannot be imported)
      "ok"                region_px is the traced region in segmented pixels
      "excluded"          the image carries a QuPath EXCLUDE tag; reason is its text
      "no tracing"        the study has a CSA folder but no file for this image
      "unusable tracing"  file present but no usable '+' region
    Mirrors morphometrics/__main__.py: 40X only, excluded when the reason is non-empty.
    """
    if magnification is not None and str(magnification).upper() != "40X":
        return None, None, None
    try:
        from morphometrics.tracing import tracing_path
        from batch_axon.overlay.process_overlay_qupath import (
            get_geojson_exclusion, get_geojson_region)
        from shapely.affinity import affine_transform
    except Exception:
        return None, None, None
    nerve_dir = tiff_dir.parent
    if not (nerve_dir / config.get("csa_folder", "CSA")).is_dir():
        return None, None, None
    path = tracing_path(nerve_dir, img_path.stem, config)
    if not path.is_file():
        return None, None, "no tracing"
    try:
        reason = get_geojson_exclusion(str(path), log, quiet=True)
        if reason:
            return None, reason, "excluded"
        region = get_geojson_region(str(path), log, quiet=True)
    except Exception:
        return None, None, "unusable tracing"
    if region is None:
        return None, None, "unusable tracing"
    s = float(scale) if scale and scale > 0 else 1.0
    # Original-image pixels → segmented pixels, pixel centres at integers.
    return affine_transform(region, [s, 0, 0, s, -0.5, -0.5]), None, "ok"


def _qc_draw_region(img, region, color, thickness=2):
    """Draw a shapely (Multi)Polygon, holes included, on an RGB uint8 image in place."""
    for g in getattr(region, "geoms", [region]):
        if getattr(g, "geom_type", "") != "Polygon" or g.is_empty:
            continue
        for ring in [g.exterior, *g.interiors]:
            pts = np.round(np.asarray(ring.coords)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)


def _qc_region_mask(region, shape):
    """1 inside a shapely (Multi)Polygon (holes excluded), 0 elsewhere."""
    m = np.zeros(shape, np.uint8)
    for g in getattr(region, "geoms", [region]):
        if getattr(g, "geom_type", "") != "Polygon" or g.is_empty:
            continue
        cv2.fillPoly(m, [np.round(np.asarray(g.exterior.coords)).astype(np.int32)], 1)
        for ring in g.interiors:
            cv2.fillPoly(m, [np.round(np.asarray(ring.coords)).astype(np.int32)], 0)
    return m


def _qc_object_scores(seg, conf):
    """
    Label each detection (a connected myelin+axon region) and score it by its
    mean confidence. Returns (labels, mean); mean[label] is NaN for specks
    smaller than QC_MIN_OBJECT_PX.
    """
    n, labels = cv2.connectedComponents((seg > 0).astype(np.uint8), connectivity=8)
    mean = np.full(n, np.nan)
    if n <= 1:
        return labels, mean
    flat  = labels.ravel()
    sizes = np.bincount(flat, minlength=n).astype(np.float64)
    sums  = np.bincount(flat, weights=conf.ravel().astype(np.float64), minlength=n)
    ok    = sizes >= QC_MIN_OBJECT_PX
    ok[0] = False
    mean[ok] = sums[ok] / sizes[ok] / 255.0
    return labels, mean


def _qc_overlay(orig, seg):
    """Overlay panel: the tissue with axon and myelin drawn on it (RGB uint8)."""
    bg = orig
    if QC_NORMALIZE_OVERLAY_BG:
        lo, hi = np.percentile(orig, (1, 99))
        if hi > lo:
            bg = np.clip((orig.astype(np.float32) - lo) * (255.0 / (hi - lo)),
                         0, 255).astype(np.uint8)
    out = cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB)

    if QC_OVERLAY_STYLE == "outline":
        for mask, color in (((seg > 0).astype(np.uint8), QC_MYELIN_RGB),
                            ((seg == 255).astype(np.uint8), QC_AXON_RGB)):
            cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cs, -1, color, 1, cv2.LINE_AA)
        return out

    f = out.astype(np.float32)
    for value, color, alpha in ((255, QC_AXON_RGB, QC_ALPHA_AXON),
                                (128, QC_MYELIN_RGB, QC_ALPHA_MYELIN)):
        a = (seg == value).astype(np.float32)
        if QC_EDGE_SOFTNESS > 0:
            # soften the edge outward only, so thin rings keep their full opacity
            a = np.maximum(a, cv2.GaussianBlur(a, (0, 0), QC_EDGE_SOFTNESS))
        a = (a * alpha)[..., None]
        f = f * (1.0 - a) + np.array(color, np.float32) * a
    return np.ascontiguousarray(np.clip(f, 0, 255).astype(np.uint8))


def save_qc_sheet(
    tiff_dir: Path,
    output_dir: Path,
    images: list,
    seg_suffix: str,
    model_name: str,
    log: AxonIDLogger = None,
    confidence: dict = None,
    scales: dict = None,
    magnification=None
):
    """
    Generate a QC sheet for one nerve: for each image, a filename bar followed
    by original | overlay (overlay only if QC_SHOW_ORIGINAL is False).
    Saved as {nerve_name}_segmentation_qc.png (QC_FORMAT = "jpg" for a smaller,
    lossy file) in the QC/ subfolder, at the working resolution of the
    segmentation (no downscaling). A nerve too tall for one image is split
    into _part1, _part2, ...

    Overlay: axon and myelin in the QC_* colors over the tissue.
      Green outline  = the image's CSA tracing (40X, when the file exists).
      Orange / yellow outline = the least-certain detections in this nerve, two
                       tiers (needs `confidence`, {image path: uint8 map, 255 = certain}).
    The bar shows text only for an excluded or untraced image.
    The sheet is a viewing aid only; the saved segmentation is never modified.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        if log:
            log.warn("Pillow not available — skipping QC sheet")
        return

    config     = load_config()
    qc_folder  = config.get("qc_folder", "QC")
    nerve_name = tiff_dir.parent.name
    qc_dir     = tiff_dir.parent / qc_folder
    qc_dir.mkdir(parents=True, exist_ok=True)

    n_images = len(images)
    if n_images == 0:
        return

    BG, BAR   = (20, 20, 20), (52, 56, 66)
    GAP       = 12     # px between the panels of one image
    BLOCK_GAP = 28     # px between images
    BAR_H     = 56     # filename bar
    BANNER_H  = 64     # nerve / model / legend, once at the top
    TAG_RGB   = (255, 170, 60)

    # ── Pass 1: build each image's panels and score its detections ──────────
    blocks, states = [], []
    any_roi = False
    any_pred = False
    for k, img_path in enumerate(images, start=1):
        block = {"name": _qc_display_name(img_path.name), "count": f"{k}/{n_images}",
                 "key": img_path.stem,
                 "panels": None, "roles": None, "tag": None, "stats": None, "warn": False}
        blocks.append(block)
        seg_path = output_dir / f"{img_path.stem}{seg_suffix}.tif"
        seg = None
        if seg_path.exists():
            seg = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
        if seg is None:
            if log:
                log.warn(f"QC: segmented file not found or unreadable: {seg_path}")
            continue

        orig = resize_img(str(img_path), is_mask=False)
        # QC only: resize the display copy of the original if dimensions differ.
        if orig.shape != seg.shape:
            orig = cv2.resize(orig, (seg.shape[1], seg.shape[0]))

        rgb     = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
        overlay = _qc_overlay(orig, seg)

        labels = mean = None
        conf = confidence.get(img_path) if confidence else None
        if conf is not None:
            if conf.shape != seg.shape:
                conf = cv2.resize(conf, (seg.shape[1], seg.shape[0]))
            labels, mean = _qc_object_scores(seg, conf)

        # CSA tracing → green outline; text only when excluded or untraced
        region, reason, status = _qc_tracing(
            tiff_dir, img_path, (scales or {}).get(img_path), magnification, config, log)
        if region is not None:
            _qc_draw_region(overlay, region, QC_ROI_RGB, QC_ROI_PX)
            any_roi = True
            if QC_ROI_ON_ORIGINAL:
                rgb = np.ascontiguousarray(rgb)
                _qc_draw_region(rgb, region, QC_ROI_RGB, QC_ROI_PX)
        elif status == "excluded":
            block["tag"] = f"EXCLUDED — {reason}" if reason else "EXCLUDED"
        elif status in ("no tracing", "unusable tracing"):
            block["tag"] = status

        panels = [rgb, overlay] if QC_SHOW_ORIGINAL else [overlay]
        roles = ["tissue", "overlay"] if QC_SHOW_ORIGINAL else ["overlay"]
        if QC_SHOW_PREDICTION:
            panels.append(_qc_side_stack(seg, conf, GAP, BG))
            roles.append("stack")
            any_pred = any_pred or conf is not None
        block["panels"], block["roles"] = panels, roles
        # Single difficulty number: share of foreground pixels (inside the tracing
        # when there is one) whose confidence is below QC_UNCERTAIN_BELOW. Fixed
        # cut-off, so it is comparable across images and nerves.
        if conf is not None:
            fg = seg > 0
            if region is not None:
                fg = fg & (_qc_region_mask(region, seg.shape) > 0)
            nfg = int(fg.sum())
            if nfg:
                nun = int((fg & (conf.astype(np.float32) / 255.0 < QC_UNCERTAIN_BELOW)).sum())
                block["_unc"] = (nun, nfg)
        # Detections that count toward the numbers: those inside the tracing when
        # there is one (the others never reach the measurements), else all.
        use = None
        if labels is not None and region is not None:
            rm  = _qc_region_mask(region, seg.shape)
            tot = np.bincount(labels.ravel(), minlength=len(mean))
            ins = np.bincount(labels[rm > 0], minlength=len(mean))
            use = ins >= 0.5 * np.maximum(tot, 1)
        states.append((block, overlay, labels, mean, use))

    # ── Pick the "least certain" cut-off for the whole nerve, then outline ──
    n_flag1 = n_flag2 = 0
    st_fl = st_n = 0
    thr1 = thr2 = None
    scores = np.empty(0)
    def _used(m, u):
        v = m if u is None else np.where(u, m, np.nan)
        return v[~np.isnan(v)]
    pooled = [_used(m, u) for _, _, _, m, u in states if m is not None]
    if pooled:
        scores = np.concatenate(pooled)
    if scores.size:
        p1, p2 = np.percentile(scores, [QC_FLAG1_PCT, QC_FLAG2_PCT])
        thr1 = max(min(float(p1), QC_FLAG_CEILING), QC_FLAG_FLOOR)
        thr2 = max(min(float(p2), QC_FLAG_CEILING), thr1)
        grow = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * QC_OUTLINE_GAP + 1, 2 * QC_OUTLINE_GAP + 1))
        for blk, overlay, labels, mean, use in states:
            if mean is None:
                continue
            with np.errstate(invalid="ignore"):
                tier1 = mean < thr1
                tier2 = (mean >= thr1) & (mean < thr2)
            tier1[0] = tier2[0] = False
            n_flag1 += int(tier1.sum())
            n_flag2 += int(tier2.sum())
            u_ = np.ones(len(mean), dtype=bool) if use is None else use
            counted = ~np.isnan(mean) & u_
            counted[0] = False
            n_det = int(counted.sum())
            n_fl  = int(((tier1 | tier2) & counted).sum())
            st_fl += n_fl
            st_n  += n_det
            # Yellow first, orange on top. The mask is grown a little so the
            # line sits just outside the fiber and leaves the myelin visible.
            for tier, color, px in ((tier2, QC_FLAG2_RGB, QC_FLAG2_PX),
                                    (tier1, QC_FLAG1_RGB, QC_FLAG1_PX)):
                if not tier.any():
                    continue
                mask = tier[labels].astype(np.uint8)
                contours, _ = cv2.findContours(cv2.dilate(mask, grow), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(overlay, contours, -1, color, px, cv2.LINE_AA)

    # Per-image number (shown above the prediction map): % of foreground pixels
    # that are uncertain. Banner shows the same figure for the whole nerve.
    nerve_ref = None
    tot_u = tot_f = 0
    for blk, _, _, _, _ in states:
        u = blk.pop("_unc", None)
        if u:
            tot_u += u[0]
            tot_f += u[1]
            blk["stats"] = f"uncertain {100 * u[0] / u[1]:.0f}%"
            blk["warn"] = u[0] / u[1] >= QC_UNCERTAIN_WARN
    if tot_f:
        nerve_ref = f"nerve: uncertain {100 * tot_u / tot_f:.0f}%"

    if log and scores.size:
        p5, p25, p50 = np.percentile(scores, [5, 25, 50])
        log.info(f"QC confidence ({nerve_name}): {scores.size} detections, median {p50:.2f} "
                 f"(p5 {p5:.2f}, p25 {p25:.2f}); {n_flag1} below {thr1:.2f} outlined in "
                 f"orange, {n_flag2} more below {thr2:.2f} in yellow")

    # ── Sheet width, then split into parts that fit the image-size limit ────
    rows_w = [sum(p.shape[1] for p in b["panels"]) + GAP * (len(b["panels"]) - 1)
              for b in blocks if b["panels"]]
    width  = max(rows_w + [1200])

    def block_h(b):
        return BAR_H + (max(p.shape[0] for p in b["panels"]) if b["panels"] else 0) + BLOCK_GAP

    use_png = QC_FORMAT.lower() == "png"
    max_px  = _QC_PNG_MAX_PX if use_png else _QC_JPEG_MAX_PX
    parts, cur, cur_h = [], [], BANNER_H
    for b in blocks:
        if cur and cur_h + block_h(b) > max_px:
            parts.append((cur, cur_h)); cur, cur_h = [], BANNER_H
        cur.append(b); cur_h += block_h(b)
    parts.append((cur, cur_h))

    font_title = _qc_font(30, bold=True)
    font_small = _qc_font(26)
    quality    = int(config.get("qc_jpeg_quality", 90))

    # Legend, left to right: filled swatch = overlay color, outlined = a line
    fill_style = "line" if QC_OVERLAY_STYLE == "outline" else "fill"
    legend = [("Axon", QC_AXON_RGB, fill_style), ("Myelin", QC_MYELIN_RGB, fill_style)]
    if n_flag1:
        legend.append((f"Least-certain {QC_FLAG1_PCT}%", QC_FLAG1_RGB, "line"))
    if n_flag2:
        legend.append((f"Next {QC_FLAG2_PCT - QC_FLAG1_PCT}%", QC_FLAG2_RGB, "line"))
    if any_roi:
        legend.append(("CSA tracing", QC_ROI_RGB, "line"))
    if any_pred:
        legend.append(("Prediction", None, "gradient"))

    saved = []
    layout = {}     # image stem -> {"part": n, "boxes": [[x, y, w, h], ...]} for the watershed QC
    for p_idx, (part_blocks, total_h) in enumerate(parts, start=1):
        canvas = np.full((total_h, width, 3), BG, dtype=np.uint8)

        # Banner: nerve + model on the left, legend on the right
        banner = Image.new("RGB", (width, BANNER_H), BG)
        d = ImageDraw.Draw(banner)
        label = f"{nerve_name}  |  model: {model_name}"
        if len(parts) > 1:
            label += f"  |  part {p_idx}/{len(parts)}"
        _qc_text(d, (16, BANNER_H // 2), label, font_title, (255, 255, 255), "lm")
        if nerve_ref:
            lw = d.textlength(label, font=font_title) if hasattr(d, "textlength") else 700
            _qc_text(d, (16 + int(lw) + 40, BANNER_H // 2), nerve_ref, font_small,
                     (150, 150, 160), "lm")
        x = width - 16
        for text, color, style in reversed(legend):          # drawn right to left
            if style == "gradient":
                # Prediction map key: "Prediction: certain [dark → bright] uncertain"
                bar_w = 90
                grad = cv2.applyColorMap(np.linspace(0, 255, bar_w).astype(np.uint8).reshape(1, -1),
                                         cv2.COLORMAP_INFERNO)[0][:, ::-1]
                for text_i, text_s in (("uncertain", None), (None, "bar"), ("certain", None),
                                       ("Prediction:", None)):
                    if text_s == "bar":
                        for i in range(bar_w):
                            d.line([x - bar_w + i, BANNER_H // 2 - 11, x - bar_w + i, BANNER_H // 2 + 11],
                                   fill=tuple(int(c) for c in grad[i]), width=1)
                        x -= bar_w + 10
                    else:
                        tw_g = d.textlength(text_i, font=font_small) if hasattr(d, "textlength") else 90
                        _qc_text(d, (x, BANNER_H // 2), text_i, font_small, (210, 210, 215), "rm")
                        x -= int(tw_g) + 10
                x -= 28
                continue
            tw = d.textlength(text, font=font_small) if hasattr(d, "textlength") else 70
            _qc_text(d, (x, BANNER_H // 2), text, font_small, (210, 210, 215), "rm")
            x -= int(tw) + 10
            box = [x - 22, BANNER_H // 2 - 11, x, BANNER_H // 2 + 11]
            if style == "fill":
                d.rectangle(box, fill=color)
            else:
                d.rectangle(box, outline=color, width=3)
            x -= 22 + 28
        canvas[:BANNER_H] = np.asarray(banner)

        y = BANNER_H
        for b in part_blocks:
            bar = Image.new("RGB", (width, BAR_H), BAR)
            d = ImageDraw.Draw(bar)
            if b["panels"]:
                _qc_text(d, (16, BAR_H // 2), b["name"], font_title, (255, 255, 255), "lm")
                nw = d.textlength(b["name"], font=font_title) if hasattr(d, "textlength") else 400
                tx = 16 + int(nw) + 40
                if b["tag"]:
                    _qc_text(d, (tx, BAR_H // 2), b["tag"], font_title, TAG_RGB, "lm")
                if b["stats"]:
                    stat_rgb = TAG_RGB if b["warn"] else (190, 190, 200)
                    if b["roles"] and b["roles"][-1] == "stack":     # above the prediction map
                        sx = sum(p.shape[1] + GAP for p in b["panels"][:-1]) + 8
                        _qc_text(d, (sx, BAR_H // 2), b["stats"], font_small, stat_rgb, "lm")
                    else:
                        cw_ = d.textlength(b["count"], font=font_small) if hasattr(d, "textlength") else 60
                        _qc_text(d, (width - 16 - int(cw_) - 40, BAR_H // 2), b["stats"],
                                 font_small, stat_rgb, "rm")
            else:
                _qc_text(d, (16, BAR_H // 2), f"{b['name']}  —  segmented file not found",
                         font_title, (255, 120, 120), "lm")
            _qc_text(d, (width - 16, BAR_H // 2), b["count"], font_small, (190, 190, 200), "rm")
            canvas[y:y + BAR_H] = np.asarray(bar)
            y += BAR_H
            if b["panels"]:
                x = 0
                boxes = []
                for panel in b["panels"]:
                    ph, pw = panel.shape[:2]
                    canvas[y:y + ph, x:x + pw] = panel
                    boxes.append([x, y, pw, ph])
                    x += pw + GAP
                layout[b["key"]] = {"part": p_idx, "boxes": boxes, "roles": b["roles"]}
                y += max(p.shape[0] for p in b["panels"])
            y += BLOCK_GAP

        suffix   = "" if len(parts) == 1 else f"_part{p_idx}"
        out_path = qc_dir / f"{nerve_name}_segmentation_qc{suffix}.{'png' if use_png else 'jpg'}"
        if use_png:
            # compress_level 3: close to the default size, noticeably faster
            Image.fromarray(canvas).save(str(out_path), "PNG",
                                         compress_level=int(config.get("qc_png_compress", 3)))
        else:
            Image.fromarray(canvas).save(
                str(out_path), "JPEG", quality=quality, subsampling=0, optimize=True
            )
        saved.append(out_path)
        del canvas

    # Where each image's panels sit in the sheet(s): read by the morphometrics QC
    # to build its sheet on the same tissue. Failure here never affects the sheet.
    try:
        import json
        side = qc_dir / f"{nerve_name}_segmentation_qc.layout.json"
        side.write_text(json.dumps({"files": [pth.name for pth in saved], "images": layout}),
                        encoding="utf-8")
    except Exception as e:                                   # pragma: no cover
        if log:
            log.warn(f"QC layout file not written: {e}")

    if log:
        for out_path in saved:
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
    confidence = {}   # QC: image path -> uncertainty source map
    scales = {}       # QC: image path -> resize scale (for the tracing outline)

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
            mask, elapsed, orig_h, orig_w, scale, (src_w, src_h), conf = segment_image(
                str(img_path), model,
                patch_size=patch_size,
                log=log,
                return_confidence=True,
            )
            confidence[img_path] = conf
            scales[img_path] = scale
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
        log=log,
        confidence=confidence,
        scales=scales,
        magnification=mag
    )
    qc_elapsed = time.time() - qc_start

    return NerveSegmentResult(
        success=(failed == 0 and success == len(images)),
        seg_seconds=seg_seconds_total,
        qc_seconds=qc_elapsed,
    )