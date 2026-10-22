"""
utils/metrics.py: pixel-level segmentation metrics.

Usage
    from utils.metrics import compute_epoch_metrics, compute_all_metrics
    m = compute_epoch_metrics(tp, fp, fn, tn)        # per epoch, cheap
    m = compute_all_metrics(preds, labels, device)   # final evaluation, adds HD95

Inputs
    Class-index predictions and labels (0 background, 1 myelin, 2 axon),
    or the confusion counts from segmentation_models_pytorch get_stats().

Outputs
    Flat dicts: Dice, IoU, precision and recall per class and as the
    unweighted mean of the three classes (macro), raw confusion counts,
    and (compute_all_metrics only) HD95 in pixels. Ratios are computed from
    counts pooled over all items, not averaged per item. A ratio whose
    denominator is zero (e.g. a class absent from prediction and label)
    is None.

Depends on
    torch, segmentation_models_pytorch, monai
"""

# ── Standard library ──────────────────────────────────────────────────────────
import math

# ── Third-party ───────────────────────────────────────────────────────────────
from monai.metrics import HausdorffDistanceMetric
import segmentation_models_pytorch as smp
import torch


# ─── Class definitions ────────────────────────────────────────────────────────
NUM_CLASSES  = 3
CLASS_BG     = 0
CLASS_MYELIN = 1
CLASS_AXON   = 2


# ─── Metric helpers ───────────────────────────────────────────────────────────
def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Return numerator / denominator, using NaN where the denominator is zero."""
    result = torch.full_like(
        numerator,
        float('nan'),
        dtype=torch.float32,
    )
    valid = denominator != 0
    result[valid] = numerator[valid].float() / denominator[valid].float()
    return result


def _compute_class_metrics(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
) -> dict:
    """
    Compute pooled per-class Dice, IoU, precision, and recall.

    Inputs are already aggregated across all evaluation items.
    """
    dice = _safe_divide(
        2 * tp,
        2 * tp + fp + fn,
    )
    iou = _safe_divide(
        tp,
        tp + fp + fn,
    )
    precision = _safe_divide(
        tp,
        tp + fp,
    )
    recall = _safe_divide(
        tp,
        tp + fn,
    )

    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
    }

def _round_metric(value: torch.Tensor) -> float | None:
    """Round a metric value, returning None when the value is undefined."""
    value = value.item()

    if math.isnan(value):
        return None

    return round(value, 4)


def _macro(values: torch.Tensor) -> float | None:
    """Return the unweighted mean across classes, ignoring undefined values."""
    return _round_metric(torch.nanmean(values))


# ─── Lightweight — per-epoch training metrics ─────────────────────────────────

def compute_epoch_metrics(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    tn: torch.Tensor
) -> dict:
    """
    Compute lightweight pixel-level metrics from confusion-matrix counts.

    Confusion-matrix counts are summed across all supplied evaluation items
    before Dice, IoU, precision, and recall are calculated. Macro metrics are
    the unweighted mean of the three class-specific metrics.

    HD95 is intentionally excluded from the per-epoch path because it is
    computed separately by compute_all_metrics().

    Args:
        tp, fp, fn, tn: output of smp.metrics.get_stats()

    Returns:
        Flat dictionary containing pooled per-class metrics, macro metrics,
        and raw per-class confusion-matrix totals.
    """
    tp_sum = tp.sum(dim=0)
    fp_sum = fp.sum(dim=0)
    fn_sum = fn.sum(dim=0)
    tn_sum = tn.sum(dim=0)

    class_metrics = _compute_class_metrics(
        tp_sum, fp_sum, fn_sum
    )

    return {
        'dice_macro':       _macro(class_metrics['dice']),
        'iou_macro':        _macro(class_metrics['iou']),
        'precision_macro':  _macro(class_metrics['precision']),
        'recall_macro':     _macro(class_metrics['recall']),

        'dice_bg':          _round_metric(class_metrics['dice'][CLASS_BG]),
        'dice_myelin':      _round_metric(class_metrics['dice'][CLASS_MYELIN]),
        'dice_axon':        _round_metric(class_metrics['dice'][CLASS_AXON]),

        'iou_bg':           _round_metric(class_metrics['iou'][CLASS_BG]),
        'iou_myelin':       _round_metric(class_metrics['iou'][CLASS_MYELIN]),
        'iou_axon':         _round_metric(class_metrics['iou'][CLASS_AXON]),

        'precision_bg':     _round_metric(class_metrics['precision'][CLASS_BG]),
        'precision_myelin': _round_metric(class_metrics['precision'][CLASS_MYELIN]),
        'precision_axon':   _round_metric(class_metrics['precision'][CLASS_AXON]),

        'recall_bg':        _round_metric(class_metrics['recall'][CLASS_BG]),
        'recall_myelin':    _round_metric(class_metrics['recall'][CLASS_MYELIN]),
        'recall_axon':      _round_metric(class_metrics['recall'][CLASS_AXON]),

        # Raw confusion-matrix counts
        'tp_bg':     int(tp_sum[CLASS_BG].item()),
        'fp_bg':     int(fp_sum[CLASS_BG].item()),
        'fn_bg':     int(fn_sum[CLASS_BG].item()),
        'tn_bg':     int(tn_sum[CLASS_BG].item()),

        'tp_myelin': int(tp_sum[CLASS_MYELIN].item()),
        'fp_myelin': int(fp_sum[CLASS_MYELIN].item()),
        'fn_myelin': int(fn_sum[CLASS_MYELIN].item()),
        'tn_myelin': int(tn_sum[CLASS_MYELIN].item()),

        'tp_axon':   int(tp_sum[CLASS_AXON].item()),
        'fp_axon':   int(fp_sum[CLASS_AXON].item()),
        'fn_axon':   int(fn_sum[CLASS_AXON].item()),
        'tn_axon':   int(tn_sum[CLASS_AXON].item()),
    }


# ─── Full suite — post-training / post-segmentation evaluation ────────────────

def compute_all_metrics(
    preds:       torch.Tensor,
    labels:      torch.Tensor,
    device,
    num_classes: int = NUM_CLASSES,
    compute_hd95: bool = True,   # skip the MONAI boundary-distance pass when not needed
) -> dict:
    """
    Compute full metric suite on a set of predictions and ground truth labels.
    Called once after training on best checkpoint val set.
    Called by segment/__main__.py for ground truth evaluation.

    Args:
        preds:       (N, H, W) int tensor — argmax class predictions
        labels:      (N, H, W) int tensor — ground truth class indices
        device:      torch device
        num_classes: number of segmentation classes (default 3)
        compute_hd95: if False, skips the MONAI Hausdorff pass entirely and    
                      returns None for every hd95_* field — for callers that   
                      need dice/iou/precision/recall/tp/fp/fn/tn only and want
                      to avoid HD95's cost (e.g. per-phenotype breakdowns).    

    Returns flat dict with all metric fields — unpack directly into result.json:
        Macro:      dice_macro, iou_macro, precision_macro, recall_macro,
                    hd95_macro, hd95_foreground (myelin+axon combined)
        Per-class:  dice_bg/myelin/axon, iou_bg/myelin/axon,
                    precision_bg/myelin/axon, recall_bg/myelin/axon,
                    hd95_bg/myelin/axon
        Raw counts: tp/fp/fn/tn_bg, tp/fp/fn/tn_myelin, tp/fp/fn/tn_axon
                    (pixel totals, pooled across all supplied evaluation
                    items — see AGGREGATION in the module docstring)

        Every hd95_* field is None when compute_hd95=False.
    """
    if num_classes != NUM_CLASSES:
        raise ValueError(
            f"AxonID metrics require {NUM_CLASSES} classes; "
            f"received num_classes={num_classes}."
        ) 
        
    preds  = preds.to(device)
    labels = labels.to(device)

    # ── smp confusion matrix ──────────────────────────────────────────────────
    tp, fp, fn, tn = smp.metrics.get_stats(
        preds, labels, mode="multiclass", num_classes=num_classes
    )

    # ── Pooled segmentation metrics ──────────────────────────────────────────
    # Primary metrics are calculated from confusion-matrix counts pooled
    # across all supplied evaluation items.
    tp_sum = tp.sum(dim=0)
    fp_sum = fp.sum(dim=0)
    fn_sum = fn.sum(dim=0)
    tn_sum = tn.sum(dim=0)

    class_metrics = _compute_class_metrics(
        tp_sum, fp_sum, fn_sum
    )

    # ── HD95 via MONAI ────────────────────────────────────────────────────────
    # HD95 is calculated independently for each supplied evaluation item.
    # Results are averaged across valid class-specific values and reported
    # in pixels. Undefined class/item distances are excluded from the mean.
    if compute_hd95:                                                     
        preds_cpu = preds.cpu()
        labels_cpu = labels.cpu()

        preds_onehot = torch.nn.functional.one_hot(
            preds_cpu,
            num_classes=num_classes,
        ).permute(0, 3, 1, 2).float()

        labels_onehot = torch.nn.functional.one_hot(
            labels_cpu,
            num_classes=num_classes,
        ).permute(0, 3, 1, 2).float()

        # reduction is set at aggregate()-time below, not here — MONAI reads
        # the reduction passed to aggregate(), so a value given to the
        # constructor is never used and would be misleading to include.
        hd_metric = HausdorffDistanceMetric(
            include_background=True,
            percentile=95,
        )

        # reduction="none" keeps per-item, per-class values unreduced. Some
        # items have no ground-truth pixels for a given class, producing NaN
        # for that (item, class) cell; nanmean() below explicitly skips those
        # rather than relying on MONAI's own reduction to handle them.
        hd_metric(y_pred=preds_onehot, y=labels_onehot)
        hd95_raw = hd_metric.aggregate(reduction="none")
        hd_metric.reset()
        if hd95_raw.ndim == 2:
            hd95_per_class = hd95_raw.nanmean(dim=0)
        else:
            hd95_per_class = hd95_raw

        hd95_bg = _round_metric(hd95_per_class[CLASS_BG])
        hd95_myel = _round_metric(hd95_per_class[CLASS_MYELIN])
        hd95_axon = _round_metric(hd95_per_class[CLASS_AXON])
        hd95_macro = _round_metric(hd95_per_class.nanmean())
        hd95_foreground = _round_metric(
            hd95_per_class[[CLASS_MYELIN, CLASS_AXON]].nanmean()
        )
    else:                                                                       
        hd95_bg = hd95_myel = hd95_axon = hd95_macro = hd95_foreground = None   

    return {
        # Macro metrics
        'dice_macro':       _macro(class_metrics['dice']),
        'iou_macro':        _macro(class_metrics['iou']),
        'precision_macro':  _macro(class_metrics['precision']),
        'recall_macro':     _macro(class_metrics['recall']),
        'hd95_macro':       hd95_macro,
        # hd95_foreground is the MEAN of hd95_myelin and hd95_axon computed
        # separately — not the boundary of a merged myelin+axon blob. It is
        # therefore NOT the same quantity as hd95_bg's "fiber outline"
        # (see note below): hd95_foreground still carries the harder,
        # internal myelin/axon boundary information that hd95_bg cannot see
        # at all, since averaging two per-class boundary distances is a
        # different operation than measuring the outer edge of their union.
        'hd95_foreground':  hd95_foreground,
        
        # Per-class HD95
        #
        # hd95_bg is effectively the fiber's OUTER boundary: background's
        # one-vs-rest boundary is the same physical contour as the combined
        # myelin+axon region's outer edge, since a mask and its complement
        # share one boundary curve. This holds cleanly only when axon is
        # always fully enclosed by myelin (never touching background
        # directly). If an unmyelinated axon is present, or myelin around an
        # axon was segmented incompletely, that axon's pixels border
        # background directly — hd95_bg then also reflects that boundary
        # error, and the "clean outer fiber edge" interpretation no longer
        # holds for that image. Treat hd95_bg's value with more caution on
        # data where unmyelinated axons or myelin segmentation gaps are
        # known or suspected to occur.
        'hd95_bg':          hd95_bg,
        'hd95_myelin':      hd95_myel,
        'hd95_axon':        hd95_axon,

        # Per-class Dice
        'dice_bg':          _round_metric(class_metrics['dice'][CLASS_BG]),
        'dice_myelin':      _round_metric(class_metrics['dice'][CLASS_MYELIN]),
        'dice_axon':        _round_metric(class_metrics['dice'][CLASS_AXON]),

        # Per-class IoU
        'iou_bg':           _round_metric(class_metrics['iou'][CLASS_BG]),
        'iou_myelin':       _round_metric(class_metrics['iou'][CLASS_MYELIN]),
        'iou_axon':         _round_metric(class_metrics['iou'][CLASS_AXON]),

        # Per-class Precision
        'precision_bg':     _round_metric(class_metrics['precision'][CLASS_BG]),
        'precision_myelin': _round_metric(class_metrics['precision'][CLASS_MYELIN]),
        'precision_axon':   _round_metric(class_metrics['precision'][CLASS_AXON]),

        # Per-class Recall
        'recall_bg':        _round_metric(class_metrics['recall'][CLASS_BG]),
        'recall_myelin':    _round_metric(class_metrics['recall'][CLASS_MYELIN]),
        'recall_axon':      _round_metric(class_metrics['recall'][CLASS_AXON]),
        
         # Raw confusion-matrix counts
        'tp_bg':     int(tp_sum[CLASS_BG].item()),
        'fp_bg':     int(fp_sum[CLASS_BG].item()),
        'fn_bg':     int(fn_sum[CLASS_BG].item()),
        'tn_bg':     int(tn_sum[CLASS_BG].item()),

        'tp_myelin': int(tp_sum[CLASS_MYELIN].item()),
        'fp_myelin': int(fp_sum[CLASS_MYELIN].item()),
        'fn_myelin': int(fn_sum[CLASS_MYELIN].item()),
        'tn_myelin': int(tn_sum[CLASS_MYELIN].item()),

        'tp_axon':   int(tp_sum[CLASS_AXON].item()),
        'fp_axon':   int(fp_sum[CLASS_AXON].item()),
        'fn_axon':   int(fn_sum[CLASS_AXON].item()),
        'tn_axon':   int(tn_sum[CLASS_AXON].item()),
    }
