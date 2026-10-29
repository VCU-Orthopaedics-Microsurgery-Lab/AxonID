"""
train/train.py: train an AxonID model with the locked recipe.

Purpose
    Runs one training job end to end: check the dataset, make patches,
    split images into training and validation, (optionally) augment, train
    UNet++/DenseNet169, keep the best epoch, evaluate it and save it as a
    checkpoint that the segmentation menu can load.

Usage
    from train.train import train_model
    model_path = train_model(images_dir, model_name, epochs, batch_size,
                             use_aug, log, mag, seed)
    (normally called through: python -m train)

Inputs
    images_dir/images/   source images (.tif, .tiff, .png)
    images_dir/masks/    BGW masks with the same file names
    Image names starting with ctrl_ / regen_ give a phenotype-stratified
    split; other names fall back to a random image-level split.

Outputs
    models/user/{model_name}.safetensors   best-validation-loss weights + training record
    images_dir/{images,masks}/cropped/...   patches (reused by later runs)
    Training log in logs/ (written by the caller's AxonIDLogger)

Behaviour
    The best epoch is the one with the lowest validation loss (improvement
    must exceed early_stop_min_delta). Training stops after
    early_stop_patience epochs without improvement or at the epoch limit.
    Final metrics are computed on the validation set with the best-epoch
    weights, not the last epoch's.

Locked (utils/recipe.py)
    Architecture, encoder, loss, class weights, optimizer, learning-rate
    schedule, early stopping, patch geometry, split fraction, augmentation.
    Epoch limit and batch size can be changed by the caller; any change is
    recorded in the checkpoint as a recipe deviation.

Depends on
    train/dataset/*, utils/recipe.py, utils/metrics.py, utils/logger.py
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import copy
from datetime import datetime
import math
from pathlib import Path
import random
import socket
import sys
import warnings

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, TensorDataset

# ── Local ─────────────────────────────────────────────────────────────────────
from train.dataset.augment import augment_dataset_np
from train.dataset.data_loader import load_all_patches
from train.dataset.preprocess import batch_process
from train.dataset.split import build_image_manifest, split_images, split_summary
from utils.filesystem import count_patches, is_stray_file
from utils.logger import AxonIDLogger
from utils.mask_contract import off_contract_count
from utils.metrics import compute_all_metrics, compute_epoch_metrics
from utils.model_io import MODEL_SUFFIX, save_model_file
from utils.model_registry import get_user_model_dir
from utils.recipe import (
    AUGMENTATION,
    RECIPE,
    RECIPE_ID,
    build_model,
    describe_parameter_groups,
    recipe_metadata,
)
from utils.version import __version__


# ─── Warning suppression ──────────────────────────────────────────────────────
# Both warnings fire per patch when a class is absent from a patch (common:
# many patches contain no axon). They do not affect results; a NaN/Inf check
# on the final metrics below replaces them.
warnings.filterwarnings("ignore", category=FutureWarning, module=r"monai\.utils\.deprecate_utils")
warnings.filterwarnings("ignore", category=UserWarning, message=r"the (ground truth|prediction) of class \d+ is all 0")

# Masks with more than this share of off-contract pixels get a warning:
# snapping a few anti-aliased edge pixels is expected, snapping many
# suggests the mask was saved with the wrong colours or as RGB.
OFF_CONTRACT_WARN_FRACTION = 0.05


def weighted_dice_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Weighted multiclass Dice loss.
    pred (N, C, H, W) raw logits | target (N, H, W) class indices | weights (C,)
    """
    pred_soft      = torch.softmax(pred, dim=1)
    target_one_hot = torch.zeros_like(pred_soft).scatter_(1, target.unsqueeze(1), 1.0)
    dims  = (0, 2, 3)
    inter = (pred_soft * target_one_hot).sum(dims)
    union = (pred_soft + target_one_hot).sum(dims)
    dice  = (2.0 * inter + 1e-6) / (union + 1e-6)
    return (weights * (1.0 - dice)).sum() / weights.sum()


def _f(value, spec: str = ".3f") -> str:
    """Format a metric that may be None (class absent) or NaN."""
    if value is None:
        return "n/a"
    try:
        if math.isnan(value):
            return "nan"
    except TypeError:
        return str(value)
    return format(value, spec)


# ─── Training logger ──────────────────────────────────────────────────────────
class TrainingLogger:
    """Prints one block per epoch and the best-checkpoint summary at the end."""

    def __init__(self, log: AxonIDLogger):
        self.log = log
        self.n_epochs = 0

    def log_epoch(self, epoch: int, m: dict, checkpoint_flag: str = ""):
        self.n_epochs += 1
        self.log.print(
            f"  Epoch{epoch + 1:>4} ({m['epoch_time']}) | learning_rate {_f(m['lr'], '.2e')}\n"
            f"  TRAIN  loss {_f(m['loss'])} | dice {_f(m['dice'])} | bg {_f(m['dice_bg'])} | "
            f"ax {_f(m['dice_axon'])} | my {_f(m['dice_myelin'])}\n"
            f"  VAL    loss {_f(m['val_loss'])} | dice {_f(m['val_dice'])} | bg {_f(m['val_dice_bg'])} | "
            f"ax {_f(m['val_dice_axon'])} | my {_f(m['val_dice_myelin'])}  {checkpoint_flag}\n"
        )

    def on_train_end(self, info: dict):
        if not self.n_epochs:
            return
        self.log.write_section("CHECKPOINT SUMMARY", (
            f"Best checkpoint @ epoch {info['epoch']}\n"
            f"  Val loss        : {_f(info['loss'])}\n"
            f"  Val bg dice     : {_f(info['bg'])}\n"
            f"  Val axon dice   : {_f(info['axon'])}\n"
            f"  Val myelin dice : {_f(info['myelin'])}\n"
            f"  Val IoU         : {_f(info['iou'])}\n"
            f"  Saved to        : {info['path']}"
        ))


# ─── Dataset preparation ──────────────────────────────────────────────────────
def prepare_dataset(images_dir: str, log: AxonIDLogger) -> dict:
    """Check the images/masks folders and report mask values that will be snapped."""
    images_dir = Path(images_dir).resolve() / "images"
    masks_dir  = images_dir.parent / "masks"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Masks folder not found: {masks_dir}")

    exts  = ("*.tif", "*.tiff", "*.png")
    imgs  = sorted(p for e in exts for p in images_dir.glob(e) if not is_stray_file(p))
    masks = sorted(p for e in exts for p in masks_dir.glob(e) if not is_stray_file(p))

    log.rule("SOURCE MASK VALUES")
    n_snapped_masks = 0
    for mask_path in masks:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        n_off = off_contract_count(mask)
        if n_off:
            n_snapped_masks += 1
            frac = n_off / mask.size
            msg = (f"  {mask_path.name}: {n_off} pixels ({frac * 100:.2f}%) are not 0/127/128/255 "
                   f"and will be snapped to the nearest class")
            if frac > OFF_CONTRACT_WARN_FRACTION:
                log.warn(msg + " (check the mask was saved as 8-bit grayscale with the BGW values)")
            else:
                log.info(msg)
    if not n_snapped_masks:
        log.success(f"All {len(masks)} masks use only BGW values")

    log.rule("DATASET VERIFICATION")
    img_stems, mask_stems = {p.stem for p in imgs}, {p.stem for p in masks}
    log.info(f"Source images: {len(img_stems & mask_stems)} pairs | "
             f"Missing masks: {len(img_stems - mask_stems)} | Missing images: {len(mask_stems - img_stems)}")

    cropped_img, cropped_mask = images_dir / "cropped", masks_dir / "cropped"
    return {
        "images_dir":   images_dir,
        "masks_dir":    masks_dir,
        "cropped_img":  cropped_img,
        "cropped_mask": cropped_mask,
        "patches_img":  cropped_img / "patches",
        "patches_mask": cropped_mask / "patches",
    }


def set_seed(seed: int) -> np.random.Generator:
    """Seed every random source used in training; returns the augmentation generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    return np.random.default_rng(seed)


def _check_extension_points():
    # Fine-tuning and deep supervision are reserved in utils/recipe.py but
    # not implemented in 1.0.0; refuse rather than silently ignore them.
    if RECIPE.init_from is not None or RECIPE.freeze != "none" or RECIPE.deep_supervision:
        raise NotImplementedError(
            "Fine-tuning (init_from / freeze) and deep supervision are not available in AxonID "
            f"{__version__}. utils/recipe.py must keep init_from=None, freeze='none', deep_supervision=False."
        )


# ─── Main training function ───────────────────────────────────────────────────
def train_model(
    images_dir: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    use_aug: bool,
    log: AxonIDLogger,
    mag: str,
    seed: int = RECIPE.default_seed,
) -> Path:
    """Train one model and return the saved checkpoint path. See the module header."""
    _check_extension_points()
    t_start = datetime.now()

    model_dir  = get_user_model_dir()
    model_path = model_dir / f"{model_name}{MODEL_SUFFIX}"
    if model_path.exists():
        # Never overwrite a trained model; checked before any work is done.
        raise FileExistsError(f"{model_path} already exists. Choose another model name or move the old file.")

    paths = prepare_dataset(images_dir, log)

    # ── Patches: make them once, reuse them on later runs ─────────────────────
    if not paths["patches_img"].exists() or count_patches(str(paths["patches_img"])) == 0:
        n_img, n_mask = batch_process(
            str(paths["images_dir"]), str(paths["masks_dir"]),
            str(paths["patches_img"]), str(paths["patches_mask"]),
            str(paths["cropped_img"]), str(paths["cropped_mask"]),
            log=log,
        )
        log.success(f"Patches created: {n_img} image, {n_mask} mask")
    else:
        log.rule("PATCH VERIFICATION")
        log.info(f"Existing patches found, preprocessing skipped ({count_patches(str(paths['patches_img']))} patches). "
                 f"Delete images/cropped and masks/cropped to regenerate them.")

    n_img_p, n_mask_p = count_patches(str(paths["patches_img"])), count_patches(str(paths["patches_mask"]))
    if n_img_p != n_mask_p:
        raise ValueError(f"Patch count mismatch: {n_img_p} images vs {n_mask_p} masks")
    log.success(f"Patch alignment verified: images {n_img_p} | masks {n_mask_p}")

    # Reused patches may come from an older AxonID with another patch size.
    sample = sorted(paths["patches_img"].glob("*.png")) + sorted(paths["patches_img"].glob("*.tif"))
    if sample:
        actual = cv2.imread(str(sample[0]), cv2.IMREAD_GRAYSCALE).shape[0]
        if actual != RECIPE.patch_size:
            raise ValueError(
                f"Existing patches in {paths['patches_img']} are {actual}px; AxonID uses "
                f"{RECIPE.patch_size}px. Delete images/cropped and masks/cropped to regenerate them."
            )

    # ── Split by source image ─────────────────────────────────────────────────
    log.rule("TRAIN / VALIDATION SPLIT")
    manifest = build_image_manifest(paths["images_dir"])
    train_m, val_m, split_mode = split_images(manifest, RECIPE.val_fraction, seed)
    summary = split_summary(train_m, val_m)
    if split_mode == "stratified":
        log.info(f"Stratified by phenotype: {summary['train_images']} train "
                 f"({summary['train_ctrl']} ctrl / {summary['train_regen']} regen) | "
                 f"{summary['val_images']} val ({summary['val_ctrl']} ctrl / {summary['val_regen']} regen) | seed={seed}")
    else:
        log.warn(f"{len(summary['unlabeled'])} image name(s) start with neither ctrl_ nor regen_ "
                 f"(e.g. {summary['unlabeled'][:3]}): using a random image-level split. Validation may not "
                 f"contain both phenotypes. Rename images ctrl_* / regen_* for a stratified split.")
        log.info(f"Random image-level split: {summary['train_images']} train | {summary['val_images']} val | seed={seed}")

    # ── Load patches ──────────────────────────────────────────────────────────
    log.rule("LOADING PATCHES")
    X_train, Y_train, X_val, Y_val, val_patch_stems = load_all_patches(
        paths["images_dir"], paths["masks_dir"],
        train_stems=summary["train_stems"], val_stems=summary["val_stems"], log=log,
    )
    Y_train = Y_train.astype(np.int64)
    Y_val   = Y_val.astype(np.int64)

    ctrl_idx  = [i for i, s in enumerate(val_patch_stems) if s.startswith("ctrl_")]
    regen_idx = [i for i, s in enumerate(val_patch_stems) if s.startswith("regen_")]

    # Seeded after loading and before augmentation and model construction,
    # the same order as the code that trained the released models.
    aug_rng = set_seed(seed)

    # ── Augmentation ──────────────────────────────────────────────────────────
    aug_count = 0
    if use_aug:
        log.rule("AUGMENTATION")
        X_train, Y_train, aug_count, aug_counts = augment_dataset_np(X_train, Y_train, rng=aug_rng)
        log.success(f"{aug_count}/{len(X_train)} patches modified ({round(aug_count / len(X_train) * 100, 1)}%)")
        log.info("  " + "  |  ".join(f"{k}: {v}" for k, v in aug_counts.items()))

    # ── Model, loss, optimizer ────────────────────────────────────────────────
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = list(RECIPE.class_weights)
    cw_tensor     = torch.tensor(class_weights, dtype=torch.float32).to(device)
    ce_loss       = torch.nn.CrossEntropyLoss(weight=cw_tensor)

    def loss_fn(pred, target):
        return RECIPE.dice_weight * weighted_dice_loss(pred, target, cw_tensor) + RECIPE.ce_weight * ce_loss(pred, target)

    model  = build_model(device=device, encoder_weights=RECIPE.encoder_weights)
    groups = describe_parameter_groups(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE.learning_rate, weight_decay=RECIPE.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=RECIPE.reduce_lr_factor,
        patience=RECIPE.reduce_lr_patience, min_lr=RECIPE.reduce_lr_min_lr,
    )

    deviations = []
    if batch_size != RECIPE.batch_size:
        deviations.append(f"batch_size={batch_size} (recipe {RECIPE.batch_size})")
    if epochs != RECIPE.max_epochs:
        deviations.append(f"epochs_limit={epochs} (recipe {RECIPE.max_epochs})")

    log.rule("TRAINING SETUP")
    log.log_dict({
        "Recipe":           RECIPE_ID,
        "Architecture":     f"{RECIPE.architecture}, {RECIPE.encoder} encoder, {RECIPE.encoder_weights} weights",
        "Parameters":       f"{groups['total']['params']:,} (encoder {groups['encoder']['params']:,} | "
                            f"decoder {groups['decoder']['params']:,} | head {groups['head']['params']:,})",
        "Class weights":    f"bg={class_weights[0]} myelin={class_weights[1]} axon={class_weights[2]}",
        "Device":           str(device),
        "Train patches":    len(X_train),
        "Val patches":      len(X_val),
        "Split":            split_mode,
        "Batch size":       batch_size if batch_size == RECIPE.batch_size else f"{batch_size} (recipe {RECIPE.batch_size})",
        "Epoch limit":      epochs,
        "Augmentation":     "ON (AxonID recipe)" if use_aug else "OFF",
        "Loss function":    f"Weighted Dice ({RECIPE.dice_weight}) + CrossEntropy ({RECIPE.ce_weight})",
        "Optimizer":        f"AdamW lr={RECIPE.learning_rate} wd={RECIPE.weight_decay}",
        "ReduceLR":         f"patience={RECIPE.reduce_lr_patience}, factor={RECIPE.reduce_lr_factor}, "
                            f"min_lr={RECIPE.reduce_lr_min_lr} (monitors val_loss)",
        "EarlyStopping":    f"patience={RECIPE.early_stop_patience}, min_delta={RECIPE.early_stop_min_delta} (monitors val_loss)",
        "Recipe deviations": "; ".join(deviations) if deviations else "none",
    })

    # ── DataLoaders ───────────────────────────────────────────────────────────
    # (N, H, W, 1) -> (N, 1, H, W) for PyTorch.
    X_train_t = torch.from_numpy(X_train.transpose(0, 3, 1, 2)).float()
    X_val_t   = torch.from_numpy(X_val.transpose(0, 3, 1, 2)).float()
    Y_train_t = torch.from_numpy(Y_train.squeeze(-1)).long()
    Y_val_t   = torch.from_numpy(Y_val.squeeze(-1)).long()

    # drop_last=False (the default): the final partial batch is used, as in
    # the training of the released models.
    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t,   Y_val_t),   batch_size=batch_size, shuffle=False)

    model_dir.mkdir(parents=True, exist_ok=True)

    remainder = len(X_train) % batch_size
    base_meta = {
        **recipe_metadata(),
        # Identity
        "model_name":          model_name,
        "version":             __version__,
        "trained_date":        datetime.now().strftime("%Y-%m-%d"),
        "magnification":       mag,
        "activation":          "none (raw logits)",
        # Data and split
        "dataset_path":        str(Path(images_dir).resolve()),
        "split_mode":          split_mode,
        "seed":                seed,
        "val_fraction":        RECIPE.val_fraction,
        "n_images":            len(manifest),
        "train_images":        summary["train_stems"],
        "val_images":          summary["val_stems"],
        "n_train_patches":     len(X_train),
        "n_val_patches":       len(X_val),
        # Training run
        "augmentation":        bool(use_aug),
        "aug_params":          AUGMENTATION.as_dict() if use_aug else None,
        "batch_size":          batch_size,
        "last_batch_fullness": f"{int(remainder / batch_size * 100) if remainder else 100}%",
        "epochs_limit":        epochs,
        "recipe_match":        not deviations,
        "recipe_deviations":   deviations,
        "parameter_groups":    groups,
        # Environment
        "gpu":                 torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch_version":       str(torch.__version__),
        "python_version":      sys.version.split()[0],
        "hostname":            socket.gethostname(),
    }

    best_val_loss = float("inf")
    best = {"epoch": 0, "bg": None, "axon": None, "myelin": None, "iou": None}
    best_checkpoint = None
    epochs_no_improve = 0
    training_logger = TrainingLogger(log)

    # ── Training loop ─────────────────────────────────────────────────────────
    log.rule("TRAINING")
    n_epochs = 0
    for epoch in range(epochs):
        epoch_start = datetime.now()
        n_epochs += 1

        model.train()
        train_loss = 0.0
        train_stats = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss  = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            train_stats.append(smp.metrics.get_stats(preds.argmax(dim=1), yb, mode="multiclass", num_classes=3))

        model.eval()
        val_loss = 0.0
        val_stats = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += loss_fn(preds, yb).item() * len(xb)
                val_stats.append(smp.metrics.get_stats(preds.argmax(dim=1), yb, mode="multiclass", num_classes=3))

        train_loss /= len(X_train_t)
        val_loss   /= len(X_val_t)
        current_lr  = optimizer.param_groups[0]["lr"]
        train_ep = compute_epoch_metrics(*(torch.cat(s) for s in zip(*train_stats)))
        val_ep   = compute_epoch_metrics(*(torch.cat(s) for s in zip(*val_stats)))

        # Checkpoint on best val_loss.
        if val_loss < best_val_loss - RECIPE.early_stop_min_delta:
            best_val_loss = val_loss
            best = {"epoch": epoch + 1, "bg": val_ep["dice_bg"], "axon": val_ep["dice_axon"],
                    "myelin": val_ep["dice_myelin"], "iou": val_ep["iou_macro"]}
            # deepcopy: state_dict() returns live tensors that keep changing
            # as training continues; the copy freezes this epoch's weights.
            best_checkpoint = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "meta": {
                    **base_meta,
                    "best_epoch":       epoch + 1,
                    "best_val_loss":    val_loss,
                    "best_bg_dice":     val_ep["dice_bg"],
                    "best_axon_dice":   val_ep["dice_axon"],
                    "best_myelin_dice": val_ep["dice_myelin"],
                    "best_val_iou":     val_ep["iou_macro"],
                    "epochs_completed": None,
                    "early_stopped":    None,
                },
            }
            # Saved at every improvement so an interrupted run still leaves
            # the best model so far on disk.
            save_model_file(model_path, best_checkpoint["model_state_dict"], best_checkpoint["meta"])
            epochs_no_improve = 0
            checkpoint_flag = " <- CHECKPOINT"
        else:
            epochs_no_improve += 1
            checkpoint_flag = ""

        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < current_lr:
            log.print(f"  ReduceLR: {current_lr:.2e} -> {new_lr:.2e} "
                      f"(val_loss no improvement for {RECIPE.reduce_lr_patience} epochs)")

        training_logger.log_epoch(epoch, {
            "epoch_time":      f"{int((datetime.now() - epoch_start).total_seconds())}s",
            "lr":              current_lr,
            "loss":            train_loss,
            "dice":            train_ep["dice_macro"],
            "dice_bg":         train_ep["dice_bg"],
            "dice_axon":       train_ep["dice_axon"],
            "dice_myelin":     train_ep["dice_myelin"],
            "val_loss":        val_loss,
            "val_dice":        val_ep["dice_macro"],
            "val_dice_bg":     val_ep["dice_bg"],
            "val_dice_axon":   val_ep["dice_axon"],
            "val_dice_myelin": val_ep["dice_myelin"],
        }, checkpoint_flag=checkpoint_flag)

        if epochs_no_improve >= RECIPE.early_stop_patience:
            log.info(f"Early stopping at epoch {epoch + 1}: no val_loss improvement for "
                     f"{RECIPE.early_stop_patience} epochs")
            break

    # ── Restore best-epoch weights ────────────────────────────────────────────
    early_stopped = n_epochs < epochs
    if best_checkpoint is None:
        raise RuntimeError(
            f"No checkpoint was saved: val_loss never improved over {n_epochs} epoch(s). This usually "
            f"means val_loss was NaN/Inf from the start (check the validation images and masks)."
        )
    best_checkpoint["meta"]["epochs_completed"] = n_epochs
    best_checkpoint["meta"]["early_stopped"]    = early_stopped
    save_model_file(model_path, best_checkpoint["model_state_dict"], best_checkpoint["meta"])
    model.load_state_dict(best_checkpoint["model_state_dict"])

    # ── Evaluation at the best epoch ──────────────────────────────────────────
    log.rule("POST-TRAINING EVALUATION (validation set, best epoch)")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb.to(device))
            all_preds.append(preds.argmax(dim=1).cpu())
            all_labels.append(yb)
    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    m = compute_all_metrics(all_preds, all_labels, device)

    bad = [k for k in ("hd95_macro", "hd95_bg", "hd95_myelin", "hd95_axon", "hd95_foreground")
           if k in m and (m[k] is None or (isinstance(m[k], float) and not math.isfinite(m[k])))]
    if bad:
        log.warn(f"HD95 is undefined for {bad}: a class is missing from the validation set's "
                 f"predictions or masks. Dice/IoU are unaffected.")

    for label, key in (("Dice", "dice"), ("IoU", "iou"), ("Precision", "precision"), ("Recall", "recall"), ("HD95", "hd95")):
        log.info(f"{label:<9} macro {_f(m.get(f'{key}_macro'), '.4f')} | bg {_f(m.get(f'{key}_bg'), '.4f')} | "
                 f"myelin {_f(m.get(f'{key}_myelin'), '.4f')} | axon {_f(m.get(f'{key}_axon'), '.4f')}")

    for name, idx in (("ctrl", ctrl_idx), ("regen", regen_idx)):
        if idx:
            pm = compute_all_metrics(all_preds[idx], all_labels[idx], device)
            log.info(f"{name:<5} ({len(idx)} patches) Dice macro {_f(pm.get('dice_macro'), '.4f')} | "
                     f"myelin {_f(pm.get('dice_myelin'), '.4f')} | axon {_f(pm.get('dice_axon'), '.4f')}")

    training_logger.on_train_end({**best, "loss": best_val_loss, "path": model_path.name})

    elapsed = (datetime.now() - t_start).total_seconds()
    log.finalize(summary={
        "Model":             str(model_path),
        "GPU":               base_meta["gpu"],
        "Training time":     f"{int(elapsed // 60)}m {int(elapsed % 60)}s",
        "Epochs":            f"{n_epochs} (early stopped)" if early_stopped else str(n_epochs),
        "Patches processed": n_epochs * len(X_train),
        "Augmented":         aug_count,
        "Best epoch":        best["epoch"],
        "Best val_loss":     _f(best_val_loss, ".4f"),
        "Dice macro":        _f(m.get("dice_macro"), ".4f"),
        "Dice myelin":       _f(m.get("dice_myelin"), ".4f"),
        "Dice axon":         _f(m.get("dice_axon"), ".4f"),
        "IoU macro":         _f(m.get("iou_macro"), ".4f"),
        "HD95 macro":        _f(m.get("hd95_macro"), ".4f"),
        "Recipe deviations": "; ".join(deviations) if deviations else "none",
    })
    return model_path
