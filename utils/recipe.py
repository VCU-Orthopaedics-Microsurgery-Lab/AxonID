"""
utils/recipe.py: the locked AxonID model and training recipe.

Purpose
    Single source of truth for every setting that defines an AxonID model:
    architecture, encoder, loss, optimizer, schedule, early stopping, class
    weights, patch geometry, split and augmentation. These values reproduce
    the released AxonID 1.0.0 models and are not configurable through
    config.toml (utils/config.py refuses them if they appear there).

Usage
    from utils.recipe import RECIPE, build_model, check_checkpoint_compatible

Contents
    RECIPE                       frozen settings (TrainingRecipe)
    AUGMENTATION                 frozen augmentation recipe (used only when augmentation is on)
    MORPHOMETRICS                frozen watershed and quality-filter settings (morphometrics)
    build_model()                the only place a network is constructed (training and inference)
    parameter_groups()           encoder / decoder / head parameter groups of a built model
    describe_parameter_groups()  parameter counts per group, for logs and metadata
    check_checkpoint_compatible() refuses checkpoints that were not built with this recipe

Inactive extension points (fixed values, not user settings)
    init_from, freeze, deep_supervision: read by the training code so that
    fine-tuning and deep supervision can be added later without restructuring.
    They are locked to "off" in 1.0.0.

Depends on
    segmentation_models_pytorch (UNet++ with a DenseNet169 encoder)
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
from dataclasses import dataclass

RECIPE_ID = "axonid-1.0-unetpp-densenet169"


@dataclass(frozen=True)
class AugmentationRecipe:
    """Augmentation used to train the released augmented model (axonid_v1_unetpp_dn169_aug)."""
    p_clean: float = 0.5            # probability a patch is left unaugmented
    hflip_prob: float = 0.05
    vflip_prob: float = 0.20
    rotation_prob: float = 0.10
    rotation_deg: float = 45.0
    gamma_prob: float = 0.10
    gamma_range: tuple = (0.8, 1.2)
    noise_prob: float = 0.05
    noise_sigma: float = 0.02       # fraction of the patch's own intensity range
    blur_prob: float = 0.05
    blur_sigma: float = 0.5         # pixels
    elastic_prob: float = 0.05
    elastic_alpha: float = 50.0
    elastic_sigma: float = 8.0

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["gamma_range"] = list(self.gamma_range)
        return d


@dataclass(frozen=True)
class TrainingRecipe:
    # Model
    architecture: str = "unet++"
    encoder: str = "densenet169"
    encoder_weights: str = "imagenet"   # initialization for new training only
    in_channels: int = 1
    classes: tuple = ("background", "myelin", "axon")
    # Loss: dice_weight * weighted Dice + ce_weight * weighted cross-entropy
    class_weights: tuple = (1.0, 2.0, 2.0)   # background, myelin, axon
    dice_weight: float = 0.5
    ce_weight: float = 0.5
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    reduce_lr_patience: int = 15
    reduce_lr_factor: float = 0.5
    reduce_lr_min_lr: float = 1e-6
    early_stop_patience: int = 40
    early_stop_min_delta: float = 0.001
    max_epochs: int = 400
    batch_size: int = 128
    checkpoint_metric: str = "val_loss"
    # Data geometry (patch step is patch_size // 2, see utils/patching.py).
    # Images larger than target_resolution (width, height) are downsampled to
    # fit it before patching, in training and in segmentation alike.
    patch_size: int = 256
    target_resolution: tuple = (1440, 1024)
    # Split
    val_fraction: float = 0.33
    default_seed: int = 1
    # Inactive extension points (locked off in 1.0.0)
    init_from: str | None = None
    freeze: str = "none"
    deep_supervision: bool = False


@dataclass(frozen=True)
class MorphometricsRecipe:
    """
    Axon separation (watershed) and acceptance rules used by morphometrics.
    Validated on the ground-truth masks of the 10 holdout images: 7746 axons
    counted against 7730 annotated (0.2%).
    """
    # Watershed seeding: seeds where distance > distance_threshold * max
    # distance (h_minima_px=None selects this mode).
    distance_threshold: float = 0.08
    dilation_disk: int = 5          # background-marker dilation radius, px
    h_minima_px: float | None = None
    min_seed_area_px: int = 2       # seeds smaller than this are dropped
    # Configurable morphology filters. Off in 1.0.0: only the two hard
    # checks below decide acceptance. The thresholds are kept so the
    # filters can be re-enabled as a recipe change, not a code change.
    quality_filters_enabled: bool = False
    min_axon_area_um2: float = 0.5
    max_axon_area_um2: float = 500.0
    min_axon_solidity: float = 0.3
    max_axon_eccentricity: float = 0.95
    min_myelin_thickness_um: float = 0.1
    gratio_min: float = 0.2
    gratio_max: float = 0.9
    min_fiber_area_um2: float = 1.0
    max_fiber_area_um2: float = 800.0
    # Hard checks, always applied: axons below this equivalent diameter are
    # rejected (needs calibration), and both g-ratios must lie in (0, 1).
    min_axon_diam_um: float = 1.01


RECIPE = TrainingRecipe()
AUGMENTATION = AugmentationRecipe()
MORPHOMETRICS = MorphometricsRecipe()
MORPHOMETRICS_ID = "axonid-1.0-watershed-v4"

# Magnifications the released models and calibration workflow support.
SUPPORTED_MAGS = ("40X",)

# Input normalization. Training and inference intentionally differ: both
# L2-normalize each 256x256 patch, but training divides by the norm of each
# patch column and inference by the norm of each patch row. Inference's
# convention was kept after a direct comparison (2026-08-14, 8 holdout images:
# 7 of 8 better, mean Dice +0.009). Do not "align" either side without re-testing.
NORMALIZATION_TRAINING = "L2 per patch column (np.linalg.norm(batch[N,H,W,1], axis=1))"
NORMALIZATION_INFERENCE = "L2 per patch row (np.linalg.norm(patch[H,W], axis=1))"


def recipe_metadata() -> dict:
    """Recipe values written into every checkpoint trained by AxonID."""
    r = RECIPE
    return {
        "recipe_id": RECIPE_ID,
        "architecture": r.architecture,
        "encoder": r.encoder,
        "encoder_weights": r.encoder_weights,
        "in_channels": r.in_channels,
        "classes": list(r.classes),
        "class_weights": list(r.class_weights),
        "dice_weight": r.dice_weight,
        "ce_weight": r.ce_weight,
        "learning_rate": r.learning_rate,
        "weight_decay": r.weight_decay,
        "reduce_lr_patience": r.reduce_lr_patience,
        "reduce_lr_factor": r.reduce_lr_factor,
        "reduce_lr_min_lr": r.reduce_lr_min_lr,
        "early_stop_patience": r.early_stop_patience,
        "early_stop_min_delta": r.early_stop_min_delta,
        "epochs_limit": r.max_epochs,
        "checkpoint_metric": r.checkpoint_metric,
        "patch_size": r.patch_size,
        "target_resolution": list(r.target_resolution),
        "normalization_training": NORMALIZATION_TRAINING,
        "normalization_inference": NORMALIZATION_INFERENCE,
        "fine_tuned_from": r.init_from,
        "freeze": r.freeze,
        "deep_supervision": r.deep_supervision,
    }


def build_model(device=None, encoder_weights: str | None = None):
    """
    Construct the AxonID network. Used by both training and segmentation, so
    the two can never build different models.

    encoder_weights: "imagenet" for new training; None when the weights will
    be loaded from a checkpoint (segmentation).
    """
    import segmentation_models_pytorch as smp

    model = smp.UnetPlusPlus(
        encoder_name=RECIPE.encoder,
        encoder_weights=encoder_weights,
        in_channels=RECIPE.in_channels,
        classes=len(RECIPE.classes),
        activation=None,           # raw logits; softmax is applied by the caller
    )
    return model.to(device) if device is not None else model


def parameter_groups(model) -> dict:
    """
    Encoder / decoder / segmentation-head modules of a built model. These are
    the groups a future fine-tuning freeze strategy acts on.
    """
    return {
        "encoder": model.encoder,
        "decoder": model.decoder,
        "head": model.segmentation_head,
    }


def describe_parameter_groups(model) -> dict:
    """{group: {"params": n, "trainable": n}} plus a "total" entry."""
    out, total, total_tr = {}, 0, 0
    for name, module in parameter_groups(model).items():
        n = sum(p.numel() for p in module.parameters())
        t = sum(p.numel() for p in module.parameters() if p.requires_grad)
        out[name] = {"params": n, "trainable": t}
        total += n
        total_tr += t
    out["total"] = {"params": total, "trainable": total_tr}
    return out


def check_checkpoint_compatible(meta: dict, model_name: str = "") -> None:
    """
    Raise ValueError unless a checkpoint was built with the locked recipe.
    Checkpoints without metadata are refused: the architecture cannot be
    inferred safely and a silent fallback would produce wrong segmentations.
    """
    label = f" ({model_name})" if model_name else ""
    if not meta:
        raise ValueError(
            f"Checkpoint{label} has no AxonID metadata. Only checkpoints trained "
            f"with AxonID {RECIPE.architecture}/{RECIPE.encoder} can be used."
        )
    arch = str(meta.get("architecture", "")).lower()
    enc = str(meta.get("encoder", "")).lower()
    if arch != RECIPE.architecture or enc != RECIPE.encoder:
        raise ValueError(
            f"Checkpoint{label} was built as {arch or '?'}/{enc or '?'}; AxonID "
            f"{RECIPE_ID} requires {RECIPE.architecture}/{RECIPE.encoder}."
        )
    n_classes = len(meta.get("classes", RECIPE.classes))
    if meta.get("in_channels", RECIPE.in_channels) != RECIPE.in_channels or n_classes != len(RECIPE.classes):
        raise ValueError(
            f"Checkpoint{label} has in_channels={meta.get('in_channels')} and "
            f"{n_classes} classes; expected {RECIPE.in_channels} and {len(RECIPE.classes)}."
        )
    if meta.get("patch_size", RECIPE.patch_size) != RECIPE.patch_size:
        raise ValueError(
            f"Checkpoint{label} was trained on {meta.get('patch_size')} px patches; "
            f"AxonID uses {RECIPE.patch_size} px."
        )
