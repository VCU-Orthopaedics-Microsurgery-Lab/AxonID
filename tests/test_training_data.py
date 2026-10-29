"""Augmentation and the locked model recipe (no training run needed)."""
import numpy as np
import pytest

from train.dataset.augment import augment_dataset_np
from utils.recipe import (
    AUGMENTATION, MORPHOMETRICS, RECIPE, check_checkpoint_compatible, recipe_metadata,
)


@pytest.fixture
def patches():
    rng = np.random.default_rng(0)
    x = rng.random((200, 64, 64, 1)).astype(np.float32)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = rng.integers(0, 3, (200, 64, 64, 1)).astype(np.float32)
    return x, y


def test_augmentation_is_reproducible(patches):
    x, y = patches
    a = augment_dataset_np(x, y, rng=np.random.default_rng(1))
    b = augment_dataset_np(x, y, rng=np.random.default_rng(1))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]


def test_augmentation_golden_counts(patches):
    # Same counts as the augmentation code that trained the released augmented
    # model, for this input and seed (checked when written). A change here means
    # a given seed no longer reproduces the released training.
    x, y = patches
    _, _, n_aug, counts = augment_dataset_np(x, y, rng=np.random.default_rng(1))
    assert n_aug == 59
    assert counts == {"hflip": 8, "vflip": 25, "rotation": 12, "elastic": 6, "gamma": 10, "noise": 7, "blur": 3}


def test_augmentation_keeps_masks_and_normalisation(patches):
    x, y = patches
    xa, ya, n_aug, counts = augment_dataset_np(x, y, rng=np.random.default_rng(1))
    assert xa.shape == x.shape and ya.shape == y.shape
    assert set(np.unique(ya)) <= {0.0, 1.0, 2.0}                 # class indices survive
    changed = [i for i in range(len(x)) if not np.array_equal(xa[i], x[i])]
    assert len(changed) == n_aug > 0
    norms = np.linalg.norm(xa[changed], axis=1)                   # per-column L2 = 1 again
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-4)
    # About half the patches stay clean (p_clean = 0.5).
    assert 0.3 < 1 - n_aug / len(x) < 0.8


def test_recipe_values_are_the_released_ones():
    assert (RECIPE.architecture, RECIPE.encoder) == ("unet++", "densenet169")
    assert RECIPE.class_weights == (1.0, 2.0, 2.0)
    assert (RECIPE.patch_size, RECIPE.target_resolution, RECIPE.batch_size) == (256, (1440, 1024), 128)
    assert AUGMENTATION.p_clean == 0.5 and AUGMENTATION.elastic_alpha == 50.0
    assert (MORPHOMETRICS.distance_threshold, MORPHOMETRICS.dilation_disk,
            MORPHOMETRICS.min_seed_area_px, MORPHOMETRICS.min_axon_diam_um) == (0.08, 5, 2, 1.01)


def test_checkpoint_compatibility_check():
    ok = recipe_metadata()
    check_checkpoint_compatible(ok)                                       # no error
    for bad in ({}, {**ok, "architecture": "unet"}, {**ok, "encoder": "resnet34"}, {**ok, "patch_size": 512}):
        with pytest.raises(ValueError):
            check_checkpoint_compatible(bad)
