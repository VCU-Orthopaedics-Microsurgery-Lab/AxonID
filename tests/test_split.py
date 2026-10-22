"""Image-level train/validation split (synthetic file names only)."""
import pytest

from train.dataset.split import get_phenotype, split_images, split_summary


def manifest(names):
    return sorted(({"stem": n, "path": None, "phenotype": get_phenotype(n)} for n in names),
                  key=lambda m: m["stem"])


BALANCED = manifest([f"ctrl_s{i:02d}" for i in range(15)] + [f"regen_s{i:02d}" for i in range(15)])


def test_stratified_balanced_sizes_and_no_overlap():
    train, val, mode = split_images(BALANCED, 0.33, seed=1)
    s = split_summary(train, val)
    assert mode == "stratified"
    assert (s["val_ctrl"], s["val_regen"], s["train_images"]) == (5, 5, 20)
    assert not set(s["train_stems"]) & set(s["val_stems"])


def test_same_seed_same_split_different_seed_different_split():
    a = split_images(BALANCED, 0.33, seed=1)[1]
    b = split_images(BALANCED, 0.33, seed=1)[1]
    c = split_images(BALANCED, 0.33, seed=2)[1]
    assert [m["stem"] for m in a] == [m["stem"] for m in b]
    assert [m["stem"] for m in a] != [m["stem"] for m in c]


def test_seed1_golden_split():
    # Fixed expected result: guards against any change to the split algorithm,
    # which would change which images a given seed puts in validation.
    val = sorted(m["stem"] for m in split_images(BALANCED, 0.33, seed=1)[1])
    assert val == GOLDEN_SEED1


def test_unbalanced_groups():
    m = manifest([f"ctrl_{i}" for i in range(7)] + [f"regen_{i}" for i in range(4)])
    s = split_summary(*split_images(m, 0.33, seed=1)[:2])
    assert (s["val_ctrl"], s["val_regen"]) == (2, 1)


def test_unlabeled_names_fall_back_to_random_image_split():
    m = manifest([f"ctrl_{i}" for i in range(4)] + ["nerve_A", "nerve_B"])
    train, val, mode = split_images(m, 0.33, seed=1)
    assert mode == "random_image" and len(val) == 2 and len(train) == 4


def test_validation_size_rounds_half_up():
    from train.dataset.split import _n_val
    assert _n_val(5, 0.5) == 3          # 2.5 rounds up (Python's round() would give 2)
    assert _n_val(15, 0.33) == 5 and _n_val(2, 0.33) == 1


def test_too_few_images_refused():
    with pytest.raises(ValueError):
        split_images(manifest(["ctrl_only"]), 0.33, seed=1)
    with pytest.raises(ValueError):
        split_images(manifest(["ctrl_a", "regen_a"]), 0.33, seed=1)   # no training image left per group


# Same result as the split code that trained the released models (checked when written).
GOLDEN_SEED1 = ['ctrl_s00', 'ctrl_s06', 'ctrl_s07', 'ctrl_s12', 'ctrl_s13', 'regen_s00', 'regen_s06', 'regen_s09', 'regen_s10', 'regen_s13']
