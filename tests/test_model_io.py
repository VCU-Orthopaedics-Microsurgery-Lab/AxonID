"""Model files: .safetensors round trip, checksums, model menu, refusing .pt."""
import hashlib
import json

import pytest
import torch

from utils.model_io import load_model_file, save_model_file
from utils.recipe import build_model, recipe_metadata


def test_round_trip_is_exact(tmp_path):
    sd = {"w": torch.randn(3, 4), "n": torch.tensor(7, dtype=torch.int64)}
    meta = {"model_name": "x", "val_images": ["ctrl_a"], "best_val_loss": 0.25, "groups": {"total": 10}}
    p = tmp_path / "m.safetensors"
    save_model_file(p, sd, meta)
    back, meta_back = load_model_file(p)
    assert back.keys() == sd.keys() and all(torch.equal(back[k], sd[k]) and back[k].dtype == sd[k].dtype for k in sd)
    assert meta_back == meta


def test_full_model_loads_through_segmentation(tmp_path):
    from segment.segment import load_model
    model = build_model(encoder_weights=None)
    p = tmp_path / "m.safetensors"
    save_model_file(p, model.state_dict(), recipe_metadata())
    loaded, meta = load_model(str(p), torch.device("cpu"))
    a, b = model.state_dict(), loaded.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not loaded.training                        # inference mode


def test_pt_files_are_refused(tmp_path):
    p = tmp_path / "old.pt"
    torch.save({"model_state_dict": {}, "meta": {}}, p)
    with pytest.raises(ValueError, match="safetensors"):
        load_model_file(p)


def test_checksums(tmp_path):
    from utils.model_registry import verify_file
    p = tmp_path / "f.bin"
    p.write_bytes(b"axonid")
    sha, md5 = hashlib.sha256(b"axonid").hexdigest(), hashlib.md5(b"axonid").hexdigest()
    assert verify_file(p, {"sha256": sha, "md5": md5}) is True
    assert verify_file(p, {"sha256": sha.upper()}) is True
    assert verify_file(p, {"sha256": "0" * 64, "md5": md5}) is False      # any mismatch fails
    assert verify_file(p, {"sha256": "YOUR_VALUE_HERE"}) is False
    assert verify_file(p, {}) is None


def test_model_menu_lists_released_then_user(tmp_path, monkeypatch):
    import utils.model_registry as R
    (tmp_path / "user").mkdir()
    (tmp_path / "models.json").write_text(json.dumps({"models": [
        {"file": "rel_b.safetensors", "label": "B", "recommended": False},
        {"file": "rel_a.safetensors", "label": "A", "recommended": True},
        {"file": "missing.safetensors", "label": "M", "recommended": False}]}))
    for name in ("rel_a.safetensors", "rel_b.safetensors", "user/mine.safetensors", "user/old.pt"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(R, "get_model_dir", lambda: tmp_path)
    labels = [(e.label, e.source) for e in R.available_models()]
    assert labels == [("A", "released"), ("B", "released"), ("mine", "user")]   # .pt not listed
    assert R.missing_released_models() == ["missing.safetensors"]
