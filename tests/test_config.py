"""config.toml loading and train_config.toml validation."""
import tomllib

import pytest

import utils.config as C


def test_tables_are_flattened(temp_config):
    temp_config('[reporting]\nprimary_gratio_method = "mean_axes"\n[folders]\ntiff_suffixes = ["_TIFF"]\n'
                '[timing]\nenabled = true\n[pixel_size_um.40X]\n1440 = 0.217\n')
    c = C.load_config()
    assert c["primary_gratio_method"] == "mean_axes" and c["tiff_suffixes"] == ["_TIFF"]
    assert c["timing"] is True
    assert C.get_pixel_size("40X", 1440) == pytest.approx(0.217)
    assert C.get_pixel_size("40X", 999) is None


@pytest.mark.parametrize("text", [
    "[watershed]\ndistance_threshold = 0.1\n",
    "clahe = true\n",
    "[quality_filters]\nenabled = true\n",
    "[training]\nlearning_rate = 0.01\n",
])
def test_locked_settings_are_refused(temp_config, text):
    temp_config(text)
    with pytest.raises(ValueError, match="cannot be changed"):
        C.load_config()


def test_syntax_error_names_the_file(temp_config):
    temp_config("pixel_size_um = {\n")
    with pytest.raises(tomllib.TOMLDecodeError, match="config.toml could not be read"):
        C.load_config()


def test_old_config_json_hint(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(C, "_CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(C, "_OLD_CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(C, "_config_cache", None)
    with pytest.raises(FileNotFoundError, match="earlier version"):
        C.load_config()


def test_shipped_config_loads():
    # The config.toml in the repository must always be valid.
    with open(C._CONFIG_PATH, "rb") as f:
        raw = tomllib.load(f)
    C.check_locked_keys(C._flatten(raw))
    assert "40X" in raw["pixel_size_um"]


# ── train_config.toml ─────────────────────────────────────────────────────────
@pytest.fixture
def train_cfg(tmp_path):
    from train.__main__ import _load_run_config
    data = tmp_path / "MyTraining"
    data.mkdir()

    def load(body: str):
        p = tmp_path / "train_config.toml"
        p.write_text(f"images_dir = '{data}'\n" + body, encoding="utf-8")
        return _load_run_config(str(p))
    return load


def test_train_config_defaults(train_cfg):
    r = train_cfg('model_name = "unit_test_model_x"\n')
    assert (r["epochs"], r["batch_size"], r["seed"], r["augmentation"], r["magnification"]) == (400, 128, 1, False, "40X")


@pytest.mark.parametrize("body,match", [
    ('arch = "unet"\n', "unsupported key"),
    ("batch_size = 0\n", "positive integer"),
    ('augmentation = "yes"\n', "true or false"),
    ('magnification = "100X"\n', "magnification"),
])
def test_train_config_refusals(train_cfg, body, match):
    with pytest.raises(ValueError, match=match):
        train_cfg(body)


def test_train_config_windows_path_hint(tmp_path):
    from train.__main__ import _load_run_config
    p = tmp_path / "train_config.toml"
    p.write_text('images_dir = "C:\\Data\\x"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="single quotes"):
        _load_run_config(str(p))
