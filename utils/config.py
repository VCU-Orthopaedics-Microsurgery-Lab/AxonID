"""
utils/config.py: read config.toml and summarize the settings each stage uses.

Usage
    from utils.config import load_config, get_pixel_size
    px = get_pixel_size("40X", 1440)     # µm/px at that ORIGINAL image width, or None

Inputs
    <repo>/config.toml (read once and cached).

Outputs
    load_config(): the settings dict. get_pixel_size(): calibration lookup.
    format_*_config_summary(): lines each entry point logs before it runs;
    the morphometrics one returns (level, text) pairs so a missing
    calibration is shown as a warning.

Locked (utils/recipe.py)
    Model, training, augmentation, working resolution, watershed and
    quality-filter settings. If config.toml contains any of them, loading
    stops with an explanation (LOCKED_KEYS) instead of silently ignoring it.

Depends on
    standard library; utils/recipe.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
from pathlib import Path
import tomllib
from typing import Any

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"
_OLD_CONFIG_PATH = _CONFIG_PATH.with_name("config.json")

# config.toml groups settings into tables for readability; the rest of AxonID
# reads them as top-level keys. These tables are flattened when loading.
_FLATTENED_TABLES = ("reporting", "folders")
_config_cache: dict[str, Any] | None = None

# Single source of truth for distributions.py's default bin edges — also
# read by _format_bin_edges() below, so the two can never drift apart.
# float("inf") == np.inf numerically; the built-in avoids adding a numpy
# dependency to this module.
DEFAULT_GRANULAR_BINS_UM = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, float("inf")]
DEFAULT_MID_BINS_UM      = [0, 2, 4, 6, 8, 10, 12, 14, 16, float("inf")]
DEFAULT_COARSE_BINS_UM   = [0, 5, 10, 15, 20, 25, 30, float("inf")]


# Settings fixed by the AxonID recipe (utils/recipe.py). They are refused in
# config.toml rather than ignored, so nobody believes a value took effect.
LOCKED_KEYS = {
    "training":      "training settings are fixed by the AxonID recipe (utils/recipe.py)",
    "augmentation":  "augmentation settings are fixed by the AxonID recipe (utils/recipe.py)",
    "patch_size":    "patch size is fixed at 256 px by the AxonID recipe (utils/recipe.py)",
    "architecture":  "the architecture is fixed (UNet++/DenseNet169) by the AxonID recipe",
    "encoder":       "the encoder is fixed (DenseNet169) by the AxonID recipe",
    "class_weights": "class weights are fixed by the AxonID recipe (utils/recipe.py)",
    "target_resolution": "the working resolution (1440x1024) is fixed by the AxonID recipe",
    "clahe":         "CLAHE was removed in AxonID 1.0.0",
    "watershed":     "watershed settings are fixed by the AxonID morphometrics recipe (utils/recipe.py)",
    "quality_filters": "quality filters are fixed by the AxonID morphometrics recipe (utils/recipe.py)",
}


def check_locked_keys(config: dict[str, Any]) -> None:
    """Raise ValueError if config.toml tries to set a recipe-locked value."""
    found = [k for k in LOCKED_KEYS if k in config]
    if found:
        reasons = "\n".join(f"  - {k}: {LOCKED_KEYS[k]}" for k in found)
        raise ValueError(
            f"config.toml contains settings that cannot be changed:\n{reasons}\n"
            f"Remove these entries from {_CONFIG_PATH} and run again."
        )


def _flatten(raw: dict[str, Any]) -> dict[str, Any]:
    """config.toml layout -> the flat settings dict the rest of AxonID reads."""
    config = {k: v for k, v in raw.items() if k not in _FLATTENED_TABLES}
    for table in _FLATTENED_TABLES:
        for key, value in raw.get(table, {}).items():
            if key in config:
                raise ValueError(f"config.toml sets '{key}' twice (once in [{table}]).")
            config[key] = value
    if isinstance(config.get("timing"), dict):   # [timing] enabled = ...
        config["timing"] = bool(config["timing"].get("enabled", False))
    return config


def load_config() -> dict[str, Any]:
    """
    Load config.toml (once) and return the settings as a flat dict.

    Raises:
        FileNotFoundError: config.toml does not exist.
        tomllib.TOMLDecodeError: config.toml is not valid TOML (the message
            gives the line and column).
        ValueError: config.toml sets a recipe-locked key (see LOCKED_KEYS).
    """
    global _config_cache

    if _config_cache is None:
        if not _CONFIG_PATH.exists():
            hint = (f" AxonID 1.0.0 reads config.toml; your {_OLD_CONFIG_PATH.name} is from an earlier "
                    f"version. Copy your pixel_size_um values into config.toml."
                    if _OLD_CONFIG_PATH.exists() else "")
            raise FileNotFoundError(f"config.toml not found at {_CONFIG_PATH}.{hint}")

        with open(_CONFIG_PATH, "rb") as f:
            try:
                raw = tomllib.load(f)
            except tomllib.TOMLDecodeError as e:
                raise tomllib.TOMLDecodeError(
                    f"config.toml could not be read ({e}). Undo your last edit or check quotes, "
                    f"brackets and '=' signs on that line."
                ) from e
        config = _flatten(raw)
        check_locked_keys(config)
        _config_cache = config

    return _config_cache


def get_pixel_size(mag: str, image_width: int) -> float | None:
    """
    Return the configured pixel size in µm/pixel.

    Args:
        mag:
            Image magnification, such as "40X" or "100X".
        image_width:
            Actual image width in pixels.

    Returns:
        Configured pixel size in µm/pixel, or None if no value is
        configured for the magnification and image width.
    """
    config = load_config()
    sizes = config.get("pixel_size_um", {}).get(mag)

    if sizes is None:
        return None

    return sizes.get(str(image_width))


def get_target_resolution() -> tuple[int, int]:
    """
    Return the working-resolution bound (width, height) that
    utils.resize.resize_img() downsamples larger images to fit within,
    preserving aspect ratio. Fixed by the recipe (utils/recipe.py): the
    released models were trained at this scale.
    """
    from utils.recipe import RECIPE
    width, height = RECIPE.target_resolution
    return (width, height)


def format_scan_config_summary() -> list[str]:
    """
    Config values that govern which folders are recognized as nerve/TIFF
    directories in the first place — read by utils.study's scan_study()/
    detect_input_level()/resolve_scan() before any magnification-specific
    processing begins. A tiff_suffixes mismatch against a study's actual
    folder naming makes affected nerves silently invisible to the whole
    pipeline (no tiff_dir found, no specific error — just an eventual
    "no nerve data"), so this is worth surfacing before scanning even
    starts, not folded into either per-stage banner.
    """
    config = load_config()
    suffixes = config.get("tiff_suffixes", ["_TIFF"])
    return [
        f"TIFF folder suffixes: {', '.join(suffixes)} "
        f"[config.toml \u2192 folders.tiff_suffixes]",
    ]


def format_segment_config_summary(mag: str) -> list[str]:
    """
    Settings that govern segmentation for this magnification. Patch size is
    fixed by the recipe; calibration and morphometric settings are reported
    by format_morphometrics_config_summary() instead.
    """
    from utils.recipe import RECIPE, RECIPE_ID
    return [
        f"Model recipe: {RECIPE_ID} | patch {RECIPE.patch_size}px, 50% overlap "
        f"[utils/recipe.py, not configurable]",
    ]


def format_morphometrics_config_summary(mag: str) -> list[tuple[str, str]]:
    """
    Settings that govern morphometrics for this magnification, logged once
    per run. Returns (level, text) pairs; level is "info" or "warn".

    A missing calibration is the most important line to see up front: it
    silently turns every diameter, CSA and g-ratio-filter measurement into
    pixel units.
    """
    from utils.recipe import MORPHOMETRICS, MORPHOMETRICS_ID

    config = load_config()

    px_sizes = {w: um for w, um in config.get("pixel_size_um", {}).get(mag, {}).items()
                if not str(w).startswith("_")}
    if px_sizes:
        px_line = ", ".join(f"{um}\u00b5m/px @ {w}px wide" for w, um in px_sizes.items())
        px_level = "info"
    else:
        px_line = f"NOT CONFIGURED for {mag}: measurements will be in pixel units"
        px_level = "warn"

    m = MORPHOMETRICS
    seeding = (f"h-minima={m.h_minima_px}px" if m.h_minima_px is not None
               else f"distance-threshold={m.distance_threshold}")
    filters = ("configurable filters ON" if m.quality_filters_enabled
               else f"configurable filters off; hard checks: axon diameter >= {m.min_axon_diam_um} \u00b5m, "
                    f"g-ratio in (0, 1)")

    return [
        (px_level, f"Pixel calibration ({mag}):  {px_line} "
                   f"(original image width; corrected for segmentation downsampling) "
                   f"[config.toml \u2192 pixel_size_um.{mag}]"),
        ("info", f"Primary g-ratio method:    {config.get('primary_gratio_method', 'equiv_diam')} "
                 f"[config.toml \u2192 reporting.primary_gratio_method]"),
        ("info", f"Diameter bins:             {_format_bin_edges(config)} "
                 f"[config.toml \u2192 morphometrics_bins]"),
        ("info", f"Morphometrics rules:       {MORPHOMETRICS_ID}: {seeding}, disk={m.dilation_disk}, "
                 f"min seed {m.min_seed_area_px}px; {filters} [utils/recipe.py]"),
    ]


def _format_bin_edges(config: dict) -> str:
    """
    Summarize the granular/mid/coarse bin edges distributions.py will
    use. Reads the same DEFAULT_*_BINS_UM constants distributions.py
    itself imports — single source of truth, no drift risk.
    """
    bins_cfg = config.get("morphometrics_bins", {})
    granular = bins_cfg.get("granular_um", DEFAULT_GRANULAR_BINS_UM)
    mid      = bins_cfg.get("mid_um",      DEFAULT_MID_BINS_UM)
    coarse   = bins_cfg.get("coarse_um",   DEFAULT_COARSE_BINS_UM)

    def _fmt(edges):
        parts = [
            "\u221e" if e in (float("inf"), 999) else str(e)
            for e in edges
        ]
        return f"{len(edges) - 1} bins [" + ", ".join(parts) + "]"

    return (
        f"granular={_fmt(granular)}; "
        f"mid={_fmt(mid)}; "
        f"coarse={_fmt(coarse)}"
    )