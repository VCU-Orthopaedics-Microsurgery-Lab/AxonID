"""
train/__main__.py: entry point for training your own AxonID model.

Purpose
    Collects the few run settings a user may choose (data folder, model
    name, augmentation on/off, and optionally epoch limit, batch size and
    seed) and runs train/train.py. Everything else is the locked recipe.

Usage
    python -m train                              interactive prompts
    python -m train --config train_config.toml   no prompts (cluster / batch jobs)

Inputs
    A training folder containing images/ and masks/ (same file names).
    train_config.toml keys (only images_dir is required):
        images_dir     training folder
        model_name     default: axonid_user_<date>_<time>
        augmentation   true / false (default false)
        epochs         epoch limit (default: recipe, 400)
        batch_size     default: recipe, 128
        seed           default: recipe, 1
        magnification  default: 40X (the only supported value in 1.0.0)

Outputs
    models/user/{model_name}.safetensors (listed automatically in python -m segment)
    logs/{model_name}_training_log.txt

Locked (utils/recipe.py)
    Everything not listed above. Unknown keys in train_config.toml are
    refused rather than ignored.

Depends on
    train/train.py, utils/recipe.py, utils/logger.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
from datetime import datetime
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
import sys
import tomllib

# ── Third-party ───────────────────────────────────────────────────────────────
from rich.align import Align
from rich.box import DOUBLE
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import torch

# sys.path must be set before the utils.* / train.* imports below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Local ─────────────────────────────────────────────────────────────────────
from train.train import train_model
from utils.console import print_panel
from utils.filesystem import get_log_dir
from utils.gpu import setup_gpu_console
from utils.input import get_int_input, get_yes_no
from utils.logger import AxonIDLogger
from utils.model_io import MODEL_SUFFIX
from utils.model_registry import get_user_model_dir
from utils.recipe import AUGMENTATION, RECIPE, SUPPORTED_MAGS

console = Console()

CONFIG_KEYS_REQUIRED = ("images_dir",)
CONFIG_KEYS_OPTIONAL = ("model_name", "augmentation", "epochs", "batch_size", "seed", "magnification")


def _default_model_name() -> str:
    return f"axonid_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _check_model_name(name: str) -> str:
    name = name.strip()
    if not name or any(c in name for c in '\\/:*?"<>|'):
        raise ValueError(f"Invalid model name '{name}': use letters, numbers, '-' and '_' only.")
    if (get_user_model_dir() / f"{name}{MODEL_SUFFIX}").exists():
        raise ValueError(f"models/user/{name}{MODEL_SUFFIX} already exists. Choose another name or move the old file.")
    return name


# ── Non-interactive config ────────────────────────────────────────────────────
def _load_run_config(config_path: str) -> dict:
    """Read and validate train_config.toml; fill defaults from the recipe."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "rb") as f:
        try:
            cfg = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(
                f"{path.name} could not be read ({e}). Windows paths need single quotes, "
                f"e.g. images_dir = 'C:\\Data\\MyTraining'."
            ) from e

    unknown = sorted(set(cfg) - set(CONFIG_KEYS_REQUIRED) - set(CONFIG_KEYS_OPTIONAL))
    if unknown:
        raise ValueError(
            f"train_config.toml has unsupported key(s) {unknown}. Allowed: "
            f"{list(CONFIG_KEYS_REQUIRED + CONFIG_KEYS_OPTIONAL)}. Model and training settings "
            f"are fixed by the AxonID recipe (utils/recipe.py)."
        )
    missing = [k for k in CONFIG_KEYS_REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"train_config.toml is missing required key(s): {missing}")

    run = {
        "images_dir":    str(cfg["images_dir"]),
        "model_name":    str(cfg.get("model_name") or _default_model_name()),
        "augmentation":  cfg.get("augmentation", False),
        "epochs":        cfg.get("epochs", RECIPE.max_epochs),
        "batch_size":    cfg.get("batch_size", RECIPE.batch_size),
        "seed":          cfg.get("seed", RECIPE.default_seed),
        "magnification": cfg.get("magnification", SUPPORTED_MAGS[0]),
    }
    if not isinstance(run["augmentation"], bool):
        raise ValueError(f"'augmentation' must be true or false, got: {run['augmentation']!r}")
    for k in ("epochs", "batch_size"):
        if not isinstance(run[k], int) or isinstance(run[k], bool) or run[k] < 1:
            raise ValueError(f"'{k}' must be a positive integer, got: {run[k]!r}")
    if not isinstance(run["seed"], int) or isinstance(run["seed"], bool) or run["seed"] < 0:
        raise ValueError(f"'seed' must be a non-negative integer, got: {run['seed']!r}")
    if run["magnification"] not in SUPPORTED_MAGS:
        raise ValueError(f"'magnification' must be one of {list(SUPPORTED_MAGS)}, got: {run['magnification']!r}")
    if not os.path.isdir(run["images_dir"]):
        raise FileNotFoundError(f"images_dir not found: {run['images_dir']}")
    run["model_name"] = _check_model_name(run["model_name"])
    return run


# ── Interactive prompts ───────────────────────────────────────────────────────
def _prompt_run() -> dict:
    while True:
        images_dir = input("Path to the training folder (contains images/ and masks/): ").strip().strip('"')
        if os.path.isdir(images_dir):
            break
        console.print(f"[red]Folder not found: {images_dir}. Please try again.[/red]")

    console.print("[yellow]Validation set: one third of the images, chosen per phenotype when image names "
                  "start with ctrl_ / regen_ (otherwise at random, by image).[/yellow]")

    if len(SUPPORTED_MAGS) == 1:
        mag = SUPPORTED_MAGS[0]
        console.print(f"\nMagnification: {mag}")
    else:
        options = "  ".join(f"[{i}] {m}" for i, m in enumerate(SUPPORTED_MAGS, 1))
        raw = input(f"\nSelect imaging magnification: {options}: ").strip()
        while raw not in {str(i) for i in range(1, len(SUPPORTED_MAGS) + 1)}:
            raw = input(f"Invalid. {options}: ").strip()
        mag = SUPPORTED_MAGS[int(raw) - 1]

    auto_name = _default_model_name()
    console.print(f"\n[dim]Auto-generated name: {auto_name}[/dim]")
    while True:
        try:
            model_name = _check_model_name(input("Model name (Enter for auto-generated): ").strip() or auto_name)
            break
        except ValueError as e:
            console.print(f"[red]{e}[/red]")

    epochs = get_int_input(
        f"Epoch limit (Enter for recipe default {RECIPE.max_epochs}; early stopping usually ends sooner): ",
        default=RECIPE.max_epochs, min_val=1,
    )

    a = AUGMENTATION
    t = Text(justify="center")
    t.append(f"{int(a.p_clean * 100)}% of patches left unchanged; the rest receive, independently:\n", style="orange1")
    t.append(f"h-flip {a.hflip_prob:.0%} | v-flip {a.vflip_prob:.0%} | rotation ±{a.rotation_deg:g}° {a.rotation_prob:.0%} | "
             f"elastic {a.elastic_prob:.0%}\n")
    t.append(f"gamma {a.gamma_range[0]}-{a.gamma_range[1]} {a.gamma_prob:.0%} | noise {a.noise_prob:.0%} | "
             f"blur {a.blur_prob:.0%}")
    print_panel(console, Panel(t, title="[bold orange1]AxonID augmentation recipe[/bold orange1]",
                               border_style="orange1", box=DOUBLE, expand=True))
    use_aug = get_yes_no("Use data augmentation?", default=False)

    console.print(
        f"\nBatch size: {RECIPE.batch_size} (recipe). The released models were trained at "
        f"{RECIPE.batch_size} on an 80 GB GPU; a GPU with less memory may need a smaller batch."
    )
    batch_size = RECIPE.batch_size
    if get_yes_no("Use a different batch size?", default=False):
        batch_size = get_int_input("Batch size: ", default=None, min_val=1)
        if batch_size != RECIPE.batch_size:
            console.print(f"[yellow]Batch size {batch_size} differs from the recipe ({RECIPE.batch_size}); "
                          f"results may differ and the checkpoint will record the deviation.[/yellow]")

    return {
        "images_dir": images_dir, "model_name": model_name, "augmentation": use_aug,
        "epochs": epochs, "batch_size": batch_size, "seed": RECIPE.default_seed, "magnification": mag,
    }


def main():
    parser = argparse.ArgumentParser(description="Train an AxonID model with the locked recipe.")
    parser.add_argument("--config", metavar="FILE", help="train_config.toml for non-interactive runs")
    args = parser.parse_args()

    print_panel(console, Panel(
        Align.center("[bold white]UNet++ Nerve Segmentation Model Training[/bold white]"),
        title="[bold cyan]AxonID — TRAIN[/bold cyan]",
        border_style="bright_cyan", box=DOUBLE, expand=True, padding=(1, 4),
    ))

    if args.config:
        console.print(f"[dim]Non-interactive mode, config: {args.config}[/dim]\n")
        run = _load_run_config(args.config)
    else:
        setup_gpu_console()
        run = _prompt_run()

    log_dir = get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(log_dir / f"{run['model_name']}_training_log.txt")
    log = AxonIDLogger(log_path=log_path, program="AxonID Train")

    log.rule("RUN CONFIGURATION")
    log.log_dict({
        "Mode":           f"non-interactive ({args.config})" if args.config else "interactive",
        "Model name":     run["model_name"],
        "Magnification":  run["magnification"],
        "Epoch limit":    run["epochs"],
        "Augmentation":   "ON" if run["augmentation"] else "OFF",
        "Batch size":     run["batch_size"],
        "Seed":           run["seed"],
        "Training folder": str(Path(run["images_dir"]).resolve()),
        "Model file":     str(get_user_model_dir() / f"{run['model_name']}{MODEL_SUFFIX}"),
        "Log file":       log_path,
    })

    try:
        train_model(
            images_dir = str(Path(run["images_dir"]).resolve()),
            model_name = run["model_name"],
            epochs     = run["epochs"],
            batch_size = run["batch_size"],
            use_aug    = run["augmentation"],
            log        = log,
            mag        = run["magnification"],
            seed       = run["seed"],
        )
    except torch.cuda.OutOfMemoryError:
        log.error(f"The GPU ran out of memory at batch size {run['batch_size']}. "
                  f"Run again with a smaller batch size (for example {max(1, run['batch_size'] // 2)}).")
        sys.exit(1)

    console.print(f"\n[dim]Log saved to: {log_path}[/dim]")


if __name__ == "__main__":
    main()
