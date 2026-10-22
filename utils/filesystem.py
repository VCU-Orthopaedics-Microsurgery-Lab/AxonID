"""
utils/filesystem.py: file discovery, natural sorting and standard repository folders.

Usage
    from utils.filesystem import list_files, natural_sort_key, get_model_dir, get_log_dir

Outputs
    pathlib.Path objects. get_model_dir() is <repo>/models, get_log_dir()
    is <repo>/logs (training logs).

Depends on
    standard library only
"""

# ── Standard library ──────────────────────────────────────────────────────────
from pathlib import Path
import re


def natural_sort_key(path: str | Path) -> list[int | str]:
    """
    Return a natural-sort key for a filename containing numbers.

    For example, img_1, img_2, and img_10 sort in numerical order rather
    than lexicographic order.

    Args:
        path:
            Filename or filesystem path.

    Returns:
        List of string and integer components suitable for sorted().
    """
    parts = re.split(r"(\d+)", Path(path).stem)
    return [int(p) if p.isdigit() else p for p in parts]


def list_files(
    directory: str | Path,
    extensions: tuple[str, ...] = (".tif", ".tiff", ".png"),
) -> list[Path]:
    """
    Return image files in a directory using natural filename sorting.

    Args:
        directory:
            Directory to search.
        extensions:
            File extensions to include.

    Returns:
        Naturally sorted list of matching file paths.
    """
    directory = Path(directory)

    return sorted(
        (
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ),
        key=natural_sort_key,
    )


def count_patches(directory: str | Path) -> int:
    """
    Count patch image files in a directory.

    Args:
        directory:
            Directory containing patch files.

    Returns:
        Number of matching patch image files.
    """
    return len(
        list_files(
            directory,
            extensions=(".png", ".tif", ".tiff", ".bmp"),
        )
    )


def get_training_dir(images_dir: str | Path) -> Path:
    """
    Return the training root directory containing images_dir.

    Args:
        images_dir:
            Training images directory.

    Returns:
        Parent directory of images_dir.
    """
    return Path(images_dir).resolve().parent


def get_model_dir(images_dir: str | Path | None = None) -> Path:
    """
    Return the repository models directory.

    Note:
        images_dir is currently unused and retained for API compatibility.
    """
    return Path(__file__).resolve().parent.parent / "models"


def get_log_dir(images_dir: str | Path | None = None) -> Path:
    """
    Return the repository logs directory.

    Note:
        images_dir is currently unused and retained for API compatibility.
    """
    return Path(__file__).resolve().parent.parent / "logs"