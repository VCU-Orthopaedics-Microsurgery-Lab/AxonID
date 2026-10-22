"""
utils/model_registry.py: which models the segmentation menu offers.

Purpose
    Combine the released models listed in models/models.json with any models
    the user trained (models/user/*.safetensors) into one menu, recommended model first.

Usage
    from utils.model_registry import available_models, verify_file

Sources
    released   models/models.json entries whose model file is present in models/.
               Read-only; updated only by AxonID releases.
    user       every .safetensors file in models/user/ (written there by python -m train).
               Not listed in models.json, so local training never edits a
               tracked file.

Notes
    Listing does not load checkpoints. Compatibility with the locked recipe is
    checked when a model is loaded (utils/recipe.check_checkpoint_compatible).

Depends on
    utils/filesystem.py   repository models/ directory, natural sort
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

# ── Local ─────────────────────────────────────────────────────────────────────
from utils.filesystem import get_model_dir, natural_sort_key
from utils.model_io import MODEL_SUFFIX


@dataclass
class ModelEntry:
    path: Path
    label: str
    source: str            # "released" | "user"
    recommended: bool = False
    md5: str | None = None
    sha256: str | None = None
    notes: str = ""


def get_user_model_dir() -> Path:
    """models/user/: where python -m train saves new models."""
    return get_model_dir() / "user"


def load_registry() -> dict:
    """Return models/models.json, or an empty registry if it is missing."""
    path = get_model_dir() / "models.json"
    if not path.exists():
        return {"models": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def available_models() -> list[ModelEntry]:
    """
    Released models that are present on disk, then user models.
    The recommended released model is always first.
    """
    model_dir = get_model_dir()
    released = []
    for m in load_registry().get("models", []):
        p = model_dir / m["file"]
        if p.exists():
            released.append(ModelEntry(
                path=p, label=m.get("label", p.stem), source="released",
                recommended=bool(m.get("recommended")), md5=m.get("md5"),
                sha256=m.get("sha256"), notes=m.get("notes", ""),
            ))
    released.sort(key=lambda e: not e.recommended)

    user_dir = get_user_model_dir()
    user = []
    if user_dir.is_dir():
        for p in sorted(user_dir.glob(f"*{MODEL_SUFFIX}"), key=natural_sort_key):
            user.append(ModelEntry(path=p, label=p.stem, source="user"))
    return released + user


def missing_released_models() -> list[str]:
    """File names listed in models.json that are not in models/ yet."""
    model_dir = get_model_dir()
    return [m["file"] for m in load_registry().get("models", [])
            if not (model_dir / m["file"]).exists()]


def _digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def verify_file(path: Path, expected) -> bool | None:
    """
    Check a model file against its expected checksums.

    expected: a models.json entry (dict) or a ModelEntry, with "sha256"
    and/or "md5". True if every given checksum matches, False if any
    differs, None if there is nothing to check (user models).
    SHA-256 detects deliberate tampering; MD5 alone only detects corruption.
    """
    get = expected.get if isinstance(expected, dict) else (lambda k: getattr(expected, k, None))
    checks = [(alg, get(alg)) for alg in ("sha256", "md5") if get(alg)]
    if not checks:
        return None
    return all(_digest(path, alg) == value.lower() for alg, value in checks)
