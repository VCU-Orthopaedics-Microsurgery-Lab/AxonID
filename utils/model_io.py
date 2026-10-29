"""
utils/model_io.py: save and load AxonID model files (.safetensors).

Purpose
    One place that writes and reads model files, in the safetensors format:
    the file holds only tensors (the network weights) and a text metadata
    block, so opening it can never run code, whatever the PyTorch version.

Usage
    from utils.model_io import save_model_file, load_model_file
    save_model_file(path, model.state_dict(), meta)
    state_dict, meta = load_model_file(path)

Inputs / Outputs
    state_dict  the network weights (name -> tensor)
    meta        the training record (recipe, split, best epoch, environment);
                stored as JSON text under the metadata key "axonid_meta"

Depends on
    safetensors, torch
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import json
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
from safetensors import safe_open
from safetensors.torch import load_file, save_file

MODEL_SUFFIX = ".safetensors"
META_KEY = "axonid_meta"


def save_model_file(path: str | Path, state_dict: dict, meta: dict) -> None:
    """Write weights and training record to a .safetensors file."""
    # safetensors stores plain, contiguous CPU tensors.
    tensors = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    # default=str: an unexpected value type in the record is written as text
    # rather than failing a save at the end of a long training run.
    save_file(tensors, str(path), metadata={META_KEY: json.dumps(meta, default=str)})


def load_model_file(path: str | Path) -> tuple[dict, dict]:
    """Read (state_dict, meta) from a .safetensors file. meta is {} if absent."""
    path = Path(path)
    if path.suffix.lower() != MODEL_SUFFIX:
        raise ValueError(
            f"{path.name}: AxonID loads {MODEL_SUFFIX} model files only. "
            f"Convert a .pt file you trust with: python -m tools.convert_model \"{path}\""
        )
    state_dict = load_file(str(path), device="cpu")
    with safe_open(str(path), framework="pt") as f:
        stored = (f.metadata() or {}).get(META_KEY)
    return state_dict, (json.loads(stored) if stored else {})
