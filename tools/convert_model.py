"""
tools/convert_model.py: convert a trusted .pt model file to .safetensors.

Usage
    python -m tools.convert_model "path/to/model.pt" ["path/to/other.pt" ...]

Inputs
    .pt checkpoints written by earlier AxonID/DeepAxon versions: a dict with
    "model_state_dict" (weights) and "meta" (training record).

Outputs
    <same folder>/<same name>.safetensors with identical weights and the same
    training record. Existing files are never overwritten. AxonID 1.0.0 loads
    only .safetensors; put converted models in models/user/ to use them.

Safety
    Reading a .pt file can run code hidden in it, so convert ONLY files you
    made yourself or received from someone you trust. The .safetensors file
    written here cannot carry code.

Depends on
    torch, utils/model_io.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from utils.model_io import MODEL_SUFFIX, load_model_file, save_model_file  # noqa: E402


def convert(src: Path) -> Path:
    dst = src.with_suffix(MODEL_SUFFIX)
    if dst.exists():
        raise FileExistsError(f"{dst} already exists; move it first.")
    ckpt = torch.load(str(src), map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict, meta = ckpt["model_state_dict"], dict(ckpt.get("meta") or {})
    else:
        state_dict, meta = ckpt, {}
    meta.setdefault("converted_from", src.name)
    save_model_file(dst, state_dict, meta)
    back, _ = load_model_file(dst)
    if back.keys() != state_dict.keys() or not all(torch.equal(back[k], state_dict[k].cpu()) for k in back):
        dst.unlink()
        raise RuntimeError(f"Verification failed for {src.name}; nothing written.")
    return dst


def main() -> int:
    paths = [Path(a.strip('"')) for a in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 1
    print("Only convert .pt files you made or trust: reading a .pt file can run code.")
    failed = 0
    for p in paths:
        try:
            print(f"OK  {p.name} -> {convert(p).name} (weights identical)")
        except Exception as e:   # report each file and continue with the rest
            print(f"FAILED  {p.name}: {e}")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
