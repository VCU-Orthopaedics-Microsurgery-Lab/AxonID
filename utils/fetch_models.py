"""
utils/fetch_models.py: download the released AxonID models into models/.

Usage
    python -m utils.fetch_models

Inputs
    models/models.json: the file name, SHA-256 and MD5 of each released
    model, and download_base_url (the GitHub release that hosts them).

Outputs
    models/<file>.safetensors for every released model that is missing or
    does not match its checksums. Each file is downloaded to <file>.part,
    checked (SHA-256 guards against tampering, MD5 against corruption), and
    only then renamed, so a bad download never replaces a good model.

Notes
    The model files are not stored in git (each is about 150 MB, over
    GitHub's file-size limit). Without internet access, download them from
    the release page in a browser and place them in models/; segmentation
    checks them the same way.

Depends on
    standard library; utils/model_registry.py
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import urllib.request
from pathlib import Path

# sys.path must include the repository root before the utils.* import below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Local ─────────────────────────────────────────────────────────────────────
from utils.filesystem import get_model_dir
from utils.model_registry import load_registry, verify_file


def _download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()


def main() -> int:
    registry = load_registry()
    base = registry.get("download_base_url", "").rstrip("/")
    model_dir = get_model_dir()
    failed = 0
    for m in registry.get("models", []):
        dest = model_dir / m["file"]
        if dest.exists() and verify_file(dest, m) is not False:
            print(f"OK       {m['file']} (already present, checksums match)")
            continue
        url = f"{base}/{m['file']}"
        part = dest.with_name(dest.name + ".part")
        print(f"Download {m['file']}\n  from {url}")
        try:
            _download(url, part)
        except Exception as e:   # network errors vary by platform; report and continue
            print(f"  FAILED: {e}\n  Download it from the release page in a browser and put it in {model_dir}")
            part.unlink(missing_ok=True)
            failed += 1
            continue
        if verify_file(part, m) is False:
            print("  FAILED: checksum does not match models.json; the download is incomplete or the wrong file.")
            part.unlink(missing_ok=True)
            failed += 1
            continue
        part.replace(dest)
        print(f"  saved {dest}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
