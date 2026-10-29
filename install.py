"""
install.py: install AxonID's dependencies and download the released models.

Usage (inside the activated axonid environment, from the AxonID folder):
    python install.py          CPU (default; works on every computer)
    python install.py --gpu    NVIDIA GPU (CUDA 12.1; Windows or Linux)

Steps
    1. check that Python 3.11 is running (PyTorch 2.5.1 has no build for
       every Python version, so a wrong version fails here, clearly)
    2. PyTorch 2.5.1, CPU or CUDA 12.1 build. Installed BEFORE the other
       packages: several of them depend on PyTorch, and pip would otherwise
       pull the newest PyTorch (several GB) first and replace it afterwards.
    3. packages from requirements.txt
    4. patchify, without its dependencies (it would otherwise change numpy)
    5. confirm PyTorch is still 2.5.1, print the installed versions and GPU
    6. download the released models into models/ (a failure here only warns;
       rerun python -m utils.fetch_models later)

Every pip call uses the Python running this script, so packages always go
into the active environment.

Depends on
    standard library only
"""

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TORCH_VERSION, TORCHVISION_VERSION = "2.5.1", "0.20.1"
PYTORCH_INDEX = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "gpu": "https://download.pytorch.org/whl/cu121",
}


def step(n: int, total: int, text: str) -> None:
    print(f"\n[{n}/{total}] {text}", flush=True)


def run(args: list[str]) -> None:
    """Run a command with this script's Python; stop the installation if it fails."""
    result = subprocess.run([sys.executable, *args], cwd=ROOT)
    if result.returncode != 0:
        print("\n  Installation stopped: see the error above.")
        sys.exit(1)


def torch_command(gpu: bool) -> list[str]:
    if gpu:
        return ["-m", "pip", "install", f"torch=={TORCH_VERSION}+cu121",
                f"torchvision=={TORCHVISION_VERSION}+cu121", "--index-url", PYTORCH_INDEX["gpu"]]
    if platform.system() == "Darwin":
        # macOS builds on the standard package index are already CPU builds.
        return ["-m", "pip", "install", f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}"]
    # Windows/Linux: the CPU index avoids downloading the much larger CUDA build.
    return ["-m", "pip", "install", f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}",
            "--index-url", PYTORCH_INDEX["cpu"]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Install AxonID 1.0.0.")
    parser.add_argument("--gpu", action="store_true",
                        help="install the NVIDIA GPU (CUDA 12.1) build of PyTorch instead of the CPU build")
    args = parser.parse_args()
    total = 6

    print("=" * 50 + "\n  AxonID 1.0.0 installation (" + ("NVIDIA GPU" if args.gpu else "CPU") + ")\n" + "=" * 50)

    step(1, total, "Checking Python...")
    if sys.version_info[:2] != (3, 11):
        print(f"  AxonID needs Python 3.11; this is Python {platform.python_version()} ({sys.executable}).\n"
              "  Create and activate the environment first:\n"
              "      conda create -n axonid python=3.11 -y\n"
              "      conda activate axonid")
        return 1
    if args.gpu and platform.system() == "Darwin":
        print("  Macs have no NVIDIA GPU support in PyTorch: run python install.py (CPU) instead.")
        return 1
    print(f"  Python {platform.python_version()} ({sys.executable})")
    if not args.gpu and shutil.which("nvidia-smi"):
        print("  Note: an NVIDIA GPU driver was found. For GPU speed (needed for training), "
              "run python install.py --gpu instead.")

    step(2, total, f"PyTorch {TORCH_VERSION} ({'CUDA 12.1' if args.gpu else 'CPU'})...")
    run(torch_command(args.gpu))

    step(3, total, "Other dependencies...")
    run(["-m", "pip", "install", "-r", "requirements.txt"])

    step(4, total, "patchify (without dependencies, so numpy is not changed)...")
    run(["-m", "pip", "install", "patchify==0.2.3", "--no-deps"])

    step(5, total, "Checking the installation...")
    run(["-c", f"import torch, sys; v = torch.__version__; print('  PyTorch', v); "
               f"sys.exit(0 if v.startswith('{TORCH_VERSION}') else 1)"])
    run(["utils/version.py"])

    step(6, total, "Downloading the released models into models/...")
    if subprocess.run([sys.executable, "-m", "utils.fetch_models"], cwd=ROOT).returncode != 0:
        print("  Model download failed: see the message above. You can rerun: python -m utils.fetch_models")

    print("\n" + "=" * 50 + "\n  Done. Next: set your calibration in config.toml (README, Calibration).\n" + "=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
