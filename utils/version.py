"""
utils/version.py: AxonID version and environment fingerprint.

Usage
    from utils.version import __version__
    python utils/version.py        prints the version and installed packages

Outputs
    __version__ (the only place the version is set) and get_env_info():
    Python, platform, PyTorch/CUDA, GPU and key package versions, written
    into log headers.

Depends on
    standard library; optional imports of the installed packages
"""

# ── Standard library ──────────────────────────────────────────────────────────
import platform
import socket
import sys

__version__     = "1.0.0"
__version_str__ = f"AxonID v{__version__}"


def get_env_info() -> dict:
    """
    Collect full environment fingerprint.
    Used by logger header and version CLI.
    Records the major runtime and package versions needed to document the
    execution environment; it is not intended to guarantee complete environment
    reconstruction from the log alone.
    """
    info = {
        "python":   sys.version.split()[0],
        "platform": platform.system(),
        "hostname": socket.gethostname(),
    }

    try:
        import torch
        info["pytorch"]        = torch.__version__
        info["cuda"]           = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            # gpu_name lists every visible device; gpu_memory_gb is device 0 only.
            num_gpus = torch.cuda.device_count()
            info["gpu_count"] = num_gpus
            info["gpu_name"]  = ", ".join(
                torch.cuda.get_device_name(i) for i in range(num_gpus)
            )
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except ImportError:
        info["pytorch"] = None

    try:
        import numpy as np
        info["numpy"] = np.__version__
    except ImportError:
        info["numpy"] = None

    try:
        import cv2
        info["opencv"] = cv2.__version__
    except ImportError:
        info["opencv"] = None

    try:
        import segmentation_models_pytorch as smp
        info["smp"] = smp.__version__
    except ImportError:
        info["smp"] = None

    try:
        import monai
        info["monai"] = monai.__version__
    except ImportError:
        info["monai"] = None

    try:
        import scipy
        info["scipy"] = scipy.__version__
    except ImportError:
        info["scipy"] = None

    try:
        import skimage
        info["scikit_image"] = skimage.__version__
    except ImportError:
        info["scikit_image"] = None

    try:
        import sklearn
        info["scikit_learn"] = sklearn.__version__
    except ImportError:
        info["scikit_learn"] = None

    try:
        import timm
        info["timm"] = timm.__version__
    except ImportError:
        info["timm"] = None

    try:
        import PIL
        info["pillow"] = PIL.__version__
    except ImportError:
        info["pillow"] = None

    try:
        import pandas as pd
        info["pandas"] = pd.__version__
    except ImportError:
        info["pandas"] = None

    try:
        import patchify
        info["patchify"] = getattr(patchify, '__version__', 'unknown')
    except ImportError:
        info["patchify"] = None

    return info


def main() -> None:
    print(f"\n{__version_str__}\n")
    for k, v in get_env_info().items():
        status = str(v) if v is not None else "NOT FOUND"
        print(f"  {k:<30} {status}")
    print()


if __name__ == "__main__":
    main()