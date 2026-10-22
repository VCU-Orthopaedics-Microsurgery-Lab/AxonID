"""
utils/gpu.py: report whether a CUDA GPU is available.

Usage
    from utils.gpu import setup_gpu_console
    use_gpu = setup_gpu_console("Training")

Outputs
    Prints the detected device(s) and returns True when CUDA is usable,
    False otherwise (including when CUDA is reported but cannot be queried).
    Does not select a device or move tensors.

Depends on
    torch, rich, utils/console.py
"""

# ── Third-party ───────────────────────────────────────────────────────────────
from rich.box import DOUBLE
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import torch

# ── Local ─────────────────────────────────────────────────────────────────────
from utils.console import print_panel

console = Console()


# NOTE: "setup_gpu_console" is the intended, permanent public API name —
# this is the call site the pipeline treats as "doing GPU setup," even
# though today it only detects CUDA and reports it; it does not select
# a device or move anything to it. Not a naming mismatch to fix.
def setup_gpu_console(activity: str = "Training and Segmentation") -> bool:
    """
    Report PyTorch CUDA availability and return whether GPU acceleration is active.

    Args:
        activity: what to name in the CPU-fallback message (e.g.
            "Segmentation" when called from segment/__main__.py). Default
            preserves the original generic wording for any other caller
            (e.g. train.py) that doesn't pass one.

    Returns:
        True if CUDA is available, False if CPU only — including if CUDA
        reports available but device info can't actually be queried, which
        is treated as unusable rather than crashing the caller.
    """

    def _cpu_fallback_panel(message: str = "No GPU detected — running on CPU only.\n") -> None:
        t = Text(justify="center")
        t.append(message, style="yellow")
        t.append(f"{activity} will be slower without GPU acceleration.")
        print_panel(console, Panel(
            t,
            title="[bold yellow]Device[/bold yellow]",
            border_style="yellow",
            box=DOUBLE,
            expand=True
        ))

    if not torch.cuda.is_available():
        _cpu_fallback_panel()
        return False

    try:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("CUDA reports available but device_count() is 0")
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        cuda_version = torch.version.cuda
    except Exception:
        # CUDA reported available, but querying device info failed —
        # HPC driver/allocation state can be flaky; fail safe to CPU
        # reporting rather than crashing the caller.
        _cpu_fallback_panel("GPU detected but device info unavailable — running on CPU only.\n")
        return False

    t = Text(justify="center")
    t.append("GPU acceleration ENABLED\n", style="green")
    t.append(f"Device(s): {', '.join(gpu_names)}\n")
    t.append(f"CUDA version: {cuda_version}")
    print_panel(console, Panel(
        t,
        title="[bold green]Device[/bold green]",
        border_style="green",
        box=DOUBLE,
        expand=True
    ))
    return True