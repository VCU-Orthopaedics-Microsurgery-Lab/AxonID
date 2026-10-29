"""
utils/console.py: Rich console helpers shared by the entry points.

Usage
    from utils.console import print_panel
    print_panel(console, Panel(...))

Depends on
    rich
"""

# ── Local ─────────────────────────────────────────────────────────────────────
from rich.console import Console, RenderableType


def print_panel(console: Console, panel: RenderableType) -> None:
    """
    Print a Rich panel with blank lines before and after it.

    Args:
        console:
            Rich Console instance used for output.
        panel:
            Rich renderable to display.
    """
    console.print()
    console.print(panel)
    console.print()