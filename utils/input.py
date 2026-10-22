"""
utils/input.py: validated command-line prompts (integer, number, yes/no, path).

Usage
    from utils.input import get_int_input, get_yes_no, get_path_input

Behaviour
    Each prompt repeats until the answer is valid; Enter returns the
    default when one is given.

Depends on
    standard library only
"""

# ── Standard library ──────────────────────────────────────────────────────────
from pathlib import Path


def get_int_input(
    prompt: str,
    default: int | None = None,
    min_val: int = 1,
) -> int:
    """
    Prompt for a validated integer.

    Args:
        prompt:
            Prompt displayed to the user.
        default:
            Value returned when the user presses Enter.
        min_val:
            Minimum accepted value.

    Returns:
        Valid integer input.
    """
    while True:
        raw = input(prompt).strip()

        if raw == "" and default is not None:
            return default

        try:
            val = int(raw)
            if val >= min_val:
                return val

            print(f"  Please enter a value ≥ {min_val}.")
        except ValueError:
            print("  Please enter a valid integer.")


def get_float_input(
    prompt: str,
    default: float | None = None,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> float:
    """
    Prompt for a validated floating-point number.

    Args:
        prompt:
            Prompt displayed to the user.
        default:
            Value returned when the user presses Enter.
        min_val:
            Minimum accepted value.
        max_val:
            Maximum accepted value.

    Returns:
        Valid floating-point input.
    """
    while True:
        raw = input(prompt).strip()

        if raw == "" and default is not None:
            return default

        try:
            val = float(raw)
            if min_val <= val <= max_val:
                return val

            print(
                f"  Please enter a value between "
                f"{min_val} and {max_val}."
            )
        except ValueError:
            print("  Please enter a valid number.")


def get_yes_no(prompt: str, default: bool = False) -> bool:
    """
    Prompt for a validated yes/no response.

    Args:
        prompt:
            Prompt displayed to the user.
        default:
            Value returned when the user presses Enter.

    Returns:
        True for yes and False for no.
    """
    hint = "[Y/n]" if default else "[y/N]"

    while True:
        raw = input(f"{prompt} {hint}: ").strip().lower()

        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False

        print("  Please enter y or n.")


def get_path_input(
    prompt: str,
    *,
    default: str | Path | None = None,
    must_exist: bool = True,
) -> Path:
    """
    Prompt for a filesystem path.

    Args:
        prompt:
            Prompt displayed to the user.
        default:
            Path returned when the user presses Enter. Subject to the
            same must_exist check as typed input — a default that
            doesn't exist is rejected the same way, not silently
            allowed through.
        must_exist:
            If True, reject paths that do not exist.

    Returns:
        Validated Path object.
    """
    while True:
        raw = input(prompt).strip()

        if raw == "" and default is not None:
            raw = str(default)
        elif not raw:
            print("  Please enter a path.")
            continue

        path = Path(raw)

        if must_exist and not path.exists():
            print(f"  Path not found: {path}")
            continue

        return path