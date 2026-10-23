"""
utils/check_update.py: is this AxonID the latest release?

Usage
    python -m utils.check_update

Outputs
    One line, for example:
        AxonID v1.1.0: up to date.
        AxonID v1.2.0 is available (you have v1.1.0): <release page>
    Exit code 0 = up to date (or newer than the latest release),
              1 = a newer release exists,
              2 = could not check (no internet, proxy, GitHub unavailable).

Behaviour
    Asks GitHub for the latest published release of the AxonID repository and
    compares its tag (vX.Y.Z) with utils/version.py. Runs only when called:
    nothing in segmentation, morphometrics or training contacts the internet
    to check for updates. A failed check never raises; it says so and points
    to the release page.

Depends on
    standard library; utils/version.py
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# The repository root must be on sys.path for "utils" to import when this file
# is run directly (python utils/check_update.py) instead of with python -m.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Local ─────────────────────────────────────────────────────────────────────
from utils.version import __version__

REPO = "VCU-Orthopaedics-Microsurgery-Lab/AxonID"
RELEASES_PAGE = f"https://github.com/{REPO}/releases"
LATEST_API = f"https://api.github.com/repos/{REPO}/releases/latest"
TIMEOUT_S = 10


def parse_version(text: str) -> tuple[int, ...] | None:
    """'v1.2.3' or '1.2.3' -> (1, 2, 3); anything after X.Y.Z is ignored. None if unreadable."""
    m = re.match(r"^\s*v?(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(x) for x in m.groups()) if m else None


def latest_release(timeout: float = TIMEOUT_S) -> tuple[str, str]:
    """(tag, release page URL) of the latest published release. Raises on any failure."""
    req = urllib.request.Request(
        LATEST_API,
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": f"AxonID/{__version__} update check"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    return data["tag_name"], data.get("html_url") or RELEASES_PAGE


def check(current: str = __version__, fetch=latest_release) -> tuple[int, str]:
    """(exit code, message). fetch is replaceable for testing."""
    try:
        tag, url = fetch()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as e:
        reason = getattr(e, "reason", None) or e
        return 2, (f"Could not check for updates ({reason}). "
                   f"You have AxonID v{current}; see {RELEASES_PAGE}")

    latest, mine = parse_version(tag), parse_version(current)
    if latest is None or mine is None:
        return 2, (f"Could not compare versions (latest release tag: {tag!r}, "
                   f"this copy: v{current}); see {RELEASES_PAGE}")
    if latest > mine:
        return 1, f"AxonID {tag} is available (you have v{current}): {url}"
    if latest == mine:
        return 0, f"AxonID v{current}: up to date."
    return 0, f"AxonID v{current} is newer than the latest release ({tag})."


def main() -> int:
    code, message = check()
    print(message)
    return code


if __name__ == "__main__":
    sys.exit(main())
