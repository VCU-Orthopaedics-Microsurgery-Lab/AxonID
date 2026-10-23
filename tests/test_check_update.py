"""Update check: version comparison and messages, without contacting GitHub."""
import urllib.error

from utils.check_update import RELEASES_PAGE, check, parse_version

PAGE = "https://github.com/VCU-Orthopaedics-Microsurgery-Lab/AxonID/releases/tag/v1.2.0"


def release(tag, url=PAGE):
    return lambda: (tag, url)


def test_parse_version():
    assert parse_version("v1.2.3") == (1, 2, 3)
    assert parse_version("1.10.0") == (1, 10, 0)
    assert parse_version("v2.0.0-rc1") == (2, 0, 0)
    assert parse_version("latest") is None


def test_newer_release_available():
    code, msg = check("1.1.0", release("v1.2.0"))
    assert code == 1 and "v1.2.0 is available" in msg and "v1.1.0" in msg and PAGE in msg


def test_up_to_date_and_ahead():
    assert check("1.2.0", release("v1.2.0"))[0] == 0
    assert "up to date" in check("1.2.0", release("v1.2.0"))[1]
    code, msg = check("1.3.0", release("v1.2.0"))
    assert code == 0 and "newer than the latest release" in msg


def test_numeric_not_text_comparison():
    assert check("1.9.0", release("v1.10.0"))[0] == 1          # "1.10" > "1.9"


def test_offline_never_raises():
    def offline():
        raise urllib.error.URLError("no internet")
    code, msg = check("1.1.0", offline)
    assert code == 2 and "Could not check" in msg and RELEASES_PAGE in msg


def test_unreadable_tag():
    code, msg = check("1.1.0", release("nightly"))
    assert code == 2 and "Could not compare" in msg
