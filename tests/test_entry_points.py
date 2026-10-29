"""Every command-line entry point imports. Catches a broken import that unit tests
of individual functions miss (the run would only fail when a user starts it)."""
import importlib

import pytest


@pytest.mark.parametrize("module", [
    "segment.__main__", "morphometrics.__main__", "batch_axon.__main__", "train.__main__",
])
def test_entry_point_imports(module):
    importlib.import_module(module)
