"""tools/archive_outputs: outputs move aside by stage, inputs are never touched, restore is safe."""
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

from tools import archive_outputs as ao

CONFIG = """
[folders]
tiff_suffixes = ["_TIFF"]
"""


def make_study(root: Path) -> Path:
    study = root / "Study"
    for nerve in ("A1/N1", "A1/N2", "A2/N1"):
        n = study / nerve
        for d in ("40X_TIFF", "4X_TIFF", "CSA", "Segmented", "Morphometrics", "QC"):
            (n / d).mkdir(parents=True)
            (n / d / "f.txt").write_text(d)
    (study / "A1" / "A1_Data.xlsx").write_text("x")
    (study / "A2" / "A2_Data.xlsx").write_text("x")
    (study / "Study_Data.xlsx").write_text("x")
    return study


def names(items, study):
    return sorted(p.relative_to(study).as_posix() for p in items)


def test_stage_selects_that_stage_and_later(tmp_path):
    study = make_study(tmp_path)
    nerves = [study / "A1" / "N1"]
    cfg = {}
    seg = names(ao.find_outputs(nerves, study, "segment", cfg), study)
    assert seg == ["A1/A1_Data.xlsx", "A1/N1/Morphometrics", "A1/N1/QC", "A1/N1/Segmented", "Study_Data.xlsx"]
    morph = names(ao.find_outputs(nerves, study, "morphometrics", cfg), study)
    assert morph == ["A1/A1_Data.xlsx", "A1/N1/Morphometrics", "A1/N1/QC", "Study_Data.xlsx"]
    comp = names(ao.find_outputs(nerves, study, "compile", cfg), study)
    assert comp == ["A1/A1_Data.xlsx", "Study_Data.xlsx"]          # only the workbooks above this scope


def test_inputs_are_never_listed(tmp_path):
    study = make_study(tmp_path)
    nerves = [study / "A1" / "N1", study / "A1" / "N2", study / "A2" / "N1"]
    for stage in ao.STAGES:
        listed = " ".join(names(ao.find_outputs(nerves, study, stage, {}), study))
        assert "TIFF" not in listed and "CSA" not in listed


def test_model_suffixed_folders_are_found(tmp_path):
    study = make_study(tmp_path)
    (study / "A1" / "N1" / "Segmented_M2").mkdir()
    (study / "A1" / "N1" / "Morphometrics_M2").mkdir()
    listed = names(ao.find_outputs([study / "A1" / "N1"], study, "segment", {}), study)
    assert "A1/N1/Segmented_M2" in listed and "A1/N1/Morphometrics_M2" in listed


def test_archive_moves_and_restore_round_trip(tmp_path):
    study = make_study(tmp_path)
    items = ao.find_outputs([study / "A1" / "N1"], study, "morphometrics", {})
    run = ao.archive(items, study, "morphometrics", now=datetime(2026, 10, 23, 14, 30, 0))
    assert run == tmp_path / "Study_archive" / "20261023_143000"
    assert not (study / "A1" / "N1" / "Morphometrics").exists()
    assert (run / "A1" / "N1" / "Morphometrics" / "f.txt").read_text() == "Morphometrics"
    assert (study / "A1" / "N1" / "CSA" / "f.txt").exists()            # inputs untouched
    assert (study / "A1" / "N2" / "Morphometrics").exists()            # other nerves untouched
    assert not (tmp_path / "Study" / "Study_archive").exists()         # archive is outside the study
    assert json.loads((run / ao.MANIFEST).read_text())["from"] == "morphometrics"

    restored, skipped = ao.restore(run)
    assert len(restored) == len(items) and skipped == []
    assert (study / "A1" / "N1" / "Morphometrics" / "f.txt").read_text() == "Morphometrics"
    assert (study / "Study_Data.xlsx").exists()


def test_restore_never_overwrites(tmp_path):
    study = make_study(tmp_path)
    items = ao.find_outputs([study / "A1" / "N1"], study, "compile", {})
    run = ao.archive(items, study, "compile", now=datetime(2026, 10, 23, 14, 30, 0))
    (study / "Study_Data.xlsx").write_text("newer")                    # rebuilt since
    restored, skipped = ao.restore(run)
    assert skipped == ["Study_Data.xlsx"]
    assert (study / "Study_Data.xlsx").read_text() == "newer"
    assert (run / "Study_Data.xlsx").exists()                          # still safe in the archive


def test_same_second_run_refuses(tmp_path):
    study = make_study(tmp_path)
    t = datetime(2026, 10, 23, 14, 30, 0)
    ao.archive(ao.find_outputs([study / "A1" / "N1"], study, "compile", {}), study, "compile", now=t)
    with pytest.raises(FileExistsError):
        ao.archive([study / "A1" / "N2"], study, "compile", now=t)


def test_cli_dry_run_then_yes(tmp_path, temp_config, monkeypatch, capsys):
    temp_config(CONFIG)
    study = make_study(tmp_path)
    nerve = study / "A1" / "N1"
    monkeypatch.setattr(sys, "argv", ["archive_outputs", "--input", str(nerve), "--from", "morphometrics", "--dry-run"])
    ao.main()
    assert (nerve / "Morphometrics").exists()                          # dry run moves nothing
    monkeypatch.setattr(sys, "argv", ["archive_outputs", "--input", str(nerve), "--from", "morphometrics", "--yes"])
    ao.main()
    assert not (nerve / "Morphometrics").exists() and (nerve / "CSA").exists()
    assert (tmp_path / "Study_archive").is_dir()