"""Hidden and system files in image folders (macOS ._name.tif, Windows Thumbs.tif) are ignored."""

from utils.filesystem import is_stray_file, list_files
from utils.study import _has_images


def test_dot_prefixed_file_is_stray(tmp_path):
    f = tmp_path / "._a.tif"
    f.write_bytes(b"x")
    assert is_stray_file(f)


def test_normal_image_is_not_stray(tmp_path):
    f = tmp_path / "a.tif"
    f.write_bytes(b"x")
    assert not is_stray_file(f)


def test_list_files_skips_stray(tmp_path):
    (tmp_path / "a.tif").write_bytes(b"x")
    (tmp_path / "._a.tif").write_bytes(b"x")
    assert [p.name for p in list_files(tmp_path, (".tif",))] == ["a.tif"]


def test_has_images_ignores_stray_source_file(tmp_path):
    src, seg = tmp_path / "40X_TIFF", tmp_path / "Segmented"
    src.mkdir(); seg.mkdir()
    (src / "a.tif").write_bytes(b"x")
    (src / "._a.tif").write_bytes(b"x")          # stray: no segmented partner expected
    (seg / "a_segmented.tif").write_bytes(b"x")
    assert _has_images(seg, src)
