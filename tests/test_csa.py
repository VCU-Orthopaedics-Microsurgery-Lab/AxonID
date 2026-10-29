"""CSA from QuPath GeoJSON, and 4X whole-nerve image discovery."""
import json
import shutil

import numpy as np
import pytest
import tifffile

from batch_axon.overlay.process_overlay_qupath import get_geojson_area
from morphometrics.analyze_nerve import get_nerve_fourx_csa


def square(x0, y0, x1, y1, name):
    return {"type": "Feature", "properties": {"name": name},
            "geometry": {"type": "Polygon",
                         "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}}


def write_geojson(path, features):
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def test_plus_minus_area(tmp_path, log):
    p = write_geojson(tmp_path / "a.geojson", [square(0, 0, 100, 100, "+"), square(10, 10, 20, 20, "-")])
    assert get_geojson_area(str(p), log) == pytest.approx(10000 - 100)


def test_overlapping_plus_regions_are_unioned(tmp_path, log):
    p = write_geojson(tmp_path / "a.geojson", [square(0, 0, 10, 10, "+"), square(5, 0, 15, 10, "+")])
    assert get_geojson_area(str(p), log) == pytest.approx(150)


def test_classification_name_is_also_read(tmp_path, log):
    f = square(0, 0, 10, 10, "")
    f["properties"] = {"classification": {"name": "+"}}
    assert get_geojson_area(str(write_geojson(tmp_path / "a.geojson", [f])), log) == pytest.approx(100)


def test_no_plus_region_gives_none(tmp_path, log):
    p = write_geojson(tmp_path / "a.geojson", [square(0, 0, 10, 10, "-")])
    assert get_geojson_area(str(p), log) is None


FOURX_CONFIG = """
[pixel_size_um.4X]
1440 = 2.0

[folders]
tiff_suffixes = ["_TIFF"]
csa_suffix_geojson = "_CSA.geojson"
"""


def make_nerve(root, folder, image_name, csa_name):
    nerve = root / "N1"
    (nerve / folder).mkdir(parents=True)
    (nerve / "CSA").mkdir()
    tifffile.imwrite(nerve / folder / image_name, np.zeros((100, 1440), np.uint8))
    write_geojson(nerve / "CSA" / csa_name, [square(0, 0, 100, 50, "+")])
    return nerve


@pytest.mark.parametrize("folder,image,csa", [
    ("4X_TIFF", "N1_4X_001.ome.tif",  "N1_4X_001_CSA.geojson"),
    ("4X_TIFF", "N1_4X_001.tif",      "N1_4X_001_CSA.geojson"),
    ("4x_tiff", "N1_4x_001.tif",      "N1_4x_001_CSA.geojson"),
    ("4X_Tiff", "N1_4X_001.ome.tiff", "N1_4X_001_CSA.geojson"),
])
def test_fourx_discovery_variants(tmp_path, temp_config, log, folder, image, csa):
    temp_config(FOURX_CONFIG)
    nerve = make_nerve(tmp_path, folder, image, csa)
    # 100 x 50 px traced, 2.0 µm/px at 1440 px wide
    assert get_nerve_fourx_csa("N1", nerve, nerve / "CSA", log) == pytest.approx(5000 * 4.0)


def test_several_fourx_images_are_added(tmp_path, temp_config, log):
    temp_config(FOURX_CONFIG)
    nerve = make_nerve(tmp_path, "4X_TIFF", "N1_4X_001.tif", "N1_4X_001_CSA.geojson")
    shutil.copy(nerve / "4X_TIFF" / "N1_4X_001.tif", nerve / "4X_TIFF" / "N1_4X_002.tif")
    shutil.copy(nerve / "CSA" / "N1_4X_001_CSA.geojson", nerve / "CSA" / "N1_4X_002_CSA.geojson")
    assert get_nerve_fourx_csa("N1", nerve, nerve / "CSA", log) == pytest.approx(2 * 5000 * 4.0)


def test_incomplete_fourx_coverage_gives_none(tmp_path, temp_config, log):
    temp_config(FOURX_CONFIG)
    nerve = make_nerve(tmp_path, "4X_TIFF", "N1_4X_001.tif", "N1_4X_001_CSA.geojson")
    shutil.copy(nerve / "4X_TIFF" / "N1_4X_001.tif", nerve / "4X_TIFF" / "N1_4X_002.tif")  # no CSA
    assert get_nerve_fourx_csa("N1", nerve, nerve / "CSA", log) is None


def test_no_fourx_folder_gives_none(tmp_path, temp_config, log):
    temp_config(FOURX_CONFIG)
    nerve = make_nerve(tmp_path, "40X_TIFF", "N1_4X_001.tif", "N1_4X_001_CSA.geojson")
    assert get_nerve_fourx_csa("N1", nerve, nerve / "CSA", log) is None
