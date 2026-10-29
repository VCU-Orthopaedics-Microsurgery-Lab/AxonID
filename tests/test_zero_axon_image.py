"""An image with no axons stays in the nerve totals as a measured 0 (with its CSA)."""
import json

import numpy as np
import pytest
import tifffile

from conftest import draw_fibers
from morphometrics.analyze_nerve import get_nerve_data
from morphometrics.distributions import bin_nerve_diameters
from morphometrics.morphometrics import get_morphometrics, save_morphometrics
from utils.calibration import Calibration

CONFIG = """
[pixel_size_um.40X]
1440 = 0.2

[folders]
tiff_suffixes = ["_TIFF"]
csa_suffix_geojson = "_CSA.geojson"
"""


def _square_geojson(path, side):
    poly = [[[0, 0], [side, 0], [side, side], [0, side], [0, 0]]]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "+"}, "geometry": {"type": "Polygon", "coordinates": poly}}]}))


def _nerve(tmp_path, log, images):
    nerve = tmp_path / "N1"
    (nerve / "CSA").mkdir(parents=True)
    for name, img in images.items():
        p = tmp_path / f"{name}_segmented.tif"
        tifffile.imwrite(p, img)
        # Calibration as if the image QuPath traced was 1440 px wide.
        cal = Calibration(0.2, 1440, 1024, 1.0, "provenance")
        df, pixel_df = get_morphometrics(str(p), "40X", log, calibration=cal)
        save_morphometrics(df, pixel_df, str(nerve / "Morphometrics"), name, calibration=cal)
        _square_geojson(nerve / "CSA" / f"{name}_CSA.geojson", 500)       # 500x500 px -> 10,000 µm²
    return nerve


def test_zero_axon_image_kept_with_csa(tmp_path, temp_config, log):
    temp_config(CONFIG)
    fibers = draw_fibers((120, 120), [(35, 35), (35, 85), (85, 35), (85, 85)], 15, 9)
    nerve = _nerve(tmp_path, log, {"img_1": fibers, "img_2": np.zeros((120, 120), np.uint8)})

    rows, agg, _ = get_nerve_data(nerve, "40X", log)
    by_name = {r["name"]: r for r in rows}
    assert set(by_name) == {"img_1", "img_2"}
    assert by_name["img_2"]["total_axons"] == 0
    assert by_name["img_2"]["csa_um2"] == pytest.approx(10000.0)
    assert by_name["img_1"]["csa_um2"] == pytest.approx(10000.0)
    assert by_name["img_2"]["gratio"] is None
    assert agg["total_images"] == 2 and agg["total_axons"] == by_name["img_1"]["total_axons"]
    assert agg["total_sample_csa_um2"] == pytest.approx(20000.0)                          # both CSAs count

    bins = bin_nerve_diameters(nerve / "Morphometrics", "N1", "40X", log)
    assert list(bins["per_image_df"]["Axon Count"]) == [by_name["img_1"]["total_axons"], 0]


def test_all_images_empty(tmp_path, temp_config, log):
    temp_config(CONFIG)
    nerve = _nerve(tmp_path, log, {"img_1": np.zeros((64, 64), np.uint8)})
    rows, agg, _ = get_nerve_data(nerve, "40X", log)
    assert [r["total_axons"] for r in rows] == [0] and agg["total_axons"] == 0
    assert rows[0]["csa_um2"] == pytest.approx(10000.0)
