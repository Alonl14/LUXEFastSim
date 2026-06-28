import numpy as np
import pandas as pd
import pytest

from pipeline.regions import REGIONS, region_mask


def _grid():
    xs = np.linspace(-3000, 1500, 40)
    ys = np.linspace(-100, 1500, 40)
    xx, yy = np.meshgrid(xs, ys)
    return pd.DataFrame({" xx": xx.ravel(), " yy": yy.ravel()})


def test_regions_constant():
    assert REGIONS == ("outer1", "outer2", "inner")


def test_masks_partition_plane_exactly_once():
    df = _grid()
    counts = np.zeros(len(df), dtype=int)
    for r in REGIONS:
        counts += region_mask(df, r).to_numpy().astype(int)
    assert (counts == 1).all()


def test_known_points():
    df = pd.DataFrame({" xx": [0.0, 600.0, -2000.0, 0.0],
                       " yy": [0.0, 0.0, 0.0, 600.0]})
    assert region_mask(df, "inner").tolist() == [True, False, False, False]
    assert region_mask(df, "outer1").tolist() == [False, True, False, True]
    assert region_mask(df, "outer2").tolist() == [False, False, True, False]


def test_unknown_region_raises():
    with pytest.raises(ValueError):
        region_mask(_grid(), "nope")
