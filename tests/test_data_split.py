import pandas as pd

from pipeline.data_split import split_by_region, split_particle
from pipeline.regions import REGIONS


def _df():
    return pd.DataFrame({
        " pdg": [22, 22, 22, 2112],
        " xx": [0.0, 600.0, -2000.0, 0.0],
        " yy": [0.0, 0.0, 0.0, 0.0],
    })


def test_split_by_region_partitions():
    parts = split_by_region(_df())
    assert set(parts.keys()) == set(REGIONS)
    total = sum(len(v) for v in parts.values())
    assert total == 4


def test_split_particle_filters_pdg_and_writes(tmp_path):
    paths = split_particle(_df(), pdg=22, out_dir=str(tmp_path), prefix="v2_")
    assert set(paths.keys()) == set(REGIONS)
    inner = pd.read_csv(paths["inner"])
    assert len(inner) == 1
    assert (inner[" pdg"] == 22).all()
    for p in paths.values():
        assert (pd.read_csv(p)[" pdg"] == 22).all()
    assert paths["inner"].endswith("v2_22_inner.csv")
