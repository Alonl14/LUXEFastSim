import json

import pytest

from pipeline.config import (
    load_base,
    load_override,
    materialize_region_config,
    materialize_all,
)
from pipeline.regions import REGIONS

BASE = {
    "shared": {
        "nQuantiles": 10, "subsample": 20, "batchSize": 8, "noiseDim": 4,
        "numEpochs": 1, "device": "cpu", "nCrit": 2, "Lambda": 5,
        "criticLearningRate": 1e-5, "generatorLearningRate": 1e-6,
        "gradMetric": "norm", "GMaxSteps": None, "emaDecay": 0.99,
    },
    "defaultFeatures": {" xx": [], " pzz": ["flip", "log"]},
    "dataRoot": "/data", "dataPrefix": "v2_",
}


def test_materialize_region_config_basic():
    cfg = materialize_region_config(BASE, {}, pdg=22, region="inner",
                                    output_dir="Output/run_x/", num_epochs=7)
    assert cfg["pdg"] == 22
    assert cfg["dataGroup"] == "inner"
    assert cfg["outputDir"] == "Output/run_x/"
    assert cfg["numEpochs"] == 7
    assert cfg["data_path"] == "/data/v2_22_inner.csv"
    assert cfg["features"] == {" xx": [], " pzz": ["flip", "log"]}
    assert cfg["emaDecay"] == 0.99
    for key in ["nQuantiles", "subsample", "batchSize", "noiseDim", "device",
                "nCrit", "Lambda", "criticLearningRate", "generatorLearningRate",
                "gradMetric", "GMaxSteps"]:
        assert key in cfg


def test_override_replaces_shared_and_features():
    override = {"shared": {"batchSize": 256}, "defaultFeatures": {" time": ["log"]}}
    cfg = materialize_region_config(BASE, override, pdg=2112, region="outer1",
                                    output_dir="o/", num_epochs=1)
    assert cfg["batchSize"] == 256
    assert cfg["features"] == {" time": ["log"]}
    assert cfg["data_path"] == "/data/v2_2112_outer1.csv"


def test_materialize_all_covers_regions():
    allcfg = materialize_all(BASE, {}, pdg=22, output_dir="o/", num_epochs=1)
    assert set(allcfg.keys()) == set(REGIONS)
    assert allcfg["outer2"]["dataGroup"] == "outer2"


def test_load_base_and_override_from_repo(tmp_path):
    base = load_base("Config/base_cfg.json")
    assert " pzz" in base["defaultFeatures"]
    ovr = load_override("Config/overrides", 22)
    assert "defaultFeatures" in ovr
    missing = load_override("Config/overrides", 999999)
    assert missing == {}
