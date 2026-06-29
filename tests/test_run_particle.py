import json
import os

from run_particle import build_configs, main
from pipeline.regions import REGIONS


def test_build_configs_for_photon():
    cfgs = build_configs(22, output_dir="Output/run_t/", num_epochs=3)
    assert set(cfgs.keys()) == set(REGIONS)
    assert cfgs["inner"]["pdg"] == 22
    assert cfgs["inner"]["numEpochs"] == 3
    assert cfgs["inner"]["data_path"].endswith("22_inner.csv")


def test_build_configs_data_root_override():
    cfgs = build_configs(11, output_dir="o/", num_epochs=1, data_root="TrainData")
    assert cfgs["inner"]["data_path"] == "TrainData/11_inner.csv"


def test_main_dry_run_writes_three_configs(tmp_path):
    out = str(tmp_path) + "/"
    main(["22", "--epochs", "2", "--output-dir", out, "--dry-run"])
    for region in REGIONS:
        path = os.path.join(out, f"cfg_{region}.json")
        assert os.path.exists(path)
        with open(path) as fp:
            cfg = json.load(fp)
        assert cfg["dataGroup"] == region
        assert cfg["pdg"] == 22


def test_split_from_writes_region_csvs(tmp_path):
    import pandas as pd
    data_root = tmp_path / "data"
    base = {
        "shared": {"nQuantiles": 10, "subsample": 20, "batchSize": 8,
                   "noiseDim": 4, "numEpochs": 1, "device": "cpu", "nCrit": 2,
                   "Lambda": 5, "criticLearningRate": 1e-5,
                   "generatorLearningRate": 1e-6, "gradMetric": "norm",
                   "GMaxSteps": None, "emaDecay": 0.99},
        "defaultFeatures": {" xx": [], " yy": []},
        "dataRoot": str(data_root), "dataPrefix": "",
    }
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base))
    master = tmp_path / "master.csv"
    pd.DataFrame({" pdg": [22, 22, 22, 2112],
                  " xx": [0.0, 600.0, -2000.0, 0.0],
                  " yy": [0.0, 0.0, 0.0, 0.0]}).to_csv(master, index=False)

    out = str(tmp_path / "run") + "/"
    main(["22", "--output-dir", out, "--dry-run",
          "--base", str(base_path), "--overrides-dir", str(tmp_path / "nope"),
          "--split-from", str(master)])

    for region in REGIONS:
        assert (data_root / f"22_{region}.csv").exists()
    inner = pd.read_csv(data_root / "22_inner.csv")
    assert (inner[" pdg"] == 22).all() and len(inner) == 1
