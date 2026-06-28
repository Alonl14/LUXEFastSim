# Pipeline Extensibility & Disc-Surface Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make adding a new particle type a single parameterized command, and improve disc-surface marginal fidelity (measured by BED) with two near-free levers — generator EMA and post-hoc per-feature marginal calibration.

**Architecture:** Keep the existing three-separate-region-models-per-particle WGAN-GP architecture. Phase 1 extracts particle/region/config handling into small, testable `pipeline/` modules behind one `run_particle.py <pdg>` entrypoint with a single source of truth for region boundaries. Phase 2 adds EMA to the trainer and a persisted per-feature calibrator applied at generation. Phase 3 adds a BED comparison harness.

**Tech Stack:** Python 3.9, PyTorch (`nn.DataParallel`-wrapped models), scikit-learn `QuantileTransformer`, joblib, pandas/numpy, pytest 8.

## Global Constraints

- **Feature column names have leading spaces** — `' xx'`, `' yy'`, `' pxx'`, `' pyy'`, `' pzz'`, `' time'`, `' pdg'`. Preserve them exactly in code, JSON keys, and test fixtures.
- **Models are `nn.DataParallel`-wrapped** — saved state dicts carry a `module.` prefix; load into a `DataParallel`-wrapped net.
- **Inference must stay essentially as fast as today** — no inference-time refiners; fidelity fixes are training-time or one 1D interpolation per feature.
- **Region training order is `outer1 → outer2 → inner`** — preserve it in the entrypoint.
- **Success metric is BED** (`utils.get_batch_ed_histograms` / `utils.get_ed`) per region on the photon (pdg 22) baseline.
- **Tests must run locally with synthetic data** — no dependency on cluster paths (`/storage/agrp/...`, `/srv01/agrp/...`).
- **Run-dir layout consumed by `utils.check_run`** — `cfg_{region}.json`, `{region}_Gen_model.pt`, `{region}_Disc_model.pt`, `{PREFIX}_{region}.npy` logs. Do not break it.

## File Structure

**Create:**
- `pipeline/__init__.py` — package marker.
- `pipeline/regions.py` — canonical region boundaries + `region_mask`; single source of truth.
- `pipeline/config.py` — config materialization from base template + per-pdg override.
- `pipeline/data_split.py` — split a master/particle DataFrame into region CSVs.
- `pipeline/calibration.py` — `MarginalCalibrator` + `fit_calibrator` helper.
- `pipeline/bed_report.py` — assemble the 3-stage BED comparison table.
- `Config/base_cfg.json` — shared hyperparameters + default transform template + region list.
- `Config/overrides/22.json` — photon override (reproduces current v5 photon config).
- `run_particle.py` — single CLI entrypoint.
- `tests/conftest.py` — adds repo root to `sys.path` for imports.
- `tests/test_regions.py`, `tests/test_config.py`, `tests/test_data_split.py`, `tests/test_ema.py`, `tests/test_calibration.py`, `tests/test_run_particle.py`, `tests/test_bed_report.py`.

**Modify:**
- `trainer.py` — module-level `ema_update`; `Trainer` maintains an EMA generator used for validation, checkpointing, and saving.
- `utils.py` — `generate_ds` and `generate_fake_real_dfs` accept/apply an optional calibrator.

---

## Phase 1 — Extensibility

### Task 1: Canonical region boundaries (`pipeline/regions.py`)

**Files:**
- Create: `pipeline/__init__.py`
- Create: `pipeline/regions.py`
- Create: `tests/conftest.py`
- Test: `tests/test_regions.py`

**Interfaces:**
- Produces:
  - `REGIONS: tuple[str, str, str]` = `("outer1", "outer2", "inner")` (training order).
  - `DISC_RX_MAX: float = 4000.0`, `TIME_MAX: float = 1e6`.
  - `region_mask(df: pd.DataFrame, region: str) -> pd.Series` — boolean mask over `df` using columns `' xx'`, `' yy'`. Raises `ValueError` for unknown region.

- [ ] **Step 1: Create the conftest so tests can import the package**

Create `tests/conftest.py`:

```python
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_regions.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_regions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline'`.

- [ ] **Step 4: Write minimal implementation**

Create `pipeline/__init__.py` (empty file).

Create `pipeline/regions.py`:

```python
"""Canonical detector-disc region boundaries.

Single source of truth shared by data splitting and evaluation. Boundaries
are half-open so the three masks partition the plane exactly once. They match
the intent of the cuts historically used in utils.check_run (yy threshold 520,
xx thresholds 500 and -1700), replacing the inconsistent cuts in MakePDGPlots.
"""

# Training order: outer regions first, then inner (preserved from FastSimCluster).
REGIONS = ("outer1", "outer2", "inner")

DISC_RX_MAX = 4000.0
TIME_MAX = 1e6


def region_mask(df, region):
    xx = df[" xx"]
    yy = df[" yy"]
    if region == "outer1":
        return (xx >= 500) | (yy >= 520)
    if region == "outer2":
        return (xx < -1700) & (yy < 520)
    if region == "inner":
        return (xx >= -1700) & (xx < 500) & (yy < 520)
    raise ValueError(f"Unknown region: {region!r}; expected one of {REGIONS}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_regions.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add pipeline/__init__.py pipeline/regions.py tests/conftest.py tests/test_regions.py
git commit -m "feat: canonical region boundaries as single source of truth"
```

---

### Task 2: Config materialization (`pipeline/config.py`)

**Files:**
- Create: `pipeline/config.py`
- Create: `Config/base_cfg.json`
- Create: `Config/overrides/22.json`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `pipeline.regions.REGIONS`.
- Produces:
  - `load_base(path: str) -> dict` — load base config JSON.
  - `load_override(overrides_dir: str, pdg: int) -> dict` — load `overrides/{pdg}.json`, or `{}` if missing.
  - `materialize_region_config(base: dict, override: dict, pdg: int, region: str, output_dir: str, num_epochs: int) -> dict` — flat cfg dict consumable by `dataset.ParticleDataset` and `trainerFactory.create_trainer`. Must contain keys: `nQuantiles`, `subsample`, `data_path`, `outputDir`, `dataGroup`, `pdg`, `batchSize`, `noiseDim`, `numEpochs`, `device`, `nCrit`, `Lambda`, `criticLearningRate`, `generatorLearningRate`, `gradMetric`, `GMaxSteps`, `features`, `emaDecay`.
  - `materialize_all(base, override, pdg, output_dir, num_epochs) -> dict[str, dict]` — `{region: cfg}` for all `REGIONS`.

**Notes:** `data_path` is built as `{dataRoot}/{dataPrefix}{pdg}_{region}.csv`. `override` may contain any of: top-level `shared` keys, `defaultFeatures`, `dataRoot`, `dataPrefix`, or a per-region `featuresByRegion` map. Keep merge shallow and explicit.

- [ ] **Step 1: Write the base config and photon override**

Create `Config/base_cfg.json` (note the leading spaces in feature keys):

```json
{
  "shared": {
    "nQuantiles": 1000000,
    "subsample": 40000000,
    "batchSize": 1024,
    "noiseDim": 256,
    "numEpochs": 50,
    "device": "cpu",
    "nCrit": 4,
    "Lambda": 5,
    "criticLearningRate": 1.5e-05,
    "generatorLearningRate": 8e-07,
    "gradMetric": "norm",
    "GMaxSteps": null,
    "emaDecay": 0.999
  },
  "defaultFeatures": {
    " xx": [],
    " yy": [],
    " pxx": [],
    " pyy": [],
    " pzz": ["flip", "log"],
    " time": ["log"]
  },
  "dataRoot": "/storage/agrp/alonle/LUXE_FastSim/GAN_InputSample",
  "dataPrefix": ""
}
```

Create `Config/overrides/22.json` (photon: same as default; present to document the pattern and pin the species):

```json
{
  "shared": {},
  "defaultFeatures": {
    " xx": [],
    " yy": [],
    " pxx": [],
    " pyy": [],
    " pzz": ["flip", "log"],
    " time": ["log"]
  }
}
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_config.py`:

```python
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
    # every key create_trainer / ParticleDataset reads is present
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.config'`.

- [ ] **Step 4: Write minimal implementation**

Create `pipeline/config.py`:

```python
"""Materialize per-region training configs from a base template + per-pdg override."""

import json
import os

from pipeline.regions import REGIONS


def load_base(path):
    with open(path, "r") as fp:
        return json.load(fp)


def load_override(overrides_dir, pdg):
    path = os.path.join(overrides_dir, f"{pdg}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as fp:
        return json.load(fp)


def _merged_shared(base, override):
    shared = dict(base.get("shared", {}))
    shared.update(override.get("shared", {}))
    return shared


def _features_for(base, override, region):
    by_region = override.get("featuresByRegion", {})
    if region in by_region:
        return by_region[region]
    if "defaultFeatures" in override:
        return override["defaultFeatures"]
    return base["defaultFeatures"]


def materialize_region_config(base, override, pdg, region, output_dir, num_epochs):
    shared = _merged_shared(base, override)
    data_root = override.get("dataRoot", base["dataRoot"])
    data_prefix = override.get("dataPrefix", base.get("dataPrefix", ""))

    cfg = dict(shared)
    cfg["pdg"] = pdg
    cfg["dataGroup"] = region
    cfg["outputDir"] = output_dir
    cfg["numEpochs"] = num_epochs
    cfg["data_path"] = os.path.join(data_root, f"{data_prefix}{pdg}_{region}.csv")
    cfg["features"] = _features_for(base, override, region)
    return cfg


def materialize_all(base, override, pdg, output_dir, num_epochs):
    return {
        region: materialize_region_config(base, override, pdg, region,
                                          output_dir, num_epochs)
        for region in REGIONS
    }
```

- [ ] **Step 5: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_config.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add pipeline/config.py Config/base_cfg.json Config/overrides/22.json tests/test_config.py
git commit -m "feat: config materialization from base template + per-pdg override"
```

---

### Task 3: Region data splitting (`pipeline/data_split.py`)

**Files:**
- Create: `pipeline/data_split.py`
- Test: `tests/test_data_split.py`

**Interfaces:**
- Consumes: `pipeline.regions.region_mask`, `REGIONS`.
- Produces:
  - `split_by_region(df: pd.DataFrame) -> dict[str, pd.DataFrame]` — `{region: subframe}` for all `REGIONS`.
  - `split_particle(master_df, pdg, out_dir, prefix="") -> dict[str, str]` — filter master to `df[' pdg'] == pdg`, write `{prefix}{pdg}_{region}.csv` per region (no index), return `{region: path}`. Creates `out_dir` if missing.

- [ ] **Step 1: Write the failing test**

Create `tests/test_data_split.py`:

```python
import pandas as pd

from pipeline.data_split import split_by_region, split_particle
from pipeline.regions import REGIONS


def _df():
    # one point per region, two pdgs
    return pd.DataFrame({
        " pdg": [22, 22, 22, 2112],
        " xx": [0.0, 600.0, -2000.0, 0.0],
        " yy": [0.0, 0.0, 0.0, 0.0],
    })


def test_split_by_region_partitions():
    parts = split_by_region(_df())
    assert set(parts.keys()) == set(REGIONS)
    total = sum(len(v) for v in parts.values())
    assert total == 4  # nothing dropped, nothing duplicated


def test_split_particle_filters_pdg_and_writes(tmp_path):
    paths = split_particle(_df(), pdg=22, out_dir=str(tmp_path), prefix="v2_")
    assert set(paths.keys()) == set(REGIONS)
    inner = pd.read_csv(paths["inner"])
    assert len(inner) == 1
    assert (inner[" pdg"] == 22).all()
    # the 2112 row must not appear in any photon file
    for p in paths.values():
        assert (pd.read_csv(p)[" pdg"] == 22).all()
    assert paths["inner"].endswith("v2_22_inner.csv")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_data_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.data_split'`.

- [ ] **Step 3: Write minimal implementation**

Create `pipeline/data_split.py`:

```python
"""Split a master sample into per-region CSVs using the canonical region masks."""

import os

from pipeline.regions import REGIONS, region_mask


def split_by_region(df):
    return {region: df[region_mask(df, region)] for region in REGIONS}


def split_particle(master_df, pdg, out_dir, prefix=""):
    os.makedirs(out_dir, exist_ok=True)
    df = master_df[master_df[" pdg"] == pdg]
    paths = {}
    for region, part in split_by_region(df).items():
        path = os.path.join(out_dir, f"{prefix}{pdg}_{region}.csv")
        part.to_csv(path, index=False)
        paths[region] = path
    return paths
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_data_split.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/data_split.py tests/test_data_split.py
git commit -m "feat: region data splitting via canonical masks"
```

- [ ] **Step 6: Make the evaluator (`utils.check_run`) use the same canonical masks**

The spec requires the splitter and evaluator to share one source of truth. `check_run` currently inlines its own region cuts. These cannot be unit-tested locally (they need cluster data), so this is a read-verified edit; the boundary differences vs. the old inline cuts are measure-zero (old used `xx <= 500`/`yy <= 520` inclusive; canonical uses `< 500`/`< 520`).

In `utils.py`, add to the imports near the top (after `import dataset`):

```python
from pipeline.regions import region_mask, DISC_RX_MAX, TIME_MAX
```

In `check_run`, replace:

```python
        posIn = (innerDF[' time'] <= 1e6) & (innerDF[' rx'] <= 4000) & (innerDF[' xx'] <= 500) & (
                innerDF[' xx'] >= -1700) & (innerDF[' yy'] <= 520)
```

with:

```python
        posIn = (innerDF[' time'] <= TIME_MAX) & (innerDF[' rx'] <= DISC_RX_MAX) & region_mask(innerDF, "inner")
```

Replace:

```python
        posOut1 = (outer1DF[' time'] <= 1e6) & (outer1DF[' rx'] <= 4000) & (
                (outer1DF[' xx'] >= 500) | (outer1DF[' yy'] >= 520))
```

with:

```python
        posOut1 = (outer1DF[' time'] <= TIME_MAX) & (outer1DF[' rx'] <= DISC_RX_MAX) & region_mask(outer1DF, "outer1")
```

Replace:

```python
        posOut2 = (outer2DF[' time'] <= 1e6) & (outer2DF[' rx'] <= 4000) & (
                (outer2DF[' xx'] < -1700) & (outer2DF[' yy'] <= 520))
```

with:

```python
        posOut2 = (outer2DF[' time'] <= TIME_MAX) & (outer2DF[' rx'] <= DISC_RX_MAX) & region_mask(outer2DF, "outer2")
```

- [ ] **Step 7: Confirm the suite still imports cleanly**

Run: `venv/bin/python -m pytest tests/ -v`
Expected: PASS (the `utils` import path now also pulls `pipeline.regions`; no test regressions).

- [ ] **Step 8: Commit**

```bash
git add utils.py
git commit -m "refactor: check_run uses canonical region masks (single source of truth)"
```

---

### Task 4: Single entrypoint (`run_particle.py`)

**Files:**
- Create: `run_particle.py`
- Test: `tests/test_run_particle.py`

**Interfaces:**
- Consumes: `pipeline.config.{load_base,load_override,materialize_all}`, `pipeline.regions.REGIONS`, `trainerFactory.create_trainer`, `utils`.
- Produces:
  - `build_configs(pdg, output_dir, num_epochs, base_path="Config/base_cfg.json", overrides_dir="Config/overrides") -> dict[str, dict]` — materialized `{region: cfg}`.
  - `write_configs(configs, output_dir) -> None` — writes `cfg_{region}.json` into `output_dir` (reuses `utils.save_cfg`).
  - `split_if_requested(master_csv, pdg, base_path, overrides_dir) -> dict[str, str]` — read the master CSV, resolve `dataRoot`/`dataPrefix` from base+override, and call `pipeline.data_split.split_particle` so the per-region CSVs land exactly where the materialized `data_path` expects them. Returns `{region: path}`.
  - `main(argv=None) -> None` — CLI: `run_particle.py <pdg> [--epochs N] [--output-dir DIR] [--base PATH] [--overrides-dir DIR] [--split-from MASTER_CSV] [--dry-run]`. If `--split-from` is given, split first. With `--dry-run`, materialize + write configs only (no dataset/training). Otherwise build trainers, run them in `REGIONS` order, and dump logs (same `log_map` as the old `FastSimCluster.py`).

**Notes:** `--dry-run` makes the orchestration testable without cluster data. `--base`/`--overrides-dir` default to the repo `Config/` paths but are overridable so tests (and local runs) can point at a writable data root. The non-dry path reuses `create_trainer(cfg)` and `trainer.run()` unchanged, preserving `outer1 → outer2 → inner` order. When pre-split CSVs already exist, omit `--split-from` and it reuses them.

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_particle.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_run_particle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_particle'`.

- [ ] **Step 3: Write minimal implementation**

Create `run_particle.py`:

```python
"""Single entrypoint: train all three region models for one particle type.

Usage:
    python run_particle.py <pdg> [--epochs N] [--output-dir DIR] [--dry-run]
"""

import argparse
import os
import time

import numpy as np

import utils
from pipeline.config import load_base, load_override, materialize_all
from pipeline.regions import REGIONS

LOG_MAP = {
    "KL_log": "KL", "D_wdist_log": "D", "Val_D_log": "ValD",
    "G_loss_log": "G", "Val_G_log": "ValG", "GP_log": "GP",
    "gradG_log": "GradG", "gradD_log": "GradD",
}


def build_configs(pdg, output_dir, num_epochs,
                  base_path="Config/base_cfg.json",
                  overrides_dir="Config/overrides"):
    base = load_base(base_path)
    override = load_override(overrides_dir, pdg)
    return materialize_all(base, override, pdg, output_dir, num_epochs)


def write_configs(configs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for cfg in configs.values():
        utils.save_cfg(cfg)


def split_if_requested(master_csv, pdg, base_path, overrides_dir):
    import pandas as pd

    from pipeline.config import load_base, load_override
    from pipeline.data_split import split_particle

    base = load_base(base_path)
    override = load_override(overrides_dir, pdg)
    root = override.get("dataRoot", base["dataRoot"])
    prefix = override.get("dataPrefix", base.get("dataPrefix", ""))
    master = pd.read_csv(master_csv)
    return split_particle(master, pdg, out_dir=root, prefix=prefix)


def _run_training(configs, output_dir):
    # imported lazily so --dry-run needs no torch/data
    from trainerFactory import create_trainer

    trainers = {r: create_trainer(cfg) for r, cfg in configs.items()}
    t0 = time.localtime()
    print(f"Start : {utils.get_time(t0)}")
    for region in REGIONS:
        print(f"--- Training {region} ---")
        trainers[region].run()
        print(f"{region} done : {utils.get_time(time.localtime(), t0)}")

    for region, tr in trainers.items():
        for attr, prefix in LOG_MAP.items():
            arr = getattr(tr, attr, None)
            if arr is not None and len(arr):
                np.save(os.path.join(tr.outputDir, f"{prefix}_{region}.npy"),
                        np.asarray(arr))
    print("All logs saved - job finished.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train fast-sim models for one particle.")
    parser.add_argument("pdg", type=int, help="PDG code of the particle")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--base", type=str, default="Config/base_cfg.json")
    parser.add_argument("--overrides-dir", type=str, default="Config/overrides")
    parser.add_argument("--split-from", type=str, default=None,
                        help="Master CSV to split into per-region inputs before training.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Materialize and write configs only; no training.")
    args = parser.parse_args(argv)

    output_dir = args.output_dir if args.output_dir.endswith("/") else args.output_dir + "/"
    if args.split_from:
        paths = split_if_requested(args.split_from, args.pdg, args.base, args.overrides_dir)
        print(f"Split master into: {paths}")
    configs = build_configs(args.pdg, output_dir, args.epochs,
                            base_path=args.base, overrides_dir=args.overrides_dir)
    write_configs(configs, output_dir)
    if args.dry_run:
        print(f"Dry run: wrote configs for pdg {args.pdg} to {output_dir}")
        return
    _run_training(configs, output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_run_particle.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full suite so far**

Run: `venv/bin/python -m pytest tests/ -v`
Expected: PASS (all Phase 1 tests).

- [ ] **Step 6: Commit**

```bash
git add run_particle.py tests/test_run_particle.py
git commit -m "feat: run_particle.py single entrypoint with dry-run"
```

---

## Phase 2 — Fidelity

### Task 5: Generator EMA in the trainer (`trainer.py`)

**Files:**
- Modify: `trainer.py` (add module-level `ema_update`; wire EMA into `Trainer`)
- Test: `tests/test_ema.py`

**Interfaces:**
- Produces:
  - `ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None` — in-place `ema = decay*ema + (1-decay)*live` for all parameters; buffers (e.g. BatchNorm running stats) copied directly from `model`.
- Changes to `Trainer`:
  - reads `cfg.get('emaDecay', None)`; if set, builds `self.ema_gen = copy.deepcopy(self.genNet)` with `requires_grad_(False)`.
  - calls `ema_update(self.ema_gen, self.genNet, decay)` after each `optG.step()`.
  - `_validate` and the best-checkpoint save use the EMA generator when present (`self.ema_gen or self.genNet`).
  - the saved `{dataGroup}_Gen_model.pt` is the EMA generator's `state_dict()` when EMA is on.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ema.py`:

```python
import copy

import torch
import torch.nn as nn

from trainer import ema_update


def test_ema_update_math():
    live = nn.Linear(3, 2, bias=False)
    ema = copy.deepcopy(live)
    with torch.no_grad():
        for p in live.parameters():
            p.copy_(torch.ones_like(p))
        for p in ema.parameters():
            p.copy_(torch.zeros_like(p))
    ema_update(ema, live, decay=0.9)
    for p in ema.parameters():
        # 0.9*0 + 0.1*1 = 0.1
        assert torch.allclose(p, torch.full_like(p, 0.1), atol=1e-6)


def test_ema_copies_buffers():
    live = nn.BatchNorm1d(4)
    ema = copy.deepcopy(live)
    with torch.no_grad():
        live.running_mean.copy_(torch.arange(4, dtype=torch.float32))
    ema_update(ema, live, decay=0.99)
    assert torch.allclose(ema.running_mean, live.running_mean)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_ema.py -v`
Expected: FAIL with `ImportError: cannot import name 'ema_update' from 'trainer'`.

- [ ] **Step 3: Add `ema_update` and EMA fields to `trainer.py`**

In `trainer.py`, add `import copy` near the top imports (after `import torch`), and add this module-level function above `class Trainer`:

```python
def ema_update(ema_model, model, decay):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p.detach(), alpha=1 - decay)
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)
```

In `trainerFactory.create_trainer`, the `Trainer` receives a *fresh* flat `cfgDict`, not the raw config — so `emaDecay` must be propagated explicitly. In the `cfgDict` literal, after `'gradMetric': cfg.get('gradMetric', 'norm'),` add:

```python
        'emaDecay': cfg.get('emaDecay', None),
```

In `Trainer.__init__`, after `self.gradMetric = cfg.get('gradMetric', 'norm')`, add:

```python
        # EMA generator (training-time only; zero inference cost)
        self.ema_decay = cfg.get('emaDecay', None)
        self.ema_gen = None
        if self.ema_decay:
            self.ema_gen = copy.deepcopy(self.genNet)
            for p in self.ema_gen.parameters():
                p.requires_grad_(False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_ema.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Wire EMA into the training loop, validation, and saving**

In `Trainer.run`, immediately after `self.optG.step()`, add:

```python
                if self.ema_gen is not None:
                    self.ema_gen.to(self.device)
                    ema_update(self.ema_gen, self.genNet, self.ema_decay)
```

Replace the best-checkpoint save block:

```python
            if self.best_ValD > -vD > 0:
                self.best_ValD = -vD
                print(f"New best Val_D: {self.best_ValD:.4f} at epoch {epoch}")
                torch.save(self.genNet.state_dict(), f"{self.outputDir}{self.dataGroup}_Gen_model.pt")
                torch.save(self.discNet.state_dict(), f"{self.outputDir}{self.dataGroup}_Disc_model.pt")
```

with:

```python
            if self.best_ValD > -vD > 0:
                self.best_ValD = -vD
                print(f"New best Val_D: {self.best_ValD:.4f} at epoch {epoch}")
                gen_to_save = self.ema_gen if self.ema_gen is not None else self.genNet
                torch.save(gen_to_save.state_dict(), f"{self.outputDir}{self.dataGroup}_Gen_model.pt")
                torch.save(self.discNet.state_dict(), f"{self.outputDir}{self.dataGroup}_Disc_model.pt")
```

In `Trainer._validate`, replace the generator used for the fake samples. Change the two `self.genNet(...)` calls inside `_validate` to use an eval generator:

```python
    def _validate(self):
        gen = self.ema_gen if self.ema_gen is not None else self.genNet
        gen.eval()
        self.discNet.eval()
        sG = sD = n = 0
        with torch.no_grad():
            for real in self.dl_val:
                real = real.to(self.device)
                bs = real.size(0)
                loss_real = -self.discNet(real).mean()
                fake = gen(torch.randn(bs, self.noiseDim, device=self.device))
                loss_fake = self.discNet(fake).mean()
                sD += (loss_real + loss_fake).item()
                sG += (-self.discNet(fake).mean()).item()
                n += 1
        self.genNet.train()
        self.discNet.train()
        return sG / n, sD / n
```

- [ ] **Step 6: Run the full suite to confirm no regressions**

Run: `venv/bin/python -m pytest tests/ -v`
Expected: PASS (all tests).

- [ ] **Step 7: Commit**

```bash
git add trainer.py tests/test_ema.py
git commit -m "feat: EMA generator used for validation, checkpointing, and saving"
```

---

### Task 6: Marginal calibrator (`pipeline/calibration.py`)

**Files:**
- Create: `pipeline/calibration.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Produces:
  - `class MarginalCalibrator` with:
    - `fit(self, gen_output: np.ndarray) -> "MarginalCalibrator"` — fit per-feature map from the generator's actual marginal to standard normal (a `QuantileTransformer(output_distribution='normal')`).
    - `transform(self, gen_output: np.ndarray) -> np.ndarray` — apply the calibration.
    - `save(self, path: str) -> None` / `load(path: str) -> "MarginalCalibrator"` (classmethod) — joblib persistence.
  - `fit_calibrator(generator_net, noise_dim, n_samples, out_path, device="cpu") -> MarginalCalibrator` — sample the generator in transformed space, fit, and save.

- [ ] **Step 1: Write the failing test**

Create `tests/test_calibration.py`:

```python
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kstest, spearmanr

from pipeline.calibration import MarginalCalibrator, fit_calibrator


def test_calibration_normalizes_off_marginal():
    rng = np.random.default_rng(0)
    # generator output that is "slightly off": shifted, scaled normal
    x = rng.normal(loc=0.3, scale=1.2, size=(20000, 1)).astype(np.float32)
    cal = MarginalCalibrator().fit(x)
    y = cal.transform(x)
    stat, pval = kstest(y[:, 0], "norm")
    assert pval > 0.01  # output marginal is standard normal


def test_calibration_is_monotonic_preserves_rank_corr():
    rng = np.random.default_rng(1)
    a = rng.normal(size=20000)
    b = a + rng.normal(scale=0.1, size=20000)  # strongly rank-correlated with a
    x = np.column_stack([a, b]).astype(np.float32)
    cal = MarginalCalibrator().fit(x)
    y = cal.transform(x)
    before = spearmanr(x[:, 0], x[:, 1]).correlation
    after = spearmanr(y[:, 0], y[:, 1]).correlation
    assert abs(before - after) < 1e-6  # per-feature monotone map preserves rank corr


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.normal(loc=1.0, size=(5000, 2)).astype(np.float32)
    cal = MarginalCalibrator().fit(x)
    p = str(tmp_path / "calib.pkl")
    cal.save(p)
    loaded = MarginalCalibrator.load(p)
    assert np.allclose(cal.transform(x), loaded.transform(x))


def test_fit_calibrator_with_dummy_generator(tmp_path):
    class Shift(nn.Module):
        def forward(self, z):
            return z[:, :2] + 0.5  # 2-feature output, shifted
    p = str(tmp_path / "c.pkl")
    cal = fit_calibrator(Shift(), noise_dim=4, n_samples=10000, out_path=p)
    z = torch.randn(10000, 4)
    out = (z[:, :2] + 0.5).numpy()
    y = cal.transform(out)
    stat, pval = kstest(y[:, 0], "norm")
    assert pval > 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.calibration'`.

- [ ] **Step 3: Write minimal implementation**

Create `pipeline/calibration.py`:

```python
"""Post-hoc per-feature marginal calibration.

The generator's per-feature output is assumed by the inverse QuantileTransformer
to be standard normal. In practice it deviates slightly, which is amplified into
off physical marginals. This calibrator maps the generator's actual per-feature
marginal back onto standard normal with a monotone (rank-preserving) map.
"""

import joblib
import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer


class MarginalCalibrator:
    def __init__(self, n_quantiles=1000):
        self._n_quantiles = n_quantiles
        self.qt = None

    def fit(self, gen_output):
        gen_output = np.asarray(gen_output)
        n_q = min(self._n_quantiles, gen_output.shape[0])
        self.qt = QuantileTransformer(output_distribution="normal", n_quantiles=n_q)
        self.qt.fit(gen_output)
        return self

    def transform(self, gen_output):
        return self.qt.transform(np.asarray(gen_output))

    def save(self, path):
        joblib.dump(self.qt, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.qt = joblib.load(path)
        return obj


def fit_calibrator(generator_net, noise_dim, n_samples, out_path, device="cpu"):
    generator_net.eval().to(device)
    with torch.no_grad():
        z = torch.randn(n_samples, noise_dim, device=device)
        out = generator_net(z).detach().cpu().numpy()
    cal = MarginalCalibrator().fit(out)
    cal.save(out_path)
    return cal
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_calibration.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add pipeline/calibration.py tests/test_calibration.py
git commit -m "feat: per-feature marginal calibrator"
```

---

### Task 7: Apply calibrator at generation (`utils.py`) + fit it in `run_particle.py`

**Files:**
- Modify: `utils.py` (`generate_ds`, `generate_fake_real_dfs`)
- Modify: `run_particle.py` (fit + save calibrator after each region trains)
- Test: extend `tests/test_calibration.py`

**Interfaces:**
- Consumes: `pipeline.calibration.MarginalCalibrator`, `fit_calibrator`.
- Changes:
  - `generate_ds(generator_net, factor, cfg, calibrator=None)` — when `calibrator` is provided, apply `calibrator.transform(generated_data)` *before* `ds.quantiles.inverse_transform`.
  - `generate_fake_real_dfs(run_id, cfg, run_dir, generator_net=None, calibrator=None)` — pass `calibrator` through to `generate_ds`.
  - `run_particle._run_training` — after a region's `trainer.run()`, call `fit_calibrator` on the saved EMA generator and write `{region}_calib.pkl` into the run dir.

- [ ] **Step 1: Write the failing test (calibrator applied before inverse transform)**

Add to `tests/test_calibration.py`:

```python
def test_generate_ds_applies_calibrator(monkeypatch):
    import numpy as np
    import torch
    import utils

    captured = {}

    class FakeQT:
        def inverse_transform(self, data):
            captured["into_inverse"] = np.asarray(data).copy()
            return np.asarray(data)

    class FakeDS:
        def __init__(self, cfg):
            self.data = np.zeros((4, 2), dtype=np.float32)
            self.quantiles = FakeQT()
            self.preprocess = None
            self.cfg = cfg
        def apply_transformation(self, cfg, inverse=False):
            pass

    monkeypatch.setattr(utils.dataset, "ParticleDataset", FakeDS)

    class Gen(torch.nn.Module):
        def forward(self, z):
            return torch.ones(z.shape[0], 2) * 5.0

    class Cal:
        def transform(self, x):
            return np.asarray(x) - 5.0  # should zero it out

    cfg = {"dataGroup": "inner", "noiseDim": 3,
           "features": {" a": [], " b": []}, "pdg": 22}
    utils.generate_ds(Gen(), factor=1, cfg=cfg, calibrator=Cal())
    assert np.allclose(captured["into_inverse"], 0.0)  # calibrator ran pre-inverse
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_calibration.py::test_generate_ds_applies_calibrator -v`
Expected: FAIL with `TypeError: generate_ds() got an unexpected keyword argument 'calibrator'`.

- [ ] **Step 3: Modify `generate_ds` in `utils.py`**

Change the signature and the inverse-transform line. Replace:

```python
def generate_ds(generator_net, factor, cfg):
```

with:

```python
def generate_ds(generator_net, factor, cfg, calibrator=None):
```

And replace:

```python
    generated_data = generated_data.detach().numpy()
    features = cfg['features'].keys()
    ds.data = pd.DataFrame(np.empty((0, len(features))), columns=features)
    data_values = ds.quantiles.inverse_transform(generated_data) if ds.quantiles is not None else generated_data
```

with:

```python
    generated_data = generated_data.detach().numpy()
    if calibrator is not None:
        generated_data = calibrator.transform(generated_data)
    features = cfg['features'].keys()
    ds.data = pd.DataFrame(np.empty((0, len(features))), columns=features)
    data_values = ds.quantiles.inverse_transform(generated_data) if ds.quantiles is not None else generated_data
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_calibration.py::test_generate_ds_applies_calibrator -v`
Expected: PASS.

- [ ] **Step 5: Thread the calibrator through `generate_fake_real_dfs`**

In `utils.py`, change:

```python
def generate_fake_real_dfs(run_id, cfg, run_dir, generator_net=None):
```

to:

```python
def generate_fake_real_dfs(run_id, cfg, run_dir, generator_net=None, calibrator=None):
```

and change the call:

```python
    fake_df, real_df = generate_ds(generator_net, factor=1, cfg=cfg)
```

to:

```python
    fake_df, real_df = generate_ds(generator_net, factor=1, cfg=cfg, calibrator=calibrator)
```

- [ ] **Step 6: Fit + save the calibrator after each region trains**

In `run_particle.py`, add the import at the top:

```python
from pipeline.calibration import fit_calibrator
```

In `_run_training`, replace the training loop:

```python
    for region in REGIONS:
        print(f"--- Training {region} ---")
        trainers[region].run()
        print(f"{region} done : {utils.get_time(time.localtime(), t0)}")
```

with:

```python
    for region in REGIONS:
        print(f"--- Training {region} ---")
        tr = trainers[region]
        tr.run()
        gen_for_calib = tr.ema_gen if tr.ema_gen is not None else tr.genNet
        calib_path = os.path.join(output_dir, f"{region}_calib.pkl")
        fit_calibrator(gen_for_calib, tr.noiseDim, n_samples=200000, out_path=calib_path)
        print(f"{region} done : {utils.get_time(time.localtime(), t0)}")
```

- [ ] **Step 7: Run the full suite**

Run: `venv/bin/python -m pytest tests/ -v`
Expected: PASS (all tests).

- [ ] **Step 8: Commit**

```bash
git add utils.py run_particle.py tests/test_calibration.py
git commit -m "feat: apply marginal calibrator at generation; fit per region after training"
```

---

## Phase 3 — Validation

### Task 8: BED comparison harness (`pipeline/bed_report.py`)

**Files:**
- Create: `pipeline/bed_report.py`
- Test: `tests/test_bed_report.py`

**Interfaces:**
- Produces:
  - `three_stage_table(bed_fn, stages=("current", "+ema", "+ema+calib")) -> dict[str, float]` — calls `bed_fn(stage)` for each stage label and returns `{stage: bed_value}`. Pure orchestration so it is testable without real data; the caller supplies `bed_fn` that wraps `utils.get_batch_ed_histograms` / `utils.get_ed`.
  - `format_table(results: dict[str, float]) -> str` — human-readable summary string.

**Notes:** BED itself is computed by the existing `utils` functions on real generated/real DataFrames (requires cluster data), so this task ships the *reporting* layer with a unit-tested pure core and leaves the data-bound wrapper to be run on the cluster.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bed_report.py`:

```python
from pipeline.bed_report import three_stage_table, format_table


def test_three_stage_table_calls_each_stage():
    seen = []
    def bed_fn(stage):
        seen.append(stage)
        return {"current": 1.0, "+ema": 0.7, "+ema+calib": 0.4}[stage]
    results = three_stage_table(bed_fn)
    assert seen == ["current", "+ema", "+ema+calib"]
    assert results["+ema+calib"] == 0.4


def test_format_table_mentions_all_stages():
    s = format_table({"current": 1.0, "+ema": 0.7, "+ema+calib": 0.4})
    assert "current" in s and "+ema+calib" in s
    assert "0.4" in s
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_bed_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline.bed_report'`.

- [ ] **Step 3: Write minimal implementation**

Create `pipeline/bed_report.py`:

```python
"""Assemble a 3-stage BED comparison (current vs +EMA vs +EMA+calibration).

The pure core takes a bed_fn(stage) -> float so it can be tested without data.
On the cluster, wrap utils.get_batch_ed_histograms / utils.get_ed in bed_fn.
"""


def three_stage_table(bed_fn, stages=("current", "+ema", "+ema+calib")):
    return {stage: bed_fn(stage) for stage in stages}


def format_table(results):
    lines = ["BED comparison (lower is better):"]
    for stage, value in results.items():
        lines.append(f"  {stage:<14} {value:.6g}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venv/bin/python -m pytest tests/test_bed_report.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full suite**

Run: `venv/bin/python -m pytest tests/ -v`
Expected: PASS (all tests across all phases).

- [ ] **Step 6: Commit**

```bash
git add pipeline/bed_report.py tests/test_bed_report.py
git commit -m "feat: 3-stage BED comparison reporting harness"
```

---

## Manual validation (on cluster, after merge)

These steps require cluster data and a GPU and are not unit-tested:

1. `python run_particle.py 22 --epochs <N> --output-dir Output/run_photon_v6/` — confirm it splits data (if needed), writes 3 configs, trains 3 region models in `outer1 → outer2 → inner` order, and writes `{region}_Gen_model.pt`, `{region}_Disc_model.pt`, `{region}_calib.pkl`.
2. Run `AnalyzeRun.py` / `check_run` for the new run with and without passing the calibrator, and record per-region BED at the three stages via the `bed_report` wrapper.
3. Acceptance: BED with `+ema+calib` beats the current v5 photon models per region; per-feature KS does not regress; correlation plots do not visibly degrade.
4. Add a second particle (e.g. `python run_particle.py 2112 ...`) to confirm a new species needs no source edits — only an optional `Config/overrides/2112.json` if its transforms differ from the default template.
