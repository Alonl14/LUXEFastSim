import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kstest, spearmanr

from pipeline.calibration import MarginalCalibrator, fit_calibrator, load_run_calibrator


def test_calibration_normalizes_off_marginal():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.3, scale=1.2, size=(20000, 1)).astype(np.float32)
    cal = MarginalCalibrator().fit(x)
    y = cal.transform(x)
    stat, pval = kstest(y[:, 0], "norm")
    assert pval > 0.01


def test_calibration_is_monotonic_preserves_rank_corr():
    rng = np.random.default_rng(1)
    a = rng.normal(size=20000)
    b = a + rng.normal(scale=0.1, size=20000)
    x = np.column_stack([a, b]).astype(np.float32)
    cal = MarginalCalibrator().fit(x)
    y = cal.transform(x)
    before = spearmanr(x[:, 0], x[:, 1]).correlation
    after = spearmanr(y[:, 0], y[:, 1]).correlation
    assert abs(before - after) < 1e-6


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.normal(loc=1.0, size=(5000, 2)).astype(np.float32)
    cal = MarginalCalibrator().fit(x)
    p = str(tmp_path / "calib.pkl")
    cal.save(p)
    loaded = MarginalCalibrator.load(p)
    assert np.allclose(cal.transform(x), loaded.transform(x))


def test_fit_calibrator_with_dummy_generator(tmp_path):
    torch.manual_seed(0)

    class Shift(nn.Module):
        def forward(self, z):
            return z[:, :2] + 0.5
    p = str(tmp_path / "c.pkl")
    cal = fit_calibrator(Shift(), noise_dim=4, n_samples=10000, out_path=p)
    z = torch.randn(10000, 4)
    out = (z[:, :2] + 0.5).numpy()
    y = cal.transform(out)
    stat, pval = kstest(y[:, 0], "norm")
    assert pval > 0.01


def test_load_run_calibrator_present_and_absent(tmp_path):
    import numpy as np
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5000, 2)).astype(np.float32)
    MarginalCalibrator().fit(x).save(str(tmp_path / "inner_calib.pkl"))

    loaded = load_run_calibrator(str(tmp_path), "inner")
    assert loaded is not None
    assert np.allclose(loaded.transform(x), MarginalCalibrator().fit(x).transform(x))
    assert load_run_calibrator(str(tmp_path), "outer1") is None


def test_generate_ds_applies_calibrator(monkeypatch):
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
            return np.asarray(x) - 5.0

    cfg = {"dataGroup": "inner", "noiseDim": 3,
           "features": {" a": [], " b": []}, "pdg": 22}
    utils.generate_ds(Gen(), factor=1, cfg=cfg, calibrator=Cal())
    assert np.allclose(captured["into_inverse"], 0.0)
