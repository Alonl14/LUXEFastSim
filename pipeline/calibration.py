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


def load_run_calibrator(run_dir, region):
    """Load {region}_calib.pkl from a run dir, or None if absent (back-compat)."""
    import os

    path = os.path.join(run_dir, f"{region}_calib.pkl")
    if os.path.exists(path):
        return MarginalCalibrator.load(path)
    return None


def unwrap_dataparallel(net):
    """Return the bare module from an nn.DataParallel wrapper (else net unchanged).

    DataParallel.forward always scatters to its configured device_ids[0] (cuda:0)
    and asserts every parameter/buffer already lives there, so it cannot be run on
    CPU — `.to('cpu')` followed by a forward raises
    `RuntimeError: module must have its parameters and buffers on device cuda:0 ...`.
    The unwrapped module runs wherever its parameters happen to be. Use this before
    any CPU-side generation/calibration forward pass.
    """
    return net.module if isinstance(net, torch.nn.DataParallel) else net


def fit_calibrator(generator_net, noise_dim, n_samples, out_path, device="cpu"):
    generator_net = unwrap_dataparallel(generator_net)
    generator_net.eval().to(device)
    with torch.no_grad():
        z = torch.randn(n_samples, noise_dim, device=device)
        out = generator_net(z).detach().cpu().numpy()
    cal = MarginalCalibrator().fit(out)
    cal.save(out_path)
    return cal
