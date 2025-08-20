# dataset.py – fit-once QT, safe transforms, optional memory saving
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import QuantileTransformer as QT

import utils


# ------------------------- transforms -------------------------

def _safe_log(x: pd.Series, inverse: bool = False, eps: float = 1e-12) -> pd.Series:
    return np.log(np.clip(x, eps, None)) if not inverse else np.exp(x)


def _flip(x: pd.Series, inverse: bool = False) -> pd.Series:
    return -x


TRANSFORM_REGISTRY = {
    "log": _safe_log,
    "flip": _flip,
}


# --------------------------- dataset --------------------------

class ParticleDataset(Dataset):
    """
    Loads CSV, adds derived features, applies configured transforms, fits QT once, returns tensors.
    - self.preprocess: DataFrame after transforms, before QT (optionally dropped for memory)
    - self.preqt     : numpy pre-QT array (optionally dropped for memory)
    - self.data      : numpy float32 array after QT
    - self.quantiles : fitted QuantileTransformer
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        # ---------- load & derive ----------
        df = pd.read_csv(cfg["data_path"])
        utils.add_features(df, cfg["pdg"])
        feature_cols = list(cfg["features"].keys())
        df = df[feature_cols]

        # ---------- apply configured transforms ----------
        for feat, fn_list in cfg["features"].items():
            for fn_name in fn_list:
                fn = TRANSFORM_REGISTRY.get(fn_name)
                if fn is None:
                    raise KeyError(f"Unknown transform '{fn_name}' for feature '{feat}'")
                df[feat] = fn(df[feat], inverse=False)

        # store for diagnostics (optionally drop later)
        self.preprocess = df.copy()
        self.preqt = self.preprocess.values

        # ---------- fit QT once, then transform ----------
        n_samples = len(df)
        req_nq = int(cfg.get("nQuantiles", 1000))
        n_quantiles = max(10, min(n_samples, req_nq))
        req_sub = int(cfg.get("subsample", 100_000))
        subsample = max(10_000, min(req_sub, n_samples)) if n_samples > 0 else req_sub

        qt = QT(output_distribution="normal", n_quantiles=n_quantiles, subsample=subsample, copy=True)
        self.quantiles = qt.fit(df)
        self.data = self.quantiles.transform(df).astype(np.float32)
        self.columns = feature_cols

        # ---------- memory saving ----------
        if not bool(cfg.get("keepPreprocess", False)):
            self.preprocess = None
            self.preqt = None

    # Dataset API
    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx, :])  # float32

    # Back-compat shim (only used by utils.generate_ds inverse path)
    @property
    def registry(self):
        return TRANSFORM_REGISTRY

    def apply_transformation(self, cfg: dict, inverse: bool = False):
        # No-op unless inverse is requested on a clone that replaced self.data with a DataFrame
        if not inverse:
            return
        if isinstance(self.data, np.ndarray):
            return
        for feat, fn_list in cfg["features"].items():
            for fn_name in reversed(fn_list):
                fn = TRANSFORM_REGISTRY.get(fn_name)
                if fn is None:
                    raise KeyError(f"Unknown transform '{fn_name}' for feature '{feat}'")
                self.data[feat] = fn(self.data[feat], inverse=True)
