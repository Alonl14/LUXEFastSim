# dataset.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import QuantileTransformer as QT

import utils


# ------------------------- transform primitives -------------------------

def _safe_log(x: pd.Series, inverse: bool = False, eps: float = 1e-12) -> pd.Series:
    """Log/Exp with small epsilon for numerical safety."""
    if not inverse:
        return np.log(np.clip(x, eps, None))
    else:
        return np.exp(x)  # inverse of log


def _flip(x: pd.Series, inverse: bool = False) -> pd.Series:
    """Negation; its own inverse."""
    return -x


TRANSFORM_REGISTRY = {
    "log": _safe_log,
    "flip": _flip,
}


# ------------------------------- dataset --------------------------------

class ParticleDataset(Dataset):
    """
    Loads CSV, adds derived features via utils.add_features, applies per-feature transforms
    defined in cfg["features"], then fits a QuantileTransformer once and transforms to
    approximately N(0,1) marginals.

    - __getitem__ returns a float32 tensor row.
    - self.preprocess stores a copy BEFORE QT (after transforms), for plotting/eval.
    - self.preqt is the numpy array before QT (for diagnostics).
    - self.quantiles is the fitted QuantileTransformer.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        # ---------- load & derive ----------
        df = pd.read_csv(cfg["data_path"])
        utils.add_features(df, cfg["pdg"])  # adds [' phi_x',' rx',' rp',...]
        # keep only the requested features in the specified order
        feature_cols = list(cfg["features"].keys())
        df = df[feature_cols]

        # ---------- apply configured transforms (pre-QT) ----------
        # Forward pass (not inverse); inverse will be applied elsewhere when needed
        for feat, fn_list in cfg["features"].items():
            for fn_name in fn_list:
                fn = TRANSFORM_REGISTRY.get(fn_name)
                if fn is None:
                    raise KeyError(f"Unknown transform '{fn_name}' for feature '{feat}'")
                df[feat] = fn(df[feat], inverse=False)

        # Stash a copy for plotting & inverse-QT reconstructions later
        self.preprocess = df.copy()
        self.preqt = self.preprocess.values  # numpy view pre-QT

        # ---------- fit QT once, then transform ----------
        n_samples = len(df)
        # Clamp n_quantiles/subsample to sensible values relative to dataset size
        req_nq = int(cfg.get("nQuantiles", 1000))
        n_quantiles = max(10, min(n_samples, req_nq))  # at least 10, at most n_samples
        subsample = int(cfg.get("subsample", 100_000))
        subsample = max(10_000, min(subsample, n_samples)) if n_samples > 0 else subsample

        qt = QT(output_distribution="normal", n_quantiles=n_quantiles, subsample=subsample, copy=True)
        self.quantiles = qt.fit(df)
        data_q = self.quantiles.transform(df).astype(np.float32)

        # store as numpy array for fast indexing; keep a DataFrame copy name for clarity if needed
        self.data = data_q  # shape: (N, F), dtype float32
        self.columns = feature_cols  # remember order

    # ------------------------- Dataset API -------------------------

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        # return torch tensor (float32), 1D: [num_features]
        row = self.data[idx, :]
        return torch.from_numpy(row)  # already float32

    # ---------------------- convenience ----------------------------

    @property
    def registry(self):
        # kept for backward-compatibility with older code that referenced ds.registry
        return TRANSFORM_REGISTRY

    def apply_transformation(self, cfg: dict, inverse: bool = False):
        """
        Backwards-compatible shim:
        - If inverse=False: no-op here (we applied transforms in __init__).
        - If inverse=True: apply the inverse transforms to self.data DataFrame stored in self.preprocess-style layout.
          This method is used by utils.generate_ds() on a DS it constructs for inverse transforms.
        """
        if not inverse:
            return  # already applied in __init__

        # Expect self.data to be a DataFrame with same columns as cfg["features"]
        if isinstance(self.data, np.ndarray):
            # If someone calls inverse on this training dataset instance, there’s nothing to do.
            # Inverse is meant for DS clones created in utils.generate_ds (they replace .data with a DataFrame).
            return

        # Inverse: reverse function order and apply inverse=True
        for feat, fn_list in cfg["features"].items():
            for fn_name in reversed(fn_list):
                fn = TRANSFORM_REGISTRY.get(fn_name)
                if fn is None:
                    raise KeyError(f"Unknown transform '{fn_name}' for feature '{feat}'")
                self.data[feat] = fn(self.data[feat], inverse=True)
