import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import QuantileTransformer as qt


class ParticleDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.preprocess = None
        self.preqt = None
        self.quantiles = None
        self._registry = {"log": my_log, "flip": flip}
        self.cfg = cfg
        
        QT = qt(output_distribution='normal', n_quantiles=cfg['nQuantiles'], subsample=cfg['subsample'])
        
        self.data = self._load_and_preprocess_data(cfg)
        self.preprocess = self.data.copy()
        
        self._load_normalization(cfg)
        self.apply_transformation(cfg)
        
        self._prepare_data_for_training(cfg, QT)


    def _load_and_preprocess_data(self, cfg):
        data = pd.read_csv(cfg['data_path'])
        print(f"Initial data shape: {data.shape}")
        
        data = self._filter_data(data, cfg)
        data = self._add_derived_features(data)
        
        return data[cfg["features"].keys()]

    def _filter_data(self, data, cfg):
        if ' pdg' in data.columns.values:
            data = data[data[' pdg'].isin([cfg['pdg']])]
        data = data[(data[' pzz'] <= 0) & (data[' time'] <= 10**6)]
        print("Applied time cut 10^6")
        return data

    def _add_derived_features(self, data):
        data[' rx'] = np.sqrt(data[' xx'].values**2 + data[' yy'].values**2)
        data[' rp'] = np.sqrt(data[' pxx'].values**2 + data[' pyy'].values**2)
        data[' phi_p'] = np.arctan2(data[' pyy'].values, data[' pxx'].values) + np.pi
        return data

    def _load_normalization(self, cfg):
        self.norm = pd.read_csv(cfg['norm_path'], index_col=0)
        self.norm['max'][' time'] = 10**6

    def _prepare_data_for_training(self, cfg, QT):
        data_type = np.float32
        print(f"Minimum values: {np.min(self.data, axis=0)}")
        self.quantiles = QT.fit(self.data)

        if cfg['applyQT']:
            self.preqt = self.data.values
            self.data = QT.fit_transform(self.data).astype(data_type)
        else:
            self.data = self.data.values.astype(data_type)

    @property
    def registry(self):
        return self._registry

    def apply_transformation(self, cfg, inverse=False):
        eps = cfg["epsilon"]
        for feature, function_list in cfg["features"].items():
            x_max, x_min = self.norm['max'][feature], self.norm['min'][feature]
            if not inverse:
                self.data[feature] = normalize(self.data[feature], x_min, x_max, eps, inverse)
            functions = function_list[::-1] if inverse else function_list[:]
            for f in functions:
                self.data[feature] = self.registry[f](self.data[feature], eps, inverse)
            if inverse:
                self.data[feature] = normalize(self.data[feature], x_min, x_max, eps, inverse)

    def __getitem__(self, item):
        return self.data[item, :]

    def __len__(self):
        return self.data.shape[0]


def normalize(data: np.ndarray, data_min: float, data_max: float, epsilon: float, inverse: bool) -> np.ndarray:
    """
    Normalize a feature to be in the range (epsilon, 1-epsilon) or perform the inverse operation.

    Args:
        data (np.ndarray): Feature to be normalized.
        data_min (float): Minimum value of the data (or from norm file if inverse).
        data_max (float): Maximum value of the data (or from norm file if inverse).
        epsilon (float): Data will be scaled to the range [epsilon, 1-epsilon].
        inverse (bool): If True, assumes data is in [epsilon, 1-epsilon] and scales it to [data_min, data_max].

    Returns:
        np.ndarray: Normalized or inverse-normalized feature.
    """
    if data_min == 0:
        b = epsilon
        a = (1 - 2 * epsilon) / data_max
    else:
        b = (1 - epsilon * (1 + data_max / data_min)) / (1 - data_max / data_min)
        a = (epsilon - b) / data_min

    return a * data + b if not inverse else (data - b) / a


def my_log(data: np.ndarray, epsilon: float, inverse: bool) -> np.ndarray:
    """
    Apply logarithmic transformation with scaling.

    Args:
        data (np.ndarray): Input data, assumed to be in range [epsilon, 1-epsilon].
        epsilon (float): Small value used for scaling, should match the one used in normalize.
        inverse (bool): If True, applies the inverse function.

    Returns:
        np.ndarray: Transformed data, still in range [epsilon, 1-epsilon].
    """
    a = (np.log((1 - epsilon) / epsilon)) ** (-1)
    b = epsilon - a * np.log(epsilon)
    
    if not inverse:
        return a * np.log(data) + b
    else:
        return np.exp((data - b) / a)


def flip(data: np.ndarray, epsilon: float, inverse: bool) -> np.ndarray:
    """
    Flip the data around its midpoint.

    Args:
        data (np.ndarray): Input data to be flipped.
        epsilon (float): Unused, included for consistency with other transformation functions.
        inverse (bool): Unused, the flip operation is its own inverse.

    Returns:
        np.ndarray: Flipped data.
    """
    return 1 - data
