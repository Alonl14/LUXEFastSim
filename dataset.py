import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import QuantileTransformer as qt
import copy

import utils
from pipeline import qt_cache


class ParticleDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.preprocess = None
        self.preqt = None
        self.quantiles = None
        self._registry = {"log": simple_log,
                          "flip": flip}
        self.cfg = cfg

        self.data = pd.read_csv(cfg['data_path'])
        utils.add_features(self.data, cfg["pdg"])
        print(f"Creating dataset w. shape: {self.data.shape}")
        self.data = self.data[cfg["features"].keys()]
        self.preprocess = self.data.copy()
        self.apply_transformation(cfg)

        # mps doesn't work with double-percision floats, cuda does
        data_type = np.float32

        # Fit the QuantileTransformer once, caching it to disk: the fit is the
        # expensive part of dataset prep and is identical across runs/analyses
        # of the same data. A single fit (then transform) also avoids the old
        # double-fit, where the stored QT differed from the one that transformed
        # the data because subsample() draws a fresh random subset each fit.
        QT = qt_cache.load_qt(cfg)
        if QT is None:
            QT = qt(output_distribution='normal', n_quantiles=cfg['nQuantiles'], subsample=cfg['subsample'])
            QT.fit(self.data)
            qt_cache.save_qt(QT, cfg)
            print(f"Fitted and cached QuantileTransformer for region {cfg['dataGroup']}")
        else:
            print(f"Loaded cached QuantileTransformer for region {cfg['dataGroup']}")
        self.quantiles = QT
        self.preqt = self.data.values
        self.data = QT.transform(self.data).astype(data_type)

    @property
    def registry(self):
        return self._registry

    def apply_transformation(self, cfg, inverse=False):
        """
        Applies transformations to data before feeding it to QT / training
        :param cfg: config for that specific run
        :param inverse: if True applies the inverse transformations listed in cfg
        :return:
        """
        # make all transformations, if inverse flip the function list since (f(g(x)))^-1 = g^-1(f^-1(x))
        for feature, function_list in cfg["features"].items():
            functions = function_list[::-1] if inverse else function_list[:]
            for f in functions:
                self.data[feature] = self.registry[f](self.data[feature], inverse)

    def __getitem__(self, item):
        return self.data[item, :]

    def __len__(self):
        return self.data.shape[0]


def simple_log(data, inverse):
    """
    applies log without scale

    :param data: ASSUMES DATA IS IN [eps, 1-eps]
    :param epsilon: should be the same as in normalize
    :param inverse: if true applies the inverse function
    :return: log(a*data+b) with a,b s.t. result is still in [eps, 1-eps]
    """

    return np.log(data) if not inverse else np.exp(data)


def flip(data, inverse):
    return -1 * data

