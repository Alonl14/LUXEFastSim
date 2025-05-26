import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import QuantileTransformer as qt
import copy


class ParticleDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.preprocess = None
        self.preqt = None
        self.quantiles = None
        self._registry = {"log": simple_log,
                          "flip": flip}
        self.cfg = cfg
        QT = qt(output_distribution='uniform', n_quantiles=cfg['nQuantiles'], subsample=cfg['subsample'])

        self.data = pd.read_csv(cfg['data_path'])
        print(f"Creating dataset w. shape: {self.data.shape}")
        self.data = self.data[cfg["features"].keys()]
        self.preprocess = self.data.copy()
        self.apply_transformation(cfg)

        # mps doesn't work with double-percision floats, cuda does
        data_type = np.float32

        self.quantiles = QT.fit(self.data)
        self.preqt = self.data.values
        self.data = QT.fit_transform(self.data)
        self.data = self.data.astype(data_type)

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

