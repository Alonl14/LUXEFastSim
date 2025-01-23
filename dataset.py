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
        self._registry = {"log": my_log,
                          "flip": flip}
        self.cfg = cfg
        QT = qt(output_distribution='normal', n_quantiles=cfg['nQuantiles'], subsample=cfg['subsample'])

        self.data = pd.read_csv(cfg['data_path'])
        print(self.data.shape)

        if ' pdg' in self.data.columns.values:
            self.data = self.data[self.data[' pdg'].isin([cfg['pdg']])]  # 22 - photons , 2112 - neutrons
        self.data = self.data[(self.data[' time'] <= 10**6) & (self.data[' pzz'] <= 0)]
        print("time cut 10^6")

        self.data[' rx'] = np.sqrt(self.data[' xx'].values ** 2 + self.data[' yy'].values ** 2)
        self.data[' rp'] = np.sqrt(self.data[' pxx'].values ** 2 + self.data[' pyy'].values ** 2)
        self.data[' phi_p'] = np.arctan2(self.data[' pyy'].values, self.data[' pxx'].values) + np.pi

        self.data = self.data[cfg["features"].keys()]

        self.preprocess = self.data.copy()

        self.norm = pd.read_csv(cfg['norm_path'], index_col=0)
        self.norm['max'][' time'] = 10**6
        self.norm['max'][' pzz'] = 0
        self.apply_transformation(cfg)

        # mps doesn't work with double-percision floats, cuda does
        data_type = np.float32
        # store quantiles for inverse (applyQT=0) or apply to data (applyQT=1)
        print(np.min(self.data, axis=0))
        self.quantiles = QT.fit(self.data)

        # store values before quantile transformation, the used quantiles, and the data itself
        if cfg['applyQT']:
            self.preqt = self.data.values
            self.data = QT.fit_transform(self.data)
            self.data = self.data.astype(data_type)
        else:
            self.data = self.data.values.astype(data_type)

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

        eps = cfg["epsilon"]

        # make all transformations, if inverse flip the function list since (f(g(x)))^-1 = g^-1(f^-1(x))
        for feature, function_list in cfg["features"].items():
            x_max = self.norm['max'][feature]
            x_min = self.norm['min'][feature]
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


def normalize(data, data_min, data_max, epsilon, inverse):
    """
    normalize a feature to be in the range (eps,1-eps) OR the inverse

    :param data: feature to be normalized
    :param data_min: min(data) OR if inverse, should be taken from norm file
    :param data_max: max(data) OR if inverse, should be taken from norm file
    :param epsilon: data will be scaled to the range [eps, 1-eps], MUST be the smallest scale in the dataset!
    :param inverse: if True, assumes data is in [eps, 1-eps] and scales it to be in [data_min, data_max]
    :return: normalized feature
    """
    if data_min == 0:
        b = epsilon
        a = (1 - 2 * epsilon) / data_max
    else:
        b = (1 - epsilon * (1 + data_max / data_min)) / (1 - data_max / data_min)
        a = (epsilon - b) / data_min

    return a * data + b if not inverse else (data - b) / a


def my_log(data, epsilon, inverse):
    """
    applies log with scale, derivation in OneNote -> FastSim -> Code Clean-up

    :param data: ASSUMES DATA IS IN [eps, 1-eps]
    :param epsilon: should be the same as in normalize
    :param inverse: if true applies the inverse function
    :return: log(a*data+b) with a,b s.t. result is still in [eps, 1-eps]
    """

    a = (np.log((1 - epsilon) / epsilon)) ** (-1)
    b = epsilon - a * np.log(epsilon)
    return a * np.log(data) + b if not inverse else np.exp((data - b) / a)


def flip(data, epsilon, inverse):
    return 1 - data