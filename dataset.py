import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ParticleDataset(Dataset):
    def __init__(self, data_path, norm_path, QT, dataGroup):
        super().__init__()

        self.data = pd.read_csv(data_path)
        if ' pdg' in self.data.columns.values:
            self.data = self.data[self.data[' pdg'].isin([2112])]  # 22 - photons , 2112 - neutrons
        self.data[' rp'] = np.sqrt(self.data[' pxx'].values ** 2 + self.data[' pyy'].values ** 2)
        self.data[' phi_p'] = np.arctan2(self.data[' pyy'].values, self.data[' pxx'].values) + np.pi
        if dataGroup == 'inner':
            self.data = self.data[[" xx", " yy", " pxx", " pyy", " rp", " phi_p", " pzz", " eneg", " time"]]  #
        if dataGroup == 'outer':
            self.data[' rx'] = np.sqrt(self.data[' xx'].values ** 2 + self.data[' yy'].values ** 2)
            self.data = self.data[[" rx", " xx", " yy", " pxx", " pyy", " rp", " phi_p", " pzz", " eneg", " time"]]

        self.norm = pd.read_csv(norm_path, index_col=0)

        epsilon = 10**(-16)  # Used to normalize all features to be in the range (eps,1-eps) non-inclusive.

        for col in self.data.columns:
            x_max = self.norm['max'][col]
            x_min = self.norm['min'][col]
            if x_min == 0:
                b = epsilon
                a = (1-2*epsilon)/x_max
            else:
                b = (1-epsilon*(1+x_max/x_min))/(1-x_max/x_min)
                a = (epsilon-b)/x_min
            self.data[col] = a * self.data[col] + b

        # store values before any transformation
        self.preprocess = self.data.copy()

        self.data[' pzz'] = 1 - self.data[' pzz']  #
        self.data[[' pzz', ' rp', ' eneg', ' time']] = -np.log(self.data[[' pzz', ' rp', ' eneg', ' time']])

        if dataGroup == 'inner':
            # Similar to y' = (y-0.83)**(5/7), makes sure we get the real root
            # self.data[' yy'] = np.copysign(np.abs(self.data[' yy'] - 0.83) ** (5. / 9),
            #                                self.data[' yy'] - 0.83)
            self.data[' yy'] = np.arctan(10*(self.data[' yy']-0.83))
            self.data[' xx'] = np.copysign(np.abs(self.data[' xx'] - 0.73) ** (5. / 9),
                                           self.data[' xx'] - 0.73)

        # if dataGroup == 'outer':
        #     self.data[[' xx', ' yy']] = self.data[[' xx', ' yy']] - 0.55
        #     self.data[' xx'], self.data[' yy'] = self.data[' yy'], -self.data[' xx']
        #     self.data[' rx'] = np.sqrt(self.data[' xx'].values ** 2 + self.data[' yy'].values ** 2)
        #     self.data[' phi_x'] = np.arctan2(self.data[' yy'].values, self.data[' xx'].values)/2
        #     self.data[' xx'], self.data[' yy'] = (self.data[' rx']*np.cos(self.data[' phi_x']),
        #                                           self.data[' rx']*np.sin(self.data[' phi_x']))
        #     self.data[' eneg'] = np.copysign(np.abs(self.data[' eneg']) ** (1 / 9),
        #                                      self.data[' eneg'])
        #     self.data[' eneg'] = np.copysign(np.abs(self.data[' eneg']-0.3)**(1/5),
        #                                      self.data[' eneg']-0.3)
        #     self.data = self.data[[" rx", " xx", " yy", " rp", " phi_p", " pzz", " eneg", " time"]]

        # store values before quantile transformation, the used quantiles, and the data itself
        self.preqt = self.data.values
        self.quantiles = QT.fit(self.data)
        self.data = QT.fit_transform(self.data)
        self.data = self.data.astype(np.float32)

    def __getitem__(self, item):
        return self.data[item, :]

    def __len__(self):
        return self.data.shape[0]

