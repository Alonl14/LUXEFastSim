from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class ParticleDataset(Dataset):
    def __init__(self, data_path, norm_path, QT, dataGroup):
        super().__init__()

        self.data = pd.read_csv(data_path)
        # self.data = self.data[self.data[' pdg'].isin([2112])]  # 22 - photons , 2112 - neutrons
        print(dataGroup)
        self.data[' rp'] = np.sqrt(self.data[' pxx'].values ** 2 + self.data[' pyy'].values ** 2)
        self.data[' phi_p'] = np.arctan2(self.data[' pyy'].values, self.data[' pxx'].values) + np.pi
        if dataGroup == 'outer':
            self.data[' rx'] = np.sqrt(self.data[' xx'].values ** 2 + self.data[' yy'].values ** 2)
            self.data[' phi_x'] = np.arctan2(self.data[' yy'].values, self.data[' xx'].values) + np.pi
            self.data = self.data[[" rx", " xx", " yy", " rp", " phi_p", " pzz", " eneg", " time"]]
        elif dataGroup == 'inner':
            self.data = self.data[[" xx", " yy", " rp", " phi_p", " pzz", " eneg", " time"]]

        self.norm = pd.read_csv(norm_path, index_col=0)

        epsilon = 10**(-12)  # Used to normalize all features to be in the range (0,1) non-inclusive.
        for col in self.data.columns:
            self.data[col] = (self.data[col] - self.norm['min'][col] + epsilon) / (self.norm['max'][col] - self.norm['min'][col] + 2 * epsilon)

        # store values before any transformation
        self.preprocess = self.data
        # The following transformation is only relevant to inner
        if dataGroup == 'inner':
            # Similar to y' = (y-0.83)**(3/7), makes sure we get the real root
            self.data[' yy'] = np.copysign(np.abs(self.data[' yy'] - 0.83) ** (3. / 7),
                                           self.data[' yy'] - 0.83)

        self.data[' pzz'] = 1 - self.data[' pzz']
        self.data[[' rp', ' pzz', ' eneg', ' time']] = -np.log(self.data[[' rp', ' pzz', ' eneg', ' time']])

        # store values before quantile transformation, the used quantiles, and the data itself
        self.preqt = self.data.values
        self.quantiles = QT.fit(self.data)
        self.data = QT.fit_transform(self.data)
        self.data = self.data.astype(np.float32)

    def __getitem__(self, item):
        return self.data[item, :]

    def __len__(self):
        return self.data.shape[0]

