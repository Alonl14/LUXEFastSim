import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noiseDim, numFeatures):
        super().__init__()

        self.noiseDim = noiseDim

        self.main = nn.Sequential(
            nn.Linear(self.noiseDim, 300, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(300, affine=True),
            nn.Linear(300, 100, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100, affine=True),
            nn.Linear(100, 75, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(75, affine=True),
            nn.Linear(75, 75, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(75, affine=True),
            nn.Linear(75, 75, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(75, affine=True),
            nn.Linear(75, 50, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50, affine=True),
            nn.Linear(50, 50, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50, affine=True),
            nn.Linear(50, 50, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50, affine=True),
            nn.Linear(50, 50, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50, affine=True),
            nn.Linear(50, 50, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50, affine=True),
            nn.Linear(50, numFeatures)
        )

    def forward(self, input):
        return self.main(input)

    def get_param_number(self):
        return sum(p.numel() for p in self.parameters())


class Generator2(nn.Module):
    def __init__(self, noiseDim, numFeatures):
        super().__init__()
        self.noiseDim = noiseDim
        self.main = nn.Sequential(
            nn.Linear(noiseDim, 512, bias=False),
            nn.BatchNorm1d(512, affine=True),
            nn.ReLU(),
            nn.Linear(512, 1024, bias=False),
            nn.BatchNorm1d(1024, affine=True),
            nn.ReLU(),
            nn.Linear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024, affine=True),
            nn.ReLU(),
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512, affine=True),
            nn.ReLU(),
            nn.Linear(512, numFeatures, bias=True)  # Only layer with bias
        )

    def forward(self, input):
        return self.main(input)

    def get_param_number(self):
        return sum(p.numel() for p in self.parameters())
