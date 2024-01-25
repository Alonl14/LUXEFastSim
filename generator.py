import torch.nn as nn


class InnerGenerator(nn.Module):
    def __init__(self, noiseDim):
        super().__init__()

        self.noiseDim = noiseDim

        self.main = nn.Sequential(
            nn.Linear(self.noiseDim, 300, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(300, affine=True),
            nn.Linear(300, 200, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200, affine=True),
            nn.Linear(200, 150, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(150, affine=True),
            nn.Linear(150, 150, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(150, affine=True),
            nn.Linear(150, 100, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100, affine=True),
            nn.Linear(100, 50, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50, affine=True),
            nn.Linear(50, 7)
        )

    def forward(self, input):
        return self.main(input)

    def get_param_number(self):
        return sum(p.numel() for p in self.parameters())


class OuterGenerator(nn.Module):
    def __init__(self, noiseDim):
        super().__init__()

        self.noiseDim = noiseDim

        self.main = nn.Sequential(
            nn.Linear(self.noiseDim, 300, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(300, affine=True),
            nn.Linear(300, 200, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200, affine=True),
            nn.Linear(200, 150, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(150, affine=True),
            nn.Linear(150, 150, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(150, affine=True),
            nn.Linear(150, 100, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100, affine=True),
            nn.Linear(100, 50, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50, affine=True),
            nn.Linear(50, 8)
        )

    def forward(self, input):
        return self.main(input)

    def get_param_number(self):
        return sum(p.numel() for p in self.parameters())