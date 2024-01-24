import torch.nn as nn


class InnerGenerator(nn.Module):
    def __init__(self, noiseDim):
        super().__init__()

        self.noiseDim = noiseDim

        self.main = nn.Sequential(
            nn.Linear(self.noiseDim, 50, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50, affine=True),
            nn.Linear(50, 25, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(25, affine=True),
            nn.Linear(25, 11, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(11, affine=True),
            nn.Linear(11, 7)
        )

    def forward(self, input):
        return self.main(input)

    def get_param_number(self):
        return (sum(p.numel() for p in self.parameters()))


class OuterGenerator(nn.Module):
    def __init__(self, noiseDim):
        super().__init__()

        self.noiseDim = noiseDim

        self.main = nn.Sequential(
            nn.Linear(self.noiseDim, 50, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50, affine=True),
            nn.Linear(50, 25, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(25, affine=True),
            nn.Linear(25, 11, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(11, affine=True),
            nn.Linear(11, 8)
        )

    def forward(self, input):
        return self.main(input)

    def get_param_number(self):
        return sum(p.numel() for p in self.parameters())