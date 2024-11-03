import torch.nn as nn


class FlowNet(nn.Module):
    def __init__(self, numFeatures):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(numFeatures + 1, 100, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100, affine=True),
            nn.Linear(100, 200, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200, affine=True),
            nn.Linear(200, 100, bias=False),
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
