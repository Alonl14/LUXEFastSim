import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, numFeatures):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(numFeatures, 150),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(150),
            nn.Linear(150, 200),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(200),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(200),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(200),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(200),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(100),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(100),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(100),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(100),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(100),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(100),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(50),
            nn.Linear(50, 50),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(50, 50),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(50, 10),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(10, 1, bias=False)
        )

    def forward(self, input):
        return self.main(input)

    def get_param_number(self):
        return sum(p.numel() for p in self.parameters())


class Discriminator2(nn.Module):
    def __init__(self, numFeatures):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(numFeatures, 256, bias=True),
            nn.InstanceNorm1d(256, affine=True),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512, bias=True),
            nn.InstanceNorm1d(512, affine=True),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256, bias=True),
            nn.InstanceNorm1d(256, affine=True),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1, bias=False)
        )

    def forward(self, input):
        return self.main(input)

    def get_param_number(self):
        return sum(p.numel() for p in self.parameters())

