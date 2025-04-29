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
        self.linear1 = nn.Linear(numFeatures, 256, bias=True)
        self.norm1 = nn.InstanceNorm1d(256, affine=True)

        self.linear2 = nn.Linear(256, 512, bias=True)
        self.norm2 = nn.InstanceNorm1d(512, affine=True)

        self.linear3 = nn.Linear(512, 256, bias=True)
        self.norm3 = nn.InstanceNorm1d(256, affine=True)

        self.linear4 = nn.Linear(256, 1, bias=False)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.linear1(x)
        x = x.unsqueeze(2)
        x = self.norm1(x).squeeze(2)
        x = self.act(x)

        x = self.linear2(x)
        x = x.unsqueeze(2)
        x = self.norm2(x).squeeze(2)
        x = self.act(x)

        x = self.linear3(x)
        x = x.unsqueeze(2)
        x = self.norm3(x).squeeze(2)
        x = self.act(x)

        x = self.linear4(x)
        return x

    def get_param_number(self):
        return sum(p.numel() for p in self.parameters())


