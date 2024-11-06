import torch.nn as nn
import torch


class Critic(nn.Module):
    def __init__(self, numFeatures, numParticleFeatures, embeddingDim):
        super().__init__()

        self.numFeatures = numFeatures
        self.particleFeatures = numParticleFeatures
        self.embeddingDim = embeddingDim

        self.common = nn.Sequential(
            nn.Linear(self.numFeatures, 100),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(100, 200),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(200, 300)
        )

        self.embed = nn.Sequential(
            nn.Linear(self.particleFeatures, 8),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8, affine=True),
            nn.Linear(8, 8),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8, affine=True),
            nn.Linear(8, self.embeddingDim)
        )

        self.specialized = nn.Sequential(
            nn.Linear(300 + self.embeddingDim, 300),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(300, 300),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(300, 300),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(300, 200),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(200, 200),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
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

    def forward(self, data):
        x1, x2 = data
        common_output = self.common(x1)
        embedded_output = self.embed(x2)
        x = torch.cat((common_output, embedded_output), dim=1)
        return tuple(self.specialized(x), x2)

    def get_param_number(self):
        return sum(p.numel() for p in self.parameters())
