import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, noiseDim, numFeatures, numParticleFeatures, embeddingDim):
        super().__init__()

        self.noiseDim = noiseDim
        self.numFeatures = numFeatures
        self.numParticleFeatures = numParticleFeatures
        self.embeddingDim = embeddingDim

        self.common = nn.Sequential(
            nn.Linear(self.noiseDim, 100, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100, affine=True),
            nn.Linear(100, 200, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200, affine=True),
            nn.Linear(200, 300, bias=False))

        self.embed = nn.Sequential(
            nn.Linear(self.numParticleFeatures, 8),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8, affine=True),
            nn.Linear(8, 8),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8, affine=True),
            nn.Linear(8, self.embeddingDim)
        )

        self.specialized = nn.Sequential(
            nn.Linear(300 + embeddingDim, 300, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(300, affine=True),
            nn.Linear(300, 300, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(300, affine=True),
            nn.Linear(300, 300, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(300, affine=True),
            nn.Linear(300, 300, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(300, affine=True),
            nn.Linear(300, 200, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200, affine=True),
            nn.Linear(200, 100, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100, affine=True),
            nn.Linear(100, 100, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100, affine=True),
            nn.Linear(100, 100, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100, affine=True),
            nn.Linear(100, 75, bias=False),
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

    def forward(self, netInput):
        x1, x2 = netInput
        common_output = self.common(x1)
        embedded_output = self.embed(x2)
        x = torch.cat((common_output, embedded_output), dim=1)
        return tuple([self.specialized(x), x2])

    def get_param_number(self):
        return sum(p.numel() for p in self.parameters())
