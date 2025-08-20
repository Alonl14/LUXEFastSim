# generator.py – cfg-driven MLP generator for tabular features
import torch.nn as nn


def _norm(norm: str, dim: int):
    n = (norm or "layer").lower()
    if n == "layer":
        return nn.LayerNorm(dim)
    if n == "batch":
        return nn.BatchNorm1d(dim, affine=True)
    if n == "none":
        return nn.Identity()
    raise ValueError(f"Unknown norm '{norm}'")


def _act(name: str):
    a = (name or "relu").lower()
    if a == "relu":
        return nn.ReLU(inplace=True)
    if a in ("lrelu", "leakyrelu", "leaky_relu"):
        return nn.LeakyReLU(0.2, inplace=True)
    if a == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation '{name}'")


class Generator(nn.Module):
    """
    Hidden sizes controlled by cfg["genLayers"] list.
    """
    def __init__(
        self,
        noiseDim: int,
        numFeatures: int,
        hidden_dims=None,
        norm: str = "layer",
        activation: str = "relu",
        bias_last: bool = True,
        bias_hidden: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.noiseDim = int(noiseDim)
        hidden_dims = list(hidden_dims or [512, 1024, 1024, 512])

        layers = []
        in_dim = self.noiseDim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h, bias=bias_hidden))
            n = _norm(norm, h)
            if not isinstance(n, nn.Identity):
                layers.append(n)
            layers.append(_act(activation))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, numFeatures, bias=bias_last))
        self.main = nn.Sequential(*layers)

    def forward(self, z):
        return self.main(z)

    def get_param_number(self) -> int:
        return sum(p.numel() for p in self.parameters())
