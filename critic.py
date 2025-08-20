# critic.py – cfg-driven WGAN critic (a.k.a. discriminator) for tabular features
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
    a = (name or "lrelu").lower()
    if a in ("lrelu", "leakyrelu", "leaky_relu"):
        return nn.LeakyReLU(0.2, inplace=True)
    if a == "relu":
        return nn.ReLU(inplace=True)
    if a == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation '{name}'")


class Critic(nn.Module):
    """
    Hidden sizes controlled by cfg["criticLayers"] list.
    """
    def __init__(
        self,
        numFeatures: int,
        hidden_dims=None,
        norm: str = "layer",
        activation: str = "lrelu",
        bias_hidden: bool = True,
        dropout: float = 0.0,
        out_bias: bool = False,
    ):
        super().__init__()
        hidden_dims = list(hidden_dims or [512, 512, 512, 512])

        layers = []
        in_dim = numFeatures
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h, bias=bias_hidden))
            n = _norm(norm, h)
            if not isinstance(n, nn.Identity):
                layers.append(n)
            layers.append(_act(activation))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1, bias=out_bias))  # scalar score
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

    def get_param_number(self) -> int:
        return sum(p.numel() for p in self.parameters())
