import copy

import torch
import torch.nn as nn

from trainer import ema_update


def test_ema_update_math():
    live = nn.Linear(3, 2, bias=False)
    ema = copy.deepcopy(live)
    with torch.no_grad():
        for p in live.parameters():
            p.copy_(torch.ones_like(p))
        for p in ema.parameters():
            p.copy_(torch.zeros_like(p))
    ema_update(ema, live, decay=0.9)
    for p in ema.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.1), atol=1e-6)


def test_ema_copies_buffers():
    live = nn.BatchNorm1d(4)
    ema = copy.deepcopy(live)
    with torch.no_grad():
        live.running_mean.copy_(torch.arange(4, dtype=torch.float32))
    ema_update(ema, live, decay=0.99)
    assert torch.allclose(ema.running_mean, live.running_mean)
