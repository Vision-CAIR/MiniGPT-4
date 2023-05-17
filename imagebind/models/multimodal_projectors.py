from torch import nn, Tensor

from typing import Union, Optional, Tuple


class BaseProjector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class LinearProjector(BaseProjector):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


class AdapterProjector(BaseProjector):
    def __init__(self, in_dim, mid_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, mid_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, out_dim, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


def create_projectors(dims):
    if len(dims) == 0:
        return nn.Identity()
    elif len(dims) == 2:
        return LinearProjector(*dims)
    elif len(dims) == 3:
        return AdapterProjector(*dims)
    else:
        raise NotImplementedError
