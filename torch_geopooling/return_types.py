from typing import NamedTuple

from torch import Tensor


class quad_pool2d(NamedTuple):
    tiles: Tensor
    weight: Tensor
