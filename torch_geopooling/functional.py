from typing import Tuple

from torch import Tensor

import torch_geopooling._C as _C
import torch_geopooling.return_types as return_types


__all__ = ["quad_pool2d"]


def quad_pool2d(
    tiles: Tensor,
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    exterior: Tuple[float, ...],
    training: bool,
    max_depth: int | None = None,
    capacity: int | None = None,
    precision: int | None = None,
) -> Tuple[Tensor, Tensor]:
    tiles, weight, bias = _C.quad_pool2d(
        tiles, input, weight, bias, exterior, training, max_depth, capacity, precision
    )
    return return_types.quad_pool2d(tiles, weight, bias)
