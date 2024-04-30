from typing import Optional
from typing import Tuple

from torch import Tensor

import torch_geopooling._C as _C
import torch_geopooling.return_types as return_types


__all__ = ["max_quad_pool2d", "quad_pool2d"]


def quad_pool2d(
    tiles: Tensor,
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    exterior: Tuple[float, ...],
    *,
    training: bool = True,
    max_depth: Optional[int] = None,
    capacity: Optional[int] = None,
    precision: Optional[int] = None,
) -> return_types.quad_pool2d:
    tiles, weight, bias = _C.quad_pool2d(
        tiles, input, weight, bias, exterior, training, max_depth, capacity, precision
    )
    return return_types.quad_pool2d(tiles, weight, bias)


def max_quad_pool2d(
    tiles: Tensor,
    input: Tensor,
    weight: Tensor,
    exterior: Tuple[float, ...],
    *,
    training: bool = True,
    max_depth: Optional[int] = None,
    capacity: Optional[int] = None,
    precision: Optional[int] = None,
) -> return_types.max_quad_pool2d:
    tiles, weight = _C.max_quad_pool2d(
        tiles, input, weight, exterior, training, max_depth, capacity, precision
    )
    return return_types.max_quad_pool2d(tiles, weight)
