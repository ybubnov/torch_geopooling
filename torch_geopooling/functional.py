from torch import Tensor

import torch_geopooling._C as _C
import torch_geopooling.return_types as return_types


__all__ = ["quad_pool2d"]


def quad_pool2d(
    tiles: Tensor,
    input: Tensor,
    weight: Tensor,
    exterior: tuple[float, ...],
    training: bool,
    max_depth: int | None = None,
    capacity: int | None = None,
    precision: int | None = None,
) -> tuple[Tensor, Tensor]:
    tiles, weight = _C.quad_pool2d(
        tiles, input, weight, exterior, training, max_depth, capacity, precision
    )
    return return_types.quad_pool2d(tiles, weight)
