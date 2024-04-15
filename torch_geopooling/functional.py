from torch import Tensor

import torch_geopooling._C as _C


def quad_pool2d(
    tiles: Tensor,
    input: Tensor,
    weight: Tensor,
    exterior: tuple[float, float, float, float],
    training: bool,
    max_depth: int | None = None,
    capacity: int | None = None,
    precision: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _C.quad_pool2d(tiles, input, weight, exterior)
