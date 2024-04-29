from typing import Tuple

import torch
from torch import nn
from torch import Tensor

from torch_geopooling import functional as F


__all__ = ["MaxQuadPool2d", "QuadPool2d"]


class _QuadPool(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        exterior: Tuple[float, float, float, float],
        max_depth: int = 17,
        capacity: int = 1,
        precision: int = 7,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.exterior = tuple(map(float, exterior))
        self.max_depth = max_depth
        self.capacity = capacity
        self.precision = precision

        self.initialize_parameters()
        self.reset_parameters()

    def initialize_parameters(self) -> None:
        raise NotImplementedError()

    def reset_parameter(self) -> None:
        raise NotImplementedError()

    def extra_repr(self) -> str:
        return (
            "{kernel_size}, exterior={exterior}, capacity={capacity}, max_depth={max_depth}, "
            "precision={precision}".format(**self.__dict__)
        )



class QuadPool2d(_QuadPool):

    def initialize_parameters(self) -> None:
        self.weight = nn.Parameter(torch.ones([self.kernel_size], dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros([self.kernel_size], dtype=torch.float64))

        self.register_buffer("tiles", torch.empty((0, 3), dtype=torch.int32))
        self.tiles: Tensor

    def reset_parameters(self) -> None:
        self.tiles.new_empty((0, 3))
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input: Tensor, x: Tensor) -> Tensor:
        result = F.quad_pool2d(
            self.tiles,
            input,
            self.weight,
            self.bias,
            self.exterior,
            self.training,
            self.max_depth,
            self.capacity,
            self.precision,
        )
        if self.training:
            self.tiles = result.tiles
        return result.weight * x + result.bias


class MaxQuadPool2d(_QuadPool):

    def initialize_parameters(self) -> None:
        self.weight = nn.Parameter(torch.ones([self.kernel_size], dtype=torch.float64))

        self.register_buffer("tiles", torch.empty((0, 3), dtype=torch.int32))
        self.tiles: Tensor

    def reset_parameters(self) -> None:
        self.tiles.new_empty((0, 3))
        nn.init.ones_(self.weight)

    def forward(self, input: Tensor, x: Tensor) -> Tensor:
        result = F.max_quad_pool2d(
            self.tiles,
            input,
            self.weight,
            self.exterior,
            self.training,
            self.max_depth,
            self.capacity,
            self.precision,
        )
        if self.training:
            self.tiles = result.tiles
        return result.weight * x
