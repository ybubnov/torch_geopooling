# Copyright (C) 2024, Yakau Bubnou
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from torch_geopooling import functional as F
from torch_geopooling.tiling import Exterior

__all__ = ["LinearQuadPool2d", "MaxQuadPool2d"]


class _QuadPool(nn.Module):
    def __init__(
        self,
        num_features: int,
        exterior: Exterior,
        max_depth: int = 17,
        capacity: int = 1,
        precision: Optional[int] = 7,
    ) -> None:
        super().__init__()
        self.num_features = num_features
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
            "num_features={num_features}, exterior={exterior}, capacity={capacity}, "
            "max_depth={max_depth}, precision={precision}".format(**self.__dict__)
        )


class LinearQuadPool2d(_QuadPool):
    """Applies linear transformations over Quadtree decomposition of input 2D coordinates.

    This module constructs an internal lookup quadtree to organize closely situated 2D points.
    Each terminal node in the resulting quadtree is paired with a weight and bias. Thus, when
    providing an input coordinate, the module retrieves the corresponding terminal node and
    returns its associated weight and bias. Then module applies a linear transformation to each
    input coordinate.

    Args:
        num_features: Number of features (linear transformations). Equals to the number of
            terminal nodes in the quadtree.
        exterior: Geometrical boundary of the learning space in (X, Y, W, H) format.
        max_depth: Maximum depth of the quadtree. Default: 17.
        capacity: Maximum number of inputs, after which a quadtree's node is subdivided and
            depth of the tree grows. Default: 1.
        precision: Optional rounding of the input coordinates. Default: 7.

    Examples:

        >>> # 48 linear transformations over a 2d space.
        >>> pool = nn.LinearQuadPool2d(48, (-10, -5, 20, 10))
        >>> # Grow tree up to 4-th level and sub-divides a node after 8 coordinates in a quad.
        >>> pool = nn.LinearQuadPool2d(48, (-10, -5, 20, 10), max_depth=4, capacity=8)
        >>> # Create 2D coordinates and feature vector associated with them.
        >>> input = torch.rand((1024, 2), dtype=torch.float64) * 10 - 5
        >>> x = torch.randn((1024,), dtype=torch.float64)
        >>> output = pool(input, x)
    """

    def initialize_parameters(self) -> None:
        self.weight = nn.Parameter(torch.ones([self.num_features], dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros([self.num_features], dtype=torch.float64))

        self.register_buffer("tiles", torch.empty((0, 3), dtype=torch.int32))
        self.tiles: Tensor

    def reset_parameters(self) -> None:
        self.tiles.new_empty((0, 3))
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input: Tensor, x: Tensor) -> Tensor:
        result = F.linear_quad_pool2d(
            self.tiles,
            input,
            self.weight,
            self.bias,
            self.exterior,
            training=self.training,
            max_depth=self.max_depth,
            capacity=self.capacity,
            precision=self.precision,
        )
        if self.training:
            self.tiles = result.tiles
        return result.weight * x + result.bias


class MaxQuadPool2d(_QuadPool):
    """Applies maximum pooling over Quadtree decomposition of input 2D coordinates.

    This module constructs an internal lookup quadtree to organize closely situated 2D points.
    Each terminal node in the resulting quadtree is assigned a weight value. For each input
    coordinate, the module queries a "terminal group" of nodes and calculates the maximum value
    from a `weight` vector associated with these nodes.

    The module then returns the product of the maximum weight and the input feature associated
    with the input coordinate.

    A terminal group refers to a collection of terminal nodes within the quadtree that share the
    same parent as the input coordinate.

    Args:
        num_features: Number of features, or terminal nodes, in the Quadtree decomposition.
        exterior: Geometrical boundary of the learning space in (X, Y, W, H) format.
        max_depth: Maximum depth of the quadtree. Default: 17.
        capacity: Maximum number of inputs, after which a quadtree's node is subdivided and
            depth of the tree grows. Default: 1.
        precision: Optional rounding of the input coordinates. Default: 7.

    Examples:

        >>> pool = nn.MaxQuadPool2d(256, (-10, -5, 20, 10), max_depth=5)
        >>> # Create 2D coordinates and feature vector associated with them.
        >>> input = torch.rand((2048, 2), dtype=torch.float64) * 10 - 5
        >>> x = torch.randn((1024,), dtype=torch.float64)
        >>> output = pool(input, x)
    """

    def initialize_parameters(self) -> None:
        self.weight = nn.Parameter(torch.ones([self.num_features], dtype=torch.float64))

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
            training=self.training,
            max_depth=self.max_depth,
            capacity=self.capacity,
            precision=self.precision,
        )
        if self.training:
            self.tiles = result.tiles
        return result.weight * x
