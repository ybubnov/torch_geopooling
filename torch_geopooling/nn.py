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

from typing import Optional, Union

import torch
from shapely.geometry import Polygon
from torch import Tensor, nn

from torch_geopooling import functional as F
from torch_geopooling.tiling import Exterior, ExteriorTuple, regular_tiling

__all__ = [
    "AdaptiveQuadPool2d",
    "AdaptiveMaxQuadPool2d",
    "AvgQuadPool2d",
    "MaxQuadPool2d",
    "QuadPool2d",
]


_Exterior = Union[Exterior, ExteriorTuple]


class _AdaptiveQuadPool(nn.Module):
    def __init__(
        self,
        num_features: int,
        feature_dim: int,
        exterior: _Exterior,
        max_depth: int = 17,
        capacity: int = 1,
        precision: Optional[int] = 7,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.exterior = tuple(map(float, exterior))
        self.max_depth = max_depth
        self.capacity = capacity
        self.precision = precision

        self.initialize_parameters()
        self.reset_parameters()

    def initialize_parameters(self) -> None:
        weight_size = [self.num_features, self.feature_dim]
        self.weight = nn.Parameter(torch.empty(weight_size, dtype=torch.float64))

        self.register_buffer("tiles", torch.empty((0, 3), dtype=torch.int32))
        self.tiles: Tensor

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight)
        self.tiles.new_empty((0, 3))

    def extra_repr(self) -> str:
        return (
            "{num_features}, {feature_dim}, "
            "exterior={exterior}, capacity={capacity}, max_depth={max_depth}, "
            "precision={precision}".format(**self.__dict__)
        )


class AdaptiveQuadPool2d(_AdaptiveQuadPool):
    """Dynamic lookup index over Quadtree decomposition of input 2D coordinates.

    This module constructs an internal lookup quadtree to organize closely situated 2D points.
    Each terminal node in the resulting quadtree is paired with a weight. Thus, when providing
    an input coordinate, the module retrieves the corresponding terminal node and returns its
    associated weight.

    Args:
        num_features: Number of features. Equals to the number of terminal nodes in the quadtree.
        feature_dim: Size of each feature vector.
        exterior: Geometrical boundary of the learning space in (X, Y, W, H) format.
        max_depth: Maximum depth of the quadtree. Default: 17.
        capacity: Maximum number of inputs, after which a quadtree's node is subdivided and
            depth of the tree grows. Default: 1.
        precision: Optional rounding of the input coordinates. Default: 7.

    Examples:

        >>> # 48 feature vectors of size 4 over a 2d space.
        >>> pool = nn.LinearQuadPool2d(48, 4, (-10, -5, 20, 10))
        >>> # Grow tree up to 4-th level and sub-divides a node after 8 coordinates in a quad.
        >>> pool = nn.LinearQuadPool2d(48, 4, (-10, -5, 20, 10), max_depth=4, capacity=8)
        >>> # Create 2D coordinates and query associated weights.
        >>> input = torch.rand((1024, 2), dtype=torch.float64) * 10 - 5
        >>> output = pool(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        result = F.quad_pool2d(
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
        return result.weight


class AdaptiveMaxQuadPool2d(_AdaptiveQuadPool):
    """Dynamic maximum pooling over Quadtree decomposition of input 2D coordinates.

    This module constructs an internal lookup quadtree to organize closely situated 2D points.
    Each terminal node in the resulting quadtree is assigned a weight value. For each input
    coordinate, the module queries a "terminal group" of nodes and calculates the maximum value
    from a `weight` vector associated with these nodes.

    A terminal group refers to a collection of terminal nodes within the quadtree that share the
    same parent as the input coordinate.

    Args:
        num_features: Number of features, or terminal nodes, in the Quadtree decomposition.
        feature_dim: Size of each feature vector.
        exterior: Geometrical boundary of the learning space in (X, Y, W, H) format.
        max_depth: Maximum depth of the quadtree. Default: 17.
        capacity: Maximum number of inputs, after which a quadtree's node is subdivided and
            depth of the tree grows. Default: 1.
        precision: Optional rounding of the input coordinates. Default: 7.

    Examples:

        >>> pool = nn.MaxQuadPool2d(256, 3, (-10, -5, 20, 10), max_depth=5)
        >>> # Create 2D coordinates and feature vector associated with them.
        >>> input = torch.rand((2048, 2), dtype=torch.float64) * 10 - 5
        >>> output = pool(input)
    """

    def forward(self, input: Tensor) -> Tensor:
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
        return result.weight


class _QuadPool(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        polygon: Polygon,
        exterior: _Exterior,
        max_depth: int = 17,
        precision: Optional[int] = 7,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.polygon = polygon
        self.exterior = tuple(map(float, exterior))
        self.max_depth = max_depth
        self.precision = precision

        # Generate regular tiling for the provided polygon and build from those
        # tiles a Quadtree from terminal nodes all way up to the root node.
        tiles_iter = regular_tiling(
            polygon, Exterior.from_tuple(exterior), z=max_depth, internal=True
        )
        tiles = torch.tensor(list(tiles_iter), dtype=torch.int32)

        self.register_buffer("tiles", tiles)
        self.tiles: Tensor

        self.initialize_parameters()
        self.reset_parameters()

    @property
    def num_features(self) -> int:
        return (self.tiles[:, 0] == self.max_depth).sum().item()

    def initialize_parameters(self) -> None:
        weight_size = [self.num_features, self.feature_dim]
        self.weight = nn.Parameter(torch.empty(weight_size, dtype=torch.float64))

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight)

    def extra_repr(self) -> str:
        return (
            "{feature_dim}, exterior={exterior}, max_depth={max_depth}, "
            "precision={precision}".format(**self.__dict__)
        )


class QuadPool2d(_QuadPool):
    def forward(self, input: Tensor) -> Tensor:
        result = F.quad_pool2d(
            self.tiles,
            input,
            self.weight,
            self.exterior,
            # This is not a mistake, since we already know the shape of the
            # quadtree, there is no need to learn it.
            training=False,
            max_depth=self.max_depth,
            precision=self.precision,
        )
        return result.weight


class MaxQuadPool2d(_QuadPool):
    def forward(self, input: Tensor) -> Tensor:
        result = F.max_quad_pool2d(
            self.tiles,
            input,
            self.weight,
            self.exterior,
            training=False,
            max_depth=self.max_depth,
            precision=self.precision,
        )
        return result.weight


class AvgQuadPool2d(_QuadPool):
    def initialize_parameters(self) -> None:
        self.weight = nn.Parameter(torch.empty([self.num_features], dtype=torch.float64))

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        result = F.avg_quad_pool2d(
            self.tiles,
            input,
            self.weight,
            self.exterior,
            training=False,
            max_depth=self.max_depth,
            precision=self.precision,
        )
        return result.weight
