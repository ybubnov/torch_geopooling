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
    "AdaptiveAvgQuadPool2d",
    "AdaptiveQuadPool2d",
    "AdaptiveMaxQuadPool2d",
    "AvgQuadPool2d",
    "MaxQuadPool2d",
    "QuadPool2d",
]


_Exterior = Union[Exterior, ExteriorTuple]


_exterior_doc = """
    Note:
        Input coordinates must be within a specified exterior geometry (including boundaries).
        For input coordinates outsize of the specified exterior, module throws an exception.
"""


_terminal_group_doc = """
    Note:
        A **terminal group** refers to a collection of terminal nodes within the quadtree that
        share the same parent tile.
"""


class _AdaptiveQuadPool(nn.Module):
    __doc__ = f"""
    Args:
        feature_dim: Size of each feature vector.
        exterior: Geometrical boundary of the learning space in (X, Y, W, H) format.
        max_terminal_nodes: Optional maximum number of terminal nodes in a quadtree. Once a
            maximum is reached, internal nodes are no longer sub-divided and tree stops growing.
        max_depth: Maximum depth of the quadtree. Default: 17.
        capacity: Maximum number of inputs, after which a quadtree's node is subdivided and
            depth of the tree grows. Default: 1.
        precision: Optional rounding of the input coordinates. Default: 7.

    Shape:
        - Input: :math:`(*, 2)`, where 2 comprises longitude and latitude coordinates.
        - Output: :math:`(*, H)`, where * is the input shape and :math:`H = \\text{{feature_dim}}`.

    {_exterior_doc}
    {_terminal_group_doc}
    """

    def __init__(
        self,
        feature_dim: int,
        exterior: _Exterior,
        max_terminal_nodes: Optional[int] = None,
        max_depth: int = 17,
        capacity: int = 1,
        precision: Optional[int] = 7,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.exterior = tuple(map(float, exterior))
        self.max_terminal_nodes = max_terminal_nodes
        self.max_depth = max_depth
        self.capacity = capacity
        self.precision = precision

        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        # The weight for adaptive operation should be sparse, since training operation
        # results in a dynamic change of the underlying quadtree.
        weight_size = (
            self.max_depth + 1,
            1 << self.max_depth,
            1 << self.max_depth,
            self.feature_dim,
        )
        self.weight = nn.Parameter(torch.sparse_coo_tensor(size=weight_size, dtype=torch.float64))

    @property
    def tiles(self) -> torch.Tensor:
        """Return tiles of the quadtree."""
        return self.weight.coalesce().detach().indices().t()[:, :-1]

    def extra_repr(self) -> str:
        return (
            "{feature_dim}, "
            "exterior={exterior}, capacity={capacity}, max_depth={max_depth}, "
            "precision={precision}".format(**self.__dict__)
        )


class AdaptiveQuadPool2d(_AdaptiveQuadPool):
    __doc__ = f"""Adaptive lookup index over quadtree decomposition of input 2D coordinates.

    This module constructs an internal lookup quadtree to organize closely situated 2D points.
    Each terminal node in the resulting quadtree is paired with a weight. Thus, when providing
    an input coordinate, the module retrieves the corresponding terminal node and returns its
    associated weight.

    {_AdaptiveQuadPool.__doc__}

    Examples:

    >>> # Feature vectors of size 4 over a 2d space.
    >>> pool = nn.AdaptiveQuadPool2d(4, (-10, -5, 20, 10))
    >>> # Grow tree up to 4-th level and sub-divides a node after 8 coordinates in a quad.
    >>> pool = nn.AdaptiveQuadPool2d(4, (-10, -5, 20, 10), max_depth=4, capacity=8)
    >>> # Create 2D coordinates and query associated weights.
    >>> input = torch.rand((1024, 2), dtype=torch.float64) * 10 - 5
    >>> output = pool(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        result = F.adaptive_quad_pool2d(
            self.weight,
            input,
            self.exterior,
            training=self.training,
            max_terminal_nodes=self.max_terminal_nodes,
            max_depth=self.max_depth,
            capacity=self.capacity,
            precision=self.precision,
        )
        if self.training:
            self.weight.data = result.weight
        return result.values


class AdaptiveMaxQuadPool2d(_AdaptiveQuadPool):
    __doc__ = f"""Adaptive maximum pooling over quadtree decomposition of input 2D coordinates.

    This module constructs an internal lookup quadtree to organize closely situated 2D points.
    Each terminal node in the resulting quadtree is paired with a weight. Thus, when providing
    an input coordinate, the module retrieves a **terminal group** of nodes and calculates the
    maximum value for each ``feature_dim``.

    {_AdaptiveQuadPool.__doc__}

    Examples:

    >>> pool = nn.AdaptiveMaxQuadPool2d(3, (-10, -5, 20, 10), max_depth=5)
    >>> # Create 2D coordinates and feature vector associated with them.
    >>> input = torch.rand((2048, 2), dtype=torch.float64) * 10 - 5
    >>> output = pool(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        result = F.adaptive_max_quad_pool2d(
            self.weight,
            input,
            self.exterior,
            training=self.training,
            max_terminal_nodes=self.max_terminal_nodes,
            max_depth=self.max_depth,
            capacity=self.capacity,
            precision=self.precision,
        )
        if self.training:
            self.weight.data = result.weight
        return result.values


class AdaptiveAvgQuadPool2d(_AdaptiveQuadPool):
    __doc__ = f"""Adaptive average pooling over quadtree decomposition of input 2D coordinates.

    This module constructs an internal lookup quadtree to organize closely situated 2D points.
    Each terminal node in the resulting quadtree is paired with a weight. Thus, when providing
    an input coordinate, the module retrieves a **terminal group** of nodes and calculates an
    average value for each ``feature_dim``.

    {_AdaptiveQuadPool.__doc__}

    Examples:

    >>> # Create pool with 7 features.
    >>> pool = nn.AdaptiveAvgQuadPool2d(7, (0, 0, 1, 1), max_depth=12)
    >>> input = torch.rand((2048, 2), dtype=torch.float64)
    >>> output = pool(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        result = F.adaptive_avg_quad_pool2d(
            self.weight,
            input,
            self.exterior,
            training=self.training,
            max_terminal_nodes=self.max_terminal_nodes,
            max_depth=self.max_depth,
            capacity=self.capacity,
            precision=self.precision,
        )
        if self.training:
            self.weight.data = result.weight
        return result.values


class _QuadPool(nn.Module):
    __doc__ = f"""
    Args:
        feature_dim: Size of each feature vector.
        polygon: Polygon that resembles boundary for the terminal nodes of a quadtree.
        exterior: Geometrical boundary of the learning space in (X, Y, W, H) format.
        max_terminal_nodes: Optional maximum number of terminal nodes in a quadtree. Once a
            maximum is reached, internal nodes are no longer sub-divided and tree stops growing.
        max_depth: Maximum depth of the quadtree. Default: 17.
        precision: Optional rounding of the input coordinates. Default: 7.

    Shape:
        - Input: :math:`(*, 2)`, where 2 comprises longitude and latitude coordinates.
        - Output: :math:`(*, H)`, where * is the input shape and :math:`H = \\text{{feature_dim}}`.

    {_exterior_doc}
    {_terminal_group_doc}

    Note:
        All terminal nodes that have an intersection with the specified polygon boundary are
        included into the quadtree.
    """

    def __init__(
        self,
        feature_dim: int,
        polygon: Polygon,
        exterior: _Exterior,
        max_terminal_nodes: Optional[int] = None,
        max_depth: int = 17,
        precision: Optional[int] = 7,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.polygon = polygon
        self.exterior = tuple(map(float, exterior))
        self.max_terminal_nodes = max_terminal_nodes
        self.max_depth = max_depth
        self.precision = precision

        # Generate regular tiling for the provided polygon and build from those
        # tiles a quadtree from terminal nodes all way up to the root node.
        tiles_iter = regular_tiling(
            polygon, Exterior.from_tuple(exterior), z=max_depth, internal=True
        )
        tiles = torch.tensor(list(tiles_iter), dtype=torch.int64)

        self.register_buffer("tiles", tiles)
        self.tiles: Tensor

        self.initialize_parameters()
        self.reset_parameters()

    def initialize_parameters(self) -> None:
        weight_size = [self.tiles.size(0), self.feature_dim]
        self.weight = nn.Parameter(torch.empty(weight_size, dtype=torch.float64))

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight)

    def extra_repr(self) -> str:
        return (
            "{feature_dim}, exterior={exterior}, max_depth={max_depth}, "
            "precision={precision}".format(**self.__dict__)
        )


class QuadPool2d(_QuadPool):
    __doc__ = f"""Lookup index over quadtree decomposition of input 2D coordinates.

    This module constructs an internal lookup tree to organize closely situated 2D points using
    a specified polygon and exterior, where polygon is treated as a *boundary* of terminal
    nodes of a quadtree.

    Each terminal node in the resulting quadtree is paired with a weight. Thus, when providing
    an input coordinate, the module retrieves the corresponding terminal node and returns its
    associated weight.

    {_QuadPool.__doc__}

    Examples:

    >>> from shapely.geometry import Polygon
    >>> # Create a pool for squared exterior 100x100 and use only a portion of that
    >>> # exterior isolated by a square 10x10.
    >>> poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    >>> pool = nn.QuadPool2d(5, poly, exterior=(0, 0, 100, 100))
    >>> input = torch.rand((2048, 2), dtype=torch.float64)
    >>> output = pool(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        result = F.quad_pool2d(
            self.tiles,
            self.weight,
            input,
            self.exterior,
            # This is not a mistake, since we already know the shape of the
            # quadtree, there is no need to learn it.
            training=False,
            max_terminal_nodes=self.max_terminal_nodes,
            max_depth=self.max_depth,
            precision=self.precision,
        )
        return result.values


class MaxQuadPool2d(_QuadPool):
    __doc__ = f"""Maximum pooling over quadtree decomposition of input 2D coordinates.

    This module constructs an internal lookup tree to organize closely situated 2D points using
    a specified polygon and exterior, where polygon is treated as a *boundary* of terminal nodes
    of a quadtree.

    Each terminal node in the resulting quadtree is paired with a weight. Thus, when providing
    an input coordinate, the module retrieves a **terminal group** of nodes and calculates the
    maximum value for each ``feature_dim``.

    {_QuadPool.__doc__}

    Examples:

    >>> from shapely.geometry import Polygon
    >>> poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    >>> pool = nn.MaxQuadPool2d(3, poly, exterior=(0, 0, 100, 100))
    >>> input = torch.rand((2048, 2), dtype=torch.float64)
    >>> output = pool(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        result = F.max_quad_pool2d(
            self.tiles,
            self.weight,
            input,
            self.exterior,
            training=False,
            max_terminal_nodes=self.max_terminal_nodes,
            max_depth=self.max_depth,
            precision=self.precision,
        )
        return result.values


class AvgQuadPool2d(_QuadPool):
    __doc__ = f"""Average pooling over quadtree decomposition of input 2D coordinates.

    This module constructs an internal lookup tree to organize closely situated 2D points using
    a specified polygon and exterior, where polygon is treated as a *boundary* of terminal
    nodes of a quadtree.

    Each terminal node in the resulting quadtree is paired with a weight. Thus, when providing
    an input coordinate, the module retrieves a **terminal group** of nodes and calculates an
    average value for each ``feature_dim``.

    {_QuadPool.__doc__}

    Examples:

    >>> from shapely.geometry import Polygon
    >>> poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    >>> pool = nn.AvgQuadPool2d(4, poly, exterior=(0, 0, 100, 100))
    >>> input = torch.rand((2048, 2), dtype=torch.float64)
    >>> output = pool(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        result = F.avg_quad_pool2d(
            self.tiles,
            self.weight,
            input,
            self.exterior,
            training=False,
            max_terminal_nodes=self.max_terminal_nodes,
            max_depth=self.max_depth,
            precision=self.precision,
        )
        return result.values
