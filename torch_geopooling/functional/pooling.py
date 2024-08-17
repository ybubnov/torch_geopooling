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

from textwrap import dedent, indent
from functools import partial
from inspect import signature
from typing import Callable, NamedTuple, Optional, Tuple

import torch
from torch import Tensor, autograd
from torch.autograd.function import FunctionCtx

import torch_geopooling._C as _C
from torch_geopooling import return_types
from torch_geopooling.tiling import ExteriorTuple

__all__ = [
    "adaptive_avg_quad_pool2d",
    "adaptive_max_quad_pool2d",
    "adaptive_quad_pool2d",
    "avg_quad_pool2d",
    "max_quad_pool2d",
    "quad_pool2d",
]


def __def__(fn: Callable, doc: str) -> Callable:
    f = partial(fn)
    f.__doc__ = doc + indent(dedent(fn.__doc__ or ""), "    ")
    f.__module__ = fn.__module__
    f.__annotations__ = fn.__annotations__
    f.__signature__ = signature(fn)  # type: ignore
    f.__defaults__ = fn.__defaults__  # type: ignore
    f.__kwdefaults__ = fn.__kwdefaults__  # type: ignore
    return f


class FunctionParams(NamedTuple):
    max_terminal_nodes: Optional[int] = None
    max_depth: Optional[int] = None
    capacity: Optional[int] = None
    precision: Optional[int] = None


ForwardType = Callable[
    [
        Tensor,  # tiles
        Tensor,  # weight
        Tensor,  # input
        ExteriorTuple,  # exterior
        bool,  # training
        Optional[int],  # max_terminal_nodes
        Optional[int],  # max_depth
        Optional[int],  # capacity
        Optional[int],  # precision
    ],
    Tuple[Tensor, Tensor, Tensor],  # (tiles, weight, values)
]


BackwardType = Callable[
    [
        Tensor,  # grad_output
        Tensor,  # tiles
        Tensor,  # weight
        Tensor,  # input
        ExteriorTuple,  # exterior
        Optional[int],  # max_terminal_nodes
        Optional[int],  # max_depth
        Optional[int],  # capacity
        Optional[int],  # precision
    ],
    Tensor,  # (grad_weight)
]


class Function(autograd.Function):
    forward_impl: ForwardType
    backward_impl: BackwardType

    @classmethod
    def forward(
        cls,
        tiles: Tensor,
        weight: Tensor,
        input: Tensor,
        exterior: ExteriorTuple,
        training: bool,
        params: FunctionParams,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return cls.forward_impl(tiles, weight, input, exterior, training, *params)

    @staticmethod
    def setup_context(ctx: FunctionCtx, inputs: Tuple, outputs: Tuple) -> None:
        _, _, input, exterior, _, params = inputs
        tiles, weight, _ = outputs

        ctx.save_for_backward(tiles.view_as(tiles), weight.view_as(weight), input.view_as(input))
        ctx.exterior = exterior
        ctx.params = params

    @classmethod
    def backward(
        cls, ctx: FunctionCtx, grad_tiles: Tensor, grad_weight: Tensor, grad_values: Tensor
    ) -> Tuple[Optional[Tensor], ...]:
        grad_weight_out = cls.backward_impl(
            grad_values, *ctx.saved_tensors, ctx.exterior, *ctx.params
        )  # type: ignore
        # Drop gradient for tiles, this should not be changed by an optimizer.
        return None, grad_weight_out, None, None, None, None

    @classmethod
    def func(
        cls,
        tiles: Tensor,
        weight: Tensor,
        input: Tensor,
        exterior: Tuple[float, ...],
        *,
        training: bool = True,
        max_terminal_nodes: Optional[int] = None,
        max_depth: Optional[int] = None,
        capacity: Optional[int] = None,
        precision: Optional[int] = None,
    ) -> return_types.quad_pool2d:
        """
        Args:
            tiles: Tiles tensor representing tiles of a quadtree (both, internal and terminal).
            weight: Weights tensor associated with each tile of a quadtree.
            input: Input 2D coordinates as pairs of x (longitude) and y (latitude).
            exterior: Geometrical boundary of the learning space in (X, Y, W, H) format.
            training: True, when executed during training, and False otherwise.
            max_terminal_nodes: Optional maximum number of terminal nodes in a quadtree. Once a
                maximum is reached, internal nodes are no longer sub-divided and tree stops
                growing.
            max_depth: Maximum depth of the quadtree. Default: 17.
            capacity: Maximum number of inputs, after which a quadtree's node is subdivided and
                depth of the tree grows. Default: 1.
            precision: Optional rounding of the input coordinates. Default: 7.
        """
        params = FunctionParams(
            max_terminal_nodes=max_terminal_nodes,
            max_depth=max_depth,
            capacity=capacity,
            precision=precision,
        )

        result = cls.apply(tiles, weight, input, exterior, training, params)
        return return_types.quad_pool2d(*result)


class QuadPool2d(Function):
    forward_impl = _C.quad_pool2d
    backward_impl = _C.quad_pool2d_backward


class MaxQuadPool2d(Function):
    forward_impl = _C.max_quad_pool2d
    backward_impl = _C.max_quad_pool2d_backward


class AvgQuadPool2d(Function):
    forward_impl = _C.avg_quad_pool2d
    backward_impl = _C.avg_quad_pool2d_backward


quad_pool2d = __def__(
    QuadPool2d.func,
    """Lookup index over quadtree decomposition of input 2D coordinates.

    See :class:`torch_geopooling.nn.QuadPool2d` for more details.
    """,
)
max_quad_pool2d = __def__(
    MaxQuadPool2d.func,
    """Maximum pooling over quadtree decomposition of input 2D coordinates.

    See :class:`torch_geopooling.nn.MaxQuadPool2d` for more details.
    """,
)
avg_quad_pool2d = __def__(
    AvgQuadPool2d.func,
    """Average pooling over quadtree decomposition of input 2D coordinates.

    See :class:`torch_geopooling.nn.AvgQuadPool2d` for more details.
    """,
)


class AdaptiveFunction(autograd.Function):
    forward_impl: ForwardType
    backward_impl: BackwardType

    @staticmethod
    def sparse_ravel(weight: Tensor) -> Tuple[Tensor, Tensor]:
        """Transform weight as coordinate sparse tensor into a tuple of tiles and feature tensor.

        The method transforms sparse encoding of quadtree (where 3 first dimensions are
        coordinates of a tile and 4-th dimension is an index of a feature in the feature
        vector), into tuple of coordinates (tiles) and dense weight tensor.

        Effectively: (17,131072,131702,5) -> (nnz,3), (nnz,5); where nnz - is a number of
        non-zero elements in the sparse tensor.
        """
        feature_dim = weight.size(-1)
        weight = weight.coalesce()

        # Transform sparse tensor into a tuple of (tiles, weight) that are directly usable
        # by the C++ extension functions.
        tiles = weight.indices().t()[::feature_dim, :-1]
        w = weight.values().reshape((-1, feature_dim))
        return tiles, w

    @staticmethod
    def sparse_unravel(tiles: Tensor, weight: Tensor, size: torch.Size) -> Tensor:
        """Perform inverse operation of `ravel`.

        Method packs tiles (coordinates) and weight (values) into a coordinate sparse tensor.
        """
        feature_dim = weight.size(-1)
        feature_indices = torch.arange(0, feature_dim).repeat(tiles.size(0))

        indices = tiles.repeat_interleave(feature_dim, dim=0)
        indices = torch.column_stack((indices, feature_indices))

        return torch.sparse_coo_tensor(indices.t(), weight.ravel(), size=size)

    @classmethod
    def forward(
        cls,
        weight: Tensor,
        input: Tensor,
        exterior: ExteriorTuple,
        training: bool,
        params: FunctionParams,
    ) -> Tuple[Tensor, Tensor]:
        tiles, w = cls.sparse_ravel(weight)

        tiles_out, w_out, values_out = cls.forward_impl(
            tiles, w, input, exterior, training, *params
        )

        weight_out = cls.sparse_unravel(tiles_out, w_out, size=weight.size())
        return weight_out.coalesce(), values_out

    @staticmethod
    def setup_context(ctx: FunctionCtx, inputs: Tuple, outputs: Tuple) -> None:
        _, input, exterior, _, params = inputs
        weight, _ = outputs

        ctx.save_for_backward(weight, input)
        ctx.exterior = exterior
        ctx.params = params

    @classmethod
    def backward(
        cls, ctx: FunctionCtx, grad_weight: Tensor, grad_values: Tensor
    ) -> Tuple[Optional[Tensor], ...]:
        weight, input = ctx.saved_tensors
        tiles, w = cls.sparse_ravel(weight)

        grad_weight_dense = cls.backward_impl(
            grad_values, tiles, w, input, ctx.exterior, *ctx.params
        )  # type: ignore
        grad_weight_sparse = cls.sparse_unravel(tiles, grad_weight_dense, size=weight.size())

        return grad_weight_sparse.coalesce(), None, None, None, None

    @classmethod
    def func(
        cls,
        weight: Tensor,
        input: Tensor,
        exterior: Tuple[float, ...],
        *,
        training: bool = True,
        max_terminal_nodes: Optional[int] = None,
        max_depth: Optional[int] = None,
        capacity: Optional[int] = None,
        precision: Optional[int] = None,
    ) -> return_types.adaptive_quad_pool2d:
        """
        Args:
            weight: Weights tensor associated with each tile of a quadtree.
            input: Input 2D coordinates as pairs of x (longitude) and y (latitude).
            exterior: Geometrical boundary of the learning space in (X, Y, W, H) format.
            training: True, when executed during training, and False otherwise.
            max_terminal_nodes: Optional maximum number of terminal nodes in a quadtree. Once a
                maximum is reached, internal nodes are no longer sub-divided and tree stops
                growing.
            max_depth: Maximum depth of the quadtree. Default: 17.
            capacity: Maximum number of inputs, after which a quadtree's node is subdivided and
                depth of the tree grows. Default: 1.
            precision: Optional rounding of the input coordinates. Default: 7.
        """
        params = FunctionParams(
            max_terminal_nodes=max_terminal_nodes,
            max_depth=max_depth,
            capacity=capacity,
            precision=precision,
        )

        result = cls.apply(weight, input, exterior, training, params)
        return return_types.adaptive_quad_pool2d(*result)


class AdaptiveQuadPool2d(AdaptiveFunction):
    forward_impl = _C.quad_pool2d
    backward_impl = _C.quad_pool2d_backward


class AdaptiveMaxQuadPool2d(AdaptiveFunction):
    forward_impl = _C.max_quad_pool2d
    backward_impl = _C.max_quad_pool2d_backward


class AdaptiveAvgQuadPool2d(AdaptiveFunction):
    forward_impl = _C.avg_quad_pool2d
    backward_impl = _C.avg_quad_pool2d_backward


adaptive_quad_pool2d = __def__(
    AdaptiveQuadPool2d.func,
    """Adaptive lookup index over quadtree decomposition of input 2D coordinates.

    See :class:`torch_geopooling.nn.AdaptiveQuadPool2d` for more details.
    """,
)
adaptive_max_quad_pool2d = __def__(
    AdaptiveMaxQuadPool2d.func,
    """Adaptive maximum pooling over quadtree decomposition of input 2D coordinates.

    See :class:`torch_geopooling.nn.AdaptiveMaxQuadPool2d` for more details.
    """,
)
adaptive_avg_quad_pool2d = __def__(
    AdaptiveAvgQuadPool2d.func,
    """Adaptive average pooling over quadtree decomposition of input 2D coordinates.

    See :class:`torch_geopooling.nn.AdaptiveAvgQuadPool2d` for more details.
    """,
)
