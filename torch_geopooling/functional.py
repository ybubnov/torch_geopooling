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

from typing import Callable, NamedTuple, Optional, Tuple

from torch import Tensor, autograd
from torch.autograd.function import FunctionCtx

import torch_geopooling._C as _C
from torch_geopooling import return_types
from torch_geopooling.tiling import ExteriorTuple

__all__ = ["avg_quad_pool2d", "max_quad_pool2d", "quad_pool2d"]


class FunctionParams(NamedTuple):
    max_depth: Optional[int] = None
    capacity: Optional[int] = None
    precision: Optional[int] = None


class Function(autograd.Function):
    forward_impl: Callable[
        [
            Tensor,  # tiles
            Tensor,  # input
            Tensor,  # weight
            ExteriorTuple,  # exterior
            bool,  # training
            Optional[int],  # max_depth
            Optional[int],  # capacity
            Optional[int],  # precision
        ],
        Tuple[Tensor, Tensor],  # (tiles, weight)
    ]

    backward_impl: Callable[
        [
            Tensor,  # grad_output
            Tensor,  # tiles
            Tensor,  # input
            Tensor,  # weight
            ExteriorTuple,  # exterior
            Optional[int],  # max_depth
            Optional[int],  # capacity
            Optional[int],  # precision
        ],
        Tuple[Tensor, Tensor],  # (tiles, weight)
    ]

    @classmethod
    def forward(
        cls,
        tiles: Tensor,
        input: Tensor,
        weight: Tensor,
        exterior: ExteriorTuple,
        training: bool,
        params: FunctionParams,
    ) -> Tuple[Tensor, Tensor]:
        return cls.forward_impl(tiles, input, weight, exterior, training, *params)

    @staticmethod
    def setup_context(ctx: FunctionCtx, inputs: Tuple, outputs: Tuple) -> None:
        tiles, input, weight, exterior, _, params = inputs
        ctx.save_for_backward(tiles.view_as(tiles), input.view_as(input), weight.view_as(weight))
        ctx.exterior = exterior
        ctx.params = params

    @classmethod
    def backward(
        cls, ctx: FunctionCtx, grad_tiles: Tensor, grad_output: Tensor
    ) -> Tuple[Optional[Tensor], ...]:
        grad_weight = cls.backward_impl(grad_output, *ctx.saved_tensors, ctx.exterior, *ctx.params)  # type: ignore
        # Drop gradient for tiles, this should not be changed by an optimizer.
        return grad_tiles, None, grad_weight, None, None, None


class QuadPool2d(Function):
    forward_impl = _C.quad_pool2d
    backward_impl = _C.quad_pool2d_backward


def quad_pool2d(
    tiles: Tensor,
    input: Tensor,
    weight: Tensor,
    exterior: Tuple[float, ...],
    *,
    training: bool = True,
    max_depth: Optional[int] = None,
    capacity: Optional[int] = None,
    precision: Optional[int] = None,
) -> return_types.quad_pool2d:
    params = FunctionParams(max_depth=max_depth, capacity=capacity, precision=precision)
    result = QuadPool2d.apply(tiles, input, weight, exterior, training, params)
    return return_types.quad_pool2d(*result)


class MaxQuadPool2d(Function):
    forward_impl = _C.max_quad_pool2d
    backward_impl = _C.max_quad_pool2d_backward


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
    params = FunctionParams(max_depth=max_depth, capacity=capacity, precision=precision)
    result = MaxQuadPool2d.apply(tiles, input, weight, exterior, training, params)
    return return_types.max_quad_pool2d(*result)


class AvgQuadPool2d(Function):
    forward_impl = _C.avg_quad_pool2d
    backward_impl = _C.avg_quad_pool2d_backward


def avg_quad_pool2d(
    tiles: Tensor,
    input: Tensor,
    weight: Tensor,
    exterior: Tuple[float, ...],
    *,
    training: bool = True,
    max_depth: Optional[int] = None,
    capacity: Optional[int] = None,
    precision: Optional[int] = None,
) -> return_types.avg_quad_pool2d:
    params = FunctionParams(max_depth=max_depth, capacity=capacity, precision=precision)
    result = AvgQuadPool2d.apply(tiles, input, weight, exterior, training, params)
    return return_types.avg_quad_pool2d(*result)
