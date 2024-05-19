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

from torch import Tensor

import torch_geopooling._C as _C
import torch_geopooling.return_types as return_types

__all__ = ["avg_quad_pool2d", "linear_quad_pool2d", "max_quad_pool2d"]


def linear_quad_pool2d(
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
) -> return_types.linear_quad_pool2d:
    tiles, weight, bias = _C.linear_quad_pool2d(
        tiles, input, weight, bias, exterior, training, max_depth, capacity, precision
    )
    return return_types.linear_quad_pool2d(tiles, weight, bias)


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
    tiles, weight = _C.avg_quad_pool2d(
        tiles, input, weight, exterior, training, max_depth, capacity, precision
    )
    return return_types.avg_quad_pool2d(tiles, weight)
