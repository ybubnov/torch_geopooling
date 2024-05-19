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

from typing import NamedTuple

from torch import Tensor


class linear_quad_pool2d(NamedTuple):
    tiles: Tensor
    weight: Tensor
    bias: Tensor


class max_quad_pool2d(NamedTuple):
    tiles: Tensor
    weight: Tensor


class avg_quad_pool2d(NamedTuple):
    tiles: Tensor
    weight: Tensor
