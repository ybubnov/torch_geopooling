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

import torch

from torch_geopooling.functional import max_quad_pool2d, quad_pool2d


def test_quad_pool2d() -> None:
    tiles = torch.empty((0, 3), dtype=torch.int32)
    input = torch.rand((100, 2), dtype=torch.float64) * 10.0
    weight = torch.randn([64, 5], dtype=torch.float64)

    result = quad_pool2d(
        tiles,
        input,
        weight,
        (0.0, 0.0, 10.0, 10.0),
        training=True,
        max_depth=16,
        capacity=1,
        precision=6,
    )
    assert result.tiles.size(0) > 0
    assert result.tiles.size(1) == 3

    assert result.weight.size() == torch.Size([input.size(0), weight.size(1)])


def test_max_quad_pool2d() -> None:
    tiles = torch.empty((0, 3), dtype=torch.int32)
    input = torch.rand((100, 2), dtype=torch.float64) * 10.0
    weight = torch.randn([64, 1], dtype=torch.float64, requires_grad=True)

    result = max_quad_pool2d(
        tiles,
        input,
        weight,
        (0.0, 0.0, 10.0, 10.0),
        training=True,
        max_depth=16,
        capacity=1,
        precision=6,
    )

    assert result.tiles.size(0) > 0
    assert result.tiles.size(1) == 3
    assert result.weight.size() == torch.Size([input.size(0), weight.size(1)])
