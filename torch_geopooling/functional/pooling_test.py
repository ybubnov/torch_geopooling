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

import pytest
import torch

from torch_geopooling.functional.pooling import (
    AdaptiveFunction,
    adaptive_quad_pool2d,
    adaptive_avg_quad_pool2d,
    adaptive_max_quad_pool2d,
    avg_quad_pool2d,
    max_quad_pool2d,
    quad_pool2d,
)


def test_adaptive_function_ravel() -> None:
    size = (2, 2, 2, 1)
    tiles = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.int64)

    weight = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=torch.float64)

    sparse = AdaptiveFunction.sparse_unravel(tiles, weight, size=size)
    torch.testing.assert_close(sparse.to_dense().to_sparse_coo(), sparse)

    tiles_out, weight_out = AdaptiveFunction.sparse_ravel(sparse)
    torch.testing.assert_close(tiles_out, tiles)
    torch.testing.assert_close(weight_out, weight)


@pytest.mark.parametrize(
    "function",
    [
        quad_pool2d,
        max_quad_pool2d,
        avg_quad_pool2d,
    ],
    ids=["id", "max", "avg"],
)
def test_quad_pool2d(function) -> None:
    tiles = torch.empty((0, 3), dtype=torch.int64)
    input = torch.rand((100, 2), dtype=torch.float64) * 10.0
    weight = torch.randn([0, 5], dtype=torch.float64)

    result = function(
        tiles,
        weight,
        input,
        (0.0, 0.0, 10.0, 10.0),
        training=True,
        max_depth=16,
        capacity=1,
        precision=6,
    )
    assert result.tiles.size(0) > 0
    assert result.tiles.size(1) == 3

    assert result.weight.size(0) == result.tiles.size(0)
    assert result.values.size() == torch.Size([input.size(0), weight.size(1)])


@pytest.mark.parametrize(
    "function",
    [
        adaptive_quad_pool2d,
        adaptive_max_quad_pool2d,
        adaptive_avg_quad_pool2d,
    ],
    ids=["id", "max", "avg"],
)
def test_adaptive_quad_pool2d(function) -> None:
    input = torch.rand((100, 2), dtype=torch.float64) * 10.0
    weight = torch.sparse_coo_tensor(size=(10, 1 << 10, 1 << 10, 4), dtype=torch.float64)

    result = function(
        weight,
        input,
        (0.0, 0.0, 10.0, 10.0),
        training=True,
        max_depth=16,
        capacity=1,
        precision=6,
    )

    assert result.weight.layout == torch.sparse_coo
    assert result.weight.indices().size(0) > 0
    assert result.values.size() == torch.Size([input.size(0), weight.size(-1)])
