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

from typing import Type

import pytest
import torch
from shapely.geometry import Polygon
from torch import nn
from torch.optim import SGD
from torch.nn import L1Loss

from torch_geopooling.nn.pooling import (
    AdaptiveAvgQuadPool2d,
    AdaptiveMaxQuadPool2d,
    AdaptiveQuadPool2d,
    AvgQuadPool2d,
    MaxQuadPool2d,
    QuadPool2d,
)


@pytest.mark.parametrize(
    "module_class",
    [
        AdaptiveQuadPool2d,
        AdaptiveMaxQuadPool2d,
        AdaptiveAvgQuadPool2d,
    ],
    ids=["id", "max", "avg"],
)
def test_adaptive_quad_pool2d_gradient(module_class: Type[nn.Module]) -> None:
    pool = module_class(5, (-180, -90, 360, 180))

    input = torch.rand((100, 2), dtype=torch.float64) * 90
    y = pool(input)

    assert pool.weight.grad is None

    loss_fn = L1Loss()
    loss = loss_fn(y, torch.ones_like(y))
    loss.backward()

    assert pool.weight.grad is not None
    assert pool.weight.grad.sum().item() == pytest.approx(-1)


def test_adaptive_quad_pool2d_optimize() -> None:
    pool = AdaptiveQuadPool2d(1, (-180, -90, 360, 180), max_depth=1)

    # Input coordinates are simply centers of the level-1 quads.
    x_true = torch.tensor(
        [[90.0, 45.0], [90.0, -45.0], [-90.0, -45.0], [-90.0, 45.0]], dtype=torch.float64
    )
    y_true = torch.tensor([[10.0], [20.0], [30.0], [40.0]], dtype=torch.float64)
    y_tile = [[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 1]]

    optim = SGD(pool.parameters(), lr=0.01)
    loss_fn = nn.L1Loss()

    for i in range(20000):
        optim.zero_grad()

        y_pred = pool(x_true)
        loss = loss_fn(y_pred, y_true)
        loss.backward()

        optim.step()

    # Ensure that model converged with a small loss.
    assert pytest.approx(0.0, abs=1e-1) == loss.detach().item()

    # Ensure that weights that pooling operation learned are the same as in the
    # target matrix (y_true).
    weight = pool.weight.to_dense()

    for i, tile in enumerate(y_tile):
        z, x, y = tile
        expect_weight = y_true[i].item()
        actual_weight = weight[z, x, y].detach().item()

        assert pytest.approx(expect_weight, abs=1e-1) == actual_weight, f"tile {tile} is wrong"


@pytest.mark.parametrize(
    "module_class",
    [
        QuadPool2d,
        MaxQuadPool2d,
        AvgQuadPool2d,
    ],
    ids=["id", "max", "avg"],
)
def test_quad_pool2d_gradient(module_class: Type[nn.Module]) -> None:
    poly = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.1), (0.0, 1.0)])
    exterior = (0.0, 0.0, 1.0, 1.0)

    pool = module_class(4, poly, exterior, max_depth=5)
    assert pool.weight.size() == torch.Size([pool.tiles.size(0), 4])

    input = torch.rand((100, 2), dtype=torch.float64)
    y = pool(input)

    assert pool.weight.grad is None

    loss_fn = L1Loss()
    loss = loss_fn(y, torch.ones_like(y))
    loss.backward()

    assert pool.weight.grad is not None
    assert pool.weight.grad.sum().item() == pytest.approx(-1)


def test_quad_pool2d_optimize() -> None:
    poly = Polygon([(-180, -90), (-180, 90), (180, 90), (180, -90)])
    pool = QuadPool2d(1, poly, (-180, -90, 360, 180), max_depth=1)

    x_true = torch.tensor(
        [[90.0, 45.0], [90.0, -45.0], [-90.0, -45.0], [-90.0, 45.0]], dtype=torch.float64
    )
    y_true = torch.tensor([[10.0], [20.0], [30.0], [40.0]], dtype=torch.float64)
    y_tile = [(1, 1, 1), (1, 1, 0), (1, 0, 0), (1, 0, 1)]

    optim = SGD(pool.parameters(), lr=0.01)
    loss_fn = nn.L1Loss()

    for i in range(20000):
        optim.zero_grad()

        y_pred = pool(x_true)
        loss = loss_fn(y_pred, y_true)
        loss.backward()

        optim.step()

    # Ensure that model converged with a small loss.
    assert pytest.approx(0.0, abs=1e-1) == loss.detach().item()

    actual_tiles = {}
    for i in range(pool.tiles.size(0)):
        tile = tuple(pool.tiles[i].detach().tolist())
        actual_tiles[tile] = pool.weight[i, 0].detach().item()

    for tile, expect_weight in zip(y_tile, y_true[:, 0].tolist()):
        actual_weight = actual_tiles[tile]
        assert pytest.approx(expect_weight, abs=1e-1) == actual_weight, f"tile {tile} is wrong"
