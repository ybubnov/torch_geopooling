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
from torch.nn import L1Loss

from torch_geopooling.nn import (
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
