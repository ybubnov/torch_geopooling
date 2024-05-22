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
from shapely.geometry import Polygon
from torch.nn import L1Loss

from torch_geopooling.nn import AdaptiveMaxQuadPool2d, AdaptiveQuadPool2d, MaxQuadPool2d


def test_adaptive_quad_pool2d_gradient() -> None:
    pool = AdaptiveQuadPool2d(1024, (-180, -90, 360, 180))

    input = torch.rand((100, 2), dtype=torch.float64) * 90
    x = torch.rand((100,), dtype=torch.float64)
    y = pool(input, x)

    assert pool.weight.grad is None
    assert pool.bias.grad is None

    loss_fn = L1Loss()
    loss = loss_fn(y, torch.ones_like(y))
    loss.backward()

    assert pool.weight.grad is not None
    assert pool.bias.grad is not None


def test_adaptive_max_quad_pool2d_gradient() -> None:
    max_pool = AdaptiveMaxQuadPool2d(1024, (-180, -90, 360, 180))

    input = torch.rand((100, 2), dtype=torch.float64) * 90
    y = max_pool(input)

    assert max_pool.weight.grad is None

    loss_fn = L1Loss()
    loss = loss_fn(y, torch.ones_like(y))
    loss.backward()

    assert max_pool.weight.grad is not None


def test_max_quad_pool2d_gradient() -> None:
    poly = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.1), (0.0, 1.0)])
    exterior = (0.0, 0.0, 1.0, 1.0)

    max_pool = MaxQuadPool2d(poly, exterior, max_depth=5)
    assert max_pool.num_features == 1 << 10

    input = torch.rand((100, 2), dtype=torch.float64)
    y = max_pool(input)

    assert max_pool.weight.grad is None

    loss_fn = L1Loss()
    loss = loss_fn(y, torch.ones_like(y))
    loss.backward()

    assert max_pool.weight.grad is not None
    assert max_pool.weight.grad.sum() == -1
