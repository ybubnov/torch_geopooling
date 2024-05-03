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
from torch.nn import L1Loss

from torch_geopooling.nn import LinearQuadPool2d, MaxQuadPool2d


def test_linear_quad_pool2d_gradient() -> None:
    pool = LinearQuadPool2d(1024, (-180, -90, 360, 180))

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


def test_max_quad_pool2d_gradient() -> None:
    max_pool = MaxQuadPool2d(1024, (-180, -90, 360, 180))

    input = torch.rand((100, 2), dtype=torch.float64) * 90
    x = torch.rand((100,), dtype=torch.float64)
    y = max_pool(input, x)

    assert max_pool.weight.grad is None

    loss_fn = L1Loss()
    loss = loss_fn(y, torch.ones_like(y))
    loss.backward()

    assert max_pool.weight.grad is not None
