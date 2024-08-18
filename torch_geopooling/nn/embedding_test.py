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
from torch import nn
from torch.optim import SGD

from torch_geopooling.nn.embedding import Embedding2d


def test_embedding2d_optimize() -> None:
    embedding = Embedding2d(
        (2, 2, 1),
        padding=(0, 0),
        exterior=(-180.0, -90.0, 360.0, 180.0),
    )

    x_true = torch.tensor(
        [[90.0, 45.0], [90.0, -45.0], [-90.0, -45.0], [-90.0, 45.0]], dtype=torch.float64
    )
    y_true = torch.tensor([[10.0], [20.0], [30.0], [40.0]], dtype=torch.float64)

    optim = SGD(embedding.parameters(), lr=0.1)
    loss_fn = nn.L1Loss()

    for i in range(10000):
        optim.zero_grad()

        y_pred = embedding(x_true)
        loss = loss_fn(y_pred[:, 0, 0, :], y_true)
        loss.backward()

        optim.step()

    assert pytest.approx(0.0, abs=1e-1) == loss.detach().item()
