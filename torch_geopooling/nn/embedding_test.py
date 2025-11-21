# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Yakau Bubnou
# SPDX-FileType: SOURCE

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
