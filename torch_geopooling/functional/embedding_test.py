# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Yakau Bubnou
# SPDX-FileType: SOURCE

import torch
from torch_geopooling.functional.embedding import embedding2d


def test_embedding2d() -> None:
    input = torch.rand((100, 2), dtype=torch.float64) * 10.0
    weight = torch.rand((1024, 1024, 3), dtype=torch.float64)

    result = embedding2d(
        input,
        weight,
        padding=(3, 2),
        exterior=(-10.0, -10.0, 20.0, 20.0),
        reflection=True,
    )

    assert result.size() == torch.Size([100, 7, 5, 3])
