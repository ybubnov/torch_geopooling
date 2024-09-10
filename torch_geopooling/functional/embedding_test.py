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
