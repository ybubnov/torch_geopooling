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

from torch_geopooling.transforms import TileWKT


def test_tile_wkt() -> None:
    tile_wkt4 = TileWKT(exterior=(0.0, 0.0, 10.0, 10.0), internal=False)
    tile_wkt5 = TileWKT(exterior=(0.0, 0.0, 10.0, 10.0), internal=True)

    tiles = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.int32,
    )

    assert len(list(tile_wkt4(tiles))) == 4
    assert len(list(tile_wkt5(tiles))) == 5
