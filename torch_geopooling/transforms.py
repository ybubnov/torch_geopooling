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

from __future__ import annotations

from typing import Iterator, Tuple

from torch import Tensor

from torch_geopooling.tiling import Tile

__all__ = ["TileWKT"]


class _TileSet(set):
    def __init__(self, tiles: Tensor) -> None:
        super().__init__(Tile(*tile.detach().tolist()) for tile in tiles)

    def is_terminal(self, tile: Tile) -> bool:
        return (
            (tile.child(0, 0) not in self)
            and (tile.child(0, 1) not in self)
            and (tile.child(1, 0) not in self)
            and (tile.child(1, 1) not in self)
        )


class TileWKT:
    """Convert a Tile to a WKT polygon given the exterior of the whole geometry.

    Module returns a tile geometry in WKT format, which comprises a polygon.

    Args:
        exterior: Exterior coordinates in (X, Y, W, H) format. The exterior is used to calculate
            boundaries of a tile to produce a final WKT.
        precision: A precision of the resulting geometry, digits after the decimal point.
        internal: When `True`, output includes internal nodes of the quadtree tiles.
            Otherwise (default) returns only geometry of terminal nodes.
    """

    def __init__(
        self,
        exterior: Tuple[float, float, float, float],
        precision: int = 7,
        internal: bool = False,
    ) -> None:
        self.exterior = tuple(map(float, exterior))
        self.precision = precision
        self.internal = internal

        self._xmin, self._ymin, self._width, self._height = exterior
        if self._width <= 0:
            raise ValueError(f"exterior width should be >0, got {self._width}")
        if self._height <= 0:
            raise ValueError(f"exterior height should be >0, got {self._height}")

    def __call__(self, tiles: Tensor) -> Iterator[str]:
        if len(tiles.size()) != 2:
            raise ValueError(f"tiles tensor must be a 2D tensor, got {tiles.size()} shape")

        if tiles.size(1) != 3:
            raise ValueError(
                f"tiles should be triplets of (z, x, y), got tensor of shape {tiles.size()}"
            )

        tileset = _TileSet(tiles)

        for tile in tiles:
            z, x, y = tile.detach().tolist()
            width = self._width / (1 << z)
            height = self._height / (1 << z)

            if (not self.internal) and (not tileset.is_terminal(Tile(z, x, y))):
                continue

            xmin = self._xmin + width * x
            xmax = round(xmin + width, self.precision)
            xmin = round(xmin, self.precision)

            ymin = self._ymin + height * y
            ymax = round(ymin + height, self.precision)
            ymin = round(ymin, self.precision)

            yield (
                "POLYGON (("
                f"{xmin} {ymin}, {xmax} {ymin}, {xmax} {ymax}, {xmin} {ymax}, {xmin} {ymin}"
                "))"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(exterior={self.exterior})"
