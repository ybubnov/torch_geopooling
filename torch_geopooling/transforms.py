from __future__ import annotations

from enum import auto
from enum import Enum
from typing import Tuple
from typing import List
from typing import Iterator
from typing import Set
from typing import NamedTuple

from torch import Tensor


__all__ = ["TileWKT"]


class _Tile(NamedTuple):
    z: int
    x: int
    y: int

    def child(self, x: int, y: int) -> _Tile:
        return _Tile(self.z + 1, self.x * 2 + x, self.y * 2 + y)


class _TileSet(set):

    def __init__(self, tiles: Tensor) -> None:
        super().__init__(_Tile(*tile.detach().tolist()) for tile in tiles)

    def is_terminal(self, tile: _Tile) -> bool:
        return (
            (tile.child(0, 0) not in self) and
            (tile.child(0, 1) not in self) and
            (tile.child(1, 0) not in self) and
            (tile.child(1, 1) not in self)
        )


class TileWKT:
    """Convert a Tile to a WKT polygon given the exterior of the whole geometry.

    Module returns a tuple comprised of a tile geometry in WKT format, which comprises a polygon.

    Args:
        exterior (tuple): Exterior coordinates in xywh format. The exterior is used to calculate
            boundaries of a tile to produce a final WKT.
        precision (float): A precision of the resulting geometry, digits after the deciman point.
        internal (bool): When True, output includes internal nodes of the Quadtree tiles.
            Otherwise (default) returns only geometry of terminal nodes.
    """

    def __init__(
        self,
        exterior: Tuple[float, float, float, float],
        precision: float = 7,
        internal: bool = False,
    ) -> None:
        self.exterior = tuple(map(float, exterior))
        self.precision = precision
        self.internal = internal

        self._xmin, self._ymin, self._width, self._height = exterior

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

            if (not self.internal) and (not tileset.is_terminal(_Tile(z, x, y))):
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
