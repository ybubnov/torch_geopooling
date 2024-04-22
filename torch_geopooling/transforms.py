from enum import auto
from enum import Enum
from typing import Tuple
from typing import List
from typing import Iterator

from torch import Tensor


__all__ = ["TileWKT"]


class TileWKT:
    """Convert a Tile to a WKT polygon given the exterior of the whole geometry.

    Module returns a tuple comprised of a tile geometry in WKT format, which comprises a polygon.

    Args:
        exterior (tuple): Exterior coordinates in xywh format. The exterior is used to calculate
            boundaries of a tile to produce a final WKT.
    """

    def __init__(self, exterior: Tuple[float, float, float, float], precision: float = 7) -> None:
        self.exterior = exterior
        self.precision = precision
        self._xmin, self._ymin, self._width, self._height = exterior

    def __call__(self, tiles: Tensor) -> Iterator[str]:
        if len(tiles.size()) != 2:
            raise ValueError(f"tiles tensor must be a 2D tensor, got {tiles.size()} shape")

        if tiles.size(1) != 3:
            raise ValueError(
                f"tiles should be triplets of (z, x, y), got tensor of shape {tiles.size()}"
            )


        for tile in tiles:
            z, x, y = tile.detach().tolist()
            width = self._width / (1 << z)
            height = self._height / (1 << z)

            xmin = self._xmin + width * x
            xmax = xmin + width

            ymin = self._ymin + height * y
            ymax = ymin + height

            yield (
                "POLYGON (("
                f"{xmin} {ymin}, {xmax} {ymin}, {xmax} {ymax}, {xmin} {ymax}, {xmin} {ymin}"
                "))"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(exterior={self.exterior})"
