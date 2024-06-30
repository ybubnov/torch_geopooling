from __future__ import annotations

from collections import deque
from itertools import product
from typing import Iterator, NamedTuple, Tuple

from shapely.geometry import Polygon

__all__ = ["Exterior", "ExteriorTuple", "Tile", "regular_tiling"]


class Tile(NamedTuple):
    z: int
    x: int
    y: int

    @classmethod
    def root(cls) -> Tile:
        return cls(0, 0, 0)

    def child(self, x: int, y: int) -> Tile:
        return Tile(self.z + 1, self.x * 2 + x, self.y * 2 + y)

    def children(self) -> Iterator[Tile]:
        for x, y in product(range(2), range(2)):
            yield self.child(x, y)


ExteriorTuple = Tuple[float, float, float, float]


class Exterior(NamedTuple):
    xmin: float
    ymin: float
    width: float
    height: float

    @classmethod
    def from_tuple(cls, exterior_tuple: ExteriorTuple) -> Exterior:
        return cls(*exterior_tuple)

    @property
    def xmax(self) -> float:
        return self.xmin + self.width

    @property
    def ymax(self) -> float:
        return self.ymin + self.height

    def slice(self, tile: Tile) -> Exterior:
        w = self.width / (1 << tile.z)
        h = self.height / (1 << tile.z)
        return Exterior(self.xmin + tile.x * w, self.ymin + tile.y * h, w, h)

    def as_polygon(self) -> Polygon:
        return Polygon(
            [
                (self.xmin, self.ymin),
                (self.xmax, self.ymin),
                (self.xmax, self.ymax),
                (self.xmin, self.ymax),
            ]
        )


def regular_tiling(
    polygon: Polygon, exterior: Exterior, z: int, internal: bool = False
) -> Iterator[Tile]:
    """Returns a regular quad-tiling (tiles of the same size).

    Method returns all tiles of level (z) that have a common intersection with a specified
    polygon.

    Args:
        polygon: A polygon to cover with tiles.
        exterior: Exterior (bounding box) of the quadtree. For example, for geospatial
            coordinates, this will be `(-180.0, -90.0, 360.0, 180.0)`.
        z: Zoom level of the tiles.
        internal: When `True`, returns internal tiles (nodes) of the quadtree up to a root
            tile (0,0,0).

    Returns:
        Iterator of tiles.
    """
    queue = deque([Tile.root()])

    while len(queue) > 0:
        tile = queue.pop()

        tile_poly = exterior.slice(tile).as_polygon()
        if not tile_poly.intersects(polygon):
            continue

        if internal or tile.z >= z:
            yield tile
        if tile.z < z:
            queue.extend(tile.children())
