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
