import torch

from torch_geopooling.functional import max_quad_pool2d, quad_pool2d


def test_quad_pool2d() -> None:
    tiles = torch.empty((0, 3), dtype=torch.int32)
    input = torch.rand((100, 2), dtype=torch.float64) * 10.0
    weight = torch.randn([64], dtype=torch.float64)
    bias = torch.randn([64], dtype=torch.float64)

    result = quad_pool2d(
        tiles,
        input,
        weight,
        bias,
        (0.0, 0.0, 10.0, 10.0),
        training=True,
        max_depth=16,
        capacity=1,
        precision=6,
    )
    assert result.tiles.size(0) > 0
    assert result.tiles.size(1) == 3

    assert result.weight.size() == torch.Size([input.size(0)])
    assert result.bias.size() == torch.Size([input.size(0)])


def test_max_quad_pool2d() -> None:
    tiles = torch.empty((0, 3), dtype=torch.int32)
    input = torch.rand((100, 2), dtype=torch.float64) * 10.0
    weight = torch.randn([64], dtype=torch.float64)

    result = max_quad_pool2d(
        tiles,
        input,
        weight,
        (0.0, 0.0, 10.0, 10.0),
        training=True,
        max_depth=16,
        capacity=1,
        precision=6,
    )

    assert result.tiles.size(0) > 0
    assert result.tiles.size(1) == 3
    assert result.weight.size() == torch.Size([input.size(0)])
