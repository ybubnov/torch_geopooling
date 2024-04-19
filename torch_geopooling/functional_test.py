import torch
from torch_geopooling.functional import quad_pool2d


def test_quad_pool2d() -> None:
    tiles = torch.tensor([[0, 0, 0]], dtype=torch.int32)
    input = torch.rand((100, 2), dtype=torch.float64) * 10.0
    weight = torch.randn([64], dtype=torch.float64)
    bias = torch.randn([64], dtype=torch.float64)

    result = quad_pool2d(tiles, input, weight, bias, (0.0, 0.0, 10.0, 10.0), True)
    assert result.tiles.size(0) > 0
    assert result.tiles.size(1) == 3

    assert result.weight.size(0) > 0
    assert result.bias.size(0) > 0
