import torch
from torch_geopooling.nn import QuadPool2d


def test_quad_pool2d() -> None:
    pool = QuadPool2d(1024, (-180, -90, 360, 180))

    input = torch.rand((100, 2), dtype=torch.float64) * 90
    x = torch.rand((100,), dtype=torch.float64)

    y = pool(input, x)
    print(y)
    assert y.requires_grad
