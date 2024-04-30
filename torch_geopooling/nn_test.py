import torch
from torch.nn import L1Loss

from torch_geopooling.nn import MaxQuadPool2d, QuadPool2d


def test_quad_pool2d_loss() -> None:
    pool = QuadPool2d(1024, (-180, -90, 360, 180))

    input = torch.rand((100, 2), dtype=torch.float64) * 90
    x = torch.rand((100,), dtype=torch.float64)
    y = pool(input, x)

    assert pool.weight.grad is None
    assert pool.bias.grad is None

    loss_fn = L1Loss()
    loss = loss_fn(y, torch.ones_like(y))
    loss.backward()

    assert pool.weight.grad is not None
    assert pool.bias.grad is not None


def test_max_quad_pool2d_loss() -> None:
    max_pool = MaxQuadPool2d(1024, (-180, -90, 360, 180))

    input = torch.rand((100, 2), dtype=torch.float64) * 90
    x = torch.rand((100,), dtype=torch.float64)
    y = max_pool(input, x)

    assert max_pool.weight.grad is None

    loss_fn = L1Loss()
    loss = loss_fn(y, torch.ones_like(y))
    loss.backward()

    assert max_pool.weight.grad is not None
