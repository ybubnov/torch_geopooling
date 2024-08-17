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

from typing import Optional, Tuple

from torch import Tensor, autograd
from torch.autograd.function import FunctionCtx

import torch_geopooling._C as _C
from torch_geopooling.tiling import ExteriorTuple


__all__ = ["embedding2d"]


class Function(autograd.Function):
    @staticmethod
    def forward(
        input: Tensor,
        weight: Tensor,
        padding: Tuple[int, int],
        exterior: ExteriorTuple,
    ) -> Tensor:
        return _C.embedding2d(input, weight, padding, exterior)

    @staticmethod
    def setup_context(ctx: FunctionCtx, inputs: Tuple, outputs: Tuple) -> None:
        input, _, padding, exterior = inputs

        ctx.save_for_backward(input)
        ctx.padding = padding
        ctx.exterior = exterior

    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> Tuple[Optional[Tensor], ...]:
        (input,) = ctx.saved_tensors
        grad_weight = _C.embedding2d_backward(grad)
        return None, grad_weight, None, None


def embedding2d(
    input: Tensor,
    weight: Tensor,
    *,
    padding: Tuple[int, int] = (0, 0),
    exterior: ExteriorTuple,
) -> Tensor:
    """
    Retrieves spatial embeddings from a fixed-size lookup table based on 2D coordinates.

    This function accepts a list of (x, y) coordinates and retrieves the corresponding
    spatial embeddings from a provided embedding matrix. The embeddings are selected
    based on the input coordinates, with an optional padding to include neighboring cells.

    Args:
        input: A list of 2D coordinates where each coordinate is represented as a tuple (x, y),
            where x is the longitude and y is the latitude.
        weight: A 3D tensor representing the embedding matrix. The first dimension  corresponds to
            the maximum possible bucket for the x coordinate, the second dimension corresponds to
            the maximum possible bucket for the y coordinate, and the third dimension corresponds
            to the embedding size.
        padding: The size of the neighborhood to query. Default is 0, meaning only the embedding
            for the exact input coordinate is retrieved.
        exterior: The geometric boundary of the learning space, specified as a tuple (X, Y, W, H),
            where X and Y represent the origin, and W and H represent the width and height of the
            space, respectively.

    Returns:
        Tensor: The retrieved spatial embeddings corresponding to the input coordinates.
    """

    return Function.apply(input, weight, padding, exterior)
