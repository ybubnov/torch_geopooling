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

from typing import Union, Tuple, cast

import torch
from torch import Tensor, nn

from torch_geopooling import functional as F
from torch_geopooling.tiling import Exterior, ExteriorTuple


__all__ = [
    "Embedding2d",
]


_Exterior = Union[Exterior, ExteriorTuple]


class Embedding2d(nn.Module):
    """
    Retrieves spatial embeddings from a fixed-size lookup table based on 2D coordinates.

    This module accepts a tensor of (x, y) coordinates and retrieves the corresponding
    spatial embeddings from a provided embedding matrix. The embeddings are selected
    based on the input coordinates, with an optional padding to include neighboring cells.

    Args:
        manifold: The size of the 2-dimensional embedding in a form (W, H, N), where
            W is a width, H is a height, and N is a feature dimension of the embedding.
        padding: The size of the neighborhood to query. Default is 0, meaning only the embedding
            for the exact input coordinate is retrieved.
        exterior: The geometric boundary of the learning space, specified as a tuple (X, Y, W, H),
            where X and Y represent the origin, and W and H represent the width and height of the
            space, respectively.
        reflection: When true, kernel is wrapped around the exterior space, otherwise kernel is
            squeezed into borders.

    Shape:
        - Input: :math:`(*, 2)`, where 2 comprises x and y coordinates.
        - Output: :math:`(*, X_{out}, Y_{out}, N)`, where * is the input shape, \
            :math:`N = \\text{manifold[2]}`, and

            :math:`X_{out} = \\text{padding}[0] \\times 2 + 1`

            :math:`Y_{out} = \\text{padding}[1] \\times 2 + 1`

    Examples:

    >>> # Create an embedding of EPSG:4326 rectangle into 1024x1024 embedding
    >>> # with 3 features in each cell.
    >>> embedding = nn.Embedding2d(
    ...     (1024, 1024, 3),
    ...     exterior=(-180.0, -90.0, 360.0, 180.0),
    ...     padding=(2, 2),
    ... )
    >>> input = torch.rand((100, 2), dtype=torch.float64) * 60.0
    >>> output = embedding(input)
    """

    def __init__(
        self,
        manifold: Tuple[int, int, int],
        exterior: _Exterior,
        padding: Tuple[int, int] = (0, 0),
        reflection: bool = True,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.exterior = cast(ExteriorTuple, tuple(map(float, exterior)))
        self.padding = padding
        self.reflection = reflection

        self.weight = nn.Parameter(torch.empty(manifold, dtype=torch.float64))
        nn.init.zeros_(self.weight)

    def extra_repr(self) -> str:
        return "{manifold}, exterior={exterior}, padding={padding}, reflection={reflection}".format(
            **self.__dict__
        )

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding2d(
            input,
            self.weight,
            exterior=self.exterior,
            padding=self.padding,
            reflection=self.reflection,
        )
