# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Yakau Bubnou
# SPDX-FileType: SOURCE

from typing import NamedTuple

from torch import Tensor


class quad_pool2d(NamedTuple):
    tiles: Tensor
    weight: Tensor
    values: Tensor


class adaptive_quad_pool2d(NamedTuple):
    weight: Tensor
    values: Tensor
