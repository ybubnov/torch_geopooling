// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>


namespace torch_geopooling {


torch::Tensor
embedding2d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::IntArrayRef& padding,
    const c10::ArrayRef<double>& exterior,
    bool reflection = true
);


torch::Tensor
embedding2d_backward(
    const torch::Tensor& grad,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::IntArrayRef& padding,
    const c10::ArrayRef<double>& exterior,
    bool reflection = true
);


} // namespace torch_geopooling
