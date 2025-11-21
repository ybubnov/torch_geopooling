// SPDX-License-Identifier: GPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Yakau Bubnou
// SPDX-FileType: SOURCE

#pragma once

#include <optional>
#include <tuple>

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>


namespace torch_geopooling {


/// Dynamic lookup index over Quadtree decomposition of input 2D coordinates.
///
/// This function constructs an internal lookup quadtree to organize closely situated 2D points.
/// Each terminal node in the resulting quadtree is paired with a weight. Thus, when providing
/// an input coordinate, the module retrieves the corresponding terminal node and returns its
/// associated weight.
///
/// This function is stateless, but training could change internal quadtree, therefore it
/// returns quadtree tiles to reconstruct the learned quadtree on the next evaluation iteration.
///
/// \return tuple of tree elements: (tiles, weight, values).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
quad_pool2d(
    const torch::Tensor& tiles,
    const torch::Tensor& weight,
    const torch::Tensor& input,
    const c10::ArrayRef<double>& exterior,
    bool training = false,
    std::optional<std::size_t> max_terminal_nodes = std::nullopt,
    std::optional<std::size_t> max_depth = std::nullopt,
    std::optional<std::size_t> capacity = std::nullopt,
    std::optional<std::size_t> precision = std::nullopt
);


torch::Tensor
quad_pool2d_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& tiles,
    const torch::Tensor& weight,
    const torch::Tensor& input,
    const c10::ArrayRef<double>& exterior,
    std::optional<std::size_t> max_terminal_nodes = std::nullopt,
    std::optional<std::size_t> max_depth = std::nullopt,
    std::optional<std::size_t> capacity = std::nullopt,
    std::optional<std::size_t> precision = std::nullopt
);


/// Dynamic maximum pooling over Quadtree decomposition of input 2D coordinates.
///
/// This function constructs an internal lookup quadtree to organize closely situated 2D points.
/// Each terminal node in the resulting quadtree is assigned a weight value. For each input
/// coordinate, the module queries a "terminal group" of nodes and calculates the maximum value
/// from a `weight` vector associated with these nodes.
///
/// Terminal node group is a set of nodes in a lookup quadtree that share the common parent.
///
/// \return tuple of two elements: (tiles, weights, values).
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
max_quad_pool2d(
    const torch::Tensor& tiles,
    const torch::Tensor& weight,
    const torch::Tensor& input,
    const c10::ArrayRef<double>& exterior,
    bool training = false,
    std::optional<std::size_t> max_terminal_nodes = std::nullopt,
    std::optional<std::size_t> max_depth = std::nullopt,
    std::optional<std::size_t> capacity = std::nullopt,
    std::optional<std::size_t> precision = std::nullopt
);


torch::Tensor
max_quad_pool2d_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& tiles,
    const torch::Tensor& weight,
    const torch::Tensor& input,
    const c10::ArrayRef<double>& exterior,
    std::optional<std::size_t> max_terminal_nodes = std::nullopt,
    std::optional<std::size_t> max_depth = std::nullopt,
    std::optional<std::size_t> capacity = std::nullopt,
    std::optional<std::size_t> precision = std::nullopt
);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
avg_quad_pool2d(
    const torch::Tensor& tiles,
    const torch::Tensor& weight,
    const torch::Tensor& input,
    const c10::ArrayRef<double>& exterior,
    bool training = false,
    std::optional<std::size_t> max_terminal_nodes = std::nullopt,
    std::optional<std::size_t> max_depth = std::nullopt,
    std::optional<std::size_t> capacity = std::nullopt,
    std::optional<std::size_t> precision = std::nullopt
);


torch::Tensor
avg_quad_pool2d_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& tiles,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::ArrayRef<double>& exterior,
    std::optional<std::size_t> max_terminal_nodes = std::nullopt,
    std::optional<std::size_t> max_depth = std::nullopt,
    std::optional<std::size_t> capacity = std::nullopt,
    std::optional<std::size_t> precision = std::nullopt
);

} // namespace torch_geopooling
