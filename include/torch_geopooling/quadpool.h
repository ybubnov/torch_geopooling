/// Copyright (C) 2024, Yakau Bubnou
///
/// This program is free software: you can redistribute it and/or modify
/// it under the terms of the GNU General Public License as published by
/// the Free Software Foundation, either version 3 of the License, or
/// (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
