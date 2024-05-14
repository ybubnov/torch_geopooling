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


#include <array>
#include <functional>
#include <iterator>
#include <unordered_map>
#include <vector>
#include <queue>

#include <ATen/Functions.h>
#include <ATen/Parallel.h>
#include <ATen/TensorAccessor.h>

#include <torch_geopooling/formatting.h>
#include <torch_geopooling/functional.h>
#include <torch_geopooling/quadpool.h>
#include <torch_geopooling/quadtree_options.h>
#include <torch_geopooling/quadtree_set.h>

#include <quadpool_op.h>


namespace torch_geopooling {


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
linear_quad_pool2d(
    const torch::Tensor& tiles,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const c10::ArrayRef<double>& exterior,
    bool training,
    std::optional<std::size_t> max_depth,
    std::optional<std::size_t> capacity,
    std::optional<std::size_t> precision
)
{
    auto options = quadtree_options()
        .max_terminal_nodes(weight.size(0))
        .max_depth(max_depth)
        .precision(precision)
        .capacity(capacity);

    quadpool_op op("linear_quad_pool2d", tiles, input, exterior, options, training);

    const int64_t grain_size = at::internal::GRAIN_SIZE;
    const int64_t total_size = input.size(0);

    std::vector<int32_t> indices(total_size);
    auto input_it = op.make_input_iterator(input);

    // This loop might raise an exception, when the tile returned by `set.find` operation
    // returns non-terminal node. This should not happen in practice.

    at::parallel_for(0, total_size, grain_size, [&](int64_t begin, int64_t end) {
        for (const auto i : c10::irange(begin, end)) {
            const auto& point = input_it[i];
            const auto& node = op.m_set.find(point);
            indices[i] = op.m_tile_index.at(node.tile());
        }
    });

    torch::Tensor tiles_out = op.forward_tiles(tiles);
    auto [weight_out, bias_out] = op.forward(indices, weight, bias);

    return std::make_tuple(tiles_out, weight_out, bias_out);
}


std::tuple<torch::Tensor, torch::Tensor>
max_quad_pool2d(
    const torch::Tensor& tiles,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::ArrayRef<double>& exterior,
    bool training,
    std::optional<std::size_t> max_depth,
    std::optional<std::size_t> capacity,
    std::optional<std::size_t> precision
)
{
    auto options = quadtree_options()
        .max_terminal_nodes(weight.size(0))
        .max_depth(max_depth)
        .precision(precision)
        .capacity(capacity);

    auto max_fn = [](const torch::Tensor& t) -> torch::Tensor {
        return at::unsqueeze(at::max(t), 0);
    };

    quadpool_stat_op op(
        "max_quad_pool2d", max_fn, tiles, input, weight, exterior, options, training
    );

    const int64_t grain_size = at::internal::GRAIN_SIZE;
    const int64_t total_size = input.size(0);

    std::vector<torch::Tensor> weight_out_vec(total_size);
    auto input_it = op.make_input_iterator(input);

    at::parallel_for(0, total_size, grain_size, [&](int64_t begin, int64_t end) {
        for (const auto i : c10::irange(begin, end)) {
            std::vector<int32_t> indices;
            const auto& point = input_it[i];

            auto node = op.m_set.find(point);
            auto stat = op.m_stat_tile_index.at(node.tile().parent());

            weight_out_vec[i] = stat;
        }
    });

    torch::Tensor tiles_out = op.forward_tiles(tiles);
    torch::Tensor weight_out = at::squeeze(torch::stack(weight_out_vec), 1);

    return std::make_tuple(tiles_out, weight_out);
}


} // namespace torch_geopooling
