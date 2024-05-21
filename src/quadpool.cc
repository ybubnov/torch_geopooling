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

#include <ATen/Dispatch.h>
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

    using coordinate_type = double;
    using index_type = int32_t;
    using result_type = torch::Tensor;

    using init_operation_reference = const quadpool_op<coordinate_type, index_type>&;
    using stat_operation_reference = const quadpool_stat_op<coordinate_type, index_type, result_type>&;

    /// Within initialization step, select a weight associated with the given tile.
    auto init_fn = [&](init_operation_reference op, const Tile& tile) -> result_type {
        return op.weight_select(tile, weight);
    };
    /// Within statistics calculation step, select weights (those are results) associated
    /// with the each node of a stat Quadtree and select a maximum value amongst them.
    ///
    /// Since we know that init function produces as a result torch::Tensor, we could just
    /// stack results and calculate maximum, without additional processing.
    auto stat_fn = [&](stat_operation_reference op, const std::vector<Tile>& tiles) -> result_type {
        auto weights = torch::stack(op.result_select(tiles, weight, /*missing_ok=*/true));
        return at::unsqueeze(at::max(weights), 0);
    };

    quadpool_stat_op<coordinate_type, index_type, result_type> op(
        "max_quad_pool2d", init_fn, stat_fn, tiles, input, exterior, options, training
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


torch::Tensor
max_quad_pool2d_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& tiles,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::ArrayRef<double>& exterior,
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

    using coordinate_type = double;
    using index_type = int32_t;
    using result_type = std::tuple<index_type, torch::Tensor>;

    using init_op_type = quadpool_op<coordinate_type, index_type>;
    using stat_op_type = quadpool_stat_op<coordinate_type, index_type, result_type>;

    /// For computation of a backward path, we need to maintain both, maximum value
    /// and index of that maximum value in the weights vector.
    auto init_fn = [&](const init_op_type& op, const Tile& tile) -> result_type {
        return op.weight_index(tile, weight);
    };
    auto stat_fn = [&](const stat_op_type& op, const std::vector<Tile>& tiles) -> result_type {
        auto results = op.result_select(tiles, weight, /*missing_ok=*/true);

        std::vector<torch::Tensor> weights;
        std::transform(
            results.begin(),
            results.end(),
            std::back_insert_iterator(weights),
            [&](result_type result) -> torch::Tensor { return std::get<1>(result); }
        );

        auto [max_tensor, index] = at::max(torch::stack(weights), 0);
        auto max_index = std::get<0>(results[index.item<int32_t>()]);

        return std::make_tuple(max_index, max_tensor);
    };

    stat_op_type op(
        "max_quad_pool2d_backward",
        init_fn, stat_fn, tiles, input, exterior, options, /*training=*/false
    );

    const int64_t input_size = input.size(0);
    const int64_t grain_size = at::internal::GRAIN_SIZE;
    const int64_t total_size = options.max_terminal_nodes();

    auto input_it = op.make_input_iterator(input);
    auto grad_weight = at::zeros(weight.sizes(), grad_output.options());

    at::parallel_for(0, total_size, grain_size, [&](int64_t begin, int64_t end) {
        for (const auto index : c10::irange(input_size)) {
            const auto& point = input_it[index];
            const auto& node = op.m_set.find(point);

            auto result = op.m_stat_tile_index.at(node.tile().parent());
            auto maxindex = std::get<0>(result);

            if (maxindex >= begin && maxindex < end) {
                grad_weight[maxindex] += grad_output[index];
            }
        }
    });

    return grad_weight;
}


/*
std::tuple<torch::Tensor, torch::Tensor>
avg_quad_pool2d(
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

    auto avg_fn = [](const torch::Tensor& t) -> torch::Tensor {
        return at::unsqueeze(at::mean(t), 0);
    };

    quadpool_stat_op op(
        "avg_quad_pool2d", avg_fn, tiles, input, weight, exterior, options, training
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
*/


} // namespace torch_geopooling
