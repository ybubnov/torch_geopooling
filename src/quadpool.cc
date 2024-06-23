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
quad_pool2d(
    const torch::Tensor& tiles,
    const torch::Tensor& weight,
    const torch::Tensor& input,
    const c10::ArrayRef<double>& exterior,
    bool training,
    std::optional<std::size_t> max_terminal_nodes,
    std::optional<std::size_t> max_depth,
    std::optional<std::size_t> capacity,
    std::optional<std::size_t> precision
)
{
    auto options = quadtree_options()
        .max_terminal_nodes(max_terminal_nodes)
        .max_depth(max_depth)
        .precision(precision)
        .capacity(capacity);

    auto op = quadpool_op("quad_pool2d", exterior, options, /*training=*/training);
    auto [tiles_out, weight_out] = op.forward(tiles, weight, input);

    const int64_t grain_size = at::internal::GRAIN_SIZE;
    const int64_t input_size = input.size(0);

    std::vector<int64_t> indices(input_size);
    auto input_it = op.make_input_iterator(input);

    // This loop might raise an exception, when the tile returned by `set.find` operation
    // returns non-terminal node. This should not happen in practice.
    at::parallel_for(0, input_size, grain_size, [&](int64_t begin, int64_t end) {
        for (const auto i : c10::irange(begin, end)) {
            indices[i] = op.index_select(input_it[i]);
        }
    });

    auto values_out = at::index_select(weight_out, 0, torch::tensor(indices));

    return std::make_tuple(tiles_out, weight_out, values_out);
}


torch::Tensor
quad_pool2d_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& tiles,
    const torch::Tensor& weight,
    const torch::Tensor& input,
    const c10::ArrayRef<double>& exterior,
    std::optional<std::size_t> max_terminal_nodes,
    std::optional<std::size_t> max_depth,
    std::optional<std::size_t> capacity,
    std::optional<std::size_t> precision
)
{
    auto options = quadtree_options()
        .max_terminal_nodes(max_terminal_nodes)
        .max_depth(max_depth)
        .precision(precision)
        .capacity(capacity);

    quadpool_op op("quad_pool2d_backward", exterior, options, /*training=*/false);
    op.forward(tiles, weight, input);

    const int64_t input_size = input.size(0);
    const int64_t weight_size = weight.size(0);
    const int64_t grain_size = at::internal::GRAIN_SIZE;

    auto input_it = op.make_input_iterator(input);
    auto grad_weight = at::zeros(weight.sizes(), grad_output.options());

    at::parallel_for(0, weight_size, grain_size, [&](int64_t begin, int64_t end) {
        for (const auto input_index : c10::irange(input_size)) {
            auto index = op.index_select(input_it[input_index]);
            if (index >= begin && index < end) {
                grad_weight[index] += grad_output[input_index];
            }
        }
    });

    return grad_weight;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
max_quad_pool2d(
    const torch::Tensor& tiles,
    const torch::Tensor& weight,
    const torch::Tensor& input,
    const c10::ArrayRef<double>& exterior,
    bool training,
    std::optional<std::size_t> max_terminal_nodes,
    std::optional<std::size_t> max_depth,
    std::optional<std::size_t> capacity,
    std::optional<std::size_t> precision
)
{
    auto options = quadtree_options()
        .max_terminal_nodes(max_terminal_nodes)
        .max_depth(max_depth)
        .precision(precision)
        .capacity(capacity);

    using coordinate_type = double;
    using result_type = torch::Tensor;

    using init_op_type = quadpool_op<coordinate_type>;
    using stat_op_type = quadpool_stat_op<coordinate_type, result_type>;

    // Within initialization step, select a weight associated with the given tile.
    auto init_fn = [&](const init_op_type& op, const Tile& tile) -> result_type {
        return op.value_select(tile);
    };
    // Within statistics calculation step, select weights (those are results) associated
    // with the each node of a stat Quadtree and select a maximum value amongst them.
    //
    // Since we know that init function produces as a result torch::Tensor, we could just
    // stack results and calculate maximum, without additional processing.
    auto stat_fn = [&](const stat_op_type& op, const std::vector<Tile>& tiles) -> result_type {
        auto weights = torch::stack(op.stats_select(tiles, /*missing_ok=*/true));
        auto [max_tensor, _] = at::max(weights, 0);
        return max_tensor;
    };

    stat_op_type op(
        "max_quad_pool2d",
        init_fn, stat_fn, exterior, options, /*training=*/training
    );

    auto [tiles_out, weight_out] = op.forward(tiles, weight, input);

    const int64_t grain_size = at::internal::GRAIN_SIZE;
    const int64_t input_size = input.size(0);

    std::vector<torch::Tensor> values_out_vec(input_size);
    auto input_it = op.make_input_iterator(input);

    at::parallel_for(0, input_size, grain_size, [&](int64_t begin, int64_t end) {
        for (const auto i : c10::irange(begin, end)) {
            std::vector<int64_t> indices;
            const auto& point = input_it[i];

            auto node = op.m_set.find(point);
            auto stat = op.m_stats.at(node.tile().parent());

            values_out_vec[i] = stat;
        }
    });

    torch::Tensor values_out = at::squeeze(torch::stack(values_out_vec), 1);

    return std::make_tuple(tiles_out, weight_out, values_out);
}


torch::Tensor
max_quad_pool2d_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& tiles,
    const torch::Tensor& weight,
    const torch::Tensor& input,
    const c10::ArrayRef<double>& exterior,
    std::optional<std::size_t> max_terminal_nodes,
    std::optional<std::size_t> max_depth,
    std::optional<std::size_t> capacity,
    std::optional<std::size_t> precision
)
{
    auto options = quadtree_options()
        .max_terminal_nodes(max_terminal_nodes)
        .max_depth(max_depth)
        .precision(precision)
        .capacity(capacity);

    using coordinate_type = double;
    using result_type = std::tuple<torch::Tensor, torch::Tensor>;

    using init_op_type = quadpool_op<coordinate_type>;
    using stat_op_type = quadpool_stat_op<coordinate_type, result_type>;

    const auto feature_size = weight.size(1);
    const auto feature_indices = at::arange(feature_size);

    // For computation of a backward path, we need to maintain both, maximum value
    // and index of that maximum value from the weights vector.
    auto init_fn = [&](const init_op_type& op, const Tile& tile) -> result_type {
        auto indices = std::vector<int64_t>(feature_size, op.index_select(tile));
        auto max_values = op.value_select(tile);

        return std::make_tuple(torch::tensor(indices), max_values);
    };
    auto stat_fn = [&](const stat_op_type& op, const std::vector<Tile>& tiles) -> result_type {
        auto stats = op.stats_select(tiles, /*missing_ok=*/true);

        std::vector<torch::Tensor> indices_vec;
        std::vector<torch::Tensor> values_vec;

        for (auto& stat : stats) {
            indices_vec.push_back(std::get<0>(stat));
            values_vec.push_back(std::get<1>(stat));
        }

        auto indices = torch::stack(indices_vec);
        auto values = torch::stack(values_vec);

        auto [max_values, max_indices] = at::max(values, 0);
        indices = indices.index({max_indices.squeeze(), feature_indices});

        return std::make_tuple(indices, max_values);
    };

    stat_op_type op(
        "max_quad_pool2d_backward",
        init_fn, stat_fn, exterior, options, /*training=*/false
    );

    op.forward(tiles, weight, input);
    const int64_t weight_size = weight.size(0);

    const int64_t input_size = input.size(0);
    const int64_t grain_size = at::internal::GRAIN_SIZE;

    auto input_it = op.make_input_iterator(input);
    auto grad_weight = at::zeros(weight.sizes(), grad_output.options());

    at::parallel_for(0, weight_size, grain_size, [&](int64_t begin, int64_t end) {
        for (const auto input_index : c10::irange(input_size)) {
            const auto& point = input_it[input_index];
            const auto& node = op.m_set.find(point);

            auto [indices, _] = op.m_stats.at(node.tile().parent());
            auto indices_acc = indices.accessor<int64_t, 1>();

            for (const auto feature_index : c10::irange(feature_size)) {
                auto maxindex = indices_acc[feature_index];
                if (maxindex >= begin && maxindex < end) {
                    grad_weight[maxindex][feature_index] += grad_output[input_index][feature_index];
                }
            }
        }
    });

    return grad_weight;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
avg_quad_pool2d(
    const torch::Tensor& tiles,
    const torch::Tensor& weight,
    const torch::Tensor& input,
    const c10::ArrayRef<double>& exterior,
    bool training,
    std::optional<std::size_t> max_terminal_nodes,
    std::optional<std::size_t> max_depth,
    std::optional<std::size_t> capacity,
    std::optional<std::size_t> precision
)
{
    auto options = quadtree_options()
        .max_terminal_nodes(max_terminal_nodes)
        .max_depth(max_depth)
        .precision(precision)
        .capacity(capacity);

    using coordinate_type = double;
    using result_type = std::tuple<torch::Tensor, torch::Tensor>;

    using init_op_type = quadpool_op<coordinate_type>;
    using stat_op_type = quadpool_stat_op<coordinate_type, result_type>;

    std::vector<int64_t> feature_size = {1, weight.size(1)};
    auto weight_options = weight.options();

    auto init_fn = [&](const init_op_type& op, const Tile& tile) -> result_type {
        auto sum = op.value_select(tile);
        auto count = torch::ones(feature_size, weight_options);
        return std::make_tuple(sum, count);
    };
    auto stat_fn = [&](const stat_op_type& op, const std::vector<Tile>& tiles) -> result_type {
        auto stats = op.stats_select(tiles, /*missing_ok=*/true);

        auto sum = torch::zeros(feature_size, weight_options);
        auto count = torch::zeros(feature_size, weight_options);

        for (const auto& stat : stats) {
            sum += std::get<0>(stat);
            count += std::get<1>(stat);
        }
        return std::make_tuple(sum, count);
    };

    stat_op_type op(
        "avg_quad_pool2d",
        init_fn, stat_fn, exterior, options, /*training=*/training
    );

    auto [tiles_out, weight_out] = op.forward(tiles, weight, input);

    const int64_t grain_size = at::internal::GRAIN_SIZE;
    const int64_t input_size = input.size(0);

    auto values_out_vec = std::vector<torch::Tensor>(input_size);
    auto input_it = op.make_input_iterator(input);

    at::parallel_for(0, input_size, grain_size, [&](int64_t begin, int64_t end) {
        for (const auto i : c10::irange(begin, end)) {
            std::vector<int64_t> indices;
            const auto& point = input_it[i];

            auto node = op.m_set.find(point);
            auto stat = op.m_stats.at(node.tile().parent());

            values_out_vec[i] = std::get<0>(stat) / std::get<1>(stat);
        }
    });

    auto values_out = at::squeeze(torch::stack(values_out_vec), 1);

    return std::make_tuple(tiles_out, weight_out, values_out);
}


torch::Tensor
avg_quad_pool2d_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& tiles,
    const torch::Tensor& weight,
    const torch::Tensor& input,
    const c10::ArrayRef<double>& exterior,
    std::optional<std::size_t> max_terminal_nodes,
    std::optional<std::size_t> max_depth,
    std::optional<std::size_t> capacity,
    std::optional<std::size_t> precision
)
{
    auto options = quadtree_options()
        .max_terminal_nodes(max_terminal_nodes)
        .max_depth(max_depth)
        .precision(precision)
        .capacity(capacity);

    auto weight_options = weight.options();

    using coordinate_type = double;
    using result_type = torch::Tensor;

    using init_op_type = quadpool_op<coordinate_type>;
    using stat_op_type = quadpool_stat_op<coordinate_type, result_type>;

    std::vector<int64_t> feature_size = {weight.size(1)};

    auto init_fn = [&](const init_op_type& op, const Tile& tile) -> result_type {
        return torch::ones(feature_size, weight_options);
    };
    auto stat_fn = [&](const stat_op_type& op, const std::vector<Tile>& tiles) -> result_type {
        auto results = op.stats_select(tiles, /*missing_ok=*/true);
        auto count = torch::zeros(feature_size, weight_options);

        for (const auto& result : results) {
            count += result;
        }
        return count;
    };

    stat_op_type op(
        "avg_quad_pool2d_backward",
        init_fn, stat_fn, exterior, options, /*training=*/false
    );

    op.forward(tiles, weight, input);

    const int64_t input_size = input.size(0);
    const int64_t grain_size = at::internal::GRAIN_SIZE;
    const int64_t weight_size = weight.size(0);

    auto input_it = op.make_input_iterator(input);
    auto grad_weight = at::zeros(weight.sizes(), grad_output.options());

    at::parallel_for(0, weight_size, grain_size, [&](int64_t begin, int64_t end) {
        for (const auto input_index : c10::irange(input_size)) {
            const auto& point = input_it[input_index];
            const auto& node = op.m_set.find(point);

            auto count = op.m_stats.at(node.tile().parent());

            for (
                auto node = op.m_set.find_terminal_group(point);
                node != op.m_set.end();
                ++node
            ) {
                auto index = op.m_indices.at(node->tile());
                if (index >= begin && index < end) {
                    grad_weight[index] += grad_output[input_index] / count;
                }
            }
        }
    });

    return grad_weight;
}

} // namespace torch_geopooling
