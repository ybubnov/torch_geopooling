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

#include <functional>
#include <queue>
#include <unordered_map>
#include <vector>

#include <ATen/Functions.h>

#include <torch_geopooling/formatting.h>
#include <torch_geopooling/functional.h>
#include <torch_geopooling/quadtree_options.h>
#include <torch_geopooling/quadtree_set.h>


namespace torch_geopooling {


/// Tensor iterator is a class that unites tensor accessor and C++ iterator trait.
///
/// This class is used to access 2-dimensional tensor as contiguous vector of array. Template
/// parameter `N` defines the size of the arrays. Iterator takes only the first `N` elements
/// from the second dimension to the output result.
template <typename Scalar, int N>
class tensor_iterator2d
{
public:
    using iterator_category = std::forward_iterator_tag;

    using iterator = tensor_iterator2d<Scalar, N>;

    using value_type = std::array<Scalar, N>;

    using reference = value_type&;

    using pointer = value_type*;

    tensor_iterator2d(const torch::Tensor& tensor)
    : m_accessor(tensor.accessor<Scalar, 2>()),
      m_begin(0),
      m_end(tensor.size(0))
    {
        TORCH_CHECK(
            tensor.size(1) == N,
            "tensor_iterator2d: incompatible shape of a size(1) = ", tensor.size(1), ", expect ", N
        );
    }

    iterator
    begin()
    {
        return *this;
    }

    iterator
    end()
    {
        iterator it = *this;
        it.m_begin = it.m_end;
        return it;
    }

    iterator&
    operator++()
    {
        m_begin = std::min(m_begin + 1, m_end);
        return *this;
    }

    value_type
    operator[](std::size_t idx) const
    {
        value_type row;
        for (auto i = 0; i < N; i++) {
            row[i] = m_accessor[idx][i];
        }
        return row;
    }

    /// Return an element from 2-dimensional array at position i,j.
    Scalar
    at(std::size_t i, std::size_t j) const
    {
        return m_accessor[i][j];
    }

    value_type
    operator*()
    {
        // TODO: Ensure that m_begin does not cause index error exception.
        return (*this)[m_begin];
    }

    bool
    operator!=(const iterator& rhs)
    {
        return !(m_accessor.data() == rhs.m_accessor.data() && m_begin == rhs.m_begin);
    }

private:
    torch::TensorAccessor<Scalar, 2> m_accessor;
    std::size_t m_begin;
    std::size_t m_end;
};


/// Structure represents a quadtree operation.
///
/// On instance creation, it creates a tile index for fast access to the weights and biases.
/// Additionally, it checks validity of input data (tiles, indices, weight, bias, etc.), to
/// ensure it can be used to compute the operation.
template <typename Coordinate = double>
struct quadpool_op
{
    using tiles_iterator = tensor_iterator2d<int64_t, 3>;
    using input_iterator = tensor_iterator2d<Coordinate, 2>;

    using quadtree_exterior = c10::ArrayRef<Coordinate>;
    using weight_indices = std::unordered_map<Tile, int64_t>;
    using weight_values = std::unordered_map<Tile, torch::Tensor>;

    using tensor_reference = const torch::Tensor&;

    std::string m_op;
    quadtree_set<Coordinate> m_set;

    weight_indices m_indices;
    weight_values m_values;

    /// An indicator that operation is executed as part of the training path.
    bool m_training;

    quadpool_op(
        std::string op,
        const quadtree_exterior& exterior,
        const quadtree_options& options,
        bool training = false
    )
    : m_op(op),
      m_set(exterior.vec(), options),
      m_indices(),
      m_values(),
      m_training(training)
    {}

    std::tuple<torch::Tensor, torch::Tensor>
    forward(tensor_reference tiles, tensor_reference values, tensor_reference input)
    {
        check_tiles(tiles);
        check_weight(values);
        check_input(input);

        TORCH_CHECK(
            tiles.size(0) == values.size(0), m_op,
            ": number of tiles should be the same as weights ", tiles.size(0),
            " != ", values.size(0)
        );

        auto tiles_it = tiles_iterator(tiles);

        m_set = quadtree_set(tiles_it.begin(), tiles_it.end(), m_set.exterior(), m_set.options());
        m_indices.clear();
        m_values.clear();

        // Tiles locations are the same as weight locations, therefore it's possible to simply
        // insert indices of the weights sequentially. Then after each sub-division of a quad
        // we will create a new weight in the weight vector.
        //
        // Question: how could we guarantee the same order of tiles in the output? Answer: we
        // don't need to since m_indices already contains all mappings.
        auto tiles_size = tiles.size(0);

        // When an empty list of tiles was passed in, add references to the root tile (it's
        // automatically added by the `torch_geopooling::quadtree_set`.
        if (tiles_size == 0) {
            auto value = torch::zeros({values.size(1)}, values.options());
            m_values.insert(std::make_pair(Tile(Tile::root), value));
            m_indices.insert(std::make_pair(Tile(Tile::root), 0));
        }

        for (int64_t i = 0; i < tiles_size; i++) {
            auto tile = Tile(tiles_it[i]);
            auto value = values[i];
            auto index = m_indices.size();

            m_values.insert(std::make_pair(tile, value));
            m_indices.insert(std::make_pair(tile, index));
        }

        torch::Tensor tiles_out, values_out;

        if (m_training) {
            auto input_it = input_iterator(input);

            m_set.insert(input_it.begin(), input_it.end(), [&](Tile parent_tile, Tile child_tile) {
                auto value = m_values.at(parent_tile);
                int64_t index = m_indices.size();

                m_values.insert(std::make_pair(child_tile, value));
                m_indices.insert(std::make_pair(child_tile, index));
            });

            // After the modification of a tree, update the final tiles and weights.
            std::size_t values_size = m_values.size();
            std::vector<torch::Tensor> tiles_vec(values_size);
            std::vector<torch::Tensor> values_vec(values_size);

            for (const auto& [tile, index] : m_indices) {
                tiles_vec[index] = torch::tensor(tile.template vec<int64_t>(), tiles.options());
                values_vec[index] = m_values.at(tile);
            }

            tiles_out = torch::stack(tiles_vec);
            values_out = torch::stack(values_vec);
            values_out = values_out.set_requires_grad(values.requires_grad());
        } else {
            tiles_out = tiles;
            values_out = values;
        }

        return std::make_tuple(tiles_out, values_out);
    }

    input_iterator
    make_input_iterator(tensor_reference input)
    {
        return input_iterator(input);
    }

    /// Select index of the tile that contains a specified point.
    int64_t
    index_select(const typename input_iterator::value_type& point)
    {
        const auto& node = m_set.find(point);
        return m_indices.at(node.tile());
    }

    int64_t
    index_select(const Tile& tile) const
    {
        return m_indices.at(tile);
    }

    torch::Tensor
    value_select(const Tile& tile) const
    {
        return m_values.at(tile);
    }

private:
    const quadtree_exterior&
    check_exterior(const quadtree_exterior& exterior) const
    {
        TORCH_CHECK(exterior.size() == 4, m_op, ": exterior must be a tuple of four doubles");
        return exterior;
    }

    tensor_reference
    check_tiles(tensor_reference tiles) const
    {
        TORCH_CHECK(
            tiles.dim() == 2, m_op, ": operation only supports 2D tiles, got ", tiles.dim(), "D"
        );
        TORCH_CHECK(tiles.size(1) == 3, m_op, ": tiles must be three-element tuples");
        TORCH_CHECK(
            tiles.dtype() == torch::kInt64, m_op, ": operation only supports Int64 tiles, got ",
            tiles.dtype()
        );
        return tiles;
    }

    tensor_reference
    check_input(tensor_reference input) const
    {
        TORCH_CHECK(
            input.dim() == 2, m_op, ": operation only supports 2D input, got ", input.dim(), "D"
        );
        TORCH_CHECK(input.size(1) == 2, m_op, ": input must be two-element tuples");
        TORCH_CHECK(
            input.dtype() == torch::kFloat64, m_op, ": operation only supports Float64 input, got ",
            input.dtype()
        );
        return input;
    }

    void
    check_weight(tensor_reference weight) const
    {
        TORCH_CHECK(
            weight.dim() == 2, m_op, ": operation only supports 2D weight, got ", weight.dim(), "D"
        );
    }
};


/// Structure represents a aggregation (statistic) quadtree operation.
///
/// Forward method creates a statistic index for fast access by a tile.
///
/// There are two functions used to build a stat quadtree: `init` and `stat`, which essentially
/// are used to compute values for terminal and intermediate nodes respectively. The major
/// implication of this separation is that `init` part should work only with `m_indices`,
/// since `m_stats` is not yet created. And `stat` function is called during construction of
/// `m_stats`.
template <typename Coordinate = double, typename Statistic = torch::Tensor>
struct quadpool_stat_op : public quadpool_op<Coordinate>
{
    using base = quadpool_op<Coordinate>;
    using tensor_reference = const torch::Tensor&;

    using stat_quadtree_index = std::unordered_map<Tile, Statistic>;

    using init_function = std::function<Statistic(const base&, const Tile&)>;
    using stat_function = std::function<Statistic(const quadpool_stat_op&, std::vector<Tile>&)>;

    init_function m_init_function;
    stat_function m_stat_function;

    /// Stat tile index is comprised of both terminal and intermediate nodes.
    stat_quadtree_index m_stats;

    quadpool_stat_op(
        std::string op,
        init_function init_function,
        stat_function stat_function,
        const typename base::quadtree_exterior& exterior,
        const quadtree_options& options,
        bool training
    )
    : quadpool_op<Coordinate>(op, exterior, options, training),
      m_init_function(init_function),
      m_stat_function(stat_function),
      m_stats()
    {}

    std::tuple<torch::Tensor, torch::Tensor>
    forward(tensor_reference tiles, tensor_reference values, tensor_reference input)
    {
        auto result = base::forward(tiles, values, input);

        std::priority_queue<Tile, std::vector<Tile>, std::less<Tile>> unvisited;

        // Iterate over terminal nodes of the quadtree index and calculate an associated
        // weight. Since this is a terminal node, then statistic will be the value from the
        // weight tensor itself.
        for (auto node : base::m_set) {
            auto tile = node.tile();
            auto stat = m_init_function(*this, tile);

            m_stats.insert(std::make_pair(tile, stat));
            unvisited.push(tile.parent());
        }

        while (!unvisited.empty()) {
            auto tile = unvisited.top();
            unvisited.pop();

            // If the tile is missing from the statistics map, then compute a statistic
            // for the all it's children and insert a new record into the map.
            //
            // Since we are using a queue, we can be sure that we iterate layer over layer
            // of the quadtree, so all children should be presented in a stat tile index.
            if (auto it = m_stats.find(tile); it != m_stats.end()) {
                continue;
            }

            auto children_tiles = tile.children();
            auto stat = m_stat_function(*this, children_tiles);
            m_stats.insert(std::make_pair(tile, stat));

            // Do not compute the root tile multiple times, once is enough.
            if (tile != Tile::root) {
                unvisited.push(tile.parent());
            }
        }

        return result;
    }

    std::vector<Statistic>
    stats_select(const std::vector<Tile>& tiles, bool missing_ok = false) const
    {
        std::vector<Statistic> stats;
        for (const auto& tile : tiles) {
            auto stat = m_stats.find(tile);
            if (stat != m_stats.end()) {
                stats.push_back(stat->second);
                continue;
            }
            if (!missing_ok) {
                throw value_error("quadpool_stat_op: tile {} not found in stat index", tile);
            }
        }

        return stats;
    }
};


} // namespace torch_geopooling
