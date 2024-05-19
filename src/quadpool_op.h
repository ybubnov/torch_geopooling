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
#include <unordered_map>
#include <vector>
#include <queue>

#include <ATen/Functions.h>

#include <torch_geopooling/formatting.h>
#include <torch_geopooling/functional.h>
#include <torch_geopooling/quadtree_set.h>
#include <torch_geopooling/quadtree_options.h>


namespace torch_geopooling {


/// Tensor iterator is a class that unites tensor accessor and C++ iterator trait.
///
/// This class is used to access 2-dimensional tensor as contiguous vector of array. Template
/// parameter `N` defines the size of the arrays. Iterator takes only the first `N` elements
/// from the second dimension to the output result.
template<typename Scalar, int N>
class tensor_iterator2d {
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
            "tensor_iterator2d: incompatible shape of a size(1) = ",
            tensor.size(1), ", expect ", N
        );
    }

    iterator
    begin()
    { return *this; }

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
template<typename Coordinate = double, typename Index = int32_t>
struct quadpool_op
{
    using tiles_iterator = tensor_iterator2d<Index, 3>;
    using input_iterator = tensor_iterator2d<Coordinate, 2>;

    using quadtree_exterior = c10::ArrayRef<Coordinate>;
    using quadtree_index = std::unordered_map<Tile, Index>;

    using tensor_reference = const torch::Tensor&;

    std::string m_op;
    quadtree_set<Coordinate> m_set;

    /// Tile index is comprised of terminal quadtree nodes.
    quadtree_index m_tile_index;

    /// An indicator that operation is executed as part of the training path.
    bool m_training;

    quadpool_op(
        std::string op,
        tiles_iterator tiles_it,
        const quadtree_exterior& exterior,
        const quadtree_options& options,
        bool training
    )
    : m_op(op),
      m_set(tiles_it.begin(), tiles_it.end(), check_exterior(exterior).vec(), options),
      m_tile_index(),
      m_training(training)
    { }

    quadpool_op(
        std::string op,
        tensor_reference tiles,
        tensor_reference input,
        const quadtree_exterior& exterior,
        const quadtree_options& options,
        bool training
    )
    : quadpool_op(op, tiles_iterator(check_tiles(tiles)), exterior, options, training)
    {
        if (training) {
            input_iterator input_it(check_input(input));
            m_set.insert(input_it.begin(), input_it.end());
        }

        // The tile index will change once the training iteration adjusts the quadtree set
        // structure. Since quads are embedded into a 1D tensor, there will be a drift of
        // weights. But with a large enough number of training iterations, these tiles map
        // eventually converges.
        //
        // Tile index is comprised only from terminal nodes.
        for (auto node_it = m_set.begin(); node_it != m_set.end(); ++node_it) {
            auto tile = node_it->tile();
            m_tile_index.insert(std::make_pair(tile, m_tile_index.size()));
        }
    }

    input_iterator
    make_input_iterator(tensor_reference input)
    {
        return input_iterator(input);
    }

    torch::Tensor
    forward_tiles(tensor_reference tiles) const
    {
        if (!m_training) {
            return tiles;
        }

        std::vector<torch::Tensor> tile_rows;

        for (auto node_it = m_set.ibegin(); node_it != m_set.iend(); ++node_it) {
            auto tile = node_it->tile();
            tile_rows.push_back(torch::tensor(tile.template vec<Index>(), tiles.options()));
        }

        return torch::stack(tile_rows);
    }

    std::tuple<torch::Tensor, torch::Tensor>
    forward(
        tensor_reference index,
        tensor_reference weight,
        tensor_reference bias
    ) const
    {
        check_weight_and_bias(weight, bias);

        torch::Tensor weight_out = at::index_select(weight, 0, index);
        torch::Tensor bias_out = at::index_select(bias, 0, index);

        return std::make_tuple(weight_out, bias_out);
    }

    std::tuple<torch::Tensor, torch::Tensor>
    forward(
        const std::vector<Index>& index,
        tensor_reference weight,
        tensor_reference bias
    ) const
    {
        return forward(torch::tensor(index), weight, bias);
    }

    torch::Tensor
    forward_weight(tensor_reference index, tensor_reference weight) const
    {
        check_weight(weight);
        return at::index_select(weight, 0, index);
    }

    torch::Tensor
    forward_weight(const std::vector<Index>& index, tensor_reference weight) const
    {
        return forward_weight(torch::tensor(index), weight);
    }

private:

    const quadtree_exterior&
    check_exterior(const quadtree_exterior& exterior) const
    {
        TORCH_CHECK(
            exterior.size() == 4,
            m_op, ": exterior must be a tuple of four doubles"
        );
        return exterior;
    }

    tensor_reference
    check_tiles(tensor_reference tiles) const
    {
        TORCH_CHECK(
            tiles.dim() == 2,
            m_op, ": operation only supports 2D tiles, got ", tiles.dim(), "D"
        );
        TORCH_CHECK(
            tiles.size(1) == 3,
            m_op, ": tiles must be three-element tuples"
        );
        TORCH_CHECK(
            tiles.dtype() == torch::kInt32,
            m_op, ": operation only supports Int32 tiles, got ", tiles.dtype()
        );
        return tiles;
    }

    tensor_reference
    check_input(tensor_reference input) const
    {
        TORCH_CHECK(
            input.dim() == 2,
            m_op, ": operation only supports 2D input, got ", input.dim(), "D"
        );
        TORCH_CHECK(
            input.size(1) == 2,
            m_op, ": input must be two-element tuples");
        TORCH_CHECK(
            input.dtype() == torch::kFloat64,
            m_op, ": operation only supports Float64 input, got ", input.dtype()
        );
        return input;
    }

    void
    check_weight(tensor_reference weight) const
    {
        TORCH_CHECK(
            weight.dim() == 1,
            m_op, ": operation only supports 1D weight, got ", weight.dim(), "D"
        );
    }

    void
    check_weight_and_bias(tensor_reference weight, tensor_reference bias) const
    {
        check_weight(weight);

        TORCH_CHECK(
            bias.dim() == 1,
            m_op, ": operation only supports 1D bias, got ", bias.dim(), "D"
        );
        TORCH_CHECK(
            weight.sizes() == bias.sizes(),
            m_op, ": weight (", weight.sizes(), ") and bias (", bias.sizes(), ") are differ in size"
        );
    }
};



/// Structure represents a aggregation (statistic) quadtree operation.
///
/// On instance create, it creates a statistic index for fast access by a tile.
template<typename Coordinate = double, typename Index = int32_t>
struct quadpool_stat_op : public quadpool_op<Coordinate, Index>
{
    using base = quadpool_op<Coordinate, Index>;

    using stat_quadtree_index = std::unordered_map<Tile, torch::Tensor>;

    using stat_function = std::function<torch::Tensor(const torch::Tensor&)>;

    /// Stat tile index is comprised of both terminal and intermediate nodes.
    stat_quadtree_index m_stat_tile_index;
    stat_function m_stat_function;

    quadpool_stat_op(
        std::string op,
        stat_function stat_function,
        typename base::tensor_reference tiles,
        typename base::tensor_reference input,
        typename base::tensor_reference weight,
        const typename base::quadtree_exterior& exterior,
        const quadtree_options& options,
        bool training
    )
    : quadpool_op<Coordinate, Index>(op, tiles, input, exterior, options, training),
      m_stat_function(stat_function),
      m_stat_tile_index()
    {
        std::priority_queue<Tile, std::vector<Tile>, std::less<Tile>> unvisited;

        // Iterate over terminal nodes of the quadtree index and calculate an associated
        // weight. Since this is a terminal node, then statistic will be the value from the
        // weight tensor itself.
        for (auto item : base::m_tile_index) {
            auto tile = item.first;
            auto index = std::vector<Index>({item.second});

            auto stat = base::forward_weight(index, weight);
            m_stat_tile_index.insert(std::make_pair(tile, stat));
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
            if (auto it = m_stat_tile_index.find(tile); it != m_stat_tile_index.end()) {
                continue;
            }

            /// Query values from children, put them into a tensor and compute a statistic
            /// using a specified function.
            std::vector<torch::Tensor> child_weights;
            for (auto child_tile : tile.children()) {
                if (auto stat = m_stat_tile_index.find(child_tile); stat != m_stat_tile_index.end()) {
                    child_weights.push_back(stat->second);
                }
            }

            auto stat = m_stat_function(torch::stack(child_weights));
            m_stat_tile_index.insert(std::make_pair(tile, stat));

            // Do not compute the root tile multiple times, once is enough.
            if (tile != Tile::root) {
                unvisited.push(tile.parent());
            }
        }
    }
};


} // namespace torch_geopooling
