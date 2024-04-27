#include <array>
#include <iterator>
#include <unordered_map>
#include <vector>

#include <ATen/TensorAccessor.h>

#include <torch_geopooling/quadpool.h>
#include <torch_geopooling/quadtree_options.h>
#include <torch_geopooling/quadtree_set.h>


namespace torch_geopooling {


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

    ~tensor_iterator2d()
    { }

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
        // TODO: ensure that iterator does not grow over the end.
        m_begin++;
        return *this;
    }

    value_type
    operator*()
    {
        value_type row;
        for (auto i = 0; i < N; i++) {
            row[i] = m_accessor[m_begin][i];
        }
        return row;
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
/// On object creation, it creates a tile index for fast access to the weights and biases.
/// Additionally, it checks validity of input data (tiles, indices, weight, bias, etc.), to
/// ensure it can be used to compute the operation.
template<typename Coordinate = double, typename TileIndex = int32_t>
struct quadtree_op
{
    using tiles_iterator = tensor_iterator2d<TileIndex, 3>;
    using input_iterator = tensor_iterator2d<Coordinate, 2>;

    using quadtree_exterior = c10::ArrayRef<Coordinate>;
    using quadtree_index = std::unordered_map<Tile, TileIndex>;

    std::string m_op;
    quadtree_set<Coordinate> m_set;

    /// Tile index is comprised of terminal quadtree nodes.
    quadtree_index m_tile_index;

    quadtree_op(
        std::string op,
        tiles_iterator tiles_it,
        const quadtree_exterior& exterior,
        const quadtree_options& options
    )
    : m_op(op),
      m_set(tiles_it.begin(), tiles_it.end(), check_exterior(exterior).vec(), options),
      m_tile_index()
    { }

    quadtree_op(
        std::string op,
        const torch::Tensor& tiles,
        const torch::Tensor& input,
        const quadtree_exterior& exterior,
        const quadtree_options& options,
        bool training
    )
    : quadtree_op(op, tiles_iterator(check_tiles(tiles)), exterior, options)
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
    make_input_iterator(const torch::Tensor& input)
    {
        return input_iterator(input);
    }

    torch::Tensor
    forward_tiles(const torch::TensorOptions& tiles_options) const
    {
        std::vector<torch::Tensor> tile_rows;

        for (auto node_it = m_set.ibegin(); node_it != m_set.iend(); ++node_it) {
            auto tile = node_it->tile();
            tile_rows.push_back(torch::tensor(tile.template vec<TileIndex>(), tiles_options));
        }

        return torch::stack(tile_rows);
    }

    std::tuple<torch::Tensor, torch::Tensor>
    forward(
        const torch::Tensor& weight,
        const torch::Tensor& bias,
        const at::indexing::TensorIndex& index
    ) const
    {
        check_weight_and_bias(weight, bias);

        torch::Tensor weight_out = weight.index(index);
        torch::Tensor bias_out = bias.index(index);

        return std::make_tuple(weight_out, bias_out);
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

    const torch::Tensor&
    check_tiles(const torch::Tensor& tiles) const
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

    const torch::Tensor&
    check_input(const torch::Tensor& input) const
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
    check_weight_and_bias(const torch::Tensor& weight, const torch::Tensor& bias) const
    {
        TORCH_CHECK(
            weight.dim() == 1,
            m_op, ": operation only supports 1D weight, got ", weight.dim(), "D"
        );
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


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
quad_pool2d(
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

    quadtree_op op("quad_pool2d", tiles, input, exterior, options, training);

    std::vector<int32_t> out_indices;

    // This loop might raise an exception, when the tile returned by `set.find` operation
    // returns non-terminal node. This should not happen in practice.
    for (const auto& point : op.make_input_iterator(input)) {
        const auto& node = op.m_set.find(point);
        const auto& index = op.m_tile_index.at(node.tile());
        out_indices.push_back(index);
    }

    torch::Tensor tiles_out = training ? op.forward_tiles(tiles.options()) : tiles;

    at::indexing::TensorIndex index(torch::tensor(out_indices));
    auto [weight_out, bias_out] = op.forward(weight, bias, index);

    return std::make_tuple(tiles_out, weight_out, bias_out);
}


/*
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
max_quad_pool2d(
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

    quadtree_op op("max_quad_pool2d", tiles, input, exterior, options, training);

    for (const auto& point : op::input_iterator(input)) {
        auto node_it = set.find_terminal_group(point);
        for (const auto& node : node_it) {
        }
    }
}
*/


} // namespace torch_geopooling
