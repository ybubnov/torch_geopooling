#include <array>
#include <iterator>
#include <unordered_map>
#include <vector>

#include <ATen/TensorAccessor.h>

#include <torch_geopooling/quadpool.h>
#include <torch_geopooling/quadtree_options.h>
#include <torch_geopooling/quadtree_set.h>


namespace torch_geopooling {


using namespace torch::indexing;


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
        return !(m_accessor.data() == rhs.m_accessor.data() && m_begin == rhs.m_end);
    }

private:
    torch::TensorAccessor<Scalar, 2> m_accessor;
    std::size_t m_begin;
    std::size_t m_end;
};


std::tuple<torch::Tensor, torch::Tensor>
quad_pool2d(
    const torch::Tensor& tiles,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::ArrayRef<float>& exterior,
    bool training,
    std::optional<std::size_t> max_depth,
    std::optional<std::size_t> capacity,
    std::optional<std::size_t> precision
)
{
    TORCH_CHECK(tiles.dim() == 2, "quad_pool2d only supports 2D tiles, got ", tiles.dim(), "D");
    TORCH_CHECK(tiles.size(1) == 3, "quad_pool2d: tiles must be three-element tuples");
    TORCH_CHECK(
        tiles.dtype() == torch::kInt32,
        "quad_pool2d only supports Int32 tiles, got: ", tiles.dtype()
    );

    TORCH_CHECK(input.dim() == 2, "quad_pool2d only supports 2D input, got ", input.dim(), "D");
    TORCH_CHECK(input.size(1) == 2, "quad_pool2d: input must be two-element tuples");
    TORCH_CHECK(
        input.dtype() == torch::kFloat32,
        "quad_pool2d only supports Float32 input, got: ", input.dtype()
    );

    TORCH_CHECK(weight.dim() == 1, "quad_pool2d only supports 1D weight, got ", weight.dim(), "D");
    TORCH_CHECK(exterior.size() == 4, "quad_pool2d: must be a tuple of four floats");

    auto options = quadtree_options()
        .max_terminal_nodes(weight.size(0))
        .max_depth(max_depth)
        .precision(precision)
        .capacity(capacity);

    tensor_iterator2d<int32_t, 3> tiles_it(tiles);
    quadtree_set<float> set(tiles_it.begin(), tiles_it.end(), exterior.vec(), options);

    std::cout << "quad_pool2d size = " << set.size() << std::endl;
    std::cout << "quad_pool2d exterior = " << set.exterior() << std::endl;
    std::cout << "quad_pool2d depth = " << set.total_depth() << std::endl;

    torch::Tensor tiles_out;

    if (training) {
        tensor_iterator2d<float, 2> input_it(input);
        set.insert(input_it.begin(), input_it.end());

        // Tiles are changing only in case of insert operation, therefore, extract tiles
        // from the set only in case of insert operation. Otherwise, just return the same
        // input tiles as a result.
        std::vector<torch::Tensor> tiles_out_rows;

        for (auto node_it = set.ibegin(); node_it != set.iend(); ++node_it) {
            auto tile = (*node_it).tile();
            tiles_out_rows.push_back(torch::tensor(tile.vec<int32_t>(), tiles.options()));
        }

        tiles_out = torch::stack(tiles_out_rows);
    } else {
        tiles_out = tiles;
    }

    std::cout << "iterate terminal nodes" << std::endl;

    // The tile index will change once the training iteration adjusts the quadtree set
    // structure. Since quads are embedded into a 1D tensor, there will be a drift of
    // weights. But with a large enough number of training iterations, these tiles map
    // eventually converges.
    //
    // Tile index is comprised only from terminal nodes.
    std::unordered_map<Tile, int32_t> tile_index;
    for (auto node_it = set.begin(); node_it != set.end(); ++node_it) {
        auto tile = (*node_it).tile();
        std::cout << "  -> tile " << tile << std::endl;
        tile_index.insert(std::make_pair(tile, tile_index.size()));
    }

    tensor_iterator2d<float, 2> points_it(input);
    std::vector<int32_t> weight_indices;

    // This loop might raise an exception, when the tile returned by `set.find` operation
    // returns non-terminal node. This should not happen in practice.
    for (const auto& point : points_it) {
        const auto& node = set.find(point);
        const auto& index = tile_index.at(node.tile());
        weight_indices.push_back(index);
    }

    torch::Tensor weight_out = weight.index(
        at::indexing::TensorIndex(torch::tensor(weight_indices))
    );
    return std::make_tuple(tiles_out, weight_out);
}


} // namespace torch_geopooling
