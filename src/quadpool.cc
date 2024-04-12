#include <array>
#include <iterator>

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
        return !(m_accessor.data() == rhs.m_accessor.data() && m_end == rhs.m_end);
    }

private:
    torch::TensorAccessor<Scalar, 2> m_accessor;
    std::size_t m_begin;
    std::size_t m_end;
};


void
quad_pool2d(
    const torch::Tensor& tiles,
    const FloatArrayRef& exterior,
    std::optional<quadtree_options> options,
    bool training
)
{
    TORCH_CHECK(tiles.dim() == 2, "quad_pool2d only supports 2D tensors, got: ", tiles.dim(), "D");
    TORCH_CHECK(tiles.size(1) == 3, "quad_pool2d: tiles must be three-element tuples");
    TORCH_CHECK(exterior.size() == 4, "quad_pool2d: must be a tuple of four Doubles");

    tensor_iterator2d<int64_t, 3> it(tiles);
    quadtree_set<double> set(it.begin(), it.end(), exterior.vec(), options);

    std::cout << "quad_pool2d" << set.size() << std::endl;
}


} // namespace torch_geopooling
