#pragma once

#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <torch/torch.h>

#include <torch_geopooling/quadtree.h>


namespace torch_geopooling {


using namespace at::sparse_csr;


class QuadTensor {
public:
    QuadTensor(
        at::IntArrayRef quad,
        std::size_t capacity = 1,
        std::size_t depth = 17,
        std::size_t precision = 7
    );

private:
    SparseCsrTensor weight;
    quadtree<double, double> indices;
};


/*
torch::Tensor
quad_pool(QuadTensor& indices, const torch::Tensor& input, bool training);
*/


} // namespace torch_geopooling
