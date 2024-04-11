#pragma once

//#include <ATen/SparseCsrTensorImpl.h>
//#include <ATen/SparseCsrTensorUtils.h>
//#include <torch/torch.h>

//#include <torch_geopooling/quadtree.h>


namespace torch_geopooling {


//using namespace at::sparse_csr;


/*
std::tuple<SparseCsrTensor, SparseCsrTensor>
empty_quad_pool(
    at::IntArrayRef quadrect,
    std::size_t capacity = 1,
    std::size_t depth = 17,
    std::size_t precision = 7
);


std::tuple<SparseCsrTensor, SparseCsrTensor, torch::Tensor>
quad_pool(
    const SparseCsrTensor& indices,
    const SparseCsrTensor& weights,
    const torch::Tensor& input,
    std::size_t capacity = 1,
    std::size_t depth = 17,
    std::size_t precision = 7,
    bool training = true,
);
*/


} // namespace torch_geopooling
