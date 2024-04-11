#pragma once

//#include <ATen/SparseCsrTensorImpl.h>
//#include <ATen/SparseCsrTensorUtils.h>
#include <c10/util/ArrayRef.h>
//#include <torch/torch.h>

//#include <torch_geopooling/quadtree.h>


namespace torch_geopooling {


//using namespace at::sparse_csr;


using FloatArrayRef = c10::ArrayRef<double>;


/*
std::tuple<SparseCsrTensor, SparseCsrTensor>
empty_quad_pool(
    const torch::Tensor& exterior,
    std::size_t capacity = 1,
    std::size_t depth = 17,
    std::size_t precision = 7
);


std::tuple<SparseCsrTensor, SparseCsrTensor, torch::Tensor>
quad_pool(
    const SparseCsrTensor& indices,
    const SparseCsrTensor& weights,
    const FloatArrayRef& exterior,
    const torch::Tensor& input,
    std::size_t capacity = 1,
    std::size_t depth = 17,
    std::size_t precision = 7,
    bool training = true,
);
*/


} // namespace torch_geopooling
