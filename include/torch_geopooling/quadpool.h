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
std::tuple<SparseCsrTensor, SparseCsrTensor, torch::Tensor>
quad_pool2d(
    const Tensor& weights,          // [0.9823, 0.8721, ...]
    const torch::Tensor& input,     // [(42.1, 32.1), (45.3, 12.3), ...]
    const FloatArrayRef& exterior,  // {-180.0, -90.0, 360.0, 180.0}
    const OptionalTensorRef tiles,  // [(0,0,0), (1,0,0), ...]
    std::size_t capacity = 1,
    std::size_t depth = 17,
    std::size_t precision = 7,
    bool training = true,
);
*/


} // namespace torch_geopooling
