#pragma once

#include <optional>

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>

#include <torch_geopooling/quadtree_options.h>


namespace torch_geopooling {


using FloatArrayRef = c10::ArrayRef<double>;


void
quad_pool2d(
    const torch::Tensor& tiles,
    const FloatArrayRef& exterior,
    std::optional<quadtree_options> options = std::nullopt,
    bool training = true
);


} // namespace torch_geopooling
