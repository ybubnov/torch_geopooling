#pragma once

#include <optional>
#include <tuple>

#include <c10/util/ArrayRef.h>
#include <torch/torch.h>


namespace torch_geopooling {


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
linear_quad_pool2d(
    const torch::Tensor& tiles,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const c10::ArrayRef<double>& exterior,
    bool training = true,
    std::optional<std::size_t> max_depth = std::nullopt,
    std::optional<std::size_t> capacity = std::nullopt,
    std::optional<std::size_t> precision = std::nullopt
);


std::tuple<torch::Tensor, torch::Tensor>
max_quad_pool2d(
    const torch::Tensor& tiles,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::ArrayRef<double>& exterior,
    bool training = true,
    std::optional<std::size_t> max_depth = std::nullopt,
    std::optional<std::size_t> capacity = std::nullopt,
    std::optional<std::size_t> precision = std::nullopt
);


} // namespace torch_geopooling
