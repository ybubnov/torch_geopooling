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

#include <cmath>
#include <vector>

#include <ATen/Functions.h>
#include <ATen/Parallel.h>
#include <ATen/TensorAccessor.h>

#include <torch_geopooling/embedding.h>

#include <embedding_op.h>


namespace torch_geopooling {


torch::Tensor
embedding2d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::IntArrayRef& padding,
    const c10::ArrayRef<double>& exterior,
    bool reflection
)
{
    auto options = embedding_options{
        .padding = padding.vec(),
        .exterior = exterior.vec(),
        .manifold = weight.sizes().vec(),
        .reflection = reflection,
    };

    check_shape_forward("embedding2d", input, weight, options);

    auto op = embedding_op(options);

    auto weight_data = weight.accessor<double, 3>();
    auto input_data = input.accessor<double, 2>();

    const auto input_size = input.size(0);
    const auto kernel_size = options.kernel_size();
    const auto feature_size = options.feature_size();

    std::vector<torch::Tensor> output(input_size);

    at::parallel_for(0, input_size, at::internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (const auto i : c10::irange(begin, end)) {
            auto x = input_data[i][0];
            auto y = input_data[i][1];

            auto kernel = torch::empty(kernel_size, weight.options());
            auto kernel_data = kernel.accessor<double, 3>();

            int64_t k0 = 0;
            for (auto j0 : op.kernel_width_iterator(x)) {
                int64_t k1 = 0;
                for (auto j1 : op.kernel_height_iterator(y)) {
                    std::tie(j0, j1) = op.reflect(j0, j1);

                    for (auto j2 : c10::irange(feature_size)) {
                        kernel_data[k0][k1][j2] = weight_data[j0][j1][j2];
                    }
                    k1++;
                }
                k0++;
            }

            output[i] = kernel;
        }
    });

    return torch::stack(output, /*dim=*/0);
}


torch::Tensor
embedding2d_backward(
    const torch::Tensor& grad,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::IntArrayRef& padding,
    const c10::ArrayRef<double>& exterior,
    bool reflection
)
{
    auto options = embedding_options{
        .padding = padding.vec(),
        .exterior = exterior.vec(),
        .manifold = weight.sizes().vec(),
        .reflection = reflection,
    };

    check_shape_backward("embedding2d_backward", grad, input, weight, options);

    auto op = embedding_op(options);

    const int64_t weight_numel = weight.numel();
    const int64_t width_size = weight.size(0);
    const int64_t input_size = input.size(0);
    const int64_t grain_size = at::internal::GRAIN_SIZE;

    auto input_data = input.accessor<double, 2>();
    auto grad_weight = at::zeros(weight.sizes(), grad.options());

    at::parallel_for(0, weight_numel, grain_size, [&](int64_t begin, int64_t end) {
        for (const auto i : c10::irange(input_size)) {
            auto x = input_data[i][0];
            auto y = input_data[i][1];

            int64_t k0 = 0;
            for (auto j0 : op.kernel_width_iterator(x)) {
                int64_t k1 = 0;
                for (auto j1 : op.kernel_height_iterator(y)) {
                    std::tie(j0, j1) = op.reflect(j0, j1);

                    int64_t pos = j0 * width_size + j1;
                    if (pos >= begin && pos < end) {
                        grad_weight[j0][j1] += grad[i][k0][k1];
                    }
                    k1++;
                }
                k0++;
            }
        }
    });

    return grad_weight;
}


} // namespace torch_geopooling
