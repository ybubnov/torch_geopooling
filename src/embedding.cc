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

#include <ATen/Parallel.h>

#include <torch_geopooling/embedding.h>
#include <torch_geopooling/quadrect.h>


namespace torch_geopooling {


int64_t
floordiv(double a, double b)
{
    return static_cast<int64_t>(std::floor(a / b));
}


int64_t
modulo(int64_t base, int64_t value)
{
    value = value < 0 ? base + value : value;
    value = value >= base ? base - value : value;
    return value;
}


struct embedding_options {
    std::vector<int64_t> padding;
    std::vector<double> exterior;

    std::vector<int64_t>
    kernel_size(int64_t feature_size) const
    {
        return {kernel_width(), kernel_height(), feature_size};
    }

    int64_t
    kernel_width() const
    {
        return padding[0] * 2 + 1;
    }

    int64_t
    kernel_height() const
    {
        return padding[1] * 2 + 1;
    }

    bool
    is_padding_neg() const
    {
        bool is_non_neg = false;
        for (const auto& p : padding) {
            is_non_neg |= (p < 0);
        }
        return is_non_neg;
    }
};


static void
check_shape_forward(
    const std::string& op,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const embedding_options& options
)
{
    TORCH_CHECK(
        options.exterior.size() == 4, op,
        ": exterior must be a tuple of four doubles comprising a rectangle (x, y, w, h)"
    );
    TORCH_CHECK(options.padding.size() == 2, op, ": padding should be comprised of 2 elements");

    TORCH_CHECK(input.dim() == 2, op, ": input must be 2D, got ", input.dim(), "D");
    TORCH_CHECK(
        input.dtype() == torch::kFloat64, op, ": operation only supports Float64 input, got ",
        input.dtype()
    );

    TORCH_CHECK(weight.dim() == 3, op, ": weight must be 3D, got ", weight.dim(), "D");
    TORCH_CHECK(
        weight.dtype() == torch::kFloat64, op, ": operation only supports Float64 weight, got ",
        weight.dtype()
    );
}


static void
check_shape_backward(
    const std::string& op,
    const torch::Tensor& grad,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const embedding_options& options
)
{
    check_shape_forward(op, input, weight, options);

    const auto grad_sizes = c10::IntArrayRef(
        {input.size(0), options.kernel_width(), options.kernel_height(), weight.size(-1)}
    );

    TORCH_CHECK(
        grad.sizes() == grad_sizes, op, ": gradient shape (", grad.sizes(),
        ") should be the same as input (", input.sizes(), ")"
    );
}


torch::Tensor
embedding2d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::ArrayRef<int64_t>& padding,
    const c10::ArrayRef<double>& exterior
)
{
    auto options = embedding_options{.padding = padding.vec(), .exterior = exterior.vec()};
    check_shape_forward("embedding2d", input, weight, options);

    auto width_size = weight.size(0);
    auto height_size = weight.size(1);
    auto feature_size = weight.size(2);

    // Bucketize input coordinates given the exterior of the quad and number of buckets
    // within the weight tensor.
    auto quad_exterior = quadrect(exterior.vec());
    auto quad_width = quad_exterior.width() / width_size;
    auto quad_height = quad_exterior.height() / height_size;

    auto weight_data = weight.accessor<double, 3>();
    auto input_data = input.accessor<double, 2>();

    auto input_size = input.size(0);
    const auto kernel_size = options.kernel_size(feature_size);
    std::vector<torch::Tensor> output(input_size);

    at::parallel_for(0, input_size, at::internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (const auto i : c10::irange(begin, end)) {
            const auto& point = input_data[i];

            auto kernel = torch::empty(kernel_size, weight.options());
            auto kernel_data = kernel.accessor<double, 3>();

            const auto w = floordiv(point[0] - quad_exterior.xmin(), quad_width);
            const auto h = floordiv(point[1] - quad_exterior.ymin(), quad_height);

            int64_t k0 = 0;
            for (auto j0 : c10::irange(w - padding[0], w + padding[0] + 1)) {
                int64_t k1 = 0;
                for (auto j1 : c10::irange(h - padding[1], h + padding[1] + 1)) {
                    j0 = modulo(width_size, j0);
                    j1 = modulo(height_size, j1);

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
    const c10::ArrayRef<double>& exterior
)
{
    auto options = embedding_options{.padding = padding.vec(), .exterior = exterior.vec()};
    check_shape_backward("embedding2d_backward", grad, input, weight, options);

    auto width_size = weight.size(0);
    auto height_size = weight.size(1);
    auto feature_size = weight.size(2);

    auto quad_exterior = quadrect(exterior.vec());
    auto quad_width = quad_exterior.width() / width_size;
    auto quad_height = quad_exterior.height() / height_size;

    const int64_t weight_numel = weight.numel();
    const int64_t input_size = input.size(0);
    const int64_t grain_size = at::internal::GRAIN_SIZE;

    auto input_data = input.accessor<double, 2>();
    auto grad_weight = at::zeros(weight.sizes(), grad.options());

    at::parallel_for(0, weight_numel, grain_size, [&](int64_t begin, int64_t end) {
        for (const auto i : c10::irange(input_size)) {
            const auto& point = input_data[i];

            const auto w = floordiv(point[0] - quad_exterior.xmin(), quad_width);
            const auto h = floordiv(point[1] - quad_exterior.ymin(), quad_height);

            int64_t k0 = 0;
            for (auto j0 : c10::irange(w - padding[0], w + padding[0] + 1)) {
                int64_t k1 = 0;
                for (auto j1 : c10::irange(h - padding[1], h + padding[1] + 1)) {
                    j0 = modulo(width_size, j0);
                    j1 = modulo(height_size, j1);

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
