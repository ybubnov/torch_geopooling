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


torch::Tensor
embedding2d(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::ArrayRef<int64_t>& padding,
    const c10::ArrayRef<double>& exterior
)
{
    const std::string op = "embedding2d";

    TORCH_CHECK(exterior.size() == 4, op, ": exterior must be a tuple of four doubles");
    TORCH_CHECK(padding.size() == 2, op, ": padding should be comprised of 2 elements");
    TORCH_CHECK(weight.dim() == 3, op, ": weight must be 3D, got ", weight.dim(), "D");
    TORCH_CHECK(
        weight.dtype() == torch::kFloat64, op, ": operation only supports Float64 weight, got ",
        weight.dtype()
    );
    TORCH_CHECK(input.dim() == 2, op, ": input must be 2D, got ", input.dim(), "D");
    TORCH_CHECK(
        input.dtype() == torch::kFloat64, op, ": operation only supports Float64 input, got ",
        input.dtype()
    );

    auto width_size = weight.size(0);
    auto height_size = weight.size(1);
    auto feature_size = weight.size(2);

    // Bucketize input coordinates given the exterior of the quad and number of buckets
    // within the weight tensor.
    auto quad_exterior = quadrect(exterior.vec());
    auto quad_width = quad_exterior.width() / width_size;
    auto quad_height = quad_exterior.height() / height_size;

    auto weight_ptr = weight.accessor<double, 3>();
    auto input_ptr = input.accessor<double, 2>();

    auto input_size = input.size(0);
    const auto kernel_size = std::vector({padding[0] * 2 + 1, padding[1] * 2 + 1, feature_size});
    std::vector<torch::Tensor> output(input_size);

    at::parallel_for(0, input_size, at::internal::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (const auto i : c10::irange(begin, end)) {
            const auto& point = input_ptr[i];

            auto kernel = torch::empty(kernel_size, weight.options());
            auto kernel_ptr = kernel.accessor<double, 3>();

            const auto x = floordiv(point[0] - quad_exterior.xmin(), quad_width);
            const auto y = floordiv(point[1] - quad_exterior.ymin(), quad_height);

            int64_t k0 = 0;
            for (auto j0 : c10::irange(x - padding[0], x + padding[0] + 1)) {
                int64_t k1 = 0;
                for (auto j1 : c10::irange(y - padding[1], y + padding[1] + 1)) {
                    j0 = modulo(width_size, j0);
                    j1 = modulo(height_size, j1);

                    int64_t k2 = 0;
                    for (auto j2 : c10::irange(feature_size)) {
                        kernel_ptr[k0][k1][k2] = weight_ptr[j0][j1][j2];
                        k2++;
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


} // namespace torch_geopooling
