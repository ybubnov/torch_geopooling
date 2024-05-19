#include <torch_geopooling/torch_geopooling.h>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "python_tuples.h"


namespace torch_geopooling {


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_quad_pool2d", &linear_quad_pool2d);

    m.def("max_quad_pool2d", &max_quad_pool2d);

    m.def("avg_quad_pool2d", &avg_quad_pool2d);
}


} // namespace torch_geopooling
