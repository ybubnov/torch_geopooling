#include <torch_geopooling/torch_geopooling.h>

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "python_tuples.h"


namespace torch_geopooling {


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("avg_quad_pool2d", &avg_quad_pool2d);
    m.def("avg_quad_pool2d_backward", &avg_quad_pool2d_backward);

    m.def("max_quad_pool2d", &max_quad_pool2d);
    m.def("max_quad_pool2d_backward", &max_quad_pool2d_backward);

    m.def("quad_pool2d", &quad_pool2d);
    m.def("quad_pool2d_backward", &quad_pool2d_backward);

    m.def("embedding2d", &embedding2d);
    m.def("embedding2d_backward", &embedding2d_backward);
}


} // namespace torch_geopooling
