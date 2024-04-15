#include <torch_geopooling/torch_geopooling.h>

#include <pybind11/pybind11.h>
#include <torch/extension.h>


namespace torch_geopooling {


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quad_pool2d", &quad_pool2d);
}


} // namespace torch_geopooling
