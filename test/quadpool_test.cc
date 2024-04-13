#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE quadpool

#include <boost/test/unit_test.hpp>
#include <torch/torch.h>

#include <torch_geopooling.h>


using namespace torch_geopooling;


BOOST_AUTO_TEST_SUITE(TestQuadPool)


BOOST_AUTO_TEST_CASE(quad_pool2d_from_tiles)
{
    auto tiles_options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(torch::kCPU);
    auto tiles = torch::tensor({
        {0, 0, 0},
        {1, 0, 0},
        {1, 0, 1},
        {1, 1, 0},
        {1, 1, 1}
    }, tiles_options);

    auto tensor_options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCPU);

    auto input = torch::tensor({{4.1, 4.5}}, tensor_options);
    auto weight = torch::randn({100}, tensor_options);

    auto [tiles_out, weight_out] = quad_pool2d(
        tiles, input, weight, {0.0, 0.0, 10.0, 10.0}, true
    );

    BOOST_REQUIRE_EQUAL(tiles_out.dim(), 2);
    BOOST_REQUIRE_EQUAL(weight_out.dim(), 1);
}


BOOST_AUTO_TEST_SUITE_END()
