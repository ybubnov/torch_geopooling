#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE quadpool

#include <boost/test/unit_test.hpp>
#include <torch/torch.h>

#include <torch_geopooling.h>


using namespace torch_geopooling;


BOOST_AUTO_TEST_SUITE(TestQuadPool)


BOOST_AUTO_TEST_CASE(quad_pool2d_training_unchanged)
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

    BOOST_REQUIRE_EQUAL(tiles_out.sizes(), torch::IntArrayRef({5, 3}));
    BOOST_REQUIRE_EQUAL(weight_out.sizes(), torch::IntArrayRef({1}));

    auto weight_acc = weight.accessor<float, 1>();
    auto weight_out_acc = weight_out.accessor<float, 1>();
    BOOST_REQUIRE_EQUAL(weight_acc[1], weight_out_acc[0]);
}


BOOST_AUTO_TEST_SUITE_END()
