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
        .dtype(torch::kFloat64)
        .device(torch::kCPU);

    auto input = torch::tensor({{4.1, 4.5}}, tensor_options);
    auto weight = torch::rand({100}, tensor_options);
    auto bias = torch::rand({100}, tensor_options);

    auto [tiles_out, weight_out, bias_out] = quad_pool2d(
        tiles, input, weight, bias, {0.0, 0.0, 10.0, 10.0}, true
    );

    BOOST_REQUIRE_EQUAL(tiles_out.dim(), 2);
    BOOST_REQUIRE_EQUAL(weight_out.dim(), 1);
    BOOST_REQUIRE_EQUAL(bias_out.dim(), 1);

    BOOST_REQUIRE_EQUAL(tiles_out.sizes(), torch::IntArrayRef({5, 3}));
    BOOST_REQUIRE_EQUAL(weight_out.sizes(), torch::IntArrayRef({1}));
    BOOST_REQUIRE_EQUAL(bias_out.sizes(), torch::IntArrayRef({1}));

    auto weight_acc = weight.accessor<double, 1>();
    auto weight_out_acc = weight_out.accessor<double, 1>();
    BOOST_REQUIRE_EQUAL(weight_acc[1], weight_out_acc[0]);

    auto bias_acc = bias.accessor<double, 1>();
    auto bias_out_acc = bias_out.accessor<double, 1>();
    BOOST_REQUIRE_EQUAL(bias_acc[1], bias_out_acc[0]);
}


BOOST_AUTO_TEST_CASE(quad_pool2d_training)
{
    auto tiles_options = torch::TensorOptions().dtype(torch::kInt32);
    auto tiles = torch::empty({0, 3}, tiles_options);

    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat64);
    auto weight = torch::randn({100}, tensor_options.requires_grad(true));
    auto bias = torch::randn({100}, tensor_options.requires_grad(true));
    auto input = torch::tensor({
        {1.0, 1.0},
        {1.7, 1.7},
        {1.0, 1.7},
        {1.7, 1.0},
        {9.9, 9.9},
        {8.0, 8.0}
    }, tensor_options);

    auto [tiles_out, weight_out, bias_out] = quad_pool2d(
        tiles, input, weight, bias, {0.0, 0.0, 10.0, 10.0}, true
    );

    BOOST_REQUIRE_EQUAL(tiles_out.dim(), 2);
    BOOST_REQUIRE_EQUAL(weight_out.dim(), 1);
    BOOST_REQUIRE_EQUAL(bias_out.dim(), 1);

    BOOST_REQUIRE_EQUAL(tiles_out.sizes(), torch::IntArrayRef({21, 3}));
    BOOST_REQUIRE_EQUAL(weight_out.sizes(), torch::IntArrayRef({input.size(0)}));
    BOOST_REQUIRE_EQUAL(bias_out.sizes(), torch::IntArrayRef({input.size(0)}));

    BOOST_CHECK(weight_out.requires_grad());
    BOOST_CHECK(bias_out.requires_grad());
}


BOOST_AUTO_TEST_SUITE_END()
