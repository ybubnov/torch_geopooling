#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE quadpool

#include <boost/test/included/unit_test.hpp>
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

    auto input = torch::tensor({{9.1, 9.5}}, tensor_options);
    auto weight = torch::rand({100, 1}, tensor_options);

    auto [tiles_out, weight_out] = quad_pool2d(
        tiles, input, weight, {0.0, 0.0, 10.0, 10.0}, true
    );

    BOOST_REQUIRE_EQUAL(tiles_out.dim(), 2);
    BOOST_REQUIRE_EQUAL(weight_out.dim(), 2);

    BOOST_REQUIRE_EQUAL(tiles_out.sizes(), torch::IntArrayRef({5, 3}));
    BOOST_REQUIRE_EQUAL(weight_out.sizes(), torch::IntArrayRef({1, 1}));

    auto weight_acc = weight.accessor<double, 2>();
    auto weight_out_acc = weight_out.accessor<double, 2>();
    BOOST_REQUIRE_EQUAL(weight_acc[3][0], weight_out_acc[0][0]);
}


BOOST_AUTO_TEST_CASE(quad_pool2d_training)
{
    auto tiles_options = torch::TensorOptions().dtype(torch::kInt32);
    auto tiles = torch::empty({0, 3}, tiles_options);

    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat64);
    auto weight = torch::randn({100, 3}, tensor_options.requires_grad(true));
    auto input = torch::tensor({
        {1.0, 1.0},
        {1.7, 1.7},
        {1.0, 1.7},
        {1.7, 1.0},
        {9.9, 9.9},
        {8.0, 8.0}
    }, tensor_options);

    auto [tiles_out, weight_out] = quad_pool2d(
        tiles, input, weight, {0.0, 0.0, 10.0, 10.0}, true
    );

    BOOST_REQUIRE_EQUAL(tiles_out.dim(), 2);
    BOOST_REQUIRE_EQUAL(weight_out.dim(), 2);

    BOOST_REQUIRE_EQUAL(tiles_out.sizes(), torch::IntArrayRef({21, 3}));
    BOOST_REQUIRE_EQUAL(weight_out.sizes(), torch::IntArrayRef({input.size(0), weight.size(1)}));

    BOOST_CHECK(weight_out.requires_grad());
}


BOOST_AUTO_TEST_CASE(quad_pool2d_backward_grad_partial)
{
    auto tiles_options = torch::TensorOptions().dtype(torch::kInt32);
    auto tiles = torch::tensor({
        {0, 0, 0},
        {1, 0, 0}, // (0,0,0)
        {1, 0, 1}, // (0,0,0) -> weight[0]
        {1, 1, 0}, // (0,0,0)
        {1, 1, 1}, // (0,0,0) -> weight[1]
        {2, 0, 0}, // (1,0,0) -> weight[2]
        {2, 0, 1}, // (1,0,0) -> weight[3]
        {2, 2, 0}, // (1,1,0) -> weight[4]
        {2, 3, 1}  // (1,1,0) -> weight[5]
    }, tiles_options);

    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat64);
    auto weight = torch::tensor({
        {2.1}, // weight[0]
        {3.0}, // weight[1]
        {1.0}, // weight[2]
        {2.0}, // weight[3]
        {4.0}, // weight[4]
        {4.5}  // weight[5]
    }, tensor_options.requires_grad(true));

    auto input = torch::tensor({
        {0.1, 0.1}, // (2,0,0) -> weight[2]
        {0.2, 0.1}, // (2,0,0) -> weight[2]
        {1.3, 0.2}, // (2,2,0) -> weight[4]
        {1.5, 1.5}, // (1,1,1) -> weight[1]
        {0.2, 1.2}, // (1,0,1) -> weight[0]
        {0.4, 1.3}  // (1,0,1) -> weight[0]
    }, tensor_options);

    auto grad_output = torch::tensor({10.0, 22.0, 30.0, 43.0, 50.0, 66.0}, tensor_options);
    grad_output = at::unsqueeze(grad_output, 1);

    auto grad_weight = quad_pool2d_backward(
        grad_output,
        tiles,
        input,
        weight,
        /*exterior=*/{0.0, 0.0, 2.0, 2.0}
    );

    BOOST_REQUIRE_EQUAL(grad_weight.sizes(), torch::IntArrayRef({6, 1}));

    auto grad_weight_acc = grad_weight.accessor<double, 2>();
    BOOST_CHECK_EQUAL(grad_weight_acc[0][0], 50.0 + 66.0);
    BOOST_CHECK_EQUAL(grad_weight_acc[1][0], 43.0);
    BOOST_CHECK_EQUAL(grad_weight_acc[2][0], 10.0 + 22.0);
    BOOST_CHECK_EQUAL(grad_weight_acc[3][0], 0.0);
    BOOST_CHECK_EQUAL(grad_weight_acc[4][0], 30.0);
    BOOST_CHECK_EQUAL(grad_weight_acc[5][0], 0.0);
}


BOOST_AUTO_TEST_SUITE_END()
