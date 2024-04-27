#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE quadpool

#include <boost/test/included/unit_test.hpp>
#include <torch/torch.h>

#include <torch_geopooling.h>


using namespace torch_geopooling;


BOOST_AUTO_TEST_SUITE(TestMaxQuadPool)


BOOST_AUTO_TEST_CASE(max_quad_pool2d_training)
{
    auto tiles_options = torch::TensorOptions().dtype(torch::kInt32);
    auto tiles = torch::empty({0, 3}, tiles_options);

    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat64);
    auto weight = torch::randn({100}, tensor_options.requires_grad(true));
    auto input_train = torch::tensor({
        {1.0, 1.0},
        {1.7, 1.7},
        {1.0, 1.7},
        {1.7, 1.0},
        {9.9, 9.9},
        {8.0, 8.0}
    }, tensor_options);

    auto [tiles_out, weight_out] = max_quad_pool2d(
        tiles, input_train, weight, {0.0, 0.0, 10.0, 10.0}, true
    );

    BOOST_REQUIRE_EQUAL(tiles_out.dim(), 2);
    BOOST_REQUIRE_EQUAL(weight_out.dim(), 1);

    BOOST_REQUIRE_EQUAL(tiles_out.sizes(), torch::IntArrayRef({21, 3}));
    BOOST_REQUIRE_EQUAL(weight_out.sizes(), torch::IntArrayRef({input_train.size(0)}));

    BOOST_CHECK(weight_out.requires_grad());

    auto input_test = torch::tensor({{1.8, 1.8}}, tensor_options);

    std::tie(tiles_out, weight_out) = max_quad_pool2d(
        tiles_out, input_test, weight, {0.0, 0.0, 10.0, 10.0}, true
    );

    BOOST_REQUIRE_EQUAL(weight_out.sizes(), torch::IntArrayRef({1}));

    auto weight_acc = weight.accessor<double, 1>();
    auto weight_out_acc = weight_out.accessor<double, 1>();

    BOOST_REQUIRE_EQUAL(
        weight_out_acc[0],
        std::max({weight_acc[8], weight_acc[9], weight_acc[10], weight_acc[11]})
    );
}


BOOST_AUTO_TEST_SUITE_END()
