#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE quadpool

#include <boost/test/included/unit_test.hpp>
#include <torch/torch.h>

#include <torch_geopooling.h>


using namespace torch_geopooling;
using namespace torch::indexing;


BOOST_AUTO_TEST_SUITE(TestAvgQuadPool)


BOOST_AUTO_TEST_CASE(avg_quad_pool2d_training)
{
    auto tiles_options = torch::TensorOptions().dtype(torch::kInt64);
    auto tiles = torch::empty({0, 3}, tiles_options);

    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat64);
    auto weight = torch::randn({0, 3}, tensor_options.requires_grad(true));
    auto input_train = torch::tensor(
        {{1.0, 1.0}, {1.7, 1.7}, {1.0, 1.7}, {1.7, 1.0}, {9.9, 9.9}, {8.0, 8.0}}, tensor_options
    );

    std::vector<double> exterior({0.0, 0.0, 10.0, 10.0});
    auto [tiles_out, weight_out, values_out]
        = avg_quad_pool2d(tiles, weight, input_train, exterior, /*training=*/true);

    BOOST_REQUIRE_EQUAL(tiles_out.sizes(), torch::IntArrayRef({21, 3}));
    BOOST_REQUIRE_EQUAL(weight_out.sizes(), torch::IntArrayRef({21, 3}));
    BOOST_REQUIRE_EQUAL(
        values_out.sizes(), torch::IntArrayRef({input_train.size(0), weight.size(1)})
    );

    BOOST_CHECK(weight_out.requires_grad());

    weight_out.mean().backward();
    BOOST_CHECK(weight_out.grad().defined());

    auto input_test = torch::tensor({{1.8, 1.8}}, tensor_options);

    std::tie(tiles_out, weight_out, values_out)
        = avg_quad_pool2d(tiles_out, weight_out, input_test, exterior, /*training=*/true);

    BOOST_REQUIRE_EQUAL(tiles_out.sizes(), torch::IntArrayRef({21, 3}));
    BOOST_REQUIRE_EQUAL(weight_out.sizes(), torch::IntArrayRef({21, 3}));
    BOOST_REQUIRE_EQUAL(
        values_out.sizes(), torch::IntArrayRef({input_test.size(0), weight.size(1)})
    );

    auto weight_mean = at::mean(weight_out.index({Slice(8, 12), Slice()}), 0);

    auto weight_acc = weight_mean.accessor<double, 1>();
    auto values_acc = values_out.accessor<double, 2>();

    BOOST_REQUIRE_EQUAL(values_acc[0][0], weight_acc[0]);
    BOOST_REQUIRE_EQUAL(values_acc[0][1], weight_acc[1]);
    BOOST_REQUIRE_EQUAL(values_acc[0][2], weight_acc[2]);
}


BOOST_AUTO_TEST_CASE(avg_quad_pool2d_backward_grad)
{
    auto tiles_options = torch::TensorOptions().dtype(torch::kInt64);
    auto tiles = torch::tensor(
        {
            {0, 0, 0},
            {1, 0, 0},
            {2, 0, 0}, // (1,0,0) -> weight[2]
            {1, 1, 0},
            {2, 0, 1}, // (1,0,0) -> weight[4]
            {2, 1, 0}, // (1,0,0) -> weight[5]
            {2, 2, 0}, // (1,1,0) -> weight[6]
            {2, 2, 1}, // (1,1,0) -> weight[7]
        },
        tiles_options
    );

    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat64);
    auto weight = torch::rand({8, 3}, tensor_options.requires_grad(true));

    auto input = torch::tensor(
        {
            {0.1, 0.1}, // (2,0,0) -> avg(weight[2], weight[4], weight[5])
            {0.2, 0.1}, // (2,0,0) -> avg(weight[2], weight[4], weight[5])
            {1.3, 0.2}  // (2,2,0) -> avg(weight[6], weight[7])
        },
        tensor_options
    );

    auto grad_output = torch::tensor(
        {{10.0, 1.0, 100.0}, {22.0, 2.2, 220.0}, {30.0, 3.0, 300.0}}, tensor_options
    );

    auto grad_weight = avg_quad_pool2d_backward(
        grad_output, tiles, weight, input,
        /*exterior=*/{0.0, 0.0, 2.0, 2.0}
    );

    BOOST_REQUIRE_EQUAL(grad_weight.sizes(), weight.sizes());

    auto grad_weight_acc = grad_weight.accessor<double, 2>();
    for (const auto i : {0, 1, 3}) {
        BOOST_CHECK_EQUAL(grad_weight_acc[i][0], 0.0);
        BOOST_CHECK_EQUAL(grad_weight_acc[i][1], 0.0);
        BOOST_CHECK_EQUAL(grad_weight_acc[i][2], 0.0);
    }

    for (const auto i : {2, 4, 5}) {
        BOOST_CHECK_EQUAL(grad_weight_acc[i][0], 10.0 / 3.0 + 22.0 / 3.0);
        BOOST_CHECK_EQUAL(grad_weight_acc[i][1], 1.0 / 3.0 + 2.2 / 3.0);
        BOOST_CHECK_EQUAL(grad_weight_acc[i][2], 100.0 / 3.0 + 220.0 / 3.0);
    }

    for (const auto i : {6, 7}) {
        BOOST_CHECK_EQUAL(grad_weight_acc[i][0], 30.0 / 2.0);
        BOOST_CHECK_EQUAL(grad_weight_acc[i][1], 3.0 / 2.0);
        BOOST_CHECK_EQUAL(grad_weight_acc[i][2], 300.0 / 2.0);
    }
}


BOOST_AUTO_TEST_SUITE_END()
