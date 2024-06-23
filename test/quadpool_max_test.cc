#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE quadpool

#include <boost/test/included/unit_test.hpp>
#include <torch/torch.h>

#include <torch_geopooling.h>


using namespace torch_geopooling;
using namespace torch::indexing;


BOOST_AUTO_TEST_SUITE(TestMaxQuadPool)


BOOST_AUTO_TEST_CASE(max_quad_pool2d_training)
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
        = max_quad_pool2d(tiles, weight, input_train, exterior, /*training=*/true);

    BOOST_REQUIRE_EQUAL(tiles_out.dim(), 2);
    BOOST_REQUIRE_EQUAL(weight_out.dim(), 2);

    BOOST_REQUIRE_EQUAL(tiles_out.sizes(), torch::IntArrayRef({21, 3}));
    BOOST_REQUIRE_EQUAL(weight_out.sizes(), torch::IntArrayRef({21, weight.size(1)}));
    BOOST_REQUIRE_EQUAL(
        values_out.sizes(), torch::IntArrayRef({input_train.size(0), weight.size(1)})
    );

    BOOST_CHECK(weight_out.requires_grad());

    weight_out.mean().backward();
    BOOST_CHECK(weight_out.grad().defined());

    auto input_test = torch::tensor({{1.8, 1.8}}, tensor_options);

    std::tie(tiles_out, weight_out, values_out)
        = max_quad_pool2d(tiles_out, weight_out, input_test, exterior, /*training=*/true);

    BOOST_REQUIRE_EQUAL(values_out.sizes(), torch::IntArrayRef({1, weight.size(1)}));

    auto [weight_max, _] = at::max(weight_out.index({Slice(8, 12), Slice()}), 0);

    auto weight_acc = weight_max.accessor<double, 1>();
    auto values_acc = values_out.accessor<double, 2>();

    BOOST_REQUIRE_EQUAL(values_acc[0][0], weight_acc[0]);
    BOOST_REQUIRE_EQUAL(values_acc[0][1], weight_acc[1]);
    BOOST_REQUIRE_EQUAL(values_acc[0][2], weight_acc[2]);
}


BOOST_AUTO_TEST_CASE(max_quad_pool2d_backward_grad)
{
    auto tiles_options = torch::TensorOptions().dtype(torch::kInt64);
    auto tiles = torch::tensor(
        {
            {0, 0, 0},
            {1, 0, 0},
            {2, 0, 0}, // (1,0,0) -> weight[2]
            {1, 1, 0},
            {2, 2, 0}, // (1,1,0) -> weight[4]
        },
        tiles_options
    );

    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat64);
    auto weight = torch::tensor(
        {{-1., -1., -1.}, {-1., -1., -1.}, {4.0, 4.1, 4.2}, {-1., -1., -1.}, {5.0, 5.1, 5.2}},
        tensor_options.requires_grad(true)
    );
    auto input = torch::tensor(
        {
            {0.1, 0.1}, // (2,0,0)
            {0.2, 0.1}, // (2,0,0)
            {1.3, 0.2}  // (2,2,0)
        },
        tensor_options
    );

    auto grad_output = torch::tensor(
        {{10.0, 1.0, 100.0}, {22.0, 2.2, 220.0}, {30.0, 3.0, 300.0}}, tensor_options
    );

    auto grad_weight = max_quad_pool2d_backward(
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

    BOOST_CHECK_EQUAL(grad_weight_acc[2][0], 10.0 + 22.0);
    BOOST_CHECK_EQUAL(grad_weight_acc[2][1], 1.0 + 2.2);
    BOOST_CHECK_EQUAL(grad_weight_acc[2][2], 100.0 + 220.0);

    BOOST_CHECK_EQUAL(grad_weight_acc[4][0], 30.0);
    BOOST_CHECK_EQUAL(grad_weight_acc[4][1], 3.0);
    BOOST_CHECK_EQUAL(grad_weight_acc[4][2], 300.0);
}


BOOST_AUTO_TEST_CASE(max_quad_pool2d_backward_grad_partial)
{
    auto tiles_options = torch::TensorOptions().dtype(torch::kInt64);
    auto tiles = torch::tensor(
        {
            {0, 0, 0}, // (0,0,0)
            {1, 0, 0}, // (0,0,0)
            {1, 0, 1}, // (0,0,0) -> weight[2]
            {1, 1, 0}, // (0,0,0)
            {1, 1, 1}, // (0,0,0) -> weight[4]
            {2, 0, 0}, // (1,0,0) -> weight[5]
            {2, 0, 1}, // (1,0,0) -> weight[6]
            {2, 2, 0}, // (1,1,0) -> weight[7]
            {2, 3, 1}  // (1,1,0) -> weight[8]
        },
        tiles_options
    );

    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat64);
    auto weight = torch::tensor(
        {
            {-1.},
            {-1.},
            {2.1}, // weight[2]
            {-1.},
            {3.0}, // weight[4]
            {1.0}, // weight[5]
            {2.0}, // weight[6]
            {4.0}, // weight[7]
            {4.5}  // weight[8]
        },
        tensor_options.requires_grad(true)
    );
    auto input = torch::tensor(
        {
            {0.1, 0.1}, // (2,0,0) -> argmax(weight[5], weight[6])
            {0.2, 0.1}, // (2,0,0) -> argmax(weight[5], weight[6])
            {1.3, 0.2}, // (2,2,0) -> argmax(weight[7], weight[8])
            {1.5, 1.5}, // (1,1,1) -> argmax(weight[0], ..., weight[8])
            {0.2, 1.2}, // (1,0,1) -> argmax(weight[0], ..., weight[8])
            {0.4, 1.3}  // (1,0,1) -> argmax(weight[0], ..., weight[8])
        },
        tensor_options
    );

    auto grad_output = torch::tensor({10.0, 22.0, 30.0, 43.0, 50.0, 66.0}, tensor_options);
    grad_output = at::unsqueeze(grad_output, 1);

    auto grad_weight = max_quad_pool2d_backward(
        grad_output, tiles, weight, input,
        /*exterior=*/{0.0, 0.0, 2.0, 2.0}
    );

    BOOST_REQUIRE_EQUAL(grad_weight.sizes(), torch::IntArrayRef({9, 1}));

    auto grad_weight_acc = grad_weight.accessor<double, 2>();
    BOOST_CHECK_EQUAL(grad_weight_acc[0][0], 0.0);
    BOOST_CHECK_EQUAL(grad_weight_acc[1][0], 0.0);
    BOOST_CHECK_EQUAL(grad_weight_acc[2][0], 0.0);
    BOOST_CHECK_EQUAL(grad_weight_acc[3][0], 0.0);
    BOOST_CHECK_EQUAL(grad_weight_acc[4][0], 0.0);
    BOOST_CHECK_EQUAL(grad_weight_acc[5][0], 0.0);
    BOOST_CHECK_EQUAL(grad_weight_acc[6][0], 10.0 + 22.0);
    BOOST_CHECK_EQUAL(grad_weight_acc[7][0], 0.0);
    BOOST_CHECK_EQUAL(grad_weight_acc[8][0], 30.0 + 43.0 + 50.0 + 66.0);
}


BOOST_AUTO_TEST_SUITE_END()
