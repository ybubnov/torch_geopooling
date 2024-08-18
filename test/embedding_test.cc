#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE embedding

#include <boost/test/included/unit_test.hpp>
#include <torch/torch.h>

#include <torch_geopooling.h>


using namespace torch::indexing;
using namespace torch_geopooling;


BOOST_AUTO_TEST_SUITE(TestEmbedding)


BOOST_AUTO_TEST_CASE(embedding2d_eval)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

    auto weight = torch::full({4, 4, 5}, 1.0, options);
    auto weight_ptr = weight.accessor<double, 3>();

    auto make_weight = [](int64_t a, int64_t b) -> double { return (a + 1) * 10 + b; };

    for (auto i : c10::irange(weight.size(0))) {
        for (auto j : c10::irange(weight.size(1))) {
            for (auto k : c10::irange(weight.size(2))) {
                weight_ptr[i][j][k] = make_weight(i, j);
            }
        }
    }

    auto expect = torch::full({2, 3, 3, 5}, 0.0, options);
    // First coordinate (1.0, 1.0) lands on (0, 0) cell.
    expect.index_put_({0, 0, 0, Ellipsis}, make_weight(3, 3)); // (3, 3)
    expect.index_put_({0, 0, 1, Ellipsis}, make_weight(3, 0)); // (3, 0)
    expect.index_put_({0, 0, 2, Ellipsis}, make_weight(3, 1)); // (3, 1)

    expect.index_put_({0, 1, 0, Ellipsis}, make_weight(0, 3)); // (0, 3)
    expect.index_put_({0, 1, 1, Ellipsis}, make_weight(0, 0)); // (0, 0)
    expect.index_put_({0, 1, 2, Ellipsis}, make_weight(0, 1)); // (0, 1)

    expect.index_put_({0, 2, 0, Ellipsis}, make_weight(1, 3)); // (1, 3)
    expect.index_put_({0, 2, 1, Ellipsis}, make_weight(1, 0)); // (1, 0)
    expect.index_put_({0, 2, 2, Ellipsis}, make_weight(1, 1)); // (1, 1)

    // Second coordinate (6.0, 6.0) lands on (2, 2) cell.
    expect.index_put_({1, 0, 0, Ellipsis}, make_weight(1, 1)); // (1, 1)
    expect.index_put_({1, 0, 1, Ellipsis}, make_weight(1, 2)); // (1, 2)
    expect.index_put_({1, 0, 2, Ellipsis}, make_weight(1, 3)); // (1, 3)

    expect.index_put_({1, 1, 0, Ellipsis}, make_weight(2, 1)); // (2, 1)
    expect.index_put_({1, 1, 1, Ellipsis}, make_weight(2, 2)); // (2, 2)
    expect.index_put_({1, 1, 2, Ellipsis}, make_weight(2, 3)); // (2, 3)

    expect.index_put_({1, 2, 0, Ellipsis}, make_weight(3, 1)); // (3, 1)
    expect.index_put_({1, 2, 1, Ellipsis}, make_weight(3, 2)); // (3, 2)
    expect.index_put_({1, 2, 2, Ellipsis}, make_weight(3, 3)); // (3, 3)

    // Weight represents the following field:
    //
    //       [0]|  [1]|  [2]|  [3]|
    // [0]|0.0  |2.5  |5.0  |7.5  |≤10.0
    // [1]|0.0  |2.5  |5.0  |7.5  |≤10.0
    // [2]|0.0  |2.5  |5.0  |7.5  |≤10.0
    // [3]|0.0  |2.5  |5.0  |7.5  |≤10.0
    auto input = torch::tensor(
        {
            {1.0, 1.0}, // center at (0, 0)
            {6.0, 6.0}, // center at (2, 2)
        },
        options
    );

    auto padding = std::vector<int64_t>({1, 1});
    auto exterior = std::vector<double>({0.0, 0.0, 10.0, 10.0});
    auto output = embedding2d(input, weight, /*padding=*/padding, /*exterior=*/exterior);

    BOOST_CHECK_EQUAL(output.sizes(), torch::IntArrayRef({2, 3, 3, 5}));
    BOOST_CHECK(torch::equal(output[0], expect[0]));
    BOOST_CHECK(torch::equal(output[1], expect[1]));
}


BOOST_AUTO_TEST_CASE(embedding2d_backward_grad)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

    auto weight = torch::full({4, 4, 5}, 1.0, options);
    auto weight_ptr = weight.accessor<double, 3>();

    auto input = torch::tensor(
        {
            {1.0, 1.0}, // center at (0, 0)
            {6.0, 6.0}, // center at (2, 2)
        },
        options
    );

    auto grad = torch::zeros({2, 3, 3, 5});
    grad.index_put_({0, Ellipsis, Ellipsis, Ellipsis}, 1.0);
    grad.index_put_({1, Ellipsis, Ellipsis, Ellipsis}, 10.0);

    auto padding = std::vector<int64_t>({1, 1});
    auto exterior = std::vector<double>({0.0, 0.0, 10.0, 10.0});

    auto grad_weight
        = embedding2d_backward(grad, input, weight, /*padding=*/padding, /*exterior=*/exterior);

    BOOST_REQUIRE_EQUAL(grad_weight.sizes(), weight.sizes());
    // Grad of weight represents the following field:
    //
    // [0]|    1|    1|    0|    0|
    // [1]|    1|   11|   10|   10|
    // [2]|    0|   10|   10|   10|
    // [3]|    1|   11|   10|   11|
    //       [0]|  [1]|  [2]|  [3]|

    auto grad_expect = torch::tensor(
        {
            {1.0, 1.0, 0.0, 1.0},
            {1.0, 11.0, 10.0, 11.0},
            {0.0, 10.0, 10.0, 10.0},
            {1.0, 11.0, 10.0, 11.0},
        },
        options
    );

    for (auto i : c10::irange(weight.size(0))) {
        for (auto j : c10::irange(weight.size(1))) {
            for (auto k : c10::irange(weight.size(2))) {
                BOOST_CHECK_EQUAL(
                    grad_weight[i][j][k].item<double>(), grad_expect[i][j].item<double>()
                );
            }
        }
    }
}


BOOST_AUTO_TEST_CASE(embedding2d_eval_large)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

    auto weight = torch::rand({1024, 1024, 3}, options);
    auto input = torch::rand({100, 2}, options) * 12.0;

    auto output
        = embedding2d(input, weight, /*padding=*/{3, 2}, /*exterior=*/{-10.0, 10.0, 20.0, 20.0});

    BOOST_CHECK_EQUAL(output.sizes(), c10::IntArrayRef({100, 7, 5, 3}));
}


BOOST_AUTO_TEST_SUITE_END()
