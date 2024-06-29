#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE quadpool

#include <boost/test/included/unit_test.hpp>
#include <c10/util/Exception.h>
#include <torch/torch.h>

#include <torch_geopooling.h>

#include <quadpool_op.h>


using namespace torch_geopooling;


template <typename Exception>
std::function<bool(const Exception&)>
exception_contains_text(const std::string error_message)
{
    return [&](const Exception& error) -> bool {
        return std::string(error.what()).find(error_message) != std::string::npos;
    };
}


BOOST_AUTO_TEST_SUITE(TestQuadPoolOperation)


BOOST_AUTO_TEST_CASE(quadpool_op_tiles_errors)
{
    auto op = quadpool_op("test_op", {0.0, 0.0, 1.0, 1.0}, quadtree_options(), /*training=*/true);

    auto tiles = torch::empty({0, 3, 5}, torch::TensorOptions().dtype(torch::kInt64));
    auto weight = torch::rand({0, 1}, torch::TensorOptions().dtype(torch::kFloat64));
    auto input = torch::rand({100, 2}, torch::TensorOptions().dtype(torch::kFloat64));

    BOOST_CHECK_EXCEPTION(
        op.forward(tiles, weight, input), c10::Error,
        exception_contains_text<c10::Error>("operation only supports 2D tiles")
    );

    tiles = torch::empty({0, 4}, torch::TensorOptions().dtype(torch::kInt64));
    BOOST_CHECK_EXCEPTION(
        op.forward(tiles, weight, input), c10::Error,
        exception_contains_text<c10::Error>("tiles must be three-element tuples")
    );

    tiles = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kFloat64));
    BOOST_CHECK_EXCEPTION(
        op.forward(tiles, weight, input), c10::Error,
        exception_contains_text<c10::Error>("operation only supports Int64 tiles")
    );
}


BOOST_AUTO_TEST_CASE(quadpool_op_weight_errors)
{
    auto op = quadpool_op("test_op", {-.1, -1., 2., 2.}, quadtree_options(), /*training=*/true);

    auto tiles = torch::ones({10, 3}, torch::TensorOptions().dtype(torch::kInt64));
    auto weight = torch::rand({5, 3}, torch::TensorOptions().dtype(torch::kFloat64));
    auto input = torch::rand({10, 2}, torch::TensorOptions().dtype(torch::kFloat64));

    BOOST_CHECK_EXCEPTION(
        op.forward(tiles, weight, input), c10::Error,
        exception_contains_text<c10::Error>("number of tiles should be the same as weights")
    );

    weight = torch::rand({5, 5, 1}, torch::TensorOptions().dtype(torch::kFloat64));
    BOOST_CHECK_EXCEPTION(
        op.forward(tiles, weight, input), c10::Error,
        exception_contains_text<c10::Error>("operation only supports 2D weight")
    );
}


BOOST_AUTO_TEST_CASE(quadpool_op_inputs)
{
    auto op = quadpool_op("test_op", {0.0, 0.0, 20.0, 20.0}, quadtree_options(), /*training=*/true);

    auto tiles = torch::empty({0, 3}, torch::TensorOptions().dtype(torch::kInt64));
    auto weight = torch::rand({0, 1}, torch::TensorOptions().dtype(torch::kFloat64));
    auto input = torch::rand({100, 2}, torch::TensorOptions().dtype(torch::kFloat64));

    BOOST_CHECK_EXCEPTION(
        op.forward(tiles, weight, input - 300.0), value_error,
        exception_contains_text<value_error>("is outside of exterior geometry")
    );

    input = torch::rand({10, 2, 5, 1}, torch::TensorOptions().dtype(torch::kFloat64));
    BOOST_CHECK_EXCEPTION(
        op.forward(tiles, weight, input), c10::Error,
        exception_contains_text<c10::Error>("operation only supports 2D input")
    );

    input = torch::rand({10, 7}, torch::TensorOptions().dtype(torch::kFloat64));
    BOOST_CHECK_EXCEPTION(
        op.forward(tiles, weight, input), c10::Error,
        exception_contains_text<c10::Error>("input must be two-element tuples")
    );

    input = torch::empty({10, 2}, torch::TensorOptions().dtype(torch::kInt32));
    BOOST_CHECK_EXCEPTION(
        op.forward(tiles, weight, input), c10::Error,
        exception_contains_text<c10::Error>("operation only supports Float64 input")
    );
}


BOOST_AUTO_TEST_CASE(quadpool_op_parentless_tiles)
{
    auto op = quadpool_op("test_op", {0.0, 0.0, 20.0, 20.0}, quadtree_options(), /*training=*/true);

    auto tiles = torch::tensor(
        {
            {1, 1, 1},
            {9, 0, 0},
            {1, 0, 0},
            {8, 0, 0},
            {4, 0, 0},
        },
        torch::TensorOptions().dtype(torch::kInt64)
    );

    auto weight = torch::rand({tiles.size(0), 1}, torch::TensorOptions().dtype(torch::kFloat64));
    auto input = torch::rand({10, 2}, torch::TensorOptions().dtype(torch::kFloat64));

    BOOST_CHECK_EXCEPTION(
        op.forward(tiles, weight, input), value_error,
        exception_contains_text<value_error>("does not have a parent")
    );
}


BOOST_AUTO_TEST_SUITE_END()
