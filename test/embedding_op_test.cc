#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE embedding

#include <boost/test/included/unit_test.hpp>
#include <c10/util/Exception.h>
#include <torch/torch.h>

#include <torch_geopooling.h>

#include <embedding_op.h>
#include "testing.h"


using namespace torch_geopooling;


BOOST_AUTO_TEST_SUITE(TestEmbeddingOperation)


BOOST_AUTO_TEST_CASE(embedding_options_exterior)
{
    auto options = embedding_options {
        .padding = {0, 0},
        .exterior = {0, 0},
        .manifold = {0, 0},
    };

    auto input = torch::empty({1, 1});
    auto weight = torch::empty({4, 4, 2});

    BOOST_CHECK_EXCEPTION(
        check_shape_forward("test_op", input, weight, options), c10::Error,
        exception_contains_text<c10::Error>("exterior must be a tuple of four doubles")
    );
}


BOOST_AUTO_TEST_CASE(embedding_options_padding)
{
    auto options = embedding_options {
        .padding = {0},
        .exterior = {0.0, 0.0, 10.0, 10.0},
        .manifold = {4, 4, 3},
    };

    auto tensor_options = torch::TensorOptions().dtype(torch::kFloat64);
    auto input = torch::empty({1, 2}, tensor_options);
    auto weight = torch::empty(options.manifold, tensor_options);

    BOOST_CHECK_EXCEPTION(
        check_shape_forward("test_op", input, weight, options), c10::Error,
        exception_contains_text<c10::Error>("padding should be comprised of 2 elements")
    );

    options.padding = {-1, 0};
    BOOST_CHECK_EXCEPTION(
        check_shape_forward("test_op", input, weight, options), c10::Error,
        exception_contains_text<c10::Error>("padding should be non-negative")
    );

    options.padding = {4, 4};
    BOOST_CHECK_EXCEPTION(
        check_shape_forward("test_op", input, weight, options), c10::Error,
        exception_contains_text<c10::Error>("padding should be inside of the manifold")
    );

    options.padding = {1, 1};
    BOOST_CHECK_NO_THROW(check_shape_forward("test_op", input, weight, options));
}


BOOST_AUTO_TEST_CASE(embedding_check_shape_forward)
{
    auto options = embedding_options {
        .padding = {0, 0},
        .exterior = {0.0, 0.0, 10.0, 10.0},
        .manifold = {16, 16, 5},
    };

    auto int64_dtype = torch::TensorOptions().dtype(torch::kInt64);
    auto float64_dtype = torch::TensorOptions().dtype(torch::kFloat64);

    auto input = torch::empty({3, 2, 1}, int64_dtype);
    auto weight = torch::empty({16, 16, 5, 1}, int64_dtype);

    BOOST_CHECK_EXCEPTION(
        check_shape_forward("test_op", input, weight, options), c10::Error,
        exception_contains_text<c10::Error>("input must be 2D")
    );

    input = torch::empty({3, 3}, int64_dtype);
    BOOST_CHECK_EXCEPTION(
        check_shape_forward("test_op", input, weight, options), c10::Error,
        exception_contains_text<c10::Error>("input must be comprised of 2D coordinates")
    );

    input = torch::empty({3, 2}, int64_dtype);
    BOOST_CHECK_EXCEPTION(
        check_shape_forward("test_op", input, weight, options), c10::Error,
        exception_contains_text<c10::Error>("operation only supports Float64 input")
    );

    input = torch::empty({3, 2}, float64_dtype);
    BOOST_CHECK_EXCEPTION(
        check_shape_forward("test_op", input, weight, options), c10::Error,
        exception_contains_text<c10::Error>("weight must be 3D")
    );

    weight = torch::empty(options.manifold, int64_dtype);
    BOOST_CHECK_EXCEPTION(
        check_shape_forward("test_op", input, weight, options), c10::Error,
        exception_contains_text<c10::Error>("operation only supports Float64 weight")
    );

    weight = torch::empty(options.manifold, float64_dtype);
    BOOST_CHECK_NO_THROW(check_shape_forward("test_op", input, weight, options));
}


BOOST_AUTO_TEST_CASE(embedding_check_shape_backward)
{
    auto options = embedding_options {
        .padding = {4, 2},
        .exterior = {0.0, 0.0, 10.0, 10.0},
        .manifold = {16, 16, 7},
    };

    auto float64_dtype = torch::TensorOptions().dtype(torch::kFloat64);

    auto grad = torch::empty({3, 6, 5}, float64_dtype);
    auto input = torch::empty({3, 2}, float64_dtype);
    auto weight = torch::empty(options.manifold, float64_dtype);

    BOOST_CHECK_EXCEPTION(
        check_shape_backward("test_op", grad, input, weight, options), c10::Error,
        exception_contains_text<c10::Error>("does not match expected shape")
    );

    grad = torch::empty({3, 9, 5, 7}, float64_dtype);
    BOOST_CHECK_NO_THROW(check_shape_backward("test_op", grad, input, weight, options));
}


BOOST_AUTO_TEST_SUITE_END()
