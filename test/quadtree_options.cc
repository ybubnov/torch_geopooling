#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE quadpool

#include <random>

#include <boost/test/included/unit_test.hpp>

#include <torch_geopooling.h>


using namespace torch_geopooling;


BOOST_AUTO_TEST_SUITE(TestQuadtreeOptions)


BOOST_AUTO_TEST_CASE(quadtree_options_has_value)
{
    auto opts = quadtree_options();

    std::random_device dev;
    std::default_random_engine engine(dev());
    std::uniform_int_distribution<std::size_t> uniform_rand(1, 1024);

    BOOST_CHECK(!opts.has_max_terminal_nodes());
    BOOST_CHECK(!opts.has_precision());

    std::size_t precision = uniform_rand(engine);
    std::size_t max_terminal_nodes = uniform_rand(engine);
    std::size_t capacity = uniform_rand(engine);
    std::size_t max_depth = uniform_rand(engine);
    opts = opts.precision(precision)
               .max_terminal_nodes(max_terminal_nodes)
               .capacity(capacity)
               .max_depth(max_depth);

    BOOST_CHECK(opts.has_max_terminal_nodes());
    BOOST_CHECK(opts.has_precision());

    BOOST_CHECK_EQUAL(opts.precision(), precision);
    BOOST_CHECK_EQUAL(opts.max_terminal_nodes(), max_terminal_nodes);
    BOOST_CHECK_EQUAL(opts.capacity(), capacity);
    BOOST_CHECK_EQUAL(opts.max_depth(), max_depth);
}


BOOST_AUTO_TEST_SUITE_END()
