#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE quadtree

#include <boost/test/included/unit_test.hpp>

#include <torch_geopooling.h>


using namespace torch_geopooling;


BOOST_AUTO_TEST_SUITE(TestTile)


BOOST_AUTO_TEST_CASE(tile_constructor)
{
    BOOST_CHECK_THROW(Tile(1024, 0, 0), value_error);
    BOOST_CHECK_THROW(Tile(2, 1024, 0), value_error);
    BOOST_CHECK_THROW(Tile(4, 0, 1024), value_error);
}


BOOST_AUTO_TEST_CASE(tile_children)
{
    auto parent = Tile(62, 10, 10);
    auto children = parent.children();

    BOOST_REQUIRE_EQUAL(children.size(), 4);

    BOOST_CHECK_EQUAL(children[0], Tile(63, 20, 20));
    BOOST_CHECK_EQUAL(children[1], Tile(63, 20, 21));
    BOOST_CHECK_EQUAL(children[2], Tile(63, 21, 20));
    BOOST_CHECK_EQUAL(children[3], Tile(63, 21, 21));
}


BOOST_AUTO_TEST_SUITE_END()
