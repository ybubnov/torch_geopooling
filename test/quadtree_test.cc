#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Bencode


#include <boost/test/unit_test.hpp>
#include <geopool.h>


BOOST_AUTO_TEST_SUITE(TestQuadtree)


BOOST_AUTO_TEST_CASE(quadtree_is_leaf)
{
    BOOST_TEST_MESSAGE("--- Empty quadtree is leaf");

    geopool::quadtree tree({-180.0, -90.0, 360.0, 180.0});

    BOOST_CHECK(tree.is_leaf());
}


BOOST_AUTO_TEST_CASE(quadtree_contains)
{
    BOOST_TEST_MESSAGE("--- Empty quadtree contains");

    geopool::quadtree tree({0, 0, 10, 10});

    BOOST_CHECK(tree.contains(std::pair(0, 0)));
    BOOST_CHECK(tree.contains(std::pair(0, 10)));
    BOOST_CHECK(tree.contains(std::pair(10, 0)));
    BOOST_CHECK(tree.contains(std::pair(10, 10)));

    BOOST_CHECK(!tree.contains(std::pair(-1, -1)));
    BOOST_CHECK(!tree.contains(std::pair(11, 11)));
}


BOOST_AUTO_TEST_CASE(quadtree_find_empty)
{
    BOOST_TEST_MESSAGE("--- Empty quadtree find");

    geopool::quadtree tree({0, 0, 10, 10});

    auto leaf = tree.find(std::pair(2, 2));

    BOOST_CHECK(leaf.exterior().centroid() == std::pair(5, 5));
}


BOOST_AUTO_TEST_SUITE_END()
