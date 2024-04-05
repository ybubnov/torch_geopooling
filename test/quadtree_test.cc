#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE quadtree


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


BOOST_AUTO_TEST_CASE(quadtree_insert_and_find)
{
    BOOST_TEST_MESSAGE("--- Find tree");

    geopool::quadtree tree({-10.0, -10.0, 20.0, 20.0});
    tree.insert(std::make_pair(0.0, 0.0), 0);
    tree.insert(std::make_pair(1.0, 1.0), 1);
    tree.insert(std::make_pair(1.5, 1.5), 2);

    auto node1 = tree.find(std::make_pair(1.5, 1.5), 3);
    BOOST_CHECK_EQUAL(node1.depth(), 3);
    BOOST_CHECK_EQUAL(node1.exterior(), geopool::quadrect({0.0, 0.0, 2.5, 2.5}));

    auto node2 = tree.find(std::make_pair(1.5, 1.5), 5);
    BOOST_CHECK_EQUAL(node2.depth(), 4);
    BOOST_CHECK_EQUAL(node2.exterior(), geopool::quadrect({1.25, 1.25, 1.25, 1.25}));
}


BOOST_AUTO_TEST_SUITE_END()
