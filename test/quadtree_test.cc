#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE quadtree

#include <boost/test/unit_test.hpp>

#include <torch_geopooling.h>


using namespace torch_geopooling;


BOOST_AUTO_TEST_SUITE(TestQuadtree)


BOOST_AUTO_TEST_CASE(quadtree_is_leaf)
{
    BOOST_TEST_MESSAGE("--- Empty quadtree is leaf");

    quadtree tree({-180.0, -90.0, 360.0, 180.0});

    BOOST_CHECK(tree.is_leaf());
}


BOOST_AUTO_TEST_CASE(quadtree_contains)
{
    BOOST_TEST_MESSAGE("--- Empty quadtree contains");

    quadtree tree({0, 0, 10, 10});

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

    quadtree tree({0, 0, 10, 10});

    auto leaf = tree.find(std::pair(2, 2));

    BOOST_CHECK(leaf.exterior().centroid() == std::pair(5, 5));
}


BOOST_AUTO_TEST_CASE(quadtree_insert_and_find)
{
    BOOST_TEST_MESSAGE("--- Find tree");

    quadtree tree({-10.0, -10.0, 20.0, 20.0});
    tree.insert(std::make_pair(0.0, 0.0), 0);
    tree.insert(std::make_pair(1.0, 1.0), 1);
    tree.insert(std::make_pair(1.5, 1.5), 2);

    auto node1 = tree.find(std::make_pair(1.5, 1.5), 3);
    BOOST_CHECK_EQUAL(node1.depth(), 3);
    BOOST_CHECK_EQUAL(node1.exterior(), quadrect({0.0, 0.0, 2.5, 2.5}));

    auto node2 = tree.find(std::make_pair(1.5, 1.5), 5);
    BOOST_CHECK_EQUAL(node2.depth(), 4);
    BOOST_CHECK_EQUAL(node2.exterior(), quadrect({1.25, 1.25, 1.25, 1.25}));
}


BOOST_AUTO_TEST_CASE(quadtree_insert_depth_1)
{
    BOOST_TEST_MESSAGE("--- Quadtree of depth 1");

    quadtree tree({0, 0, 10, 10});

    tree.insert(std::make_pair(1, 1), 0);
    tree.insert(std::make_pair(1, 9), 1);
    tree.insert(std::make_pair(9, 9), 2);
    tree.insert(std::make_pair(9, 1), 3);

    BOOST_CHECK_EQUAL(tree.size(), 4);
    BOOST_CHECK_EQUAL(tree.total_depth(), 1);
}


BOOST_AUTO_TEST_CASE(quadtree_insert_depth_3)
{
    BOOST_TEST_MESSAGE("--- Quadtree of depth 3");

    quadtree tree({0.0, 0.0, 10.0, 10.0});

    tree.insert(std::make_pair(1.0, 1.0), 0);
    tree.insert(std::make_pair(1.7, 1.7), 1);
    tree.insert(std::make_pair(1.0, 1.7), 2);
    tree.insert(std::make_pair(1.7, 1.0), 3);

    BOOST_CHECK_EQUAL(tree.size(), 4);
    BOOST_CHECK_EQUAL(tree.total_depth(), 3);

    auto node1 = tree.find(std::make_pair(1.0, 1.0));
    BOOST_CHECK_EQUAL(node1.depth(), 3);
    BOOST_CHECK_EQUAL(node1.tile(), Tile(3, 0, 0));
    BOOST_CHECK_EQUAL(node1.exterior(), quadrect({0.0, 0.0, 1.25, 1.25}));

    auto node2 = tree.find(std::make_pair(1.7, 1.7));
    BOOST_CHECK_EQUAL(node2.depth(), 3);
    BOOST_CHECK_EQUAL(node2.tile(), Tile(3, 1, 1));
    BOOST_CHECK_EQUAL(node2.exterior(), quadrect({1.25, 1.25, 1.25, 1.25}));

    auto node3 = tree.find(std::make_pair(1.0, 1.7));
    BOOST_CHECK_EQUAL(node3.depth(), 3);
    BOOST_CHECK_EQUAL(node3.tile(), Tile(3, 0, 1));
    BOOST_CHECK_EQUAL(node3.exterior(), quadrect({0.0, 1.25, 1.25, 1.25}));

    auto node4 = tree.find(std::make_pair(1.7, 1.0));
    BOOST_CHECK_EQUAL(node4.depth(), 3);
    BOOST_CHECK_EQUAL(node4.tile(), Tile(3, 1, 0));
    BOOST_CHECK_EQUAL(node4.exterior(), quadrect({1.25, 0.0, 1.25, 1.25}));
}


BOOST_AUTO_TEST_CASE(quadtree_find_by_tile)
{
    BOOST_TEST_MESSAGE("--- Quadtree find node by tile");

    quadtree tree({0.0, 0.0, 10.0, 10.0});

    tree.insert(std::make_pair(1.0, 1.0), 0);
    tree.insert(std::make_pair(1.7, 1.7), 1);
    tree.insert(std::make_pair(1.0, 1.7), 2);
    tree.insert(std::make_pair(1.7, 1.0), 3);
    tree.insert(std::make_pair(9.9, 9.9), 4);
    tree.insert(std::make_pair(8.0, 8.0), 5);

    auto node1 = tree.find(Tile(0, 0, 0));
    BOOST_CHECK_EQUAL(node1.tile(), Tile(0, 0, 0));

    auto node2 = tree.find(Tile(3, 0, 1));
    BOOST_CHECK_EQUAL(node2.tile(), Tile(3, 0, 1));

    auto node3 = tree.find(Tile(3, 7, 7));
    BOOST_CHECK_EQUAL(node3.tile(), Tile(3, 7, 7));

    auto node4 = tree.find(Tile(3, 6, 6));
    BOOST_CHECK_EQUAL(node4.tile(), Tile(3, 6, 6));

    auto node5 = tree.find(Tile(3, 4, 4));
    BOOST_CHECK_EQUAL(node5.tile(), Tile(2, 2, 2));
}


BOOST_AUTO_TEST_SUITE_END()
