#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE quadtree_set

#include <boost/test/unit_test.hpp>

#include <torch_geopooling.h>


using namespace torch_geopooling;


BOOST_AUTO_TEST_SUITE(TestQuadtreeSet)


BOOST_AUTO_TEST_CASE(quadtree_set_contains)
{
    BOOST_TEST_MESSAGE("--- Check empty quadtree set contains points");

    QuadtreeSet set({0, 0, 10, 10});

    BOOST_CHECK(set.contains(std::pair(0, 0)));
    BOOST_CHECK(set.contains(std::pair(0, 10)));
    BOOST_CHECK(set.contains(std::pair(10, 0)));
    BOOST_CHECK(set.contains(std::pair(10, 10)));

    BOOST_CHECK(!set.contains(std::pair(-1, -1)));
    BOOST_CHECK(!set.contains(std::pair(11, 11)));
}


BOOST_AUTO_TEST_CASE(quadtree_set_find_in_empty)
{
    BOOST_TEST_MESSAGE("--- Find nodes in empty quadtree set");

    QuadtreeSet set({0, 0, 10, 10});

    auto node = set.find(std::pair(2, 2));
    BOOST_CHECK_EQUAL(node.exterior(), quadrect(0, 0, 10, 10));
}


BOOST_AUTO_TEST_CASE(quadtree_set_insert_and_find)
{
    BOOST_TEST_MESSAGE("--- Insert point and find node in quatree set");

    QuadtreeSet set({-10.0, -10.0, 20.0, 20.0});
    set.insert(std::make_pair(0.0, 0.0));
    set.insert(std::make_pair(1.0, 1.0));
    set.insert(std::make_pair(1.5, 1.5));

    auto node1 = set.find(std::make_pair(1.5, 1.5), 3);
    BOOST_CHECK_EQUAL(node1.tile().z(), 3);
    BOOST_CHECK_EQUAL(node1.exterior(), quadrect({0.0, 0.0, 2.5, 2.5}));

    auto node2 = set.find(std::make_pair(1.5, 1.5), 5);
    BOOST_CHECK_EQUAL(node2.tile().z(), 4);
    BOOST_CHECK_EQUAL(node2.exterior(), quadrect({1.25, 1.25, 1.25, 1.25}));
}


BOOST_AUTO_TEST_CASE(quadtree_set_insert_depth_1)
{
    BOOST_TEST_MESSAGE("--- Quadtree set of depth 1");

    QuadtreeSet set({0, 0, 10, 10});

    set.insert(std::make_pair(1, 1));
    set.insert(std::make_pair(1, 9));
    set.insert(std::make_pair(9, 9));
    set.insert(std::make_pair(9, 1));

    BOOST_CHECK_EQUAL(set.size(), 4);
    BOOST_CHECK_EQUAL(set.total_depth(), 1);
}


BOOST_AUTO_TEST_CASE(quadtree_set_insert_depth_3)
{
    BOOST_TEST_MESSAGE("--- Quadtree set of depth 3");

    QuadtreeSet set({0.0, 0.0, 10.0, 10.0});

    set.insert(std::make_pair(1.0, 1.0));
    set.insert(std::make_pair(1.7, 1.7));
    set.insert(std::make_pair(1.0, 1.7));
    set.insert(std::make_pair(1.7, 1.0));

    BOOST_CHECK_EQUAL(set.size(), 4);
    BOOST_CHECK_EQUAL(set.total_depth(), 3);

    auto node1 = set.find(std::make_pair(1.0, 1.0));
    BOOST_CHECK_EQUAL(node1.depth(), 3);
    BOOST_CHECK_EQUAL(node1.tile(), Tile(3, 0, 0));
    BOOST_CHECK_EQUAL(node1.exterior(), quadrect({0.0, 0.0, 1.25, 1.25}));

    auto node2 = set.find(std::make_pair(1.7, 1.7));
    BOOST_CHECK_EQUAL(node2.depth(), 3);
    BOOST_CHECK_EQUAL(node2.tile(), Tile(3, 1, 1));
    BOOST_CHECK_EQUAL(node2.exterior(), quadrect({1.25, 1.25, 1.25, 1.25}));

    auto node3 = set.find(std::make_pair(1.0, 1.7));
    BOOST_CHECK_EQUAL(node3.depth(), 3);
    BOOST_CHECK_EQUAL(node3.tile(), Tile(3, 0, 1));
    BOOST_CHECK_EQUAL(node3.exterior(), quadrect({0.0, 1.25, 1.25, 1.25}));

    auto node4 = set.find(std::make_pair(1.7, 1.0));
    BOOST_CHECK_EQUAL(node4.depth(), 3);
    BOOST_CHECK_EQUAL(node4.tile(), Tile(3, 1, 0));
    BOOST_CHECK_EQUAL(node4.exterior(), quadrect({1.25, 0.0, 1.25, 1.25}));
}


BOOST_AUTO_TEST_CASE(quadtree_set_find_by_tile)
{
    BOOST_TEST_MESSAGE("--- Find node by tile in non-empty quadtree set");

    QuadtreeSet set({0.0, 0.0, 10.0, 10.0});

    set.insert(std::make_pair(1.0, 1.0));
    set.insert(std::make_pair(1.7, 1.7));
    set.insert(std::make_pair(1.0, 1.7));
    set.insert(std::make_pair(1.7, 1.0));
    set.insert(std::make_pair(9.9, 9.9));
    set.insert(std::make_pair(8.0, 8.0));

    auto node1 = set.find(Tile(0, 0, 0));
    BOOST_CHECK_EQUAL(node1.tile(), Tile(0, 0, 0));

    auto node2 = set.find(Tile(3, 0, 1));
    BOOST_CHECK_EQUAL(node2.tile(), Tile(3, 0, 1));

    auto node3 = set.find(Tile(3, 7, 7));
    BOOST_CHECK_EQUAL(node3.tile(), Tile(3, 7, 7));

    auto node4 = set.find(Tile(3, 6, 6));
    BOOST_CHECK_EQUAL(node4.tile(), Tile(3, 6, 6));

    auto node5 = set.find(Tile(3, 4, 4));
    BOOST_CHECK_EQUAL(node5.tile(), Tile(2, 2, 2));
}


BOOST_AUTO_TEST_CASE(quadtree_set_from_tiles)
{
    BOOST_TEST_MESSAGE("--- Initialize quadtree set from tiles");

    std::vector<Tile> tiles = {
        Tile(0, 0, 0),
        Tile(1, 0, 0),
        Tile(1, 1, 1),
        Tile(2, 0, 0),
    };

    QuadtreeSet set(tiles.begin(), tiles.end(), quadrect(0.0, 0.0, 10.0, 10.0));

    auto node1 = set.find(Tile(0, 0, 0));
    BOOST_CHECK_EQUAL(node1.tile(), Tile(0, 0, 0));
    BOOST_CHECK_EQUAL(node1.size(), 0);

    auto node2 = set.find(Tile(3, 7, 7));
    BOOST_CHECK_EQUAL(node2.tile(), Tile(1, 1, 1));
    BOOST_CHECK_EQUAL(node2.size(), 0);
}


BOOST_AUTO_TEST_CASE(quadtree_set_missing_parent)
{
    BOOST_TEST_MESSAGE("--- Ensure quadtree set initialization fails on missing parent");

    std::vector<Tile> tiles = {
        Tile(0, 0, 0),
        Tile(1, 1, 1),
        Tile(2, 0, 0),
    };

    BOOST_CHECK_THROW(QuadtreeSet(tiles.begin(), tiles.end(), quadrect(0.0, 0.0, 10.0, 10.0)));
}


BOOST_AUTO_TEST_SUITE_END()
