#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE quadtree_set

#include <unordered_set>

#include <fmt/format.h>
#include <boost/test/included/unit_test.hpp>

#include <torch_geopooling.h>


using namespace torch_geopooling;


BOOST_AUTO_TEST_SUITE(Testquadtree_set)


BOOST_AUTO_TEST_CASE(quadtree_set_contains)
{
    quadtree_set set({0, 0, 10, 10});

    BOOST_CHECK(set.contains(std::pair(0, 0)));
    BOOST_CHECK(set.contains(std::pair(0, 10)));
    BOOST_CHECK(set.contains(std::pair(10, 0)));
    BOOST_CHECK(set.contains(std::pair(10, 10)));

    BOOST_CHECK(!set.contains(std::pair(-1, -1)));
    BOOST_CHECK(!set.contains(std::pair(11, 11)));
}


BOOST_AUTO_TEST_CASE(quadtree_set_find_in_empty)
{
    quadtree_set set({0, 0, 10, 10});

    auto node = set.find(std::pair(2, 2));
    BOOST_CHECK_EQUAL(node.exterior(), quadrect(0, 0, 10, 10));
}


BOOST_AUTO_TEST_CASE(quadtree_set_insert_and_find)
{
    quadtree_set set({-10.0, -10.0, 20.0, 20.0});
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
    quadtree_set set({0, 0, 10, 10});

    set.insert(std::make_pair(1, 1));
    set.insert(std::make_pair(1, 9));
    set.insert(std::make_pair(9, 9));
    set.insert(std::make_pair(9, 1));

    BOOST_CHECK_EQUAL(set.size(), 4);
    BOOST_CHECK_EQUAL(set.total_depth(), 1);
}


BOOST_AUTO_TEST_CASE(quadtree_set_insert_depth_3)
{
    quadtree_set set({0.0, 0.0, 10.0, 10.0});

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
    quadtree_set set({0.0, 0.0, 10.0, 10.0});

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


BOOST_AUTO_TEST_CASE(quadtree_set_find_terminal_group)
{
    quadtree_set set({0.0, 0.0, 10.0, 10.0});

    set.insert(std::make_pair(1.0, 1.0)); // depth = 0
    set.insert(std::make_pair(1.7, 1.7)); // depth = 1
    set.insert(std::make_pair(1.0, 1.7)); // depth = 2
    set.insert(std::make_pair(1.7, 1.0)); // depth = 3

    // Query a node from the bottom of the quadtree, which has exactly 4 neighbours.
    std::unordered_set<Tile> nodes1;
    auto point1 = std::make_pair(1.8, 1.8);

    for (auto it = set.find_terminal_group(point1); it != set.end(); ++it) {
        Tile tile = it->tile();
        bool is_terminal = !(
            set.contains(tile.child(0, 0)) ||
            set.contains(tile.child(0, 1)) ||
            set.contains(tile.child(1, 0)) ||
            set.contains(tile.child(1, 1))
        );

        BOOST_REQUIRE_MESSAGE(is_terminal, fmt::format("tile {} is not terminal", tile));
        nodes1.insert(tile);
    }

    BOOST_CHECK_EQUAL(nodes1.size(), 4);

    // Repeat the same, but now take node from the lower-depth node. In this case, the
    // terminal group will be larger as we query high-partitioned neighbour quads.
    std::unordered_set<Tile> nodes2;
    auto point2 = std::make_pair(9.9, 9.9);

    for (auto it = set.find_terminal_group(point2); it != set.end(); ++it) {
        Tile tile = it->tile();
        bool is_terminal = !(
            set.contains(tile.child(0, 0)) ||
            set.contains(tile.child(0, 1)) ||
            set.contains(tile.child(1, 0)) ||
            set.contains(tile.child(1, 1))
        );

        BOOST_REQUIRE_MESSAGE(is_terminal, fmt::format("tile {} is not terminal", tile));
        nodes2.insert(tile);
    }

    BOOST_CHECK_EQUAL(nodes2.size(), 10);
}


BOOST_AUTO_TEST_CASE(quadtree_set_from_tiles)
{
    std::vector<Tile> tiles = {
        Tile(0, 0, 0),
        Tile(1, 0, 0),
        Tile(1, 1, 1),
        Tile(2, 0, 0),
    };

    quadtree_set set(tiles.begin(), tiles.end(), quadrect(0.0, 0.0, 10.0, 10.0));

    auto node1 = set.find(Tile(0, 0, 0));
    BOOST_CHECK_EQUAL(node1.tile(), Tile(0, 0, 0));
    BOOST_CHECK_EQUAL(node1.size(), 0);

    auto node2 = set.find(Tile(3, 7, 7));
    BOOST_CHECK_EQUAL(node2.tile(), Tile(1, 1, 1));
    BOOST_CHECK_EQUAL(node2.size(), 0);
}


BOOST_AUTO_TEST_CASE(quadtree_set_missing_parent)
{
    std::vector<Tile> tiles = {
        Tile(0, 0, 0),
        Tile(1, 1, 1),
        Tile(2, 0, 0),
    };

    BOOST_CHECK_THROW(
        quadtree_set(tiles.begin(), tiles.end(), quadrect(0.0, 0.0, 10.0, 10.0)),
        value_error
    );
}


BOOST_AUTO_TEST_SUITE_END()
