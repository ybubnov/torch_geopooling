#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE quadtree_set

#include <boost/test/unit_test.hpp>

#include <torch_geopooling.h>


using namespace torch_geopooling;


BOOST_AUTO_TEST_SUITE(TestQuadtreeSet)


BOOST_AUTO_TEST_CASE(quadtree_set_contains)
{
    BOOST_TEST_MESSAGE("--- Empty quadtree set contains");

    QuadtreeSet set({0, 0, 10, 10});

    BOOST_CHECK(set.contains(std::pair(0, 0)));
    BOOST_CHECK(set.contains(std::pair(0, 10)));
    BOOST_CHECK(set.contains(std::pair(10, 0)));
    BOOST_CHECK(set.contains(std::pair(10, 10)));

    BOOST_CHECK(!set.contains(std::pair(-1, -1)));
    BOOST_CHECK(!set.contains(std::pair(11, 11)));
}


BOOST_AUTO_TEST_CASE(quadtree_set_find_empty)
{
    BOOST_TEST_MESSAGE("--- Empty quadtree find");

    QuadtreeSet set({0, 0, 10, 10});

    auto node = set.find(std::pair(2, 2));
    BOOST_CHECK_EQUAL(node.exterior(), quadrect(0, 0, 10, 10));
}


BOOST_AUTO_TEST_CASE(quadtree_insert_and_find)
{
    BOOST_TEST_MESSAGE("--- Find tree");

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


BOOST_AUTO_TEST_SUITE_END()
