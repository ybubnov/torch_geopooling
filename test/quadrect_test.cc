#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE quadpool

#include <boost/test/included/unit_test.hpp>

#include <torch_geopooling.h>


using namespace torch_geopooling;


BOOST_AUTO_TEST_SUITE(TestQuadRect)


BOOST_AUTO_TEST_CASE(quad_all)
{
    auto q1 = quad(5);

    BOOST_CHECK_EQUAL(q1.at(0, 0), 5);
    BOOST_CHECK_EQUAL(q1.at(0, 1), 5);
    BOOST_CHECK_EQUAL(q1.at(1, 0), 5);
    BOOST_CHECK_EQUAL(q1.at(1, 1), 5);

    auto q2 = quad<float>(1.0, 2.0, 3.0, 4.0);

    BOOST_CHECK_EQUAL(q2.at(0, 0), 1.0);
    BOOST_CHECK_EQUAL(q2.at(0, 1), 2.0);
    BOOST_CHECK_EQUAL(q2.at(1, 0), 3.0);
    BOOST_CHECK_EQUAL(q2.at(1, 1), 4.0);

    BOOST_CHECK_THROW(q2.at(2, 3), out_of_range);

    auto v2 = std::vector({1.0, 2.0, 3.0, 4.0});
    BOOST_CHECK(equal(q2.begin(), q2.end(), v2.begin()));
}


BOOST_AUTO_TEST_CASE(quadrect_all)
{
    auto rect1 = quadrect(0.0, 0.0, 10.0, 20.0);

    BOOST_CHECK_EQUAL(rect1.width(), 10.0);
    BOOST_CHECK_EQUAL(rect1.height(), 20.0);

    BOOST_CHECK_EQUAL(rect1.centroid(), std::make_pair(5.0, 10.0));

    auto split = rect1.symmetric_split();

    BOOST_CHECK_EQUAL(split.at(0, 0), quadrect(0.0, 0.0, 5.0, 10.0));
    BOOST_CHECK_EQUAL(split.at(0, 1), quadrect(0.0, 10.0, 5.0, 10.0));
    BOOST_CHECK_EQUAL(split.at(1, 0), quadrect(5.0, 0.0, 5.0, 10.0));
    BOOST_CHECK_EQUAL(split.at(1, 1), quadrect(5.0, 10.0, 5.0, 10.0));

    auto rect2 = rect1.slice(Tile::root);
    BOOST_CHECK_EQUAL(rect1, rect2);

    BOOST_CHECK_THROW(quadrect({10.0, 0.0, -10.0, 10.0}), value_error);
    BOOST_CHECK_THROW(quadrect({10.0, 0.0, 10.0, -10.0}), value_error);
}


BOOST_AUTO_TEST_CASE(quadrect_slice)
{
    auto rect0 = quadrect(0, 0, 2048, 2048);

    auto rect1 = rect0.slice(Tile(1, 0, 0));
    BOOST_CHECK_EQUAL(rect1, quadrect(0, 0, 1024, 1024));

    auto rect2 = rect0.slice(Tile(2, 0, 0));
    BOOST_CHECK_EQUAL(rect2, quadrect(0, 0, 512, 512));

    auto rect11 = rect0.slice(Tile(11, 0, 0));
    BOOST_CHECK_EQUAL(rect11, quadrect(0, 0, 1, 1));

    auto rectx = rect0.slice(Tile(11, (1 << 11) - 1, (1 << 11) - 1));
    BOOST_CHECK_EQUAL(rectx, quadrect(2047, 2047, 1, 1));
}


BOOST_AUTO_TEST_SUITE_END()
