#pragma once

#include <algorithm>
#include <iterator>
#include <initializer_list>
#include <stdexcept>
#include <queue>

#include <geopool/exception.h>


namespace geopool {


template <typename T>
class quad {
public:
    using value_type = T;

    using container_type = std::array<value_type, 4>;

    using size_type = typename container_type::size_type;

    using const_iterator = typename container_type::const_iterator;

    quad(const value_type& fill)
    : m_elements({fill, fill, fill, fill})
    { }

    quad(value_type t0, value_type t1, value_type t2, value_type t3)
    : m_elements({t0, t1, t2, t3})
    { }

    const value_type&
    at(size_type x, size_type y)
    {
        if (x > 1) {
            throw out_of_range("quad: x ({}) >1", x);
        }
        if (y > 1) {
            throw out_of_range("quad: x ({}) >1", y);
        }
        return m_elements.at(x * 2 + y);
    }

    const_iterator
    cbegin()
    { return m_elements.cbegin(); }

    const_iterator
    cend()
    { return m_elements.cend(); }

private:
    container_type m_elements;

};


template <typename T>
class quadrect {
public:
    using value_type = T;
    using point_type = std::pair<T, T>;

    quadrect(const std::initializer_list<value_type>& list)
    {
        auto list_size = list.size();
        if (list_size != 4) {
            throw value_error("quadrect: size of initializer list ({}) should be {}", list_size);
        }

        std::array<value_type, 4> xywh;
        std::copy(list.begin(), list.end(), xywh.begin());

        auto [x, y, w, h] = xywh;

        if (w <= T(0)) {
            throw value_error("quadrect: width ({}) should be a positive number", w);
        }
        if (h <= T(0)) {
            throw value_error("quadrect: height ({}) should be a positive number", h);
        }

        m_xmin = x;
        m_ymin = y;
        m_xmax = x + w;
        m_ymax = y + h;
    }

    quadrect(const std::tuple<value_type, value_type, value_type, value_type>& xywh)
    : quadrect(std::get<0>(xywh), std::get<1>(xywh), std::get<2>(xywh), std::get<3>(xywh))
    { }

    quadrect(T xmin, T ymin, T width, T height)
    : quadrect({xmin, ymin, width, height})
    { }

    inline value_type
    width() const
    { return m_xmax - m_xmin; }

    inline value_type
    height() const
    { return m_ymax - m_ymin; }

    point_type
    centroid() const
    {
        auto w = width() / 2;
        auto h = height() / 2;
        return point_type(m_xmin + w, m_ymin + h);
    }

    bool
    contains(const point_type& point) const
    {
        return (
            (point.first >= m_xmin) &&
            (point.first <= m_xmax) &&
            (point.second >= m_ymin) &&
            (point.second <= m_ymax)
        );
    }

    quad<quadrect>
    symmetric_split() const
    {
        auto w = width() / 2;
        auto h = height() / 2;

        return quad<quadrect>(
            {m_xmin, m_ymin, w, h},
            {m_xmin, m_ymin + h, w, h},
            {m_xmin + w, m_ymin, w, h},
            {m_xmin + w, m_ymin + h, w, h}
        );
    }

private:

    value_type m_xmin, m_ymin, m_xmax, m_ymax;
};


template <typename Coordinate, typename V>
class quadtree;


template <typename Coordinate, typename V>
class quadtree_iterator
: public std::iterator<std::forward_iterator_tag, const quadtree<Coordinate, V> >
{
public:
    using value_type = quadtree<Coordinate, V>;

    using reference = value_type&;

    using pointer = value_type*;

    using iterator = quadtree_iterator<Coordinate, V>;

    explicit
    quadtree_iterator(const value_type& tree)
    : m_queue(1)
    { m_queue.push_back(tree); }

    quadtree_iterator()
    : m_queue(0)
    { }

    ~quadtree_iterator()
    { }

    iterator
    operator++(int)
    { return next(); }

    iterator&
    operator++()
    { return next(); }

    reference
    operator*() const
    { return m_queue.front(); }

    pointer
    operator->() const
    { return m_queue.front(); }

    bool
    operator!=(const iterator& rhs)
    { return !(m_queue.empty() && rhs.m_queue.empty()); }

private:
    friend class quadtree<Coordinate, V>;

    std::queue<value_type> m_queue;

    iterator&
    next()
    {
        auto node = m_queue.front();
        if (!node.is_leaf()) {
            for (auto const& child : node.quad) {
                m_queue.push_back(child);
            }
        }

        m_queue.pop();
        return *this;
    }
};


template <typename Coordinate, typename T = long>
class quadtree
{
public:
    using value_type = T;

    using coordinate = Coordinate;

    using point_type = std::pair<coordinate, coordinate>;

    using const_node_iterator = quadtree_iterator<coordinate, const value_type>;

    quadtree(const std::initializer_list<coordinate> xywh)
    : quadtree(quadrect(xywh))
    { }

    quadtree(
        const quadrect<coordinate>& quad,
        std::size_t max_depth = 17,
        std::size_t capacity = 1,
        std::size_t z = 0,
        std::size_t x = 0,
        std::size_t y = 0
    )
    : m_quadrect(quad),
      m_max_depth(max_depth),
      m_capacity(capacity),
      m_z(z),
      m_x(x),
      m_y(y),
      m_children(nullptr),
      m_data(0)
    { }

    inline bool
    is_leaf() const
    { return m_children == nullptr; }

    const quadrect<coordinate>&
    exterior() const
    { return m_quadrect; }

    bool
    contains(const point_type& point) const
    { return m_quadrect.contains(point); }

    const_node_iterator
    cbegin() const
    { return quadtree_iterator<Coordinate, T>(*this); }

    const_node_iterator
    cend() const
    { return quadtree_iterator<Coordinate, T>(*this); }

    void
    insert(const point_type& point, const value_type& value)
    {
        assert_contains(point);
    }

    const quadtree<coordinate, value_type>&
    find(const point_type& point, std::size_t max_depth = -1) const
    {
        assert_contains(point);

        if (((m_z >= max_depth) && max_depth > 0) || m_children == nullptr) {
            return *this;
        }

        auto centroid = m_quadrect.centroid();
        std::size_t x, y = 0;
        if (point.first > centroid.first) {
            x += 1;
        }
        if (point.second > centroid.second) {
            y += 1;
        }

        return m_children->at(x, y).find(point);
    }

private:
    quadrect<Coordinate> m_quadrect;

    std::size_t m_z, m_x, m_y;
    std::size_t m_max_depth;
    std::size_t m_capacity;

    std::shared_ptr<quad<quadtree>> m_children;
    std::vector<T> m_data;

    void
    assert_contains(const point_type& point) const
    {
        if (!contains(point)) {
            throw value_error(
                "point ({}, {}) is outside of quad geometry", point.first, point.second
            );
        }
    }
};


};
