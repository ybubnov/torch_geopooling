#pragma once

#include <algorithm>
#include <iterator>
#include <initializer_list>
#include <stdexcept>
#include <queue>
#include <iostream>

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

    quad(value_type t00, value_type t01, value_type t10, value_type t11)
    : m_elements({t00, t01, t10, t11})
    { }

    value_type&
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

    const value_type&
    at(size_type x, size_type y) const
    { return at(x, y); }

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

    bool
    operator==(const quadrect& rhs) const
    {
        return (
            m_xmin == rhs.m_xmin && m_xmax == rhs.m_xmax
            && m_ymin == rhs.m_ymin && m_ymax == rhs.m_ymax
        );
    }

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

    friend std::ostream&
    operator<< (std::ostream &os, const quadrect& rect)
    {
        os << "quadrect(" << rect.m_xmin << ", " << rect.m_ymin << ", ";
        os << rect.width() << ", " << rect.height() << ")";
        return os;
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
        auto tree = m_queue.front();
        if (!tree.is_leaf()) {
            for (auto const& node: tree.m_nodes) {
                m_queue.push_back(node);
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
    using key_type = std::pair<Coordinate, Coordinate>;

    using mapped_type = T;

    using value_type = std::pair<key_type, mapped_type>;

    using node_type = quad<quadtree<Coordinate, T>>;

    using const_node_iterator = quadtree_iterator<Coordinate, const T>;

    quadtree(const std::initializer_list<Coordinate> xywh)
    : quadtree(quadrect(xywh))
    { }

    quadtree(
        const quadrect<Coordinate>& quad,
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
      m_nodes(nullptr),
      m_data(0)
    { }

    inline bool
    is_leaf() const
    { return m_nodes == nullptr; }

    const quadrect<Coordinate>&
    exterior() const
    { return m_quadrect; }

    bool
    contains(const key_type& point) const
    { return m_quadrect.contains(point); }

    const_node_iterator
    cbegin() const
    { return quadtree_iterator<Coordinate, T>(*this); }

    const_node_iterator
    cend() const
    { return quadtree_iterator<Coordinate, T>(*this); }

    std::size_t
    depth() const
    { return m_z; }

    void
    insert(const value_type& value)
    {
        auto point = value.first;
        assert_contains(point);

        auto& node = find(point);
        node.m_data.push_back(value);

        if (
            node.m_nodes == nullptr
            && (node.m_data.size() > m_capacity)
            && (node.m_z <= m_max_depth)
        ) {
            node.subdivide();
        }
    }

    void
    insert(const key_type& point, const mapped_type& value)
    { insert(std::make_pair(point, value)); }

    quadtree<Coordinate, T>&
    find(const key_type& point, std::size_t max_depth = -1)
    {
        assert_contains(point);

        if (((m_z >= max_depth) && max_depth >= 0) || m_nodes == nullptr) {
            return *this;
        }

        auto centroid = m_quadrect.centroid();
        std::size_t x = 0, y = 0;
        if (point.first > centroid.first) {
            x += 1;
        }
        if (point.second > centroid.second) {
            y += 1;
        }

        auto& tree = m_nodes->at(x, y);
        return tree.find(point, max_depth);
    }

    const quadtree<Coordinate, T>&
    find(const key_type& point, std::size_t max_depth = -1) const
    { return find(point, max_depth); }

private:
    quadrect<Coordinate> m_quadrect;

    std::size_t m_z, m_x, m_y;
    std::size_t m_max_depth;
    std::size_t m_capacity;

    std::shared_ptr<node_type> m_nodes;
    std::vector<value_type> m_data;

    void
    assert_contains(const key_type& point) const
    {
        if (!contains(point)) {
            throw value_error(
                "quadtree: point ({}, {}) is outside of quad geometry", point.first, point.second
            );
        }
    }

    quadtree<Coordinate, T>
    make_subtree(quad<quadrect<Coordinate>>& quadrects, std::size_t x, std::size_t y) const
    {
        return quadtree(quadrects.at(x, y), m_max_depth, m_capacity, m_z+1, m_x * 2 + x, m_y * 2 + y);
    }

    void
    subdivide()
    {
        if (m_nodes != nullptr) {
            throw value_error("quadtree: tree is already split");
        }

        // Symmetrically split the quad into the equal-area elements and move
        // the data from the parent node to sub-nodes. At the end, clear the
        // data from the parent.
        quad<quadrect<Coordinate>> quadrects = m_quadrect.symmetric_split();

        node_type nodes(
            make_subtree(quadrects, 0, 0),
            make_subtree(quadrects, 0, 1),
            make_subtree(quadrects, 1, 0),
            make_subtree(quadrects, 1, 1)
        );

        m_nodes = std::make_shared<node_type>(std::move(nodes));

        for (auto data : m_data) {
            auto& node = find(data.first, -1);
            node.insert(data);
        }

        m_data.clear();
    }
};


};
