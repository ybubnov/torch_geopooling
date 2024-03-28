#pragma once

#include <iterator>
#include <stdexcept>

#include <geopool/exception.hpp>


namespace geopool {


template <typename T>
class quad {
private:
    container_type m_elements;

public:
    using value_type = T;

    using container_type = std::array<T>;

    using size_type = typename container_type::size_type;

    using const_iterator = typename container_type::const_iterator;

    quad(const value_type& fill)
    : m_elements({fill, fill, fill, fill})
    { }

    quad(value_type t0, value_type, t1, value_type, t2, value_type, t3)
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
}


template <typename T>
class quadrect {
private:
    using value_type = T;
    using point_type = std::pair<T, T>;

    T m_xmin, m_ymin, m_xmax, m_ymax;

public:
    quadrect(T xmin, T ymin, T width, T height)
    : m_xmin(xmin), m_ymin(ymin)
    {
        if (width <= T(0)) {
            throw value_error("quadrect: width ({}) should be a positive number", width);
        }
        if (height <= T(0)) {
            throw value_error("quadrect: height ({}) should be a positive number", height);
        }

        m_xmax = m_xmin + width;
        m_ymax = m_ymin + height;
    }

    quad(const std::tuple<T, T, T, T>& xywh)
    : quad(std::get<0>(xywh), std::get<1>(xywh), std::get<2>(xywh), std::get<3>(xywh))
    { }


    inline T
    width() const
    {
        return m_xmax - m_xmin;
    }

    inline T
    height() const
    {
        return m_ymax - m_ymin;
    }

    point_type
    centroid() const
    {
        auto width = width() / 2;
        auto height = height() / 2;
        return point_type(m_xmin + width, m_ymin + height);
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
        auto width = width() / 2;
        auto height = height() / 2;

        return quad<quadrect>(
            {m_xmin, m_ymin, width, height},
            {m_xmin, m_ymin + height, width, height},
            {m_xmin + width, m_ymin, width, height},
            {m_xmin + width, m_ymin + height, width, height}
        );
    }
};


template <typename Coordinate, typename V>
class quadtree;


template <typename Coordinate, typename V>
class quadtree_iterator
: public std::iterator<std::forward_iterator_tag, const quadtree<Coordinate, V>& >
{
public:
    using iterator = quadtree_iterator<Coordinate, V>;

    explicit
    quadtree_iterator(const quadtree& tree)
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
    { return !(m_queue.empty() && rhs.m_queue.empty()) }

private:
    friend class quadtree;

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


template <typename Coordinate, typename T>
class quadtree
{
public:
    using value_type = T;

    using coordinate = Coordinate;

    using point_type = std::pair<coordinate, coordinate>;

    quadtree(const quadrect& quad, std::size_t max_depth = 17, std::size_t capacity = 1)
    : quadtree(quad, max_depth, capacity)
    { }

    quadtree(
        const quadrect& quad,
        std::size_t max_depth = 17,
        std::size_t capacity = 1,
        std::size_t z = 0,
        std::size_t x = 0,
        std::size_t y = 0
    )
    : m_quad(quad),
      m_max_depth(max_depth),
      m_capacity(capacity),
      m_z(z),
      m_x(x),
      m_y(y),
      m_children(nullptr),
      m_data(0),
    { }

    bool
    is_leaf() const
    { return m_children == nullptr; }

    bool
    contains(point_type point) const
    { return m_quad.contains(point); }

    const_node_iterator
    cbegin() const
    { return quadtree_iterator<Coordinate, T>(*this); }

    const_node_iterator
    cend() const
    { return quadtree_iterator<Coordinate, T>(*this); }

private:
    quadrect<Coordinate> m_quad;

    std::size_t m_z, m_x, m_y;
    std::size_t m_max_depth;
    std::size_t m_capacity;

    std::unique_ptr<quad<quadtree>> m_children;
    std::vector<T> m_data;
};


};
