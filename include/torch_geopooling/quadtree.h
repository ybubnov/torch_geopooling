#pragma once

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <queue>
#include <unordered_map>
#include <utility>

#include <torch_geopooling/exception.h>
#include <torch_geopooling/functional.h>
#include <torch_geopooling/quadrect.h>
#include <torch_geopooling/tile.h>


namespace torch_geopooling {


template
<typename Coordinate, typename T, class Container>
class quadtree_iterator;


template
<
    typename Coordinate,
    typename T = long,
    class Container = std::unordered_map<std::pair<Coordinate, Coordinate>, T>
>
class quadtree
{
public:
    using key_type = std::pair<Coordinate, Coordinate>;

    using mapped_type = T;

    using value_type = std::pair<key_type, mapped_type>;

    using container_type = Container;

    using node_type = quad<quadtree<Coordinate, T, Container>>;

    using node_iterator = quadtree_iterator<Coordinate, T, Container>;

    quadtree(const std::initializer_list<Coordinate> xywh)
    : quadtree(quadrect(xywh))
    { }

    quadtree(
        const quadrect<Coordinate>& quad,
        std::size_t max_depth = 17,
        std::size_t capacity = 1,
        std::size_t z = 0,
        std::size_t x = 0,
        std::size_t y = 0,
        std::size_t precision = 7
    )
    : m_quadrect(quad),
      m_max_depth(max_depth),
      m_capacity(capacity),
      m_z(z),
      m_x(x),
      m_y(y),
      m_nodes(nullptr),
      m_precision(precision)
    { }

    Tile
    tile() const
    { return Tile(m_z, m_x, m_y); }

    inline bool
    is_terminal() const
    { return m_nodes == nullptr; }

    const quadrect<Coordinate>&
    exterior() const
    { return m_quadrect; }

    bool
    contains(const key_type& point) const
    { return m_quadrect.contains(point); }

    node_iterator
    begin() const
    { return node_iterator(*this); }

    node_iterator
    end() const
    { return node_iterator(); }

    node_iterator
    begin()
    { return node_iterator(*this); }

    node_iterator
    end()
    { return node_iterator(); }

    std::size_t
    depth() const
    { return m_z; }

    std::size_t
    total_depth() const
    {
        std::size_t depth = 0;
        for (auto const& node : std::as_const(*this)) {
            depth = std::max(depth, node.m_z);
        }
        return depth;
    }

    void
    insert(const value_type& value)
    {
        auto point = round(value.first, m_precision);
        assert_contains(point);

        auto& node = find(point);
        node.m_data.insert(std::make_pair(point, value.second));
        node.subdivide();
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

    quadtree<Coordinate, T, Container>&
    find(const Tile& t)
    {
        std::size_t mid_width = 1 << t.z();
        mid_width >>= 1;

        std::size_t xmid = mid_width, ymid = mid_width;

        auto node = this;
        while (!node->is_terminal() && t.z() > node->m_z) {
            mid_width >>= 1;

            std::size_t x = 0, y = 0;
            if (t.x() >= xmid) {
                x += 1;
                xmid += mid_width;
            } else {
                xmid -= mid_width;
            }
            if (t.y() >= ymid) {
                y += 1;
                ymid += mid_width;
            } else {
                ymid -= mid_width;
            }

            node = &(node->m_nodes->at(x, y));
        }

        return *node;
    }

    const quadtree<Coordinate, T>&
    find(const key_type& point, std::size_t max_depth = -1) const
    { return find(point, max_depth); }

    std::size_t
    size() const
    {
        std::size_t num_elements = 0;
        for (auto const& node : std::as_const(*this)) {
            num_elements += node.m_data.size();
        }
        return num_elements;
    }

private:
    friend class quadtree_iterator<Coordinate, T, Container>;

    quadrect<Coordinate> m_quadrect;

    std::size_t m_z, m_x, m_y;
    std::size_t m_max_depth;
    std::size_t m_capacity;
    std::size_t m_precision;

    std::shared_ptr<node_type> m_nodes;
    container_type m_data;

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
        return quadtree(quadrects.at(x, y), m_max_depth, m_capacity, m_z+1, m_x*2+x, m_y*2+y);
    }

    void
    subdivide()
    {
        if (m_nodes != nullptr || m_data.size() <= m_capacity || m_z >= m_max_depth) {
            return;
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
            if (node.tile() != tile()) {
                node.insert(data);
            }
        }

        m_data.clear();
    }
};


template
<typename Coordinate, typename T, class Container>
class quadtree_iterator
{
public:
    using iterator_category = std::forward_iterator_tag;

    using value_type = quadtree<Coordinate, T, Container>;

    using reference = value_type&;

    using pointer = value_type*;

    using iterator = quadtree_iterator<Coordinate, T, Container>;

    explicit
    quadtree_iterator(const value_type& tree)
    : m_queue()
    { m_queue.push(tree); }

    quadtree_iterator()
    : m_queue()
    { }

    ~quadtree_iterator()
    { }

    iterator&
    operator++()
    { return next(); }

    reference
    operator*()
    { return m_queue.front(); }

    pointer
    operator->()
    { return m_queue.front(); }

    bool
    operator!=(const iterator& rhs)
    { return !(m_queue.empty() && rhs.m_queue.empty()); }

private:
    std::queue<value_type> m_queue;

    iterator&
    next()
    {
        auto tree = m_queue.front();
        if (!tree.is_terminal()) {
            for (auto const& node: *tree.m_nodes) {
                m_queue.push(node);
            }
        }

        m_queue.pop();
        return *this;
    }
};


} // namespace torch_geopooling
