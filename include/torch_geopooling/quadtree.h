#pragma once

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <optional>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <torch_geopooling/exception.h>
#include <torch_geopooling/functional.h>
#include <torch_geopooling/quadrect.h>
#include <torch_geopooling/quadtree_options.h>
#include <torch_geopooling/tile.h>


namespace torch_geopooling {


template <typename Coordinate, typename T, class Container> class quadtree_iterator;


template <
    typename Coordinate,
    typename T = long,
    class Container = std::unordered_map<std::pair<Coordinate, Coordinate>, T>>
class quadtree {
public:
    using key_type = std::pair<Coordinate, Coordinate>;

    using mapped_type = T;

    using value_type = std::pair<key_type, mapped_type>;

    using container_type = Container;

    using node_type = quad<quadtree<Coordinate, T, Container>>;

    using node_iterator = quadtree_iterator<Coordinate, T, Container>;

    using exterior_type = quadrect<Coordinate>;

    quadtree(
        const exterior_type& exterior,
        const Tile& tile = Tile::root,
        std::optional<quadtree_options> options = std::nullopt
    )
    : m_exterior(exterior),
      m_tile(tile),
      m_options(options.value_or(quadtree_options())),
      m_nodes(nullptr)
    {}

    quadtree(
        const std::initializer_list<Coordinate> xywh,
        std::optional<quadtree_options> options = std::nullopt
    )
    : quadtree(exterior_type(xywh), Tile::root, options)
    {}

    Tile
    tile() const
    {
        return m_tile;
    }

    inline bool
    is_terminal() const
    {
        return m_nodes == nullptr;
    }

    const exterior_type&
    exterior() const
    {
        return m_exterior;
    }

    bool
    contains(const key_type& point) const
    {
        return m_exterior.contains(point);
    }

    node_iterator
    begin() const
    {
        return node_iterator(*this);
    }

    node_iterator
    end() const
    {
        return node_iterator();
    }

    node_iterator
    begin()
    {
        return node_iterator(*this);
    }

    node_iterator
    end()
    {
        return node_iterator();
    }

    std::size_t
    depth() const
    {
        return m_tile.z();
    }

    std::size_t
    total_depth() const
    {
        std::size_t depth = 0;
        for (auto const& node : std::as_const(*this)) {
            depth = std::max(depth, node.m_tile.z());
        }
        return depth;
    }

    void
    insert(const value_type& value)
    {
        auto point = value.first;
        if (m_options.has_precision()) {
            point = round(point, m_options.precision());
        }

        assert_contains(point);

        auto& node = find(point);
        node.m_data.insert(std::make_pair(point, value.second));
        node.subdivide();
    }

    void
    insert(const key_type& point, const mapped_type& value)
    {
        insert(std::make_pair(point, value));
    }

    quadtree<Coordinate, T, Container>&
    find(const key_type& point, std::optional<std::size_t> max_depth = std::nullopt)
    {
        assert_contains(point);

        auto maximum_depth = max_depth.value_or(m_options.max_depth());
        if (((m_tile.z() >= maximum_depth) && maximum_depth >= 0) || m_nodes == nullptr) {
            return *this;
        }

        auto centroid = m_exterior.centroid();
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
        while (!node->is_terminal() && t.z() > node->m_tile.z()) {
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
    find(const key_type& point, std::optional<std::size_t> max_depth = std::nullopt) const
    {
        return find(point, max_depth);
    }

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

    std::shared_ptr<node_type> m_nodes;
    container_type m_data;
    exterior_type m_exterior;

    quadtree_options m_options;
    Tile m_tile;

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
    make_subtree(quad<exterior_type>& exteriors, std::size_t x, std::size_t y) const
    {
        return quadtree(exteriors.at(x, y), m_tile.child(x, y), m_options);
    }

    void
    subdivide()
    {
        if (m_nodes != nullptr || m_data.size() <= m_options.capacity()
            || m_tile.z() >= m_options.max_depth()) {
            return;
        }

        // Symmetrically split the quad into the equal-area elements and move
        // the data from the parent node to sub-nodes. At the end, clear the
        // data from the parent.
        quad<exterior_type> exteriors = m_exterior.symmetric_split();

        node_type nodes(
            make_subtree(exteriors, 0, 0), make_subtree(exteriors, 0, 1),
            make_subtree(exteriors, 1, 0), make_subtree(exteriors, 1, 1)
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


template <typename Coordinate, typename T, class Container> class quadtree_iterator {
public:
    using iterator_category = std::forward_iterator_tag;

    using iterator = quadtree_iterator<Coordinate, T, Container>;

    using value_type = quadtree<Coordinate, T, Container>;

    using reference = value_type&;

    using pointer = value_type*;

    explicit quadtree_iterator(const value_type& tree)
    : m_queue()
    {
        m_queue.push(tree);
    }

    quadtree_iterator()
    : m_queue()
    {}

    ~quadtree_iterator() {}

    iterator&
    operator++()
    {
        return next();
    }

    reference
    operator*()
    {
        return m_queue.front();
    }

    pointer
    operator->()
    {
        return m_queue.front();
    }

    bool
    operator!=(const iterator& rhs)
    {
        return !(m_queue.empty() && rhs.m_queue.empty());
    }

private:
    std::queue<value_type> m_queue;

    iterator&
    next()
    {
        auto tree = m_queue.front();
        if (!tree.is_terminal()) {
            for (auto const& node : *tree.m_nodes) {
                m_queue.push(node);
            }
        }

        m_queue.pop();
        return *this;
    }
};


} // namespace torch_geopooling
