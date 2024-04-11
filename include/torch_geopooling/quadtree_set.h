#pragma once

#include <queue>
#include <initializer_list>
#include <unordered_map>
#include <unordered_set>

#include <torch_geopooling/functional.h>
#include <torch_geopooling/tile.h>
#include <torch_geopooling/quadrect.h>


namespace torch_geopooling {


template<typename Coordinate>
class QuadtreeNode {
public:
    using key_type = std::pair<Coordinate, Coordinate>;

    using exterior_type = quadrect<Coordinate>;

    using container_type = std::unordered_set<key_type>;

    using point_iterator = typename container_type::iterator;

    QuadtreeNode(Tile tile, exterior_type exterior, std::size_t max_depth)
    : m_tile(tile), m_exterior(exterior), m_max_depth(max_depth), m_points()
    { }

    ~QuadtreeNode() = default;

    const Tile&
    tile() const
    { return m_tile; }

    std::size_t
    depth() const
    { return m_tile.z(); }

    exterior_type
    exterior() const
    { return m_exterior; }

    std::size_t
    x() const
    { return cell_size() * m_tile.x(); }

    std::size_t
    y() const
    { return cell_size() * m_tile.y(); }

    void
    insert(const key_type& key)
    { m_points.insert(key); }

    void
    clear()
    { m_points.clear(); }

    std::size_t
    size() const
    { return m_points.size(); }

    point_iterator
    begin()
    { return m_points.begin(); }

    point_iterator
    end()
    { return m_points.end(); }

private:
    Tile m_tile;
    std::size_t m_max_depth;

    exterior_type m_exterior;
    container_type m_points;

    std::size_t
    cell_size() const
    { return (1 << (m_max_depth - m_tile.z())); }
};


template<typename Coordinate>
class QuadtreeSet {
public:
    using key_type = std::pair<Coordinate, Coordinate>;

    using node_type = QuadtreeNode<Coordinate>;

    using exterior_type = quadrect<Coordinate>;

    QuadtreeSet(
        const exterior_type& exterior,
        std::size_t max_depth = 17,
        std::size_t capacity = 1,
        std::size_t precision = 7
    )
    : m_max_depth(max_depth),
      m_capacity(capacity),
      m_precision(precision),
      m_nodes(),
      m_total_depth(0)
    {
        Tile tile = Tile::root;
        m_nodes.insert(std::make_pair(tile, node_type(tile, exterior, max_depth)));
    }

    template<typename InputIt>
    QuadtreeSet(
        InputIt first,
        InputIt last,
        const exterior_type& exterior,
        std::size_t max_depth = 17,
        std::size_t capacity = 1,
        std::size_t precision = 7
    )
    : QuadtreeSet(exterior, max_depth, capacity, precision)
    {
        while (first != last) {
            auto node_tile = *first;
            auto node_exterior = make_exterior(exterior, node_tile);

            m_nodes.insert(std::make_pair(
                node_tile, node_type(node_tile, node_exterior, max_depth)
            ));

            first++;
            m_total_depth = std::max(node_tile.z(), m_total_depth);
        }

        // Verify integrity of the resulting quadtree set by assuring that every children
        // has a parent up until the root tile (0, 0, 0).
        for (auto node : m_nodes) {
            auto parent_tile = node.first.parent();

            while (parent_tile != Tile::root) {
                if (auto n = m_nodes.find(parent_tile); n == m_nodes.end()) {
                    throw value_error(
                        "QuadtreeSet: tile {} does not have a parent {}",
                        node.first, parent_tile
                    );
                }
                parent_tile = parent_tile.parent();
            }
        }
    }

    QuadtreeSet(const std::initializer_list<Coordinate> xywh)
    : QuadtreeSet(quadrect(xywh))
    { }

    ~QuadtreeSet() = default;

    bool
    contains(const key_type& point) const
    {
        const auto node = m_nodes.at(Tile::root);
        return node.exterior().contains(point);
    }

    void
    insert(const key_type& key)
    {
        auto point = std::round(key, m_precision);
        assert_contains(point);

        auto& node = find(point);
        node.insert(point);
        subdivide(node);
    }

    node_type&
    find(const key_type& point, std::size_t max_depth = -1)
    {
        assert_contains(point);

        Tile tile = Tile::root;
        node_type* node = &m_nodes.at(tile);

        max_depth = std::min(max_depth, m_max_depth);

        while (tile.z() < max_depth) {
            auto centroid = node->exterior().centroid();

            std::size_t x = 0, y = 0;
            if (point.first > centroid.first) {
                x += 1;
            }
            if (point.second > centroid.second) {
                y += 1;
            }

            auto child_tile = tile.child(x, y);

            if (auto n = m_nodes.find(child_tile); n != m_nodes.end()) {
                tile = n->first;
                node = &n->second;
            } else {
                break;
            }
        }

        return *node;
    }

    node_type&
    find(const Tile& tile)
    {
        Tile node_tile = tile;
        auto node = m_nodes.find(node_tile);

        while (node == m_nodes.end() && node_tile != Tile::root) {
            node_tile = node_tile.parent();
            node = m_nodes.find(node_tile);
        }

        return m_nodes.at(node_tile);
    }

    std::size_t
    total_depth() const
    { return m_total_depth; }

    std::size_t
    size() const
    {
        std::size_t num_elements = 0;
        for (auto& value : m_nodes) {
            num_elements += value.second.size();
        }
        return num_elements;
    }

private:
    std::unordered_map<Tile, node_type> m_nodes;

    std::size_t m_max_depth;
    std::size_t m_capacity;
    std::size_t m_precision;
    std::size_t m_total_depth;

    void
    assert_contains(const key_type& point) const
    {
        if (!contains(point)) {
            throw value_error(
                "quadtree: point ({}, {}) is outside of exterior geometry", point.first, point.second
            );
        }
    }

    exterior_type
    make_exterior(const exterior_type exterior, const Tile& tile) const
    {
        auto width = exterior.width() / tile.z();
        auto height = exterior.height() / tile.z();
        return exterior_type(tile.x() * width, tile.y() * height, width, height);
    }

    bool
    has_children(const node_type& node) const
    {
        return (
            (m_nodes.find(node.tile().child(0, 0)) != m_nodes.end()) &&
            (m_nodes.find(node.tile().child(0, 1)) != m_nodes.end()) &&
            (m_nodes.find(node.tile().child(1, 0)) != m_nodes.end()) &&
            (m_nodes.find(node.tile().child(1, 1)) != m_nodes.end())
        );
    }

    void
    subdivide(node_type& node)
    {
        if (node.size() <= m_capacity || node.tile().z() >= m_max_depth || has_children(node)) {
            return;
        }

        auto quadrects = node.exterior().symmetric_split();
        for (std::size_t x : {0, 1}) {
            for (std::size_t y : {0, 1}) {
                auto tile = node.tile().child(x, y);
                auto n = node_type(tile, quadrects.at(x, y), m_max_depth);

                m_total_depth = std::max(tile.z(), m_total_depth);
                m_nodes.insert(std::make_pair(tile, n));
            }
        }

        for (auto point : node) {
            insert(point);
        }

        node.clear();
    }
};


} // namespace torch_geopooling
