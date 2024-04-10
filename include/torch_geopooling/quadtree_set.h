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
    : m_tile(tile), m_exterior(exterior), m_max_depth(max_depth)
    { }

    Tile
    tile() const
    { return m_tile; }

    std::size_t
    x() const
    { return cell_size() * m_tile.x(); }

    exterior_type
    exterior() const
    { return m_exterior; }

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

    QuadtreeSet(
        const quadrect<Coordinate>& exterior,
        std::size_t max_depth = 17,
        std::size_t capacity = 1,
        std::size_t precision = 7
    )
    : m_max_depth(max_depth), m_capacity(capacity), m_precision(precision), m_nodes()
    {
        Tile tile(0, 0, 0);
        m_nodes.insert(std::make_pair(tile, node_type(tile, exterior, max_depth)));
    }

    QuadtreeSet(const std::initializer_list<Coordinate> xywh)
    : QuadtreeSet(quadrect(xywh))
    { }

    bool
    contains(const key_type& point) const
    {
        const auto node = m_nodes.at(Tile(0, 0, 0));
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

        Tile tile(0, 0, 0);
        node_type& node = m_nodes.at(tile);

        max_depth = std::min(max_depth, m_max_depth);

        while (tile.z() <= max_depth) {
            auto centroid = node.exterior().centroid();

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
                node = n->second;
            } else {
                break;
            }

        }

        return node;
    }

private:
    std::unordered_map<Tile, node_type> m_nodes;

    std::size_t m_max_depth;
    std::size_t m_capacity;
    std::size_t m_precision;

    void
    assert_contains(const key_type& point) const
    {
        if (!contains(point)) {
            throw value_error(
                "quadtree: point ({}, {}) is outside of exterior geometry", point.first, point.second
            );
        }
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
                auto node = node_type(tile, quadrects.at(x, y), m_max_depth);
                m_nodes.insert(std::make_pair(tile, node));
            }
        }

        for (auto point: node) {
            std::cout << "POINT(" << point.first << ", " << point.second << ")" << std::endl;
            auto& n = find(point);
            std::cout << "  tile" << n.tile() << std::endl;
        /*
            if (n.tile() != node.tile()) {
                n.insert(point);
            }
        */
        }

        node.clear();
    }
};


} // namespace torch_geopooling
