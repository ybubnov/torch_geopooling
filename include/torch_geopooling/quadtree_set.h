#pragma once

#include <limits>
#include <optional>
#include <queue>
#include <initializer_list>
#include <unordered_map>
#include <unordered_set>

#include <torch_geopooling/functional.h>
#include <torch_geopooling/tile.h>
#include <torch_geopooling/quadrect.h>
#include <torch_geopooling/quadtree_options.h>


namespace torch_geopooling {


template<typename Coordinate>
class quadtree_node {
public:
    using key_type = std::pair<Coordinate, Coordinate>;

    using exterior_type = quadrect<Coordinate>;

    using container_type = std::unordered_set<key_type>;

    using point_iterator = typename container_type::iterator;

    quadtree_node(Tile tile, exterior_type exterior, std::size_t max_depth)
    : m_tile(tile), m_exterior(exterior), m_max_depth(max_depth), m_points()
    { }

    ~quadtree_node() = default;

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
class quadtree_set {
public:
    using key_type = std::pair<Coordinate, Coordinate>;

    using node_type = quadtree_node<Coordinate>;

    using exterior_type = quadrect<Coordinate>;

    quadtree_set(
        const exterior_type& exterior,
        std::optional<quadtree_options> options = std::nullopt
    )
    : m_options(options.value_or(quadtree_options())),
      m_nodes(),
      m_total_depth(0),
      m_num_terminal_nodes(0)
    {
        Tile tile = Tile::root;
        node_type node(tile, exterior, m_options.max_depth());
        m_nodes.insert(std::make_pair(tile, node));
    }

    template<typename InputIt>
    quadtree_set(
        InputIt first,
        InputIt last,
        const exterior_type& exterior,
        std::optional<quadtree_options> options = std::nullopt
    )
    : quadtree_set(exterior, options)
    {
        while (first != last) {
            auto node_tile = *first;
            auto node_exterior = make_exterior(exterior, node_tile);
            auto node = node_type(node_tile, node_exterior, m_options.max_depth());

            m_nodes.insert(std::make_pair(node_tile, node));

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
                        "quadtree_set: tile {} does not have a parent {}",
                        node.first, parent_tile
                    );
                }
                parent_tile = parent_tile.parent();
            }
        }
    }

    quadtree_set(
        const std::initializer_list<Coordinate> xywh,
        std::optional<quadtree_options> options = std::nullopt
    )
    : quadtree_set(exterior_type(xywh), options)
    { }

    ~quadtree_set() = default;

    bool
    contains(const key_type& point) const
    {
        const auto node = m_nodes.at(Tile::root);
        return node.exterior().contains(point);
    }

    void
    insert(const key_type& key)
    {
        auto point = key;
        if (m_options.has_precision()) {
            point = std::round(point, m_options.precision());
        }

        assert_contains(point);

        auto& node = find(point);
        node.insert(point);
        subdivide(node);
    }

    node_type&
    find(const key_type& point, std::optional<std::size_t> max_depth = std::nullopt)
    {
        assert_contains(point);

        Tile tile = Tile::root;
        node_type* node = &m_nodes.at(tile);

        max_depth = std::min(max_depth.value_or(m_total_depth + 1), m_options.max_depth());

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
    quadtree_options m_options;

    std::size_t m_total_depth;

    // Terminal nodes are updated once the split of the node happens (it's always increases
    // the number of terminal node by 3).
    std::size_t m_num_terminal_nodes;

    void
    assert_contains(const key_type& point) const
    {
        if (!contains(point)) {
            throw value_error(
                "quadtree_set: point ({}, {}) is outside of exterior geometry",
                point.first, point.second
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
        auto full = (
            m_options.hash_max_terminal_nodes() &&
            m_options.max_terminal_nodes() >= m_num_terminal_nodes
        );

        if (
            full ||
            node.size() <= m_options.capacity() ||
            node.tile().z() >= m_options.max_depth() ||
            has_children(node)
        ) {
            return;
        }

        auto quadrects = node.exterior().symmetric_split();
        for (std::size_t x : {0, 1}) {
            for (std::size_t y : {0, 1}) {
                auto tile = node.tile().child(x, y);
                auto n = node_type(tile, quadrects.at(x, y), m_options.max_depth());

                m_nodes.insert(std::make_pair(tile, n));
                m_total_depth = std::max(tile.z(), m_total_depth);
                m_num_terminal_nodes += 1;
            }
        }

        // Current node is no longer terminal, therefore subtract -1 here.
        m_num_terminal_nodes -= 1;

        for (auto point : node) {
            insert(point);
        }

        node.clear();
    }
};


} // namespace torch_geopooling
