#pragma once

#include <initializer_list>
#include <iterator>
#include <limits>
#include <optional>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include <torch_geopooling/exception.h>
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

    quadtree_node(Tile tile, exterior_type exterior)
    : m_tile(tile), m_exterior(exterior), m_points()
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

    exterior_type m_exterior;
    container_type m_points;
};


template
<typename Coordinate>
class quadtree_set_iterator;


template<typename Coordinate>
class quadtree_set {
public:
    using key_type = std::pair<Coordinate, Coordinate>;
    using key_array_type = std::array<Coordinate, 2>;

    using node_type = quadtree_node<Coordinate>;

    using exterior_type = quadrect<Coordinate>;

    using container_type = std::unordered_map<Tile, node_type>;

    using iterator = quadtree_set_iterator<Coordinate>;

    quadtree_set(
        const exterior_type& exterior,
        std::optional<quadtree_options> options = std::nullopt
    )
    : m_nodes(),
      m_options(options.value_or(quadtree_options())),
      m_total_depth(0),
      m_num_terminal_nodes(1)
    {
        Tile tile = Tile::root;
        node_type node(tile, exterior);
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
            Tile node_tile(*first);
            auto node_exterior = make_exterior(exterior, node_tile);
            auto node = node_type(node_tile, node_exterior);

            m_nodes.insert(std::make_pair(node_tile, node));

            ++first;
            m_total_depth = std::max(node_tile.z(), m_total_depth);
        }

        // Verify integrity of the resulting quadtree set by assuring that every children
        // has a parent up until the root tile (0, 0, 0).
        for (auto node : m_nodes) {
            auto node_tile = node.first;
            auto parent_tile = node_tile.parent();

            while (parent_tile != Tile::root) {
                if (auto n = m_nodes.find(parent_tile); n == m_nodes.end()) {
                    throw value_error(
                        "quadtree_set: tile {} does not have a parent {}",
                        node_tile, parent_tile
                    );
                }
                parent_tile = parent_tile.parent();
            }

            if (!has_children(node_tile)) {
                m_num_terminal_nodes += 1;
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

    iterator
    begin()
    { return iterator(this, Tile::root); }

    iterator
    end()
    { return iterator(); }

    iterator
    ibegin()
    { return iterator(this, Tile::root, true); }

    iterator
    iend()
    { return iterator(); }

    const exterior_type
    exterior() const
    {
        const auto node = m_nodes.at(Tile::root);
        return node.exterior();
    }

    bool
    contains(const key_type& point) const
    {
        return exterior().contains(point);
    }

    bool
    contains(const Tile& tile) const
    {
        auto node = m_nodes.find(tile);
        return node != m_nodes.end();
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

    void
    insert(const key_array_type& key)
    {
        insert(key_type(key[0], key[1]));
    }

    template<typename InputIt>
    void
    insert(InputIt first, InputIt last) {
        while (first != last) {
            insert(*first);
            ++first;
        }
    }

    /// @brief Find terminal group of nodes sharing the same parent.
    ///
    /// This method finds a terminal node containing the specified point, gets it's parent
    /// and iterates over all terminal nodes of the parent. When node does not belong to a
    /// terminal node, method returns `end()`, or empty iterator.
    ///
    /// @param point a 2-dimensional point.
    /// @param max_depth a maximum depth of the look operation. When specified, the point lookup
    ///     process is limited by the specified depth.
    ///
    /// @returns The iterator over a group of terminal nodes.
    iterator
    find_terminal_group(
        const key_type& point,
        std::optional<std::size_t> max_depth = std::nullopt
    )
    {
        assert_contains(point);

        const auto& node = find(point, max_depth);
        if (has_children(node.tile())) {
            return end();
        }

        return quadtree_set_iterator(this, node.tile().parent());
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
    find(const key_array_type& point, std::optional<std::size_t> max_depth = std::nullopt)
    {
        return find(key_type(point[0], point[1]), max_depth);
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
    friend class quadtree_set_iterator<Coordinate>;

    container_type m_nodes;
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
    has_children(const Tile& tile) const
    {
        return (
            contains(tile.child(0, 0)) ||
            contains(tile.child(0, 1)) ||
            contains(tile.child(1, 0)) ||
            contains(tile.child(1, 1))
        );
    }

    void
    subdivide(node_type& node)
    {
        // Subdivision increases the number of terminal nodes by 3, which might not be
        // possible to do, since it will break the promised limit of terminal nodes.
        auto full = (
            m_options.hash_max_terminal_nodes() &&
            (m_num_terminal_nodes + 3) >= m_options.max_terminal_nodes()
        );

        if (
            full ||
            node.size() <= m_options.capacity() ||
            node.tile().z() >= m_options.max_depth() ||
            has_children(node.tile())
        ) {
            return;
        }

        auto exteriors = node.exterior().symmetric_split();
        for (std::size_t x : {0, 1}) {
            for (std::size_t y : {0, 1}) {
                auto tile = node.tile().child(x, y);
                auto n = node_type(tile, exteriors.at(x, y));

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


/// Forward iterator of quadtree set.
///
/// Class provides iterator traits and can be used in for-loops to iterate both internal and
/// terminal nodes of a quadtree.
///
/// Effectively iterator utilizes breadth-first graph traversal algorithm.
template<typename Coordinate>
class quadtree_set_iterator
{
public:
    using iterator_category = std::forward_iterator_tag;

    using iterator = quadtree_set_iterator<Coordinate>;

    using value_type = const quadtree_node<Coordinate>;

    using reference = value_type&;

    using pointer = value_type*;

    explicit
    quadtree_set_iterator(
        const quadtree_set<Coordinate>* set,
        const Tile tile = Tile::root,
        bool include_internal = false
    )
    : m_queue(),
      m_set(set),
      m_include_internal(include_internal)
    {
        m_queue.push(tile);
        forward();
    }

    quadtree_set_iterator()
    : m_queue(),
      m_set(nullptr),
      m_include_internal(false)
    { }

    ~quadtree_set_iterator()
    { }

    // TODO: remember the size of a quadtree and throw exception, if size of the underlying
    // set was changed by insert operation.
    iterator&
    operator++()
    { return next(); }

    reference
    operator*()
    { return get(); }

    pointer
    operator->()
    { return &get(); }

    bool
    operator!=(const iterator& rhs)
    {
        return !(m_queue.size() == rhs.m_queue.size());
    }

private:
    std::queue<Tile> m_queue;
    const quadtree_set<Coordinate>* m_set;
    bool m_include_internal;

    using node_type = quadtree_node<Coordinate>;

    reference
    get()
    {
        if (m_queue.empty()) {
            throw out_of_range("quadtree_set_iterator: access to empty iterator");
        }
        return m_set->m_nodes.at(m_queue.front());
    }

    /// Rewind the state of the iterator to the expected state.
    ///
    /// If the iterator should not return internal nodes, method rewinds tiles until the
    /// queue contains in front a terminal node. The rewinding forward might reach the end
    /// of the iterator.
    void
    forward()
    {
        if (!m_include_internal) {
            while (!m_queue.empty() && m_set->has_children(m_queue.front())) {
                next_tile();
            }
        }
    }

    /// Iterate nodes of the quadtree. For each internal node, queue growth by it's children.
    iterator&
    next_tile()
    {
        if (m_queue.empty()) {
            return *this;
        }

        auto tile = m_queue.front();
        m_queue.pop();

        if (m_set->has_children(tile)) {
            for (std::size_t x : {0, 1}) {
                for (std::size_t y : {0, 1}) {
                    m_queue.push(tile.child(x, y));
                }
            }
        }
        return *this;
    }

    /// Next moves the iterator one step forward.
    ///
    /// When the front of the queue contains internal node, but iterator is configured
    /// only for terminal nodes, then method rewinds nodes until the first terminal node.
    iterator&
    next()
    {
        next_tile();
        forward();
        return *this;
    }
};


} // namespace torch_geopooling
