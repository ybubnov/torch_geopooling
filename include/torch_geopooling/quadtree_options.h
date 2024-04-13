#pragma once

#include <optional>
#include <limits>


namespace torch_geopooling {


struct quadtree_options {

    quadtree_options()
    : m_max_depth(17),
      m_capacity(1),
      m_max_terminal_nodes(std::nullopt),
      m_precision(std::nullopt)
    { }

    bool
    hash_max_terminal_nodes() const noexcept
    { return m_max_terminal_nodes.has_value(); }

    bool
    has_precision() const noexcept
    { return m_precision.has_value(); }

    quadtree_options
    max_depth(std::optional<std::size_t> max_depth) const noexcept
    {
        quadtree_options r = *this;
        if (max_depth.has_value()) {
            r.set_max_depth(max_depth.value());
        }
        return r;
    }

    quadtree_options
    capacity(std::optional<std::size_t> capacity) const noexcept
    {
        quadtree_options r = *this;
        if (capacity.has_value()) {
            r.set_capacity(capacity.value());
        }
        return r;
    }

    quadtree_options
    max_terminal_nodes(std::optional<std::size_t> max_terminal_nodes) const noexcept
    {
        quadtree_options r = *this;
        if (max_terminal_nodes.has_value()) {
            r.set_max_terminal_nodes(max_terminal_nodes.value());
        }
        return r;
    }

    quadtree_options
    precision(std::optional<std::size_t> precision) const noexcept
    {
        quadtree_options r = *this;
        if (precision.has_value()) {
            r.set_precision(precision.value());
        }
        return r;
    }

    std::size_t
    max_depth() const noexcept
    { return m_max_depth; }

    std::size_t
    capacity() const noexcept
    { return m_capacity; }

    std::size_t
    max_terminal_nodes() const noexcept
    { return m_max_terminal_nodes.value_or(std::numeric_limits<std::size_t>::max()); }

    std::size_t
    precision()
    { return m_precision.value_or(std::numeric_limits<std::size_t>::digits10); }

private:
    std::size_t m_max_depth;
    std::size_t m_capacity;
    std::optional<std::size_t> m_max_terminal_nodes;
    std::optional<std::size_t> m_precision;

    void
    set_max_depth(std::size_t max_depth)
    { m_max_depth = max_depth; }

    void
    set_capacity(std::size_t capacity)
    { m_capacity = capacity; }

    void
    set_max_terminal_nodes(std::size_t max_terminal_nodes)
    { m_max_terminal_nodes = max_terminal_nodes; }

    void
    set_precision(std::size_t precision)
    { m_precision = precision; }
};


} // namespace torch_geopooling
