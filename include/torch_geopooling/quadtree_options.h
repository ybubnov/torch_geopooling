#pragma once

#include <optional>
#include <limits>


namespace torch_geopooling {


struct QuadtreeOptions {

    QuadtreeOptions()
     : m_max_depth(17),
       m_capacity(1),
       m_max_terminal_nodes(std::nullopt),
       m_precision(std::nullopt)
    { }

    bool
    hash_max_terminal_nodes() const noexcept
    {
        return m_max_terminal_nodes.has_value();
    }

    bool
    has_precision() const noexcept
    {
        return m_precision.has_value();
    }

    QuadtreeOptions
    max_depth(std::size_t max_depth) const noexcept
    {
        QuadtreeOptions r = *this;
        r.set_max_depth(max_depth);
        return r;
    }

    QuadtreeOptions
    capacity(std::size_t capacity) const noexcept
    {
        QuadtreeOptions r = *this;
        r.set_capacity(capacity);
        return r;
    }

    QuadtreeOptions
    max_terminal_nodes(std::size_t max_terminal_nodes) const noexcept
    {
        QuadtreeOptions r = *this;
        r.set_max_terminal_nodes(max_terminal_nodes);
        return r;
    }

    QuadtreeOptions
    precision(std::size_t precision) const noexcept
    {
        QuadtreeOptions r = *this;
        r.set_precision(precision);
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
