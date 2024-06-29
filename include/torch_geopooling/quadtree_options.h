/// Copyright (C) 2024, Yakau Bubnou
///
/// This program is free software: you can redistribute it and/or modify
/// it under the terms of the GNU General Public License as published by
/// the Free Software Foundation, either version 3 of the License, or
/// (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <limits>
#include <optional>


namespace torch_geopooling {


/// Options for configuring quadtree.
struct quadtree_options {

    /// Construct options using default quadtree values.
    quadtree_options()
    : m_max_depth(17),
      m_capacity(1),
      m_max_terminal_nodes(std::nullopt),
      m_precision(std::nullopt)
    {}

    bool
    has_max_terminal_nodes() const noexcept
    {
        return m_max_terminal_nodes.has_value();
    }

    bool
    has_precision() const noexcept
    {
        return m_precision.has_value();
    }

    /// Set the maximum depth of the tree.
    ///
    /// \param max_depth a maximum depth of a tree.
    quadtree_options
    max_depth(std::optional<std::size_t> max_depth) const noexcept
    {
        quadtree_options r = *this;
        if (max_depth.has_value()) {
            r.set_max_depth(max_depth.value());
        }
        return r;
    }

    /// Set the capacity of terminal nodes.
    ///
    /// This parameter controls how many different points could be situated in the terminal node
    /// of the tree until the split (sub-division). Meaning, if the number of points within a
    /// terminal node reaches the specified capacity, the tree growth deeper.
    ///
    /// \param capacity a capacity of a terminal node.
    quadtree_options
    capacity(std::optional<std::size_t> capacity) const noexcept
    {
        quadtree_options r = *this;
        if (capacity.has_value()) {
            r.set_capacity(capacity.value());
        }
        return r;
    }

    /// Set the maximum number of terminal nodes.
    ///
    /// This parameter could be used to control the size of the deep and wide trees. When used
    /// as part of quad pooling operations it might bias the shape of data that the operation
    /// could learn, therefore use this parameter carefully.
    ///
    /// \param max_terminal_nodes a maximum number of terminal nodes in a quadtree.
    quadtree_options
    max_terminal_nodes(std::optional<std::size_t> max_terminal_nodes) const noexcept
    {
        quadtree_options r = *this;
        if (max_terminal_nodes.has_value()) {
            r.set_max_terminal_nodes(max_terminal_nodes.value());
        }
        return r;
    }

    /// Set the precision of coordinates.
    ///
    /// On insertion of a point into a quadtree, it's coordinates will be rounded to the specified
    /// precision. Sometimes it's necessary to treat points with slightly different coordinates
    /// as the same. This could be achieved using this option.
    ///
    /// \param precision a float number precision used to round coordinates (longitude and
    /// latitude).
    quadtree_options
    precision(std::optional<std::size_t> precision) const noexcept
    {
        quadtree_options r = *this;
        if (precision.has_value()) {
            r.set_precision(precision.value());
        }
        return r;
    }

    /// Returns the maximum allowed depth of a quadtree.
    ///
    /// \return maximum depth of a tree (17 by default).
    std::size_t
    max_depth() const noexcept
    {
        return m_max_depth;
    }

    /// Returns the capacity of a tree node.
    ///
    /// \return capacity of a tree node (1 by default).
    std::size_t
    capacity() const noexcept
    {
        return m_capacity;
    }

    /// Returns the maximum allowed number of terminal nodes in a quadtree.
    ///
    /// \return maximum number of terminal nodes in the tree if set, otherwise maximum
    ///     numeric limit of unsigned integer.
    std::size_t
    max_terminal_nodes() const noexcept
    {
        return m_max_terminal_nodes.value_or(std::numeric_limits<std::size_t>::max());
    }

    /// Returns precision of geographic coordinates.
    ///
    /// \return precision of points within quadtree if set, otherwise maximum numeric
    ///     limit of unsigned integer.
    std::size_t
    precision()
    {
        return m_precision.value_or(std::numeric_limits<std::size_t>::digits10);
    }

private:
    std::size_t m_max_depth;
    std::size_t m_capacity;
    std::optional<std::size_t> m_max_terminal_nodes;
    std::optional<std::size_t> m_precision;

    void
    set_max_depth(std::size_t max_depth)
    {
        m_max_depth = max_depth;
    }

    void
    set_capacity(std::size_t capacity)
    {
        m_capacity = capacity;
    }

    void
    set_max_terminal_nodes(std::size_t max_terminal_nodes)
    {
        m_max_terminal_nodes = max_terminal_nodes;
    }

    void
    set_precision(std::size_t precision)
    {
        m_precision = precision;
    }
};


} // namespace torch_geopooling
