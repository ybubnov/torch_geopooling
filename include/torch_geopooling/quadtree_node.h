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

#include <unordered_set>

#include <torch_geopooling/quadrect.h>
#include <torch_geopooling/tile.h>


namespace torch_geopooling {


template <typename Coordinate>
class quadtree_node {
public:
    using key_type = std::pair<Coordinate, Coordinate>;

    using exterior_type = quadrect<Coordinate>;

    using container_type = std::unordered_set<key_type>;

    using point_iterator = typename container_type::iterator;

    quadtree_node(Tile tile, exterior_type exterior)
    : m_tile(tile),
      m_exterior(exterior),
      m_points()
    {}

    const Tile&
    tile() const
    {
        return m_tile;
    }

    std::size_t
    depth() const
    {
        return m_tile.z();
    }

    exterior_type
    exterior() const
    {
        return m_exterior;
    }

    void
    insert(const key_type& key)
    {
        m_points.insert(key);
    }

    void
    clear()
    {
        m_points.clear();
    }

    std::size_t
    size() const
    {
        return m_points.size();
    }

    point_iterator
    begin()
    {
        return m_points.begin();
    }

    point_iterator
    end()
    {
        return m_points.end();
    }

private:
    Tile m_tile;

    exterior_type m_exterior;
    container_type m_points;
};


} // namespace torch_geopooling
