#include <torch_geopooling/tile.h>


namespace torch_geopooling {


Tile::Tile()
: m_z(0), m_x(0), m_y(0)
{ }


Tile::Tile(std::size_t z, std::size_t x, std::size_t y)
: m_z(z), m_x(x), m_y(y)
{ }


std::size_t
Tile::z() const
{ return m_z; }


std::size_t
Tile::x() const
{ return m_x; }


std::size_t
Tile::y() const
{ return m_y; }


Tile
Tile::parent() const
{
    return Tile(m_z - 1, m_x >> 1, m_y >> 1);
}


Tile
Tile::child(std::size_t x, std::size_t y) const
{
    // TODO: assert x, y;
    return Tile(m_z + 1, m_x * 2 + x, m_y * 2 + y);
}


bool
Tile::operator==(const Tile& rhs) const
{
    return m_z == rhs.m_z && m_x == rhs.m_x && m_y == rhs.m_y;
}


bool
Tile::operator!=(const Tile& rhs) const
{
    return !(*this == rhs);
}


} // namespace torch_geopooling
