#include <torch_geopooling/tile.h>


namespace torch_geopooling {


tile::tile()
: m_z(0), m_x(0), m_y(0)
{ }


tile::tile(std::size_t z, std::size_t x, std::size_t y)
: m_z(z), m_x(x), m_y(y)
{ }


std::size_t
tile::z() const
{ return m_z; }


std::size_t
tile::x() const
{ return m_x; }


std::size_t
tile::y() const
{ return m_y; }


tile
tile::parent() const
{
    return tile(m_z - 1, m_x >> 1, m_y >> 1);
}


bool
tile::operator==(const tile& rhs) const
{
    return m_z == rhs.m_z && m_x == rhs.m_x && m_y == rhs.m_y;
}


bool
tile::operator!=(const tile& rhs) const
{
    return !(*this == rhs);
}


} // namespace torch_geopooling
