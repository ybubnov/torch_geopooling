#include <torch_geopooling/exception.h>
#include <torch_geopooling/tile.h>


namespace torch_geopooling {


const Tile Tile::root = Tile(0, 0, 0);


Tile::Tile()
: m_z(0), m_x(0), m_y(0)
{ }


Tile::Tile(std::size_t z, std::size_t x, std::size_t y)
: m_z(z), m_x(x), m_y(y)
{
    auto max_size = 1 << m_z;
    if (x >= max_size) {
        throw value_error(
            "Tile: x ({}) exceeds max size ({}) for z-scale ({})", m_x, max_size, m_z
        );
    }
    if (y >= max_size) {
        throw value_error(
            "Tile: y ({}) exceeds max size ({}) for z-scale ({})", m_y, max_size, m_z
        );
    }
}


Tile::Tile(const std::array<std::size_t, 3>& zxy)
: Tile(zxy[0], zxy[1], zxy[1])
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
    return Tile(std::max(m_z, std::size_t(1)) - 1, m_x >> 1, m_y >> 1);
}


Tile
Tile::child(std::size_t x, std::size_t y) const
{
    if (x > 1) {
        throw value_error("Tile: x ({}) value should be either 0 or 1", x);
    }
    if (y > 1) {
        throw value_error("Tile: y ({}) value shoule be either 0 or 1", y);
    }
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
