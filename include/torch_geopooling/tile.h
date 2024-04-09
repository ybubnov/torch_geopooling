#pragma once

#include <iostream>


namespace torch_geopooling {


class tile {
public:
    tile();

    tile(std::size_t z, std::size_t x, std::size_t y);

    std::size_t z() const;

    std::size_t x() const;

    std::size_t y() const;

    tile parent() const;

    bool
    operator== (const tile& rhs) const;

    bool
    operator!= (const tile& rhs) const;

    friend std::ostream&
    operator<<(std::ostream& os, const tile& t)
    {
        os << "tile(" << t.m_z << ", " << t.m_x << ", " << t.m_y << ")";
        return os;
    }

private:
    std::size_t m_z, m_x, m_y;
};


} // namespace torch_geopooling
