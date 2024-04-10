#pragma once

#include <iostream>

#include <torch_geopooling/functional.h>


namespace torch_geopooling {


class Tile {
public:
    Tile();

    Tile(std::size_t z, std::size_t x, std::size_t y);

    std::size_t z() const;

    std::size_t x() const;

    std::size_t y() const;

    Tile parent() const;

    Tile child(std::size_t x, std::size_t y) const;

    bool
    operator== (const Tile& rhs) const;

    bool
    operator!= (const Tile& rhs) const;

    friend std::ostream&
    operator<<(std::ostream& os, const Tile& t)
    {
        os << "Tile(" << t.m_z << ", " << t.m_x << ", " << t.m_y << ")";
        return os;
    }

private:
    std::size_t m_z, m_x, m_y;
};


} // namespace torch_geopooling



namespace std {


template<>
struct hash<torch_geopooling::Tile>
{
    using argument_type = torch_geopooling::Tile;

    std::size_t
    operator()(const argument_type& argument) const noexcept {
        std::size_t seed = 0;
        hash_combine(seed, argument.z());
        hash_combine(seed, argument.x());
        hash_combine(seed, argument.y());
        return seed;
    }
};


} // namespace std
