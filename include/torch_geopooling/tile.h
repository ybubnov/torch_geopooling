#pragma once

#include <cstdint>
#include <iostream>
#include <vector>

#include <fmt/core.h>

#include <torch_geopooling/exception.h>
#include <torch_geopooling/functional.h>


namespace torch_geopooling {


class Tile {
public:
    Tile();

    Tile(std::size_t z, std::size_t x, std::size_t y);

    template<typename T>
    Tile(const std::array<T, 3>& zxy)
    : Tile(
        static_cast<std::size_t>(zxy[0]),
        static_cast<std::size_t>(zxy[1]),
        static_cast<std::size_t>(zxy[2])
      )
    {
        auto [z, x, y] = zxy;
        if (z < 0) {
            throw value_error("Tile: z ({}) must be more than 0", z);
        }
        if (x < 0) {
            throw value_error("Tile: x ({}) must be more than 0", x);
        }
        if (y < 0) {
            throw value_error("Tile: y ({}) must be more than 0", y);
        }
    }

    std::size_t z() const;

    std::size_t x() const;

    std::size_t y() const;

    Tile parent() const;

    Tile child(std::size_t x, std::size_t y) const;

    template<typename T>
    std::vector<T>
    vec()
    {
        std::vector<T> zxy({
            static_cast<T>(m_z),
            static_cast<T>(m_y),
            static_cast<T>(m_x)
        });
        return zxy;
    }

    bool
    operator==(const Tile& rhs) const;

    bool
    operator!=(const Tile& rhs) const;

    friend std::ostream&
    operator<<(std::ostream& os, const Tile& t)
    {
        os << "Tile(" << t.m_z << ", " << t.m_x << ", " << t.m_y << ")";
        return os;
    }

    const static Tile root;
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


template<>
struct fmt::formatter<torch_geopooling::Tile>
{
    template<typename ParseContext>
    constexpr auto parse(ParseContext& context) {
        return context.begin();
    }

    template<typename FormatContext>
    auto format(torch_geopooling::Tile const& tile, FormatContext& context) const
    {
        return fmt::format_to(context.out(), "Tile({}, {}, {})", tile.z(), tile.x(), tile.y());
    }
};
