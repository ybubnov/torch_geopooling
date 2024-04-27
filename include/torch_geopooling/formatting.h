#pragma once


#include <array>

#include <fmt/format.h>


template<typename T>
struct fmt::formatter<std::array<T, 2>>
{
    template<typename ParseContext>
    constexpr auto parse(ParseContext& context) {
        return context.begin();
    }

    template<typename FormatContext>
    auto format(std::array<T, 2> const& array2, FormatContext& context) const
    {
        return fmt::format_to(context.out(), "{}, {}", array2[0], array2[1]);
    }
};
