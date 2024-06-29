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


#include <array>

#include <fmt/format.h>


template <typename T>
struct fmt::formatter<std::array<T, 2>> {
    template <typename ParseContext>
    constexpr auto
    parse(ParseContext& context)
    {
        return context.begin();
    }

    template <typename FormatContext>
    auto
    format(std::array<T, 2> const& array2, FormatContext& context) const
    {
        return fmt::format_to(context.out(), "{}, {}", array2[0], array2[1]);
    }
};
