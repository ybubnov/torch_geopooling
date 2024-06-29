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

#include <functional>


namespace std {


template <typename SizeT, typename T>
void
hash_combine(SizeT& seed, T value)
{
    hash<T> make_hash;
    seed ^= make_hash(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}


template <typename T>
struct hash<pair<T, T>> {
    using argument_type = pair<T, T>;

    size_t
    operator()(const argument_type& argument) const noexcept
    {
        size_t seed = 0;
        hash_combine(seed, argument.first);
        hash_combine(seed, argument.second);
        return seed;
    }
};


template <class Arithmetic1, class Arithmetic2>
Arithmetic1
round(Arithmetic1 value, Arithmetic2 precision)
{
    Arithmetic2 power = pow(10.0, precision);
    return static_cast<Arithmetic1>((value * power) / power);
}


template <class Arithmetic>
long
round(long value, Arithmetic precision)
{
    return value;
}


template <class Arithmetic1, class Arithmetic2>
pair<Arithmetic1, Arithmetic1>
round(const pair<Arithmetic1, Arithmetic1>& pair, Arithmetic2 precision)
{
    return make_pair(round(pair.first, precision), round(pair.second, precision));
}


} // namespace std
