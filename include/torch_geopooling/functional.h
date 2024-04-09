#pragma once

#include <functional>


template<typename T>
struct std::hash<std::pair<T, T>>
{
    using argument_type = std::pair<T, T>;

    std::size_t
    operator()(const argument_type& argument) const noexcept {
        std::size_t h1 = std::hash<T>{}(argument.first);
        std::size_t h2 = std::hash<T>{}(argument.second);
        return h1 ^ (h2 << 1);
    }
};


template<class Arithmetic1, class Arithmetic2>
Arithmetic1
round(Arithmetic1 value, Arithmetic2 precision)
{
    Arithmetic2 power = std::pow(10.0, precision);
    return static_cast<Arithmetic1>((value * power) / power);
}


template<class Arithmetic>
long
round(long value, Arithmetic precision)
{
    return value;
}


template<class Arithmetic1, class Arithmetic2>
std::pair<Arithmetic1, Arithmetic1>
round(const std::pair<Arithmetic1, Arithmetic1>& pair, Arithmetic2 precision)
{
    return std::make_pair(round(pair.first, precision), round(pair.second, precision));
}
