#pragma once

#include <functional>


namespace std {


template<typename SizeT, typename T>
void
hash_combine(SizeT& seed, T value)
{
    std::hash<T> hash;
    seed ^= hash(value) + 0x9e3779b9 + (seed<<6) + (seed>>2);
}


template<typename T>
struct std::hash<std::pair<T, T>>
{
    using argument_type = std::pair<T, T>;

    std::size_t
    operator()(const argument_type& argument) const noexcept {
        std::size_t seed = 0;
        hash_combine(seed, argument.first);
        hash_combine(seed, argument.second);
        return seed;
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


} // namespace std
