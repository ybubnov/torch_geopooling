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
struct hash<pair<T, T>>
{
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
