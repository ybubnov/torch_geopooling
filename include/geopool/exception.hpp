#pragma once

#include <stdexcept>

#include <fmt/core.h>


namespace geopool {


class exception: public std::exception {
private:
    std::string m_msg;

public:

    explicit
    exception(const std::string& s);

    virtual
    ~exception();

    virtual const char*
    what() const noexcept (true);
};


inline
exception::exception(const std::string& s)
: m_msg(s)
{ }


inline
exception::exception()
{ }


inline
const char*
exception::what() const noexcept (true)
{
    return m_msg.c_str();
}


class value_error: public exception {
public:
    explicit
    value_error(const std::string& s);

    template<typename ...T>
    value_error(fmt::format_string<T...> fmt, T&& ...args)
    : exception(fmt::format(fmt, args))
    { }
}


inline
value_error::value_error(const std::string& s)
: exception(s)
{ }


class out_of_range: public std::out_of_range {
    template<typename ...T>
    out_of_range(fmt::format_string<T...> fmt, T&& ...args)
    : std::out_of_range(fmt::format(fmt, args))
    { }
}


}
