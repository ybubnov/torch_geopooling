#pragma once

#include <stdexcept>

#include <fmt/core.h>


namespace torch_geopooling {


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
const char*
exception::what() const noexcept (true)
{ return m_msg.c_str(); }


inline
exception::~exception()
{ }


class value_error: public exception {
public:
    explicit
    value_error(const std::string& s);

    template<typename... UTypes>
    value_error(fmt::format_string<UTypes...> fmt, UTypes&&... args)
    : exception(fmt::vformat(fmt, fmt::make_format_args(args...)))
    { }
};


inline
value_error::value_error(const std::string& s)
: exception(s)
{ }


class out_of_range: public exception {
public:
    explicit
    out_of_range(const std::string& s);

    template<typename ...T>
    out_of_range(fmt::format_string<T...> fmt, T&&... args)
    : exception(fmt::vformat(fmt, fmt::make_format_args(args...)))
    { }
};

inline
out_of_range::out_of_range(const std::string& s)
: exception(s)
{ }


} // namespace torch_geopooling
