#pragma once

#include <algorithm>
#include <initializer_list>
#include <iostream>

#include <torch_geopooling/exception.h>
#include <torch_geopooling/tile.h>


namespace torch_geopooling {


template <typename T>
class quad {
public:
    using value_type = T;

    using container_type = std::array<value_type, 4>;

    using size_type = typename container_type::size_type;

    using const_iterator = typename container_type::const_iterator;

    quad(const value_type& fill)
    : m_elements({fill, fill, fill, fill})
    {}

    quad(value_type t00, value_type t01, value_type t10, value_type t11)
    : m_elements({t00, t01, t10, t11})
    {}

    value_type&
    at(size_type x, size_type y)
    {
        if (x > 1) {
            throw out_of_range("quad: x ({}) >1", x);
        }
        if (y > 1) {
            throw out_of_range("quad: x ({}) >1", y);
        }
        return m_elements.at(x * 2 + y);
    }

    const value_type&
    at(size_type x, size_type y) const
    {
        return at(x, y);
    }

    const_iterator
    begin() const
    {
        return m_elements.cbegin();
    }

    const_iterator
    end() const
    {
        return m_elements.cend();
    }

private:
    container_type m_elements;
};


template <typename T>
class quadrect {
public:
    using value_type = T;
    using point_type = std::pair<T, T>;

    quadrect(const std::vector<value_type>& vec)
    {
        auto vec_size = vec.size();
        if (vec_size != 4) {
            throw value_error("quadrect: size of input ({}) should be {}", vec_size);
        }

        std::array<value_type, 4> xywh;
        std::copy(vec.begin(), vec.end(), xywh.begin());

        auto [x, y, w, h] = xywh;

        if (w < T(0)) {
            throw value_error("quadrect: width ({}) should be a non-negative number", w);
        }
        if (h < T(0)) {
            throw value_error("quadrect: height ({}) should be a non-negative number", h);
        }

        m_xmin = x;
        m_ymin = y;
        m_xmax = x + w;
        m_ymax = y + h;
    }

    quadrect(const std::initializer_list<value_type>& list)
    : quadrect(std::vector(list))
    {}

    quadrect(const std::tuple<value_type, value_type, value_type, value_type>& xywh)
    : quadrect(std::get<0>(xywh), std::get<1>(xywh), std::get<2>(xywh), std::get<3>(xywh))
    {}

    quadrect(T xmin, T ymin, T width, T height)
    : quadrect({xmin, ymin, width, height})
    {}

    ~quadrect() = default;

    bool
    operator==(const quadrect& rhs) const
    {
        return (
            m_xmin == rhs.m_xmin && m_xmax == rhs.m_xmax && m_ymin == rhs.m_ymin
            && m_ymax == rhs.m_ymax
        );
    }

    inline value_type
    width() const
    {
        return m_xmax - m_xmin;
    }

    inline value_type
    height() const
    {
        return m_ymax - m_ymin;
    }

    point_type
    centroid() const
    {
        auto w = width() / 2;
        auto h = height() / 2;
        return point_type(m_xmin + w, m_ymin + h);
    }

    bool
    contains(const point_type& point) const
    {
        return (
            (point.first >= m_xmin) && (point.first <= m_xmax) && (point.second >= m_ymin)
            && (point.second <= m_ymax)
        );
    }

    /// Split rectangle into 4 sub-rectangles of the equal size
    ///
    /// Method returns a quad of 4 rectangles that together assemble the parent's rectangle.
    quad<quadrect>
    symmetric_split() const
    {
        value_type w = width() / 2;
        value_type h = height() / 2;

        return quad<quadrect>(
            {m_xmin, m_ymin, w, h}, {m_xmin, m_ymin + h, w, h}, {m_xmin + w, m_ymin, w, h},
            {m_xmin + w, m_ymin + h, w, h}
        );
    }

    /// Slice from the rectangle the given Tile.
    ///
    /// Method assumes that the rectangle is a Tile on a root level (0,0,0), then calculates
    /// a coordinates of a new rectangle and returns it.
    quadrect
    slice(const Tile& tile) const
    {
        value_type w = width() / (1 << tile.z());
        value_type h = height() / (1 << tile.z());

        auto result = quadrect(m_xmin + tile.x() * w, m_ymin + tile.y() * h, w, h);
        return result;
    }

    friend std::ostream&
    operator<<(std::ostream& os, const quadrect& rect)
    {
        os << "QuadRect(" << rect.m_xmin << ", " << rect.m_ymin << ", ";
        os << rect.width() << ", " << rect.height() << ")";
        return os;
    }

private:
    value_type m_xmin, m_ymin, m_xmax, m_ymax;
};


} // namespace torch_geopooling
