#pragma once

#include <algorithm>
#include <initializer_list>
#include <iostream>

#include <torch_geopooling/exception.h>
#include <torch_geopooling/tile.h>


namespace torch_geopooling {


/// A container representing 2-dimensional matrix 2x2.
///
/// \tparam T The type of elements stored in the quad.
template <typename T>
class quad {
public:
    /// T
    using value_type = T;

    /// Internal container type.
    using container_type = std::array<value_type, 4>;

    /// Unsigned integer type (usually std::size_t).
    using size_type = typename container_type::size_type;

    using const_iterator = typename container_type::const_iterator;

    /// Constructs a quad with all elements initialized to the given value.
    ///
    /// \param fill the value to initialize all elements.
    quad(const value_type& fill)
    : m_elements({fill, fill, fill, fill})
    {}

    /// Constructs a quad with specified values for each element.
    ///
    /// \param t00 value for element at (0, 0).
    /// \param t01 value for element at (0, 1).
    /// \param t10 value for element at (1, 0).
    /// \param t11 value for element at (1, 1).
    quad(value_type t00, value_type t01, value_type t10, value_type t11)
    : m_elements({t00, t01, t10, t11})
    {}

    /// Access the element at the specified coordinates.
    ///
    /// \param x the row index (0 or 1).
    /// \param y the column index (0 or 1).
    /// \return reference to the element at (x, y).
    /// \throws out_of_range if x or y is greater than 1.
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

    /// Returns a constant iterator to the beginning of the quad.
    ///
    /// \return constant iterator to the beginning.
    const_iterator
    begin() const
    {
        return m_elements.cbegin();
    }

    /// Returns a constant iterator to the end of the quad.
    ///
    /// /return constant iterator to the end.
    const_iterator
    end() const
    {
        return m_elements.cend();
    }

private:
    container_type m_elements;
};


/// A class representing a rectangular region.
///
/// \tparam the type of point coordinates (usually double).
template <typename T>
class quadrect {
public:
    /// T.
    using value_type = T;

    /// Point coordinate type (x, y).
    using point_type = std::pair<T, T>;

    /// Constructs a quadrect from 4 values (xmin, ymin, width, height).
    ///
    /// \param vector comprised of xmin, ymin, width, and height.
    /// \throws value_error if the size of vec is not 4 or if width/height is negative.
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

    /// Constructs a quadrect from an initializer list of 4 values (xmin, ymin, width, height).
    ///
    /// \param list initalizer list containing xmin, ymin, width, and height.
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

    /// Compares two quadrect objects for equality.
    ///
    /// \param rhs the other quadrect to compare with.
    /// \return true if both quadrects are equal, false otherwise.
    bool
    operator==(const quadrect& rhs) const
    {
        return (
            m_xmin == rhs.m_xmin && m_xmax == rhs.m_xmax && m_ymin == rhs.m_ymin
            && m_ymax == rhs.m_ymax
        );
    }

    /// Returns width of the rectangle.
    ///
    /// \return width of the rectangle.
    inline value_type
    width() const
    {
        return m_xmax - m_xmin;
    }

    /// Returns height of the rectangle.
    ///
    /// \return height of the rectangle.
    inline value_type
    height() const
    {
        return m_ymax - m_ymin;
    }

    /// Returns the centroid of the rectangle.
    ///
    /// \return coordinates of a point (x, y) representing a center of a rectangle.
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

    /// Split rectangle into 4 sub-rectangles of the equal size.
    ///
    /// \return a quad of 4 rectangles that together assemble the parent's rectangle.
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
    /// coordinates of a new rectangle and returns it.
    ///
    /// \param tile the tile slice from the rectangle.
    /// \return a slice of the rectangle for the given tile.
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
