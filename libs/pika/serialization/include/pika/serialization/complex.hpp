//  Copyright (c) 2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/serialization/serialization_fwd.hpp>
#include <pika/serialization/traits/is_bitwise_serializable.hpp>
#include <pika/serialization/traits/is_not_bitwise_serializable.hpp>

#include <complex>
#include <type_traits>

namespace pika { namespace serialization {

    template <typename T>
    void serialize(input_archive& ar, std::complex<T>& c, unsigned)
    {
        T real, imag;
        ar >> real >> imag;
        c.real(real);
        c.imag(imag);
    }

    template <typename T>
    void serialize(output_archive& ar, std::complex<T> const& c, unsigned)
    {
        ar << c.real() << c.imag();
    }
}}    // namespace pika::serialization

namespace pika { namespace traits {

    template <typename T>
    struct is_bitwise_serializable<std::complex<T>>
      : is_bitwise_serializable<typename std::remove_const<T>::type>
    {
    };

    template <typename T>
    struct is_not_bitwise_serializable<std::complex<T>>
      : std::integral_constant<bool,
            !is_bitwise_serializable_v<std::complex<T>>>
    {
    };
}}    // namespace pika::traits
