//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/serialization/access.hpp>

#include <type_traits>

namespace pika { namespace traits {

    // This traits can be used in systems that assume that all types are bitwise
    // serializable by default (like SHAD), while still enforcing normal
    // serialization for a given type.

#if defined(PIKA_SERIALIZATION_HAVE_ALL_TYPES_ARE_BITWISE_SERIALIZABLE)
    // By default, any abstract type or type that exposes serialization will be
    // treated as non-bitwise copyable.
    template <typename T>
    struct is_not_bitwise_serializable
      : std::integral_constant<bool,
            std::is_abstract<T>::value ||
                pika::serialization::access::has_serialize<T>::value ||
                pika::serialization::has_serialize_adl<T>::value>
    {
    };

    template <typename T>
    inline constexpr bool is_not_bitwise_serializable_v =
        is_not_bitwise_serializable<T>::value;
#else
    template <typename T>
    struct is_not_bitwise_serializable : std::true_type
    {
    };

    template <typename T>
    inline constexpr bool is_not_bitwise_serializable_v = true;
#endif

}}    // namespace pika::traits

#define PIKA_IS_NOT_BITWISE_SERIALIZABLE(T)                                    \
    namespace pika { namespace traits {                                        \
            template <>                                                        \
            struct is_not_bitwise_serializable<T> : std::true_type             \
            {                                                                  \
            };                                                                 \
        }                                                                      \
    }                                                                          \
    /**/
