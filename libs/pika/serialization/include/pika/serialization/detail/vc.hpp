//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#if defined(PIKA_HAVE_DATAPAR_VC)
#include <pika/serialization/array.hpp>
#include <pika/serialization/serialize.hpp>
#include <pika/serialization/traits/is_bitwise_serializable.hpp>
#include <pika/serialization/traits/is_not_bitwise_serializable.hpp>

#include <cstddef>
#include <type_traits>

#include <Vc/global.h>

#if defined(Vc_IS_VERSION_1) && Vc_IS_VERSION_1

#include <Vc/Vc>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace serialization {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    void serialize(input_archive& ar, Vc::Vector<T, Abi>& v, unsigned)
    {
        ar& make_array((T*) &v.data(), v.size());
    }

    template <typename T, std::size_t N, typename V, std::size_t W>
    void serialize(input_archive& ar, Vc::SimdArray<T, N, V, W>& v, unsigned)
    {
        ar& make_array((T*) &internal_data0(v),
            Vc::SimdArrayTraits<T, N>::storage_type0::Size);

        if (Vc::SimdArrayTraits<T, N>::storage_type1::Size != 0)
        {
            ar& make_array((T*) &internal_data1(v),
                Vc::SimdArrayTraits<T, N>::storage_type1::Size);
        }
    }

    template <typename T, std::size_t N, typename V>
    void serialize(input_archive& ar, Vc::SimdArray<T, N, V, N>& v, unsigned)
    {
        ar& make_array((T*) &internal_data(v),
            Vc::SimdArray<T, N, V, N>::storage_type::Size);
    }

    template <typename T>
    void serialize(input_archive& ar, Vc::Scalar::Vector<T>& v, unsigned)
    {
        ar& make_array((T*) &v.data(), v.size());
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Abi>
    void serialize(output_archive& ar, Vc::Vector<T, Abi> const& v, unsigned)
    {
        ar& make_array((T const*) &v.data(), v.size());
    }

    template <typename T, std::size_t N, typename V, std::size_t W>
    void serialize(
        output_archive& ar, Vc::SimdArray<T, N, V, W> const& v, unsigned)
    {
        ar& make_array((T const*) &internal_data0(v),
            Vc::SimdArrayTraits<T, N>::storage_type0::Size);

        if (Vc::SimdArrayTraits<T, N>::storage_type1::Size != 0)
        {
            ar& make_array((T const*) &internal_data1(v),
                Vc::SimdArrayTraits<T, N>::storage_type1::Size);
        }
    }

    template <typename T, std::size_t N, typename V>
    void serialize(
        output_archive& ar, Vc::SimdArray<T, N, V, N> const& v, unsigned)
    {
        ar& make_array((T const*) &internal_data(v),
            Vc::SimdArray<T, N, V, N>::storage_type::Size);
    }

    template <typename T>
    void serialize(output_archive& ar, Vc::Scalar::Vector<T> const& v, unsigned)
    {
        ar& make_array((T const*) &v.data(), v.size());
    }
}}    // namespace pika::serialization

namespace pika { namespace traits {

    template <typename T, typename Abi>
    struct is_bitwise_serializable<Vc::Vector<T, Abi>>
      : is_bitwise_serializable<typename std::remove_const<T>::type>
    {
    };

    template <typename T>
    struct is_bitwise_serializable<Vc::Scalar::Vector<T>>
      : is_bitwise_serializable<typename std::remove_const<T>::type>
    {
    };

    template <typename T, std::size_t N, typename V, std::size_t W>
    struct is_bitwise_serializable<Vc::SimdArray<T, N, V, W>>
      : is_bitwise_serializable<typename std::remove_const<T>::type>
    {
    };

    template <typename T, typename Abi>
    struct is_not_bitwise_serializable<Vc::Vector<T, Abi>>
      : std::integral_constant<bool,
            !is_bitwise_serializable_v<Vc::Vector<T, Abi>>>
    {
    };

    template <typename T>
    struct is_not_bitwise_serializable<Vc::Scalar::Vector<T>>
      : std::integral_constant<bool,
            !is_bitwise_serializable_v<Vc::Scalar::Vector<T>>>
    {
    };

    template <typename T, std::size_t N, typename V, std::size_t W>
    struct is_not_bitwise_serializable<Vc::SimdArray<T, N, V, W>>
      : std::integral_constant<bool,
            !is_bitwise_serializable_v<Vc::SimdArray<T, N, V, W>>>
    {
    };
}}    // namespace pika::traits

#else

#include <array>

#include <Vc/datapar>

///////////////////////////////////////////////////////////////////////////////
namespace pika { namespace serialization {

    template <typename T, typename Abi>
    void serialize(input_archive& ar, Vc::datapar<T, Abi>& v, unsigned)
    {
        std::array<T, Vc::datapar<T, Abi>::size()> data;
        ar& data;
        v.memload(data.data(), Vc::flags::vector_aligned);
    }

    template <typename T, typename Abi>
    void serialize(output_archive& ar, Vc::datapar<T, Abi> const& v, unsigned)
    {
        std::array<T, Vc::datapar<T, Abi>::size()> data;
        v.memstore(data.data(), Vc::flags::vector_aligned);
        ar& data;
    }
}}    // namespace pika::serialization

namespace pika { namespace traits {

    template <typename T, typename Abi>
    struct is_bitwise_serializable<Vc::datapar<T, Abi>>
      : is_bitwise_serializable<typename std::remove_const<T>::type>
    {
    };

    template <typename T, typename Abi>
    struct is_not_bitwise_serializable<Vc::datapar<T, Abi>>
      : std::integral_constant<bool,
            !is_bitwise_serializable_v<Vc::datapar<T, Abi>>>
    {
    };
}}    // namespace pika::traits

#endif    // Vc_IS_VERSION_1

#endif
