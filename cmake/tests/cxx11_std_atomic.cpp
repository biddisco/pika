////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <atomic>
#include <cstdint>

template <typename T>
void test_atomic()
{
    std::atomic<T> a;

    a.store(T{});

    {
        [[maybe_unused]] T i = a.load();
    }

    {
        [[maybe_unused]] T i = a.exchange(T{});
    }

    {
        T expected{};
        [[maybe_unused]] bool b = a.compare_exchange_weak(expected, T{});
    }

    {
        T expected{};
        [[maybe_unused]] bool b = a.compare_exchange_strong(expected, T{});
    }

    {
        [[maybe_unused]] T i = a.fetch_sub(T{1});
    }

    {
        [[maybe_unused]] T i = a.fetch_add(T{1});
    }
}

int main()
{
    std::atomic_flag af = ATOMIC_FLAG_INIT;
    if (af.test_and_set()) af.clear();

    test_atomic<int>();
    test_atomic<std::uint8_t>();
    test_atomic<std::uint16_t>();
    test_atomic<std::uint32_t>();
    test_atomic<std::uint64_t>();

    [[maybe_unused]] std::memory_order mo;
    mo = std::memory_order_relaxed;
    mo = std::memory_order_acquire;
    mo = std::memory_order_release;
    mo = std::memory_order_acq_rel;
    mo = std::memory_order_seq_cst;
    (void) mo;
}
