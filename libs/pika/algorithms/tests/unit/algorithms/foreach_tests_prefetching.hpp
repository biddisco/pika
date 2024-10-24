//  Copyright (c) 2016 Zahra Khatami
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/execution.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/for_each.hpp>
#include <pika/parallel/util/prefetching.hpp>

#include <cstddef>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_for_each_prefetching(ExPolicy&& policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::size_t prefetch_distance_factor = 2;
    std::vector<double> c(10007, 1.0);

    std::vector<std::size_t> range(10007);
    std::iota(range.begin(), range.end(), 0);

    auto ctx = pika::parallel::util::make_prefetcher_context(
        range.begin(), range.end(), prefetch_distance_factor, c);

    pika::for_each(std::forward<ExPolicy>(policy), ctx.begin(), ctx.end(),
        [&](std::size_t i) { c[i] = 42.1; });

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](double v) -> void {
        PIKA_TEST_EQ(v, 42.1);
        ++count;
    });
    PIKA_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_for_each_prefetching_async(ExPolicy&& p, IteratorTag)
{
    std::size_t prefetch_distance_factor = 2;
    std::vector<double> c(10007, 1.0);

    std::vector<std::size_t> range(10007);
    std::iota(range.begin(), range.end(), 0);

    auto ctx = pika::parallel::util::make_prefetcher_context(
        range.begin(), range.end(), prefetch_distance_factor, c);

    auto f = pika::for_each(std::forward<ExPolicy>(p), ctx.begin(), ctx.end(),
        [&](std::size_t i) { c[i] = 42.1; });
    f.wait();

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](double v) -> void {
        PIKA_TEST_EQ(v, 42.1);
        ++count;
    });
    PIKA_TEST_EQ(count, c.size());
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_for_each_prefetching_exception(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::size_t prefetch_distance_factor = 2;
    std::vector<double> c(10007, 1.0);

    std::vector<std::size_t> range(10007);
    std::iota(range.begin(), range.end(), 0);

    auto ctx = pika::parallel::util::make_prefetcher_context(
        range.begin(), range.end(), prefetch_distance_factor, c);

    bool caught_exception = false;
    try
    {
        pika::for_each(policy, ctx.begin(), ctx.end(),
            [](std::size_t) { throw std::runtime_error("test"); });

        PIKA_TEST(false);
    }
    catch (pika::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_for_each_prefetching_exception_async(ExPolicy p, IteratorTag)
{
    std::size_t prefetch_distance_factor = 2;
    std::vector<double> c(10007, 1.0);

    std::vector<std::size_t> range(10007);
    std::iota(range.begin(), range.end(), 0);

    auto ctx = pika::parallel::util::make_prefetcher_context(
        range.begin(), range.end(), prefetch_distance_factor, c);

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = pika::for_each(p, ctx.begin(), ctx.end(),
            [](std::size_t) { throw std::runtime_error("test"); });
        returned_from_algorithm = true;
        f.get();

        PIKA_TEST(false);
    }
    catch (pika::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(p, e);
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
    PIKA_TEST(returned_from_algorithm);
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_for_each_prefetching_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::size_t prefetch_distance_factor = 2;
    std::vector<double> c(10007, 1.0);

    std::vector<std::size_t> range(10007);
    std::iota(range.begin(), range.end(), 0);

    auto ctx = pika::parallel::util::make_prefetcher_context(
        range.begin(), range.end(), prefetch_distance_factor, c);

    bool caught_exception = false;
    try
    {
        pika::for_each(policy, ctx.begin(), ctx.end(),
            [](std::size_t) { throw std::bad_alloc(); });

        PIKA_TEST(false);
    }
    catch (std::bad_alloc const&)
    {
        caught_exception = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_for_each_prefetching_bad_alloc_async(ExPolicy p, IteratorTag)
{
    std::size_t prefetch_distance_factor = 2;
    std::vector<double> c(10007, 1.0);

    std::vector<std::size_t> range(10007);
    std::iota(range.begin(), range.end(), 0);

    auto ctx = pika::parallel::util::make_prefetcher_context(
        range.begin(), range.end(), prefetch_distance_factor, c);

    bool caught_exception = false;
    bool returned_from_algorithm = false;

    try
    {
        auto f = pika::for_each(p, ctx.begin(), ctx.end(),
            [](std::size_t) { throw std::bad_alloc(); });
        returned_from_algorithm = true;
        f.get();

        PIKA_TEST(false);
    }
    catch (std::bad_alloc const&)
    {
        caught_exception = true;
    }
    catch (...)
    {
        PIKA_TEST(false);
    }

    PIKA_TEST(caught_exception);
    PIKA_TEST(returned_from_algorithm);
}
