//  Copyright (c) 2018 Christopher Ogle
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/init.hpp>
#include <pika/iterator_support/tests/iter_sent.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/container_algorithms/fill.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
void test_fill_sent()
{
    std::vector<std::size_t> c(200);
    std::iota(std::begin(c), std::end(c), std::rand());

    pika::ranges::fill(
        std::begin(c), sentinel<std::size_t>{*(std::begin(c) + 100)}, 10);

    // verify values
    std::size_t count = 0;
    std::for_each(
        std::begin(c), std::begin(c) + 100, [&count](std::size_t v) -> void {
            PIKA_TEST_EQ(v, std::size_t(10));
            ++count;
        });

    PIKA_TEST_EQ(count, (size_t) 100);
}

template <typename ExPolicy>
void test_fill_sent(ExPolicy policy)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(200);
    std::iota(std::begin(c), std::end(c), std::rand());

    pika::ranges::fill(policy, std::begin(c),
        sentinel<std::size_t>{*(std::begin(c) + 100)}, 10);

    // verify values
    std::size_t count = 0;
    std::for_each(
        std::begin(c), std::begin(c) + 100, [&count](std::size_t v) -> void {
            PIKA_TEST_EQ(v, std::size_t(10));
            ++count;
        });

    PIKA_TEST_EQ(count, (size_t) 100);
}

template <typename IteratorTag>
void test_fill(IteratorTag)
{
    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    pika::ranges::fill(c, 10);

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](std::size_t v) -> void {
        PIKA_TEST_EQ(v, std::size_t(10));
        ++count;
    });

    PIKA_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_fill(ExPolicy policy, IteratorTag)
{
    static_assert(pika::is_execution_policy<ExPolicy>::value,
        "pika::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    pika::ranges::fill(policy, c, 10);

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](std::size_t v) -> void {
        PIKA_TEST_EQ(v, std::size_t(10));
        ++count;
    });

    PIKA_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_fill_async(ExPolicy p, IteratorTag)
{
    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    pika::future<void> f = pika::ranges::fill(p, c, 10);
    f.wait();

    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c), [&count](std::size_t v) -> void {
        PIKA_TEST_EQ(v, std::size_t(10));
        ++count;
    });

    PIKA_TEST_EQ(count, c.size());
}

template <typename IteratorTag>
void test_fill()
{
    using namespace pika::execution;

    test_fill(IteratorTag());

    test_fill(seq, IteratorTag());
    test_fill(par, IteratorTag());
    test_fill(par_unseq, IteratorTag());

    test_fill_async(seq(task), IteratorTag());
    test_fill_async(par(task), IteratorTag());

    test_fill_sent();
    test_fill_sent(seq);
    test_fill_sent(par);
    test_fill_sent(par_unseq);
}

void fill_test()
{
    test_fill<std::random_access_iterator_tag>();
    test_fill<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////
int pika_main(pika::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    fill_test();
    return pika::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace pika::program_options;
    options_description desc_commandline(
        "Usage: " PIKA_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run");

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"pika.os_threads=all"};

    // Initialize and run pika
    pika::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    PIKA_TEST_EQ_MSG(pika::init(pika_main, argc, argv, init_args), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
