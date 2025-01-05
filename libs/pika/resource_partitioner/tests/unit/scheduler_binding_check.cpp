//  Copyright (c) 2020 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Test that loops over each core assigned to the program and launches
// tasks bound to that core incrementally.
// Tasks should always report the right core number when they run.

#include <pika/debugging/print.hpp>
#include <pika/execution.hpp>
#include <pika/init.hpp>
#include <pika/modules/resource_partitioner.hpp>
#include <pika/modules/schedulers.hpp>
#include <pika/runtime.hpp>
#include <pika/testing.hpp>
#include <pika/thread.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;

namespace pika {
    // use <true>/<false> to enable/disable debug printing
    using sbc_print_on = enable_print<false>;
    static sbc_print_on deb_schbin("SCHBIND");
}    // namespace pika

// counts down on destruction
struct dec_counter
{
    explicit dec_counter(std::atomic<int>& counter)
      : counter_(counter)
    {
    }
    ~dec_counter() { --counter_; }
    //
    std::atomic<int>& counter_;
};

void threadLoop()
{
#if defined(PIKA_HAVE_VALGRIND)
    unsigned const iterations = 256;
#else
    unsigned const iterations = 2048;
#endif
    std::atomic<int> count_down(iterations);

    auto f = [&count_down](std::size_t iteration, std::size_t thread_expected) {
        dec_counter dec(count_down);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::size_t thread_actual = pika::get_worker_thread_num();
        pika::deb_schbin.debug(ffmt<s10>("Iteration"),
            ffmt<dec4>(iteration),
            ffmt<s20>("Running on thread"), thread_actual,
            ffmt<s10>("Expected"), thread_expected);
        PIKA_TEST_EQ(thread_actual, thread_expected);
    };

    std::size_t threads = pika::get_num_worker_threads();
    // launch tasks on threads using numbering 0,1,2,3...0,1,2,3
    for (std::size_t i = 0; i < iterations; ++i)
    {
        auto sched = ex::with_hint(ex::with_stacksize(ex::with_priority(ex::thread_pool_scheduler{},
                                                          pika::execution::thread_priority::bound),
                                       pika::execution::thread_stacksize::default_),
            pika::execution::thread_schedule_hint(std::int16_t(i % threads)));
        tt::sync_wait(ex::just(i, i % threads) | ex::continues_on(sched) | ex::then(f));
    }

    do {
        pika::this_thread::yield();
        pika::deb_schbin.debug(
            ffmt<s15>("count_down"), ffmt<dec4>(count_down));
    } while (count_down > 0);

    pika::deb_schbin.debug(
        ffmt<s15>("complete"), ffmt<dec4>(count_down));
    PIKA_TEST_EQ(count_down.load(), 0);
}

int pika_main()
{
    auto const current = pika::threads::detail::get_self_id_data()->get_scheduler_base();
    std::cout << "Scheduler is " << current->get_description() << std::endl;
    if (std::string("core-shared_priority_queue_scheduler") != current->get_description())
    {
        std::cout << "The scheduler might not work properly " << std::endl;
    }

    threadLoop();

    pika::finalize();
    pika::deb_schbin.debug(ffmt<s15>("Finalized"));
    return 0;
}

int main(int argc, char* argv[])
{
    pika::init_params init_args;

    init_args.rp_callback = [](auto& rp, pika::program_options::variables_map const&) {
        // setup the default pool with a numa/binding aware scheduler
        rp.create_thread_pool("default", pika::resource::scheduling_policy::shared_priority,
            pika::threads::scheduler_mode::default_mode);
    };

    return pika::init(pika_main, argc, argv, init_args);
}
