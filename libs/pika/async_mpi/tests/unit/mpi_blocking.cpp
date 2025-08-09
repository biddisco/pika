//  Copyright (c) 2023 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <pika/assert.hpp>
#include <pika/command_line_handling/get_env_var_as.hpp>
#include <pika/debugging/print.hpp>
#include <pika/execution.hpp>
#include <pika/execution_base/this_thread.hpp>
#include <pika/init.hpp>
#include <pika/mpi.hpp>
#include <pika/program_options.hpp>
#include <pika/synchronization/counting_semaphore.hpp>
#include <pika/testing.hpp>
#include <pika/thread.hpp>
//
#include <boost/lockfree/stack.hpp>
#include <fmt/format.h>
#include <fmt/printf.h>
//
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

/*
 * This test exercises the MPI sender/receiver capabilities
 * blocking mode
*/

using pika::program_options::options_description;
using pika::program_options::value;
using pika::program_options::variables_map;

using namespace pika::debug::detail;

namespace ex = pika::execution::experimental;
namespace tt = pika::this_thread::experimental;
namespace mpix = pika::mpi::experimental;

// ------------------------------------------------------------
// a debug level of zero disables messages with a priority>0
// a debug level of N shows messages with priority<N
template <int Level>
inline constexpr print_threshold<Level, 1> blk_deb("MPI_BLK");

// ------------------------------------------------------------
std::atomic<std::uint32_t> message_buffers_size_{0};
// ------------------------------------------------------------
struct message_buffer
{
    std::vector<std::uint64_t> data_;
    message_buffer(std::size_t size, std::size_t numranks, std::size_t rank)
      : data_(size * numranks, rank)
    {
    }
    std::uint64_t* data() { return data_.data(); }
};

// ------------------------------------------------------------
message_buffer* get_msg_buffer(std::size_t size, std::size_t numranks, std::size_t rank)
{
    message_buffer* buffer;
    // construct our buffer object in that space
    buffer = new message_buffer(size, numranks, rank);
    message_buffers_size_++;
    blk_deb<6>.debug(str<>("message_buffers"), std::uint32_t(message_buffers_size_.load()));
    return buffer;
}

// ------------------------------------------------------------
void release_msg_buffer(message_buffer* buffer)
{
    --message_buffers_size_;
    delete buffer;
    blk_deb<6>.debug(str<>("message_buffers"), std::uint32_t(message_buffers_size_.load()));
}

// ------------------------------------------------------------
// this is called on a pika thread after the runtime starts up
int pika_main(pika::program_options::variables_map& vm)
{
    // --------------------------
    // Enable polling on mpi pool, install an error handler
    // --------------------------
    mpix::enable_polling enable_polling(mpix::exception_mode::install_handler);
    //
    std::int32_t rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // MPI_Comm barrier_com;
    // auto res = MPI_Comm_dup(MPI_COMM_WORLD, &barrier_com);
    // PIKA_TEST_EQ(res, MPI_SUCCESS);

    // --------------------------
    // Get user options/flags
    // --------------------------
    std::uint32_t const iterations = vm["iterations"].as<std::uint32_t>();
    std::uint32_t message_count = vm["message-bytes"].as<std::uint32_t>();

    // we want to test blocking mode
    auto mode = pika::detail::to_underlying(mpix::detail::handler_method::blocking);

    for (uint32_t i = 0; i < iterations; ++i)
    {
        // each iteration every rank contributes 'rank' to the total
        // we will sum 0 + 1 + 2 + 3 + 4 + 5 .... N, message_bytes times
        message_buffer* input = get_msg_buffer(message_count, size, rank);
        message_buffer* output = get_msg_buffer(message_count, size, rank);

        auto s1 = ex::just(input->data(), output->data(), message_count * size, MPI_UINT64_T,
                      MPI_SUM, MPI_COMM_WORLD) |
            mpix::transform_mpi(MPI_Iallreduce, mode)    //
            | ex::then([rank, size, message_count, input, output](/*int res*/) {
                  // total should be 0 + 1 + 2 ... + N-1 = N(N-1)/2
                  std::uint64_t expected = size * (size - 1) / 2;
                  bool ok = true;
                  for (uint32_t n = 0; n < message_count * rank; ++n)
                  {
                      PIKA_TEST_EQ(input->data_[n], static_cast<uint64_t>(rank));
                      ok = ok && (input->data_[n] == static_cast<uint64_t>(rank));

                      PIKA_TEST_EQ(output->data_[n], expected);
                      ok = ok && (output->data_[n] == expected);
                  }
                  release_msg_buffer(input);
                  release_msg_buffer(output);
                  return ok;
              })    //
            | ex::ensure_started();

        blk_deb<0>.debug(str<>("MPI_Iallreduce"), "begin wait");
        bool result = tt::sync_wait(std::move(s1));
        blk_deb<0>.debug(str<>("MPI_Iallreduce"), "end wait");
        PIKA_TEST_EQ(result, true);
    }

    auto s2 = ex::just(MPI_COMM_WORLD)               //
        | mpix::transform_mpi(MPI_Ibarrier, mode)    //
        | ex::ensure_started();
    blk_deb<0>.debug(str<>("MPI_Ibarrier"), "begin wait");
    tt::sync_wait(std::move(s2));
    blk_deb<0>.debug(str<>("MPI_Ibarrier"), "end wait");

    PIKA_TEST_EQ(message_buffers_size_.load(), 0u);
    pika::finalize();
    return EXIT_SUCCESS;
}

//----------------------------------------------------------------------------
// the normal int main function that is called at startup and runs on an OS
// thread the user must call pika::init to start the pika runtime which
// will execute pika_main on a pika thread
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description cmdline("usage: " PIKA_APPLICATION_STRING " [options]");

    // clang-format off
    cmdline.add_options()("iterations",
        value<std::uint32_t>()->default_value(5000),
        "number of iterations to test");

    cmdline.add_options()("message-bytes",
        pika::program_options::value<std::uint32_t>()->default_value(64),
        "Specify the buffer size to use (total = message-bytes * numranks)");
    // clang-format on

    // -----------------
    // process command line options early so we can use them for mpi_init
    namespace po = pika::program_options;
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(cmdline).allow_unregistered().run(), vm);
    po::notify(vm);

    // -----------------
    // Init MPI
    int provided, preferred = mpix::get_preferred_thread_mode();
    MPI_Init_thread(&argc, &argv, preferred, &provided);
    if (provided != preferred)
    {
        blk_deb<0>.error(str<>("Caution"), "Provided MPI is not as requested");
    }

    // -----------------
    // Initialize and run pika.
    pika::init_params init_args;
    init_args.desc_cmdline = cmdline;
    auto result = pika::init(pika_main, argc, argv, init_args);
    PIKA_TEST_EQ(result, 0);

    // -----------------
    // Finalize MPI
    MPI_Finalize();
    blk_deb<0>.debug(str<>("Exit status"), result);
    return result;
}
