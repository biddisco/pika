//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file pika/runtime/threads/thread_data_fwd.hpp

#pragma once

#include <pika/config.hpp>
#include <pika/coroutines/coroutine_fwd.hpp>
#include <pika/coroutines/thread_enums.hpp>
#include <pika/coroutines/thread_id_type.hpp>
#include <pika/functional/function.hpp>
#include <pika/functional/unique_function.hpp>
#include <pika/modules/errors.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#if defined(PIKA_HAVE_APEX)
namespace pika { namespace util { namespace external_timer {
    struct task_wrapper;
}}}    // namespace pika::util::external_timer
#endif

namespace pika { namespace threads {

    class thread_data;
    class thread_data_stackful;
    class thread_data_stackless;

    namespace policies {
        struct scheduler_base;
    }
    class PIKA_EXPORT thread_pool_base;

    /// \cond NOINTERNAL
    using thread_id_ref_type = thread_id_ref;
    using thread_id_type = thread_id;

    using coroutine_type = coroutines::coroutine;
    using stackless_coroutine_type = coroutines::stackless_coroutine;

    using thread_result_type = std::pair<thread_schedule_state, thread_id_type>;
    using thread_arg_type = thread_restart_state;

    using thread_function_sig = thread_result_type(thread_arg_type);
    using thread_function_type =
        util::unique_function_nonser<thread_function_sig>;

    using thread_self = coroutines::detail::coroutine_self;
    using thread_self_impl_type = coroutines::detail::coroutine_impl;

#if defined(PIKA_HAVE_APEX)
    PIKA_EXPORT std::shared_ptr<pika::util::external_timer::task_wrapper>
    get_self_timer_data(void);
    PIKA_EXPORT void set_self_timer_data(
        std::shared_ptr<pika::util::external_timer::task_wrapper> data);
#endif
    /// \endcond
}}    // namespace pika::threads
