//  Copyright (c) 2006, Giovanni P. Deretta
//  Copyright (c) 2007 Robert Perricone
//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2013-2016 Thomas Heller
//  Copyright (c) 2017 Christopher Taylor
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(__linux) || defined(linux) || defined(__linux__) || defined(__FreeBSD__)

# include <pika/config.hpp>
# include <pika/assert.hpp>
# include <pika/coroutines/detail/get_stack_pointer.hpp>
# include <pika/coroutines/detail/posix_utility.hpp>
# include <pika/coroutines/detail/swap_context.hpp>
# include <pika/util/get_and_reset_value.hpp>

# include <fmt/format.h>

# include <atomic>
# include <cstddef>
# include <cstdint>
# include <cstdlib>
# include <stdexcept>
# include <sys/param.h>

# if defined(PIKA_HAVE_VALGRIND)
#  if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#   if defined(PIKA_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#    pragma GCC diagnostic push
#   endif
#   pragma GCC diagnostic ignored "-Wpointer-arith"
#  endif
#  include <valgrind/valgrind.h>
# endif

# if defined(PIKA_HAVE_ADDRESS_SANITIZER)
#  include <sanitizer/asan_interface.h>
# endif

/*
 * Defining PIKA_COROUTINE_NO_SEPARATE_CALL_SITES will disable separate
 * invoke, and yield swap_context functions. Separate calls sites
 * increase performance by 25% at least on P4 for invoke+yield back loops
 * at the cost of a slightly higher instruction cache use and is thus enabled by
 * default.
 */

# if defined(__x86_64__)
extern "C" void swapcontext_stack(void***, void**) noexcept;
extern "C" void swapcontext_stack2(void***, void**) noexcept;
# else
extern "C" void swapcontext_stack(void***, void**) noexcept __attribute((regparm(2)));
extern "C" void swapcontext_stack2(void***, void**) noexcept __attribute((regparm(2)));
# endif

///////////////////////////////////////////////////////////////////////////////
namespace pika::threads::coroutines {
    namespace detail {
        // some platforms need special preparation of the main thread
        struct prepare_main_thread
        {
            constexpr prepare_main_thread() {}
        };
    }    // namespace detail

    namespace detail::lx {
        template <typename T>
        PIKA_FORCEINLINE void trampoline(void* fun)
        {
            (*static_cast<T*>(fun))();
            std::abort();
        }

        template <typename CoroutineImpl>
        class x86_linux_context_impl;

        class x86_linux_context_impl_base : detail::context_impl_base
        {
        public:
            x86_linux_context_impl_base()
              : m_sp(nullptr)
# if defined(PIKA_HAVE_ADDRESS_SANITIZER)
              , asan_fake_stack(nullptr)
              , asan_stack_bottom(nullptr)
              , asan_stack_size(0)
# endif
            {
            }

            void prefetch() const
            {
# if defined(__x86_64__)
                PIKA_ASSERT(sizeof(void*) == 8);
# else
                PIKA_ASSERT(sizeof(void*) == 4);
# endif

                // Silence this warning since the explicit cast is ignored until clang-tidy version
                // 19 (https://github.com/llvm/llvm-project/pull/94524)
                // NOLINTBEGIN(bugprone-multi-level-implicit-pointer-conversion)
                __builtin_prefetch(static_cast<void*>(m_sp), 1, 3);
                __builtin_prefetch(static_cast<void*>(m_sp), 0, 3);
                __builtin_prefetch(
                    static_cast<void*>(static_cast<void**>(m_sp) + 64 / sizeof(void*)), 1, 3);
                __builtin_prefetch(
                    static_cast<void*>(static_cast<void**>(m_sp) + 64 / sizeof(void*)), 0, 3);
# if !defined(__x86_64__)
                __builtin_prefetch(
                    static_cast<void*>(static_cast<void**>(m_sp) + 32 / sizeof(void*)), 1, 3);
                __builtin_prefetch(
                    static_cast<void*>(static_cast<void**>(m_sp) + 32 / sizeof(void*)), 0, 3);
                __builtin_prefetch(
                    static_cast<void*>(static_cast<void**>(m_sp) - 32 / sizeof(void*)), 1, 3);
                __builtin_prefetch(
                    static_cast<void*>(static_cast<void**>(m_sp) - 32 / sizeof(void*)), 0, 3);
# endif
                __builtin_prefetch(
                    static_cast<void*>(static_cast<void**>(m_sp) - 64 / sizeof(void*)), 1, 3);
                __builtin_prefetch(
                    static_cast<void*>(static_cast<void**>(m_sp) - 64 / sizeof(void*)), 0, 3);
                // NOLINTEND(bugprone-multi-level-implicit-pointer-conversion)
            }

            /**
             * Free function. Saves the current context in \p from
             * and restores the context in \p to.
             * \note This function is found by ADL.
             */
            friend void swap_context(x86_linux_context_impl_base& from,
                x86_linux_context_impl_base const& to, default_hint);

            friend void swap_context(x86_linux_context_impl_base& from,
                x86_linux_context_impl_base const& to, yield_hint);

# if defined(PIKA_HAVE_ADDRESS_SANITIZER)
            void start_switch_fiber(void** fake_stack)
            {
                __sanitizer_start_switch_fiber(fake_stack, asan_stack_bottom, asan_stack_size);
            }
            void start_yield_fiber(void** fake_stack, x86_linux_context_impl_base& caller)
            {
                __sanitizer_start_switch_fiber(
                    fake_stack, caller.asan_stack_bottom, caller.asan_stack_size);
            }
            void finish_yield_fiber(void* fake_stack)
            {
                __sanitizer_finish_switch_fiber(fake_stack, &asan_stack_bottom, &asan_stack_size);
            }
            void finish_switch_fiber(void* fake_stack, x86_linux_context_impl_base& caller)
            {
                __sanitizer_finish_switch_fiber(
                    fake_stack, &caller.asan_stack_bottom, &caller.asan_stack_size);
            }
# endif

        protected:
            void** m_sp;

# if defined(PIKA_HAVE_ADDRESS_SANITIZER)
        public:
            void* asan_fake_stack;
            void const* asan_stack_bottom;
            std::size_t asan_stack_size;
# endif
        };

        template <typename CoroutineImpl>
        class x86_linux_context_impl : public x86_linux_context_impl_base
        {
        public:
            enum
            {
                default_stack_size = 4 * PIKA_EXEC_PAGESIZE
            };

            using context_impl_base = x86_linux_context_impl_base;

            /**
             * Create a context that on restore invokes Functor on
             *  a new stack. The stack size can be optionally specified.
             */
            explicit x86_linux_context_impl(std::ptrdiff_t stack_size = -1)
              : m_stack_size(
                    stack_size == -1 ? static_cast<std::ptrdiff_t>(default_stack_size) : stack_size)
              , m_stack(nullptr)
            {
            }

            void init()
            {
                if (m_stack != nullptr) return;

                m_stack = posix::alloc_stack(static_cast<std::size_t>(m_stack_size));
                if (m_stack == nullptr)
                {
                    throw std::runtime_error("could not allocate memory for stack");
                }

                posix::watermark_stack(m_stack, static_cast<std::size_t>(m_stack_size));

                using fun = void(void*);
                fun* funp = trampoline<CoroutineImpl>;

                m_sp = (static_cast<void**>(m_stack) +
                           static_cast<std::size_t>(m_stack_size) / sizeof(void*)) -
                    context_size;

                m_sp[cb_idx] = this;
                m_sp[funp_idx] = reinterpret_cast<void*>(funp);

# if defined(PIKA_HAVE_VALGRIND) && !defined(NVALGRIND)
                {
                    void* eos = static_cast<char*>(m_stack) + m_stack_size;
                    m_sp[valgrind_id_idx] =
                        reinterpret_cast<void*>(VALGRIND_STACK_REGISTER(m_stack, eos));
                }
# endif
# if defined(PIKA_HAVE_ADDRESS_SANITIZER)
                asan_stack_size = m_stack_size;
                asan_stack_bottom = const_cast<void const*>(m_stack);
# endif
            }

            ~x86_linux_context_impl()
            {
                if (m_stack)
                {
# if defined(PIKA_HAVE_VALGRIND) && !defined(NVALGRIND)
                    VALGRIND_STACK_DEREGISTER(reinterpret_cast<std::size_t>(m_sp[valgrind_id_idx]));
# endif
                    posix::free_stack(m_stack, static_cast<std::size_t>(m_stack_size));
                }
            }

            // Return the size of the reserved stack address space.
            std::ptrdiff_t get_stacksize() const { return m_stack_size; }

            void reset_stack()
            {
                PIKA_ASSERT(m_stack);
                if (posix::reset_stack(m_stack, static_cast<std::size_t>(m_stack_size)))
                {
# if defined(PIKA_HAVE_COROUTINE_COUNTERS)
                    increment_stack_unbind_count();
# endif
                }
            }

            void rebind_stack()
            {
                PIKA_ASSERT(m_stack);
# if defined(PIKA_HAVE_COROUTINE_COUNTERS)
                increment_stack_recycle_count();
# endif

                // On rebind, we initialize our stack to ensure a virgin stack
                m_sp = (static_cast<void**>(m_stack) +
                           static_cast<std::size_t>(m_stack_size) / sizeof(void*)) -
                    context_size;

                using fun = void(void*);
                fun* funp = trampoline<CoroutineImpl>;
                m_sp[cb_idx] = this;
                m_sp[funp_idx] = reinterpret_cast<void*>(funp);
# if defined(PIKA_HAVE_ADDRESS_SANITIZER)
                asan_stack_size = m_stack_size;
                asan_stack_bottom = const_cast<void const*>(m_stack);
# endif
            }

            std::ptrdiff_t get_available_stack_space()
            {
                return get_stack_ptr() - reinterpret_cast<std::size_t>(m_stack) - context_size;
            }

            using counter_type = std::atomic<std::int64_t>;

# if defined(PIKA_HAVE_COROUTINE_COUNTERS)
        private:
            static counter_type& get_stack_unbind_counter()
            {
                static counter_type counter(0);
                return counter;
            }

            static counter_type& get_stack_recycle_counter()
            {
                static counter_type counter(0);
                return counter;
            }

            static std::uint64_t increment_stack_unbind_count()
            {
                return ++get_stack_unbind_counter();
            }

            static std::uint64_t increment_stack_recycle_count()
            {
                return ++get_stack_recycle_counter();
            }

        public:
            static std::uint64_t get_stack_unbind_count(bool reset)
            {
                return ::pika::detail::get_and_reset_value(get_stack_unbind_counter(), reset);
            }

            static std::uint64_t get_stack_recycle_count(bool reset)
            {
                return ::pika::detail::get_and_reset_value(get_stack_recycle_counter(), reset);
            }
# endif

            friend void swap_context(x86_linux_context_impl_base& from,
                x86_linux_context_impl_base const& to, default_hint);

            friend void swap_context(x86_linux_context_impl_base& from,
                x86_linux_context_impl_base const& to, yield_hint);

        private:
# if defined(__x86_64__)
            /** structure of context_data:
             * 11: additional alignment (or valgrind_id if enabled)
             * 10: parm 0 of trampoline
             * 9:  dummy return address for trampoline
             * 8:  return addr (here: start addr)
             * 7:  rbp
             * 6:  rbx
             * 5:  rsi
             * 4:  rdi
             * 3:  r12
             * 2:  r13
             * 1:  r14
             * 0:  r15
             **/
#  if defined(PIKA_HAVE_VALGRIND) && !defined(NVALGRIND)
            static std::size_t const valgrind_id_idx = 11;
#  endif

            static std::size_t const context_size = 12;
            static std::size_t const cb_idx = 10;
            static std::size_t const funp_idx = 8;
# else
            /** structure of context_data:
             * 7: valgrind_id (if enabled)
             * 6: parm 0 of trampoline
             * 5: dummy return address for trampoline
             * 4: return addr (here: start addr)
             * 3: ebp
             * 2: ebx
             * 1: esi
             * 0: edi
             **/
#  if defined(PIKA_HAVE_VALGRIND) && !defined(NVALGRIND)
            static std::size_t const context_size = 8;
            static std::size_t const valgrind_id_idx = 7;
#  else
            static std::size_t const context_size = 7;
#  endif

            static std::size_t const cb_idx = 6;
            static std::size_t const funp_idx = 4;
# endif

            std::ptrdiff_t m_stack_size;
            void* m_stack;
        };

        /**
         * Free function. Saves the current context in \p from
         * and restores the context in \p to.
         * \note This function is found by ADL.
         */
        inline void swap_context(
            x86_linux_context_impl_base& from, x86_linux_context_impl_base const& to, default_hint)
        {
            //        PIKA_ASSERT(*(void**)to.m_stack == (void*)~0);
            to.prefetch();
            swapcontext_stack(&from.m_sp, to.m_sp);
        }

        inline void swap_context(
            x86_linux_context_impl_base& from, x86_linux_context_impl_base const& to, yield_hint)
        {
            //        PIKA_ASSERT(*(void**)from.m_stack == (void*)~0);
            to.prefetch();
# if !defined(PIKA_COROUTINE_NO_SEPARATE_CALL_SITES)
            swapcontext_stack2(&from.m_sp, to.m_sp);
# else
            swapcontext_stack(&from.m_sp, to.m_sp);
# endif
        }
    }    // namespace detail::lx
}    // namespace pika::threads::coroutines

# if defined(PIKA_HAVE_VALGRIND)
#  if defined(__GNUG__) && !defined(__INTEL_COMPILER)
#   if defined(PIKA_GCC_DIAGNOSTIC_PRAGMA_CONTEXTS)
#    pragma GCC diagnostic pop
#   endif
#  endif
# endif

#else

# error This header can only be included when compiling for linux systems.

#endif
