//  Copyright (c) 2006, Giovanni P. Deretta
//  Copyright (c) 2011, Bryce Adelstein-Lelbach
//
//  This code may be used under either of the following two licences:
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE. OF SUCH DAMAGE.
//
//  Or:
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>

// include unist.d conditionally to check for POSIX version. Not all OSs have the
// unistd header...
#if defined(PIKA_HAVE_UNISTD_H)
# include <unistd.h>
#endif

#if defined(_POSIX_VERSION)
/**
 * Most of these utilities are really pure C++, but they are useful
 * only on posix systems.
 */
# include <fmt/format.h>

# include <cerrno>
# include <cstddef>
# include <cstdlib>
# include <cstring>
# include <stdexcept>
# include <string>

# if defined(_POSIX_MAPPED_FILES) && _POSIX_MAPPED_FILES > 0
#  include <errno.h>
#  include <sys/mman.h>
#  include <sys/param.h>

#  include <stdexcept>
# endif

# if defined(__FreeBSD__)
#  include <sys/param.h>
#  define PIKA_EXEC_PAGESIZE PAGE_SIZE
# endif

# if defined(__APPLE__)
#  include <unistd.h>
#  define PIKA_EXEC_PAGESIZE static_cast<std::size_t>(sysconf(_SC_PAGESIZE))
# endif

# if !defined(PIKA_EXEC_PAGESIZE)
#  define PIKA_EXEC_PAGESIZE EXEC_PAGESIZE
# endif

/**
 * Stack allocation routines and trampolines for setcontext
 */
namespace pika::threads::coroutines::detail::posix {
    PIKA_EXPORT extern bool use_guard_pages;

    inline void check_stack_size(std::size_t size)
    {
        if (0 != (size % PIKA_EXEC_PAGESIZE))
        {
            throw std::runtime_error(fmt::format(
                "stack size of {} is not page aligned, page size is {}", size, PIKA_EXEC_PAGESIZE));
        }

        if (0 >= size)
        {
            throw std::runtime_error(fmt::format("stack size of {} is invalid", size));
        }
    }

# if defined(PIKA_HAVE_THREAD_STACK_MMAP) && defined(_POSIX_MAPPED_FILES) && _POSIX_MAPPED_FILES > 0

    inline void* to_stack_with_guard_page(void* stack)
    {
        if (use_guard_pages)
        {
            // NOLINTBEGIN(bugprone-multi-level-implicit-pointer-conversion)
            return static_cast<void*>(
                static_cast<void**>(stack) - (PIKA_EXEC_PAGESIZE / sizeof(void*)));
            // NOLINTEND(bugprone-multi-level-implicit-pointer-conversion)
        }

        return stack;
    }

    inline void* to_stack_without_guard_page(void* stack)
    {
        if (use_guard_pages)
        {
            // NOLINTBEGIN(bugprone-multi-level-implicit-pointer-conversion)
            return static_cast<void*>(
                static_cast<void**>(stack) + (PIKA_EXEC_PAGESIZE / sizeof(void*)));
            // NOLINTEND(bugprone-multi-level-implicit-pointer-conversion)
        }

        return stack;
    }

    inline void add_guard_page(void* stack)
    {
        if (use_guard_pages)
        {
            int r = ::mprotect(stack, PIKA_EXEC_PAGESIZE, PROT_NONE);
            if (r != 0)
            {
                std::string error_message = "mprotect on a stack allocation failed with errno " +
                    std::to_string(errno) + " (" + std::strerror(errno) + ")";
                throw std::runtime_error(error_message);
            }
        }
    }

    inline std::size_t stack_size_with_guard_page(std::size_t size)
    {
        if (use_guard_pages) { return size + PIKA_EXEC_PAGESIZE; }

        return size;
    }

    inline void* alloc_stack(std::size_t size)
    {
        check_stack_size(size);

        void* real_stack = ::mmap(nullptr, stack_size_with_guard_page(size), PROT_READ | PROT_WRITE,
#  if defined(__APPLE__)
            MAP_PRIVATE | MAP_ANON | MAP_NORESERVE,
#  elif defined(__FreeBSD__)
            MAP_PRIVATE | MAP_ANON,
#  else
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
#  endif
            -1, 0);

        if (real_stack == MAP_FAILED)
        {
            if (ENOMEM == errno && use_guard_pages)
            {
                char const* error_message =
                    "mmap failed to allocate thread stack due to insufficient resources. "
                    "Increasing /proc/sys/vm/max_map_count or disabling guard pages with the "
                    "configuration option pika.stacks.use_guard_pages=0 may reduce memory "
                    "consumption.";
                throw std::runtime_error(error_message);
            }

            std::string error_message = "mmap failed to allocate thread stack with errno " +
                std::to_string(errno) + " (" + std::strerror(errno) + ")";
            throw std::runtime_error(error_message);
        }

        add_guard_page(real_stack);
        return to_stack_without_guard_page(real_stack);
    }

    inline void watermark_stack(void* stack, std::size_t size)
    {
        PIKA_ASSERT(size >= PIKA_EXEC_PAGESIZE);

        // Fill the bottom 8 bytes of the first page with 1s.
        void** watermark =
            static_cast<void**>(stack) + ((size - PIKA_EXEC_PAGESIZE) / sizeof(void*));
        *watermark = reinterpret_cast<void*>(0xDEAD'BEEF'DEAD'BEEFull);
    }

    inline bool reset_stack(void* stack, std::size_t size)
    {
        void** watermark =
            static_cast<void**>(stack) + ((size - PIKA_EXEC_PAGESIZE) / sizeof(void*));

        // If the watermark has been overwritten, then we've gone past the first
        // page.
        if ((reinterpret_cast<void*>(0xDEAD'BEEF'DEAD'BEEFull)) != *watermark)
        {
            // We never free up the first page, as it's initialized only when the
            // stack is created.
            int r = ::madvise(stack, size - PIKA_EXEC_PAGESIZE, MADV_DONTNEED);
            if (r != 0)
            {
                std::string error_message = "madvise on a stack allocation failed with errno " +
                    std::to_string(errno) + " (" + std::strerror(errno) + ")";
                throw std::runtime_error(error_message);
            }
            return true;
        }

        return false;
    }

    inline void free_stack(void* stack, std::size_t size)
    {
        int r = ::munmap(to_stack_with_guard_page(stack), stack_size_with_guard_page(size));
        if (r != 0)
        {
            std::string error_message = "munmap failed to deallocate thread stack with errno " +
                std::to_string(errno) + " (" + std::strerror(errno) + ")";
            throw std::runtime_error(error_message);
        }
    }

# else    // non-mmap()

    //this should be a fine default.
    static std::size_t const stack_alignment = sizeof(void*) > 16 ? sizeof(void*) : 16;

    struct stack_aligner
    {
        alignas(stack_alignment) char dummy[stack_alignment];
    };

    /**
     * Stack allocator and deleter functions.
     * Better implementations are possible using
     * mmap (might be required on some systems) and/or
     * using a pooling allocator.
     * NOTE: the SuSv3 documentation explicitly allows
     * the use of malloc to allocate stacks for makectx.
     * We use new/delete for guaranteed alignment.
     */
    inline void* alloc_stack(std::size_t size)
    {
        check_stack_size(size);

        return new stack_aligner[size / sizeof(stack_aligner)];
    }

    inline void watermark_stack(void* /* stack */, std::size_t /* size */) {}    // no-op

    inline bool reset_stack(void* /* stack */, std::size_t /* size */) { return false; }

    inline void free_stack(void* stack, std::size_t /* size */)
    {
        delete[] static_cast<stack_aligner*>(stack);
    }

# endif    // non-mmap() implementation of alloc_stack()/free_stack()

    /**
     * The splitter is needed for 64 bit systems.
     * \note The current implementation does NOT use
     * (for debug reasons).
     * Thus it is not 64 bit clean.
     * Use it for 64 bits systems.
     */
    template <typename T>
    union splitter
    {
        int int_[2];
        T* ptr;

        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
        splitter(int first_, int second_)
        {
            int_[0] = first_;
            int_[1] = second_;
        }

        int first() { return int_[0]; }

        int second() { return int_[1]; }

        splitter(T* ptr_)
          : ptr(ptr_)
        {
        }

        void operator()() { (*ptr)(); }
    };

    template <typename T>
    inline void trampoline_split(int first, int second)
    {
        splitter<T> split(first, second);
        split();
    }

    template <typename T>
    inline void trampoline(void* fun)
    {
        (*static_cast<T*>(fun))();
    }
}    // namespace pika::threads::coroutines::detail::posix

#else
# error This header can only be included when compiling for posix systems.
#endif
