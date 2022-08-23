//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/for_each.hpp

#pragma once

#if defined(DOXYGEN)
namespace pika {
    /// Applies \a f to the result of dereferencing every iterator in the
    /// range [first, last).
    ///
    /// \note   Complexity: Applies \a f exactly \a last - \a first times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// \tparam InIter      The type of the source begin and end iterator used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). F must meet requirements of
    ///                     \a MoveConstructible.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// \returns            \a f.
    template <typename InIter, typename F>
    F for_each(InIter first, InIter last, F&& f);

    /// Applies \a f to the result of dereferencing every iterator in the
    /// range [first, last).
    ///
    /// \note   Complexity: Applies \a f exactly \a last - \a first times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// Unlike its sequential form, the parallel overload of
    /// \a for_each does not return a copy of its \a Function parameter,
    /// since parallelization may not permit efficient state
    /// accumulation.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam FwdIte      The type of the source begin and end iterator used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a for_each requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a for_each algorithm returns a
    ///           \a pika::future<void> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns void otherwise.
    template <typename ExPolicy, typename FwdIter, typename F>
    typename util::detail::algorithm_result<ExPolicy, void>::type for_each(
        ExPolicy&& policy, FwdIter first, FwdIter last, F&& f);

    /// Applies \a f to the result of dereferencing every iterator in the range
    /// [first, first + count), starting from first and proceeding to
    /// first + count - 1.
    ///
    /// \note   Complexity: Applies \a f exactly \a count times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// \tparam InIter      The type of the source begin and end iterator used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an input iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). F must meet requirements of
    ///                     \a MoveConstructible.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// \returns            \a first + \a count for non-negative values of
    ///                     \a count and \a first for negative values.
    template <typename InIter, typename Size, typename F>
    InIter for_each_n(InIter first, Size count, F&& f);

    /// Applies \a f to the result of dereferencing every iterator in the range
    /// [first, first + count), starting from first and proceeding to
    /// first + count - 1.
    ///
    /// \note   Complexity: Applies \a f exactly \a count times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// Unlike its sequential form, the parallel overload of
    /// \a for_each does not return a copy of its \a Function parameter,
    /// since parallelization may not permit efficient state
    /// accumulation.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a for_each requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a for_each_n algorithm returns a
    ///           \a pika::future<FwdIter> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a FwdIter
    ///           otherwise.
    ///           It returns \a first + \a count for non-negative values of
    ///           \a count and \a first for negative values.
    template <typename ExPolicy, typename FwdIter, typename Size, typename F>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type for_each_n(
        ExPolicy&& policy, FwdIter first, Size count, F&& f);

}    // namespace pika

#else

#include <pika/config.hpp>
#include <pika/algorithms/traits/projected.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/functional/detail/invoke.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/modules/execution.hpp>
#include <pika/modules/executors.hpp>
#include <pika/parallel/algorithms/detail/dispatch.hpp>
#include <pika/parallel/algorithms/detail/distance.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>
#include <pika/parallel/util/foreach_partitioner.hpp>
#include <pika/parallel/util/loop.hpp>
#include <pika/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

namespace pika::parallel::detail {
    ///////////////////////////////////////////////////////////////////////////
    // for_each_n
    /// \cond NOINTERNAL
    template <typename F, typename Proj = util::projection_identity>
    struct invoke_projected
    {
        F& f_;
        Proj& proj_;

        template <typename Iter>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr void operator()(Iter curr)
        {
            PIKA_INVOKE(f_, PIKA_INVOKE(proj_, *curr));
        }
    };

    template <typename F>
    struct invoke_projected<F, util::projection_identity>
    {
        PIKA_HOST_DEVICE constexpr invoke_projected(F& f) noexcept
          : f_(f)
        {
        }

        PIKA_HOST_DEVICE constexpr invoke_projected(
            F& f, util::projection_identity) noexcept
          : f_(f)
        {
        }

        F& f_;

        template <typename Iter>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr void operator()(Iter curr)
        {
            PIKA_INVOKE(f_, *curr);
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename F,
        typename Proj = util::projection_identity>
    struct for_each_iteration
    {
        using execution_policy_type = std::decay_t<ExPolicy>;
        using fun_type = std::decay_t<F>;
        using proj_type = Proj;

        fun_type f_;
        proj_type proj_;

        template <typename F_, typename Proj_>
        PIKA_HOST_DEVICE for_each_iteration(F_&& f, Proj_&& proj)
          : f_(PIKA_FORWARD(F_, f))
          , proj_(PIKA_FORWARD(Proj_, proj))
        {
        }

#if !defined(__NVCC__) && !defined(__CUDACC__)
        for_each_iteration(for_each_iteration const&) = default;
        for_each_iteration(for_each_iteration&&) = default;
#else
        PIKA_HOST_DEVICE for_each_iteration(for_each_iteration const& rhs)
          : f_(rhs.f_)
          , proj_(rhs.proj_)
        {
        }

        PIKA_HOST_DEVICE for_each_iteration(for_each_iteration&& rhs)
          : f_(PIKA_MOVE(rhs.f_))
          , proj_(PIKA_MOVE(rhs.proj_))
        {
        }
#endif
        for_each_iteration& operator=(for_each_iteration const&) = default;
        for_each_iteration& operator=(for_each_iteration&&) = default;

        template <typename Iter>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr void operator()(
            Iter part_begin, std::size_t part_size, std::size_t)
        {
            util::loop_n<execution_policy_type>(part_begin, part_size,
                invoke_projected<fun_type, proj_type>{f_, proj_});
        }
    };

    template <typename ExPolicy, typename F>
    struct for_each_iteration<ExPolicy, F, util::projection_identity>
    {
        using execution_policy_type = std::decay_t<ExPolicy>;
        using fun_type = std::decay_t<F>;

        fun_type f_;

        template <typename F_,
            typename Enable = std::enable_if_t<
                !std::is_same_v<std::decay_t<F_>, for_each_iteration>>>
        PIKA_HOST_DEVICE explicit for_each_iteration(F_&& f)
          : f_(PIKA_FORWARD(F_, f))
        {
        }

        template <typename F_, typename Proj_>
        PIKA_HOST_DEVICE for_each_iteration(F_&& f, Proj_&&)
          : f_(PIKA_FORWARD(F_, f))
        {
        }

#if !defined(__NVCC__) && !defined(__CUDACC__)
        for_each_iteration(for_each_iteration const&) = default;
        for_each_iteration(for_each_iteration&&) = default;
#else
        PIKA_HOST_DEVICE for_each_iteration(for_each_iteration const& rhs)
          : f_(rhs.f_)
        {
        }

        PIKA_HOST_DEVICE for_each_iteration(for_each_iteration&& rhs)
          : f_(PIKA_MOVE(rhs.f_))
        {
        }
#endif
        for_each_iteration& operator=(for_each_iteration const&) = default;
        for_each_iteration& operator=(for_each_iteration&&) = default;

        template <typename Iter>
        PIKA_HOST_DEVICE PIKA_FORCEINLINE constexpr void operator()(
            Iter part_begin, std::size_t part_size, std::size_t)
        {
            util::loop_n_ind<execution_policy_type>(part_begin, part_size, f_);
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename Iter>
    struct for_each_n : public detail::algorithm<for_each_n<Iter>, Iter>
    {
        constexpr for_each_n() noexcept
          : for_each_n::algorithm("for_each_n")
        {
        }

        template <typename ExPolicy, typename InIter, typename F, typename Proj>
        PIKA_HOST_DEVICE static constexpr Iter sequential(
            ExPolicy&&, InIter first, std::size_t count, F&& f, Proj&& proj)
        {
            return util::loop_n<std::decay_t<ExPolicy>>(
                first, count, invoke_projected<F, std::decay_t<Proj>>{f, proj});
        }

        template <typename ExPolicy, typename InIter, typename F>
        PIKA_HOST_DEVICE static constexpr Iter sequential(ExPolicy&&,
            InIter first, std::size_t count, F&& f, util::projection_identity)
        {
            return util::loop_n_ind<std::decay_t<ExPolicy>>(
                first, count, PIKA_FORWARD(F, f));
        }

        template <typename ExPolicy, typename FwdIter, typename F,
            typename Proj = util::projection_identity>
        static constexpr
            typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
            parallel(ExPolicy&& policy, FwdIter first, std::size_t count, F&& f,
                Proj&& proj /* = Proj()*/)
        {
            if (count != 0)
            {
                auto f1 = for_each_iteration<ExPolicy, F, std::decay_t<Proj>>(
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Proj, proj));

                return util::foreach_partitioner<ExPolicy>::call(
                    PIKA_FORWARD(ExPolicy, policy), first, count, PIKA_MOVE(f1),
                    util::projection_identity());
            }

            return util::detail::algorithm_result<ExPolicy, FwdIter>::get(
                PIKA_MOVE(first));
        }
    };
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    // for_each
    /// \cond NOINTERNAL
    template <typename Iter>
    struct for_each : public detail::algorithm<for_each<Iter>, Iter>
    {
        constexpr for_each() noexcept
          : for_each::algorithm("for_each")
        {
        }

        template <typename ExPolicy, typename InIterB, typename InIterE,
            typename F, typename Proj>
        static constexpr InIterB sequential(
            ExPolicy&& policy, InIterB first, InIterE last, F&& f, Proj&& proj)
        {
            if constexpr (pika::traits::is_random_access_iterator_v<InIterB>)
            {
                PIKA_UNUSED(policy);
                return util::loop_n<std::decay_t<ExPolicy>>(first,
                    static_cast<std::size_t>(detail::distance(first, last)),
                    invoke_projected<F, std::decay_t<Proj>>{f, proj});
            }
            else
            {
                return util::loop(PIKA_FORWARD(ExPolicy, policy), first, last,
                    invoke_projected<F, std::decay_t<Proj>>{f, proj});
            }
        }

        template <typename ExPolicy, typename InIterB, typename InIterE,
            typename F>
        static constexpr InIterB sequential(ExPolicy&& policy, InIterB first,
            InIterE last, F&& f, util::projection_identity)
        {
            if constexpr (pika::traits::is_random_access_iterator_v<InIterB>)
            {
                PIKA_UNUSED(policy);
                return util::loop_n_ind<std::decay_t<ExPolicy>>(first,
                    static_cast<std::size_t>(detail::distance(first, last)),
                    PIKA_FORWARD(F, f));
            }
            else
            {
                return util::loop_ind(PIKA_FORWARD(ExPolicy, policy), first,
                    last, PIKA_FORWARD(F, f));
            }
        }

        template <typename ExPolicy, typename FwdIterB, typename FwdIterE,
            typename F, typename Proj>
        static constexpr
            typename util::detail::algorithm_result<ExPolicy, FwdIterB>::type
            parallel(ExPolicy&& policy, FwdIterB first, FwdIterE last, F&& f,
                Proj&& proj)
        {
            if (first != last)
            {
                auto f1 = for_each_iteration<ExPolicy, F, std::decay_t<Proj>>(
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Proj, proj));

                return util::foreach_partitioner<ExPolicy>::call(
                    PIKA_FORWARD(ExPolicy, policy), first,
                    detail::distance(first, last), PIKA_MOVE(f1),
                    util::projection_identity());
            }

            return util::detail::algorithm_result<ExPolicy, FwdIterB>::get(
                PIKA_MOVE(first));
        }
    };
    /// \endcond
}    // namespace pika::parallel::detail

namespace pika {
    // Note: The implementation of the non-segmented algorithms here relies on
    //       tag_fallback_invoke. For this reason the tag_invoke overloads for
    //       the segmented algorithms (and other specializations) take
    //       precedence over the implementations here. This has the advantage
    //       that the non-segmented algorithms do not need to be explicitly
    //       disabled for other, possibly external specializations.
    //
    inline constexpr struct for_each_t final
      : pika::detail::tag_parallel_algorithm<for_each_t>
    {
    private:
        // clang-format off
        template <typename InIter,
            typename F,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator<InIter>::value
            )>
        // clang-format on
        friend F tag_fallback_invoke(
            pika::for_each_t, InIter first, InIter last, F&& f)
        {
            static_assert((pika::traits::is_input_iterator<InIter>::value),
                "Requires at least input iterator.");

            if (first != last)
            {
                pika::parallel::detail::for_each<InIter>().call(
                    pika::execution::seq, first, last, f,
                    pika::parallel::util::projection_identity());
            }
            return PIKA_FORWARD(F, f);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter,
            typename F,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename pika::parallel::util::detail::algorithm_result<
            ExPolicy>::type
        tag_fallback_invoke(pika::for_each_t, ExPolicy&& policy, FwdIter first,
            FwdIter last, F&& f)
        {
            static_assert((pika::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            if (first == last)
            {
                using result =
                    pika::parallel::util::detail::algorithm_result<ExPolicy>;
                return result::get();
            }

            return pika::parallel::util::detail::algorithm_result<ExPolicy>::
                get(pika::parallel::detail::for_each<FwdIter>().call(
                    PIKA_FORWARD(ExPolicy, policy), first, last,
                    PIKA_FORWARD(F, f),
                    pika::parallel::util::projection_identity()));
        }
    } for_each{};

    ///////////////////////////////////////////////////////////////////////////
    inline constexpr struct for_each_n_t final
      : pika::detail::tag_parallel_algorithm<for_each_n_t>
    {
    private:
        // clang-format off
        template <typename InIter, typename Size, typename F,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_input_iterator<InIter>::value
            )>
        // clang-format on
        friend InIter tag_fallback_invoke(
            pika::for_each_n_t, InIter first, Size count, F&& f)
        {
            static_assert((pika::traits::is_input_iterator<InIter>::value),
                "Requires at least input iterator.");

            // if count is representing a negative value, we do nothing
            if (parallel::detail::is_negative(count))
            {
                return first;
            }

            return pika::parallel::detail::for_each_n<InIter>().call(
                pika::execution::seq, first, std::size_t(count),
                PIKA_FORWARD(F, f), parallel::util::projection_identity());
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Size, typename F,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy<ExPolicy>::value &&
                pika::traits::is_forward_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_invoke(pika::for_each_n_t, ExPolicy&& policy,
            FwdIter first, Size count, F&& f)
        {
            static_assert((pika::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (parallel::detail::is_negative(count))
            {
                using result =
                    parallel::util::detail::algorithm_result<ExPolicy, FwdIter>;
                return result::get(PIKA_MOVE(first));
            }

            return pika::parallel::detail::for_each_n<FwdIter>().call(
                PIKA_FORWARD(ExPolicy, policy), first, std::size_t(count),
                PIKA_FORWARD(F, f), parallel::util::projection_identity());
        }
    } for_each_n{};
}    // namespace pika

#if defined(PIKA_HAVE_THREAD_DESCRIPTION)
namespace pika::detail {
    template <typename ExPolicy, typename F, typename Proj>
    struct get_function_address<
        parallel::detail::for_each_iteration<ExPolicy, F, Proj>>
    {
        static constexpr std::size_t call(
            parallel::detail::for_each_iteration<ExPolicy, F, Proj> const&
                f) noexcept
        {
            return get_function_address<std::decay_t<F>>::call(f.f_);
        }
    };

    template <typename ExPolicy, typename F, typename Proj>
    struct get_function_annotation<
        parallel::detail::for_each_iteration<ExPolicy, F, Proj>>
    {
        static constexpr char const* call(
            parallel::detail::for_each_iteration<ExPolicy, F, Proj> const&
                f) noexcept
        {
            return get_function_annotation<std::decay_t<F>>::call(f.f_);
        }
    };

#if PIKA_HAVE_ITTNOTIFY != 0 && !defined(PIKA_HAVE_APEX)
    template <typename ExPolicy, typename F, typename Proj>
    struct get_function_annotation_itt<
        parallel::detail::for_each_iteration<ExPolicy, F, Proj>>
    {
        static util::itt::string_handle call(
            parallel::detail::for_each_iteration<ExPolicy, F, Proj> const&
                f) noexcept
        {
            return get_function_annotation_itt<std::decay_t<F>>::call(f.f_);
        }
    };
#endif
}    // namespace pika::detail
#endif
#endif
