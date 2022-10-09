//  Copyright (c) 2014-2017 Hartmut Kaiser
//  Copyright (c)      2017 Taeguk Kwon
//  Copyright (c)      2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(DOXYGEN)
namespace pika {
    // clang-format off

    ///////////////////////////////////////////////////////////////////////////
    /// Reorders the elements in the range [first, last) in such a way that
    /// all elements for which the predicate \a pred returns true precede
    /// the elements for which the predicate \a pred returns false.
    /// Relative order of the elements is not preserved.
    ///
    /// \note   Complexity: At most 2 * (last - first) swaps.
    ///         Exactly \a last - \a first applications of the predicate and
    ///         projection.
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a partition requires \a Pred to meet
    ///                     the requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a pika::parallel::detail::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     unary predicate for partitioning the source
    ///                     iterators. The signature of
    ///                     this predicate should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a partition algorithm invoked without
    /// an execution policy object execute in sequential order in the calling
    /// thread.
    ///
    /// \returns  The \a partition algorithm returns returns \a FwdIter.
    ///           The \a partition algorithm returns the iterator to
    ///           the first element of the second group.
    ///
    template <typename FwdIter, typename Pred, typename Proj>
    FwdIter partition(ExPolicy&& policy, FwdIter first, FwdIter last,
        Pred&& pred, Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Reorders the elements in the range [first, last) in such a way that
    /// all elements for which the predicate \a pred returns true precede
    /// the elements for which the predicate \a pred returns false.
    /// Relative order of the elements is not preserved.
    ///
    /// \note   Complexity: At most 2 * (last - first) swaps.
    ///         Exactly \a last - \a first applications of the predicate and
    ///         projection.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a partition requires \a Pred to meet
    ///                     the requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a pika::parallel::detail::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     unary predicate for partitioning the source
    ///                     iterators. The signature of
    ///                     this predicate should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a partition algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a partition algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a partition algorithm returns a \a pika::future<FwdIter>
    ///           if the execution policy is of type \a parallel_task_policy
    ///           and returns \a FwdIter otherwise.
    ///           The \a partition algorithm returns the iterator to
    ///           the first element of the second group.
    ///
    template <typename ExPolicy, typename FwdIter, typename Pred, typename Proj>
    pika::parallel::detail::algorithm_result_t<ExPolicy, FwdIter>
    partition(ExPolicy&& policy, FwdIter first, FwdIter last, Pred&& pred,
        Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Permutes the elements in the range [first, last) such that there exists
    /// an iterator i such that for every iterator j in the range [first, i)
    /// INVOKE(f, INVOKE (proj, *j)) != false, and for every iterator k in the
    /// range [i, last), INVOKE(f, INVOKE (proj, *k)) == false
    ///
    /// \note   Complexity: At most (last - first) * log(last - first) swaps,
    ///         but only linear number of swaps if there is enough extra memory.
    ///         Exactly \a last - \a first applications of the predicate and
    ///         projection.
    ///
    /// \tparam BidirIter   The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a transform requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a pika::parallel::detail::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            Unary predicate which returns true if the element
    ///                     should be ordered before other elements.
    ///                     Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). The signature
    ///                     of this predicate should be equivalent to:
    ///                     \code
    ///                     bool fun(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The type \a Type must be such that an object of
    ///                     type \a BidirIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a f is invoked.
    ///
    /// The invocations of \a f in the parallel \a stable_partition algorithm
    /// invoked without an execution policy object executes in sequential order
    /// in the calling thread.
    ///
    /// \returns  The \a stable_partition algorithm returns an iterator i such
    ///           that for every iterator j in the range [first, i), f(*j) !=
    ///           false INVOKE(f, INVOKE(proj, *j)) != false, and for every
    ///           iterator k in the range [i, last), f(*k) == false
    ///           INVOKE(f, INVOKE (proj, *k)) == false. The relative order of
    ///           the elements in both groups is preserved.
    ///
    template <typename BidirIter, typename F, typename Proj>
    BidirIter stable_partition(BidirIter first, BidirIter last, F&& f,
        Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Permutes the elements in the range [first, last) such that there exists
    /// an iterator i such that for every iterator j in the range [first, i)
    /// INVOKE(f, INVOKE (proj, *j)) != false, and for every iterator k in the
    /// range [i, last), INVOKE(f, INVOKE (proj, *k)) == false
    ///
    /// \note   Complexity: At most (last - first) * log(last - first) swaps,
    ///         but only linear number of swaps if there is enough extra memory.
    ///         Exactly \a last - \a first applications of the predicate and
    ///         projection.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the invocations of \a f.
    /// \tparam BidirIter   The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a transform requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a pika::parallel::detail::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            Unary predicate which returns true if the element
    ///                     should be ordered before other elements.
    ///                     Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). The signature
    ///                     of this predicate should be equivalent to:
    ///                     \code
    ///                     bool fun(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The type \a Type must be such that an object of
    ///                     type \a BidirIter can be dereferenced and then
    ///                     implicitly converted to \a Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a f is invoked.
    ///
    /// The invocations of \a f in the parallel \a stable_partition algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy executes in sequential order in the
    /// calling thread.
    ///
    /// The invocations of \a f in the parallel \a stable_partition algorithm
    /// invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a stable_partition algorithm returns an iterator i such
    ///           that for every iterator j in the range [first, i), f(*j) !=
    ///           false INVOKE(f, INVOKE(proj, *j)) != false, and for every
    ///           iterator k in the range [i, last), f(*k) == false
    ///           INVOKE(f, INVOKE (proj, *k)) == false. The relative order of
    ///           the elements in both groups is preserved.
    ///           If the execution policy is of type \a parallel_task_policy
    ///           the algorithm returns a future<> referring to this iterator.
    ///
    template <typename ExPolicy, typename BidirIter, typename F,
        typename Proj>
    pika::parallel::detail::algorithm_result_t<ExPolicy, BidirIter>
    stable_partition(ExPolicy&& policy, BidirIter first, BidirIter last,
        F&& f, Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements in the range, defined by [first, last),
    /// to two different ranges depending on the value returned by
    /// the predicate \a pred. The elements, that satisfy the predicate \a pred
    /// are copied to the range beginning at \a dest_true. The rest of
    /// the elements are copied to the range beginning at \a dest_false.
    /// The order of the elements is preserved.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of the
    ///         predicate \a f.
    ///
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range for the elements that satisfy
    ///                     the predicate \a pred (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter3    The type of the iterator representing the
    ///                     destination range for the elements that don't
    ///                     satisfy the predicate \a pred (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a partition_copy requires \a Pred to
    ///                     meet the requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a pika::parallel::detail::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest_true    Refers to the beginning of the destination range
    ///                     for the elements that satisfy the predicate \a pred
    /// \param dest_false   Refers to the beginning of the destination range
    ///                     for the elements that don't satisfy the predicate
    ///                     \a pred.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     unary predicate for partitioning the source
    ///                     iterators. The signature of
    ///                     this predicate should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a partition_copy algorithm invoked
    /// without an execution policy object execute in sequential order in the
    /// calling thread.
    ///
    /// \returns  The \a partition_copy algorithm returns
    ///           \a std::pair<OutIter1, OutIter2>.
    ///           The \a partition_copy algorithm returns the pair of
    ///           the destination iterator to the end of the \a dest_true
    ///           range, and the destination iterator to the end of the \a
    ///           dest_false range.
    ///
    template <typename FwdIter1, typename FwdIter2,
        typename FwdIter3, typename Pred, typename Proj>
    std::pair<FwdIter2 ,FwdIter3>
    partition_copy(FwdIter1 first, FwdIter1 last,
        FwdIter2 dest_true, FwdIter3 dest_false, Pred&& pred, Proj&& proj);

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements in the range, defined by [first, last),
    /// to two different ranges depending on the value returned by
    /// the predicate \a pred. The elements, that satisfy the predicate \a pred,
    /// are copied to the range beginning at \a dest_true. The rest of
    /// the elements are copied to the range beginning at \a dest_false.
    /// The order of the elements is preserved.
    ///
    /// \note   Complexity: Performs not more than \a last - \a first
    ///         assignments, exactly \a last - \a first applications of the
    ///         predicate \a f.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range for the elements that satisfy
    ///                     the predicate \a pred (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter3    The type of the iterator representing the
    ///                     destination range for the elements that don't
    ///                     satisfy the predicate \a pred (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a partition_copy requires \a Pred to
    ///                     meet the requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a pika::parallel::detail::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest_true    Refers to the beginning of the destination range
    ///                     for the elements that satisfy the predicate \a pred
    /// \param dest_false   Refers to the beginning of the destination range
    ///                     for the elements that don't satisfy the predicate
    ///                     \a pred.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     unary predicate for partitioning the source
    ///                     iterators. The signature of
    ///                     this predicate should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a FwdIter1 can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a partition_copy algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a partition_copy algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a partition_copy algorithm returns a
    ///           \a pika::future<std::pair<OutIter1, OutIter2>>
    ///           if the execution policy is of type \a parallel_task_policy
    ///           and returns
    ///           \a std::pair<OutIter1, OutIter2> otherwise.
    ///           The \a partition_copy algorithm returns the pair of
    ///           the destination iterator to the end of the \a dest_true
    ///           range, and the destination iterator to the end of the \a
    ///           dest_false range.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename FwdIter3, typename Pred, typename Proj>
    pika::parallel::detail::algorithm_result_t<ExPolicy, std::pair<FwdIter2, FwdIter3>>
    partition_copy(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
        FwdIter2 dest_true, FwdIter3 dest_false, Pred&& pred, Proj&& proj);

    // clang-format on
}    // namespace pika

#else    // DOXYGEN

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/concepts/concepts.hpp>
#include <pika/functional/invoke.hpp>
#include <pika/functional/traits/is_invocable.hpp>
#include <pika/futures/future.hpp>
#include <pika/iterator_support/traits/is_iterator.hpp>
#include <pika/modules/async.hpp>
#include <pika/synchronization/spinlock.hpp>
#include <pika/type_support/unused.hpp>

#include <pika/algorithms/traits/projected.hpp>
#include <pika/execution/executors/execution.hpp>
#include <pika/execution/executors/execution_information.hpp>
#include <pika/execution/executors/execution_parameters.hpp>
#include <pika/executors/exception_list.hpp>
#include <pika/executors/execution_policy.hpp>
#include <pika/parallel/algorithms/detail/advance_and_get_distance.hpp>
#include <pika/parallel/algorithms/detail/advance_to_sentinel.hpp>
#include <pika/parallel/algorithms/detail/dispatch.hpp>
#include <pika/parallel/algorithms/detail/distance.hpp>
#include <pika/parallel/util/detail/algorithm_result.hpp>
#include <pika/parallel/util/detail/chunk_size.hpp>
#include <pika/parallel/util/detail/handle_local_exceptions.hpp>
#include <pika/parallel/util/detail/sender_util.hpp>
#include <pika/parallel/util/invoke_projected.hpp>
#include <pika/parallel/util/loop.hpp>
#include <pika/parallel/util/projection_identity.hpp>
#include <pika/parallel/util/scan_partitioner.hpp>
#include <pika/parallel/util/transfer.hpp>
#include <pika/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iterator>
#include <list>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace pika::parallel::detail {
    template <typename Tuple>
    constexpr PIKA_FORCEINLINE
        std::pair<typename std::tuple_element<1, Tuple>::type,
            typename std::tuple_element<2, Tuple>::type>
        tuple_to_pair(Tuple&& t)
    {
        return std::make_pair(std::get<1>(t), std::get<2>(t));
    }

    template <typename Tuple>
    pika::future<std::pair<typename std::tuple_element<1, Tuple>::type,
        typename std::tuple_element<2, Tuple>::type>>
    tuple_to_pair(pika::future<Tuple>&& f)
    {
        using result_type =
            std::pair<typename std::tuple_element<1, Tuple>::type,
                typename std::tuple_element<2, Tuple>::type>;

        return pika::make_future<result_type>(
            PIKA_MOVE(f), [](Tuple&& t) -> result_type {
                return tuple_to_pair(PIKA_MOVE(t));
            });
    }

    ///////////////////////////////////////////////////////////////////////////
    // stable_partition
    /// \cond NOINTERNAL
    struct stable_partition_helper
    {
        template <typename ExPolicy, typename RandIter, typename F,
            typename Proj>
        pika::future<RandIter>
        operator()(ExPolicy&& policy, RandIter first, RandIter last,
            std::size_t size, F&& f, Proj&& proj, std::size_t chunks)
        {
            if (chunks < 2)
            {
                return execution::async_execute(policy.executor(),
                    [first, last, f = PIKA_FORWARD(F, f),
                        proj = PIKA_FORWARD(Proj, proj)]() -> RandIter {
                        return std::stable_partition(
                            first, last, invoke_projected<F, Proj>(f, proj));
                    });
            }

            std::size_t mid_point = size / 2;
            chunks /= 2;

            RandIter mid = first;
            std::advance(mid, mid_point);

            pika::future<RandIter> left =
                execution::async_execute(policy.executor(), *this, policy,
                    first, mid, mid_point, f, proj, chunks);
            pika::future<RandIter> right =
                execution::async_execute(policy.executor(), *this, policy, mid,
                    last, size - mid_point, f, proj, chunks);

            return dataflow(
                policy.executor(),
                [mid](pika::future<RandIter>&& left,
                    pika::future<RandIter>&& right) -> RandIter {
                    if (left.has_exception() || right.has_exception())
                    {
                        std::list<std::exception_ptr> errors;
                        if (left.has_exception())
                        {
                            pika::parallel::detail::handle_local_exceptions<
                                ExPolicy>::call(left.get_exception_ptr(),
                                errors);
                        }
                        if (right.has_exception())
                        {
                            pika::parallel::detail::handle_local_exceptions<
                                ExPolicy>::call(right.get_exception_ptr(),
                                errors);
                        }

                        if (!errors.empty())
                        {
                            throw exception_list(PIKA_MOVE(errors));
                        }
                    }
                    RandIter first = left.get();
                    RandIter last = right.get();

                    std::rotate(first, mid, last);

                    // for some library implementations std::rotate
                    // does not return the new middle point
                    std::advance(first, std::distance(mid, last));
                    return first;
                },
                PIKA_MOVE(left), PIKA_MOVE(right));
        }
    };

    template <typename BidirIter, typename Sent, typename F, typename Proj>
    static BidirIter
    stable_partition_seq(BidirIter first, Sent last, F&& f, Proj&& proj)
    {
        using value_type = typename std::iterator_traits<BidirIter>::value_type;
        std::vector<value_type> falseValues;

        BidirIter next = first;
        while (first != last)
        {
            if (PIKA_INVOKE(f, PIKA_INVOKE(proj, *first)))
            {
                *next = PIKA_MOVE(*first);
                ++next;
            }
            else
            {
                falseValues.emplace_back(PIKA_MOVE(*first));
            }

            ++first;
        }

        move(std::begin(falseValues), std::end(falseValues), next);
        return next;
    }

    template <typename Iter>
    struct stable_partition_algo
      : public algorithm<stable_partition_algo<Iter>, Iter>
    {
        stable_partition_algo()
          : stable_partition_algo::algorithm("stable_partition")
        {
        }

        template <typename ExPolicy, typename BidirIter, typename Sent,
            typename F, typename Proj>
        static BidirIter
        sequential(ExPolicy&&, BidirIter first, Sent last, F&& f, Proj&& proj)
        {
            return stable_partition_seq(
                first, last, PIKA_FORWARD(F, f), PIKA_FORWARD(Proj, proj));
        }

        template <typename ExPolicy, typename RandIter, typename Sent,
            typename F, typename Proj>
        static typename algorithm_result<ExPolicy, RandIter>::type parallel(
            ExPolicy&& policy, RandIter first, Sent last, F&& f, Proj&& proj)
        {
            using algorithm_result = algorithm_result<ExPolicy, RandIter>;
            using difference_type =
                typename std::iterator_traits<RandIter>::difference_type;

            future<RandIter> result;
            auto last_iter = first;

            try
            {
                // advances last_iter to last and gets distance
                difference_type size =
                    detail::advance_and_get_distance(last_iter, last);

                if (size == 0)
                {
                    result = pika::make_ready_future(PIKA_MOVE(last_iter));
                }
                else
                {
                    std::size_t const cores = execution::processing_units_count(
                        policy.parameters(), policy.executor());

                    std::size_t chunk_size = execution::get_chunk_size(
                        policy.parameters(), policy.executor(),
                        [](std::size_t) { return 0; }, cores, size);

                    std::size_t max_chunks =
                        execution::maximal_number_of_chunks(policy.parameters(),
                            policy.executor(), cores, size);

                    adjust_chunk_size_and_max_chunks(
                        cores, size, chunk_size, max_chunks);

                    result = stable_partition_helper()(
                        PIKA_FORWARD(ExPolicy, policy), first, last_iter, size,
                        PIKA_FORWARD(F, f), PIKA_FORWARD(Proj, proj),
                        max_chunks);
                }
            }
            catch (...)
            {
                result = pika::make_exceptional_future<RandIter>(
                    std::current_exception());
            }

            if (result.has_exception())
            {
                return algorithm_result::get(
                    detail::handle_exception<ExPolicy, RandIter>::call(
                        PIKA_MOVE(result)));
            }

            return algorithm_result::get(PIKA_MOVE(result));
        }
    };
    /// \endcond

    /////////////////////////////////////////////////////////////////////////////
    // partition
    /// \cond NOINTERNAL

    // sequential partition with projection function for bidirectional iterator.
    template <typename BidirIter, typename Pred, typename Proj,
        PIKA_CONCEPT_REQUIRES_(
            pika::traits::is_bidirectional_iterator_v<BidirIter>)>
    BidirIter sequential_partition(
        BidirIter first, BidirIter last, Pred&& pred, Proj&& proj)
    {
        while (true)
        {
            while (
                first != last && PIKA_INVOKE(pred, PIKA_INVOKE(proj, *first)))
                ++first;
            if (first == last)
                break;

            while (
                first != --last && !PIKA_INVOKE(pred, PIKA_INVOKE(proj, *last)))
                ;
            if (first == last)
                break;

#if defined(PIKA_HAVE_CXX20_STD_RANGES_ITER_SWAP)
            std::ranges::iter_swap(first++, last);
#else
            std::iter_swap(first++, last);
#endif
        }

        return first;
    }

    // sequential partition with projection function for forward iterator.
    template <typename FwdIter, typename Pred, typename Proj,
        PIKA_CONCEPT_REQUIRES_(pika::traits::is_forward_iterator_v<FwdIter> &&
            !pika::traits::is_bidirectional_iterator_v<FwdIter>)>
    FwdIter
    sequential_partition(FwdIter first, FwdIter last, Pred&& pred, Proj&& proj)
    {
        while (first != last && PIKA_INVOKE(pred, PIKA_INVOKE(proj, *first)))
            ++first;

        if (first == last)
            return first;

        for (FwdIter it = std::next(first); it != last; ++it)
        {
            if (PIKA_INVOKE(pred, PIKA_INVOKE(proj, *it)))
            {
#if defined(PIKA_HAVE_CXX20_STD_RANGES_ITER_SWAP)
                std::ranges::iter_swap(first++, it);
#else
                std::iter_swap(first++, it);
#endif
            }
        }

        return first;
    }

    struct partition_helper
    {
        template <typename FwdIter>
        struct block
        {
            FwdIter first;
            FwdIter last;
            // Maybe 'std::int64_t' is enough to avoid overflow.
            std::int64_t block_no;

            block()
              : first()
              , last()
              , block_no(-1)
            {
            }

            block(FwdIter first, FwdIter last, std::int64_t block_no = -1)
              : first(first)
              , last(last)
              , block_no(block_no)
            {
            }

            bool empty() const
            {
                return first == last;
            }

            // The blocks are sorted by their positions.
            // The block_no implies block's position.
            // If the block is on leftside of boundary, block_no is negative number.
            // Otherwise, block_no is positive number.
            // The farther from the boundary, the larger its absolute value.
            // The example of sorted blocks below (the number means block_no):
            //     -3  -2  -1  1  2  3  4
            bool operator<(block<FwdIter> const& other) const
            {
                if ((this->block_no < 0 && other.block_no < 0) ||
                    (this->block_no > 0 && other.block_no > 0))
                    return this->block_no > other.block_no;
                else
                    return this->block_no < other.block_no;
            }
        };

        template <typename Iter, typename Enable = void>
        class block_manager;

        // block manager for random access iterator.
        template <typename RandIter>
        class block_manager<RandIter,
            std::enable_if_t<
                pika::traits::is_random_access_iterator_v<RandIter>>>
        {
        public:
            block_manager(RandIter first, RandIter last, std::size_t block_size)
              : first_(first)
              , left_(0)
              , right_(std::distance(first, last))
              , block_size_(block_size)
            {
            }
            block_manager(const block_manager&) = delete;
            block_manager& operator=(const block_manager&) = delete;

            // Get block from the end of leftside of boundary.
            block<RandIter> get_left_block()
            {
                std::lock_guard<decltype(mutex_)> lk(mutex_);

                if (left_ >= right_)
                    return {first_, first_};

                std::size_t begin_index = left_;
                std::size_t end_index = (std::min)(left_ + block_size_, right_);

                left_ += end_index - begin_index;

                RandIter begin_iter = std::next(first_, begin_index);
                RandIter end_iter = std::next(first_, end_index);

                boundary_ = end_iter;

                return {begin_iter, end_iter, left_block_no_--};
            }

            // Get block from the end of rightside of boundary.
            block<RandIter> get_right_block()
            {
                std::lock_guard<decltype(mutex_)> lk(mutex_);

                if (left_ >= right_)
                    return {first_, first_};

                std::size_t begin_index =
                    (std::max)(right_ - block_size_, left_);
                std::size_t end_index = right_;

                right_ -= end_index - begin_index;

                RandIter begin_iter = std::next(first_, begin_index);
                RandIter end_iter = std::next(first_, end_index);

                boundary_ = begin_iter;

                return {begin_iter, end_iter, right_block_no_++};
            }

            RandIter boundary()
            {
                return boundary_;
            }

        private:
            RandIter first_, boundary_;
            std::size_t left_, right_;
            std::size_t block_size_;
            std::int64_t left_block_no_{-1}, right_block_no_{1};
            pika::spinlock mutex_;
        };

        // block manager for forward access iterator.
        template <typename FwdIter>
        class block_manager<FwdIter,
            std::enable_if_t<pika::traits::is_forward_iterator_v<FwdIter> &&
                !pika::traits::is_random_access_iterator_v<FwdIter>>>
        {
        public:
            // In constructor, prepare all blocks for fast acquirements of blocks.
            block_manager(FwdIter first, FwdIter last, std::size_t block_size)
              : boundary_(first)
              , blocks_(
                    (std::distance(first, last) + block_size - 1) / block_size)
            {
                left_ = 0;
                right_ = blocks_.size();

                if (blocks_.size() == 1)
                {
                    blocks_.front() = {first, last};
                    return;
                }

                FwdIter next = std::next(first, block_size);

                blocks_.front() = {first, next};

                for (std::size_t i = 1; i < blocks_.size() - 1; ++i)
                {
                    first = next;
                    next = std::next(first, block_size);
                    blocks_[i] = {first, next};
                }

                blocks_.back() = {next, last};
            }

            block_manager(const block_manager&) = delete;
            block_manager& operator=(const block_manager&) = delete;

            // Get block from the end of leftside of boundary.
            block<FwdIter> get_left_block()
            {
                std::lock_guard<decltype(mutex_)> lk(mutex_);

                if (left_ >= right_)
                    return {boundary_, boundary_};

                boundary_ = blocks_[left_].last;
                blocks_[left_].block_no = left_block_no_--;

                // NOLINTNEXTLINE(bugprone-macro-repeated-side-effects)
                return PIKA_MOVE(blocks_[left_++]);
            }

            // // Get block from the end of rightside of boundary.
            block<FwdIter> get_right_block()
            {
                std::lock_guard<decltype(mutex_)> lk(mutex_);

                if (left_ >= right_)
                    return {boundary_, boundary_};

                boundary_ = blocks_[--right_].first;
                blocks_[right_].block_no = right_block_no_++;

                return PIKA_MOVE(blocks_[right_]);
            }

            FwdIter boundary()
            {
                return boundary_;
            }

        private:
            FwdIter boundary_;
            std::vector<block<FwdIter>> blocks_;
            std::size_t left_, right_;
            std::int64_t left_block_no_{-1}, right_block_no_{1};
            pika::spinlock mutex_;
        };

        // std::swap_ranges doesn't support overlapped ranges in standard.
        // But, actually general implementations of std::swap_ranges are useful
        //     in specific cases.
        // The problem is that standard doesn't guarantee that implementation.
        // The swap_ranges_forward is the general implementation of
        //     std::swap_ranges for guaranteeing utilizations in specific cases.
        // If dest is previous to first, the range [first, last) can be
        //     successfully moved to the range [dest, dest+distance(first, last)).
        template <class FwdIter1, class FwdIter2>
        static FwdIter2
        swap_ranges_forward(FwdIter1 first, FwdIter1 last, FwdIter2 dest)
        {
            while (first != last)
            {
#if defined(PIKA_HAVE_CXX20_STD_RANGES_ITER_SWAP)
                std::ranges::iter_swap(first++, dest++);
#else
                std::iter_swap(first++, dest++);
#endif
            }

            return dest;
        }

        // The function which performs sub-partitioning.
        template <typename FwdIter, typename Pred, typename Proj>
        static block<FwdIter> partition_thread(
            block_manager<FwdIter>& block_manager, Pred pred, Proj proj)
        {
            block<FwdIter> left_block, right_block;

            left_block = block_manager.get_left_block();
            right_block = block_manager.get_right_block();

            while (true)
            {
                while ((!left_block.empty() ||
                           !(left_block = block_manager.get_left_block())
                                .empty()) &&
                    PIKA_INVOKE(pred, PIKA_INVOKE(proj, *left_block.first)))
                {
                    ++left_block.first;
                }

                while ((!right_block.empty() ||
                           !(right_block = block_manager.get_right_block())
                                .empty()) &&
                    !PIKA_INVOKE(pred, PIKA_INVOKE(proj, *right_block.first)))
                {
                    ++right_block.first;
                }

                if (left_block.empty())
                    return right_block;
                if (right_block.empty())
                    return left_block;

#if defined(PIKA_HAVE_CXX20_STD_RANGES_ITER_SWAP)
                std::ranges::iter_swap(left_block.first++, right_block.first++);
#else
                std::iter_swap(left_block.first++, right_block.first++);
#endif
            }
        }

        // The function which collapses remaining blocks.
        // Performs sequential sub-partitioning to remaining blocks for
        //     reducing the number and size of remaining blocks.
        template <typename FwdIter, typename Pred, typename Proj>
        static void
        collapse_remaining_blocks(std::vector<block<FwdIter>>& remaining_blocks,
            Pred& pred, Proj& proj)
        {
            if (remaining_blocks.empty())
                return;

            auto left_iter = std::begin(remaining_blocks);
            auto right_iter = std::end(remaining_blocks) - 1;

            if (left_iter->block_no > 0 || right_iter->block_no < 0)
                return;

            while (true)
            {
                while (true)
                {
                    if (left_iter->empty())
                    {
                        ++left_iter;
                        if (left_iter == std::end(remaining_blocks) ||
                            left_iter->block_no > 0)
                            break;
                    }
                    if (!PIKA_INVOKE(
                            pred, PIKA_INVOKE(proj, *left_iter->first)))
                        break;
                    ++left_iter->first;
                }

                while (true)
                {
                    if (right_iter->empty())
                    {
                        if (right_iter == std::begin(remaining_blocks) ||
                            (--right_iter)->block_no < 0)
                            break;
                    }
                    if (PIKA_INVOKE(
                            pred, PIKA_INVOKE(proj, *right_iter->first)))
                        break;
                    ++right_iter->first;
                }

                if (left_iter == std::end(remaining_blocks) ||
                    left_iter->block_no > 0)
                    break;
                if (right_iter->empty() || right_iter->block_no < 0)
                    break;

#if defined(PIKA_HAVE_CXX20_STD_RANGES_ITER_SWAP)
                std::ranges::iter_swap(left_iter->first++, right_iter->first++);
#else
                std::iter_swap(left_iter->first++, right_iter->first++);
#endif
            }

            if (left_iter < right_iter ||
                (!right_iter->empty() && left_iter == right_iter))
            {
                remaining_blocks.erase(
                    right_iter->empty() ? right_iter : right_iter + 1,
                    std::end(remaining_blocks));

                remaining_blocks.erase(std::begin(remaining_blocks), left_iter);
            }
            else
            {
                remaining_blocks.clear();
            }
        }

        // The function which merges remaining blocks that are placed
        //     leftside of boundary. Requires bidirectional iterator.
        // Move remaining blocks to the adjacent left of the boundary.
        // In the end, all remaining blocks are merged into one block which
        //     is adjacent to the left of boundary.
        // clang-format off
            template <typename BidirIter,
                PIKA_CONCEPT_REQUIRES_(
                    pika::traits::is_bidirectional_iterator_v<BidirIter>
                )>
        // clang-format on
        static block<BidirIter> merge_leftside_remaining_blocks(
            std::vector<block<BidirIter>>& remaining_blocks, BidirIter boundary,
            BidirIter first)
        {
            PIKA_ASSERT(!remaining_blocks.empty());
            PIKA_UNUSED(first);

            auto boundary_rbegin = std::reverse_iterator<BidirIter>(boundary);
            for (auto it = remaining_blocks.rbegin();
                 it != remaining_blocks.rend(); ++it)
            {
                auto rbegin = std::reverse_iterator<BidirIter>(it->last);
                auto rend = std::reverse_iterator<BidirIter>(it->first);

                if (boundary_rbegin == rbegin)
                {
                    boundary_rbegin = rend;
                    continue;
                }

                boundary_rbegin =
                    swap_ranges_forward(rbegin, rend, boundary_rbegin);
            }

            return {boundary_rbegin.base(), boundary};
        }

        // The function which merges remaining blocks that are placed
        //     leftside of boundary. Requires forward iterator.
        // Move remaining blocks to the adjacent left of the boundary.
        // In the end, all remaining blocks are merged into one block which
        //     is adjacent to the left of boundary.
        // clang-format off
            template <typename FwdIter,
                PIKA_CONCEPT_REQUIRES_(
                    pika::traits::is_forward_iterator_v<FwdIter> &&
                    !pika::traits::is_bidirectional_iterator_v<FwdIter>
                )>
        // clang-format on
        static block<FwdIter> merge_leftside_remaining_blocks(
            std::vector<block<FwdIter>>& remaining_blocks, FwdIter boundary,
            FwdIter first)
        {
            PIKA_ASSERT(!remaining_blocks.empty());

            std::vector<FwdIter> dest_iters(remaining_blocks.size());
            std::vector<std::size_t> dest_iter_indexes(remaining_blocks.size());
            std::vector<std::size_t> remaining_block_indexes(
                remaining_blocks.size());
            std::vector<std::size_t> counts(remaining_blocks.size());
            std::size_t count_sum = 0u;

            for (std::size_t i = 0; i < counts.size(); ++i)
            {
                counts[i] = std::distance(
                    remaining_blocks[i].first, remaining_blocks[i].last);
                count_sum += counts[i];
            }

            remaining_block_indexes[0] =
                std::distance(first, remaining_blocks[0].first);
            for (std::size_t i = 1; i < remaining_block_indexes.size(); ++i)
            {
                remaining_block_indexes[i] = remaining_block_indexes[i - 1] +
                    counts[i - 1] +
                    std::distance(remaining_blocks[i - 1].last,
                        remaining_blocks[i].first);
            }

            std::size_t boundary_end_index = std::distance(first, boundary);
            std::size_t boundary_begin_index = boundary_end_index - count_sum;

            dest_iters[0] = std::next(first, boundary_begin_index);
            dest_iter_indexes[0] = boundary_begin_index;

            for (std::size_t i = 0; i < dest_iters.size() - 1; ++i)
            {
                dest_iters[i + 1] = std::next(dest_iters[i], counts[i]);
                dest_iter_indexes[i + 1] = dest_iter_indexes[i] + counts[i];
            }

            for (std::int64_t i = std::int64_t(dest_iters.size() - 1); i >= 0;
                 --i)
            {
                if (remaining_blocks[i].first == dest_iters[i])
                    continue;

                if (remaining_block_indexes[i] + counts[i] <=
                    dest_iter_indexes[i])
                {
                    // when the ranges are not overlapped each other.
                    swap_ranges_forward(remaining_blocks[i].first,
                        remaining_blocks[i].last, dest_iters[i]);
                }
                else
                {
                    // when the ranges are overlapped each other.
                    swap_ranges_forward(remaining_blocks[i].first,
                        dest_iters[i], remaining_blocks[i].last);
                }
            }

            return {dest_iters[0], boundary};
        }

        // The function which merges remaining blocks into
        //     one block which is adjacent to boundary.
        template <typename FwdIter>
        static block<FwdIter>
        merge_remaining_blocks(std::vector<block<FwdIter>>& remaining_blocks,
            FwdIter boundary, FwdIter first)
        {
            if (remaining_blocks.empty())
                return {boundary, boundary};

            if (remaining_blocks.front().block_no < 0)
            {
                // when blocks are placed in left side of boundary.
                return merge_leftside_remaining_blocks(
                    remaining_blocks, boundary, first);
            }
            else
            {
                // when blocks are placed in right side of boundary.
                FwdIter boundary_end = boundary;
                for (auto& block : remaining_blocks)
                {
                    if (block.first == boundary_end)
                    {
                        boundary_end = block.last;
                        continue;
                    }

                    boundary_end = swap_ranges_forward(
                        block.first, block.last, boundary_end);
                }

                return {boundary, boundary_end};
            }
        }

        template <typename ExPolicy, typename FwdIter, typename Pred,
            typename Proj>
        static FwdIter call(ExPolicy&& policy, FwdIter first, FwdIter last,
            Pred&& pred, Proj&& proj)
        {
            if (first == last)
                return first;

            std::size_t const cores = execution::processing_units_count(
                policy.parameters(), policy.executor());

            // TODO: Find more better block size.
            const std::size_t block_size = std::size_t(20000);
            block_manager<FwdIter> block_manager(first, last, block_size);

            std::vector<pika::future<block<FwdIter>>> remaining_block_futures(
                cores);

            // Main parallel phase: perform sub-partitioning in each thread.
            for (std::size_t i = 0; i < remaining_block_futures.size(); ++i)
            {
                remaining_block_futures[i] = execution::async_execute(
                    policy.executor(), [&block_manager, pred, proj]() {
                        return partition_thread(block_manager, pred, proj);
                    });
            }

            // Wait sub-partitioning to be all finished.
            pika::wait_all_nothrow(remaining_block_futures);

            // Handle exceptions in parallel phrase.
            std::list<std::exception_ptr> errors;

            handle_local_exceptions<ExPolicy>::call(
                remaining_block_futures, errors);

            std::vector<block<FwdIter>> remaining_blocks(
                remaining_block_futures.size());

            // Get remaining blocks from the result of sub-partitioning.
            for (std::size_t i = 0; i < remaining_block_futures.size(); ++i)
                remaining_blocks[i] = remaining_block_futures[i].get();

            // Remove blocks that are empty.
            FwdIter boundary = block_manager.boundary();
            remaining_blocks.erase(std::remove_if(std::begin(remaining_blocks),
                                       std::end(remaining_blocks),
                                       [](block<FwdIter> const& block) -> bool {
                                           return block.empty();
                                       }),
                std::end(remaining_blocks));

            // Sort remaining blocks to be listed from left to right.
            std::sort(std::begin(remaining_blocks), std::end(remaining_blocks));

            // Collapse remaining blocks each other.
            collapse_remaining_blocks(remaining_blocks, pred, proj);

            // Merge remaining blocks into one block
            //     which is adjacent to boundary.
            block<FwdIter> unpartitioned_block =
                merge_remaining_blocks(remaining_blocks, boundary, first);

            // Perform sequential partition to unpartitioned range.
            FwdIter real_boundary =
                sequential_partition(unpartitioned_block.first,
                    unpartitioned_block.last, pred, proj);

            return real_boundary;
        }
    };

    template <typename ExPolicy, typename FwdIter, typename Pred, typename Proj>
    pika::future<FwdIter> parallel_partition(ExPolicy&& policy, FwdIter first,
        FwdIter last, Pred&& pred, Proj&& proj)
    {
        pika::future<FwdIter> f = execution::async_execute(
            policy.executor(), [=]() mutable -> FwdIter {
                try
                {
                    return partition_helper::call(
                        policy, first, last, pred, proj);
                }
                catch (...)
                {
                    handle_local_exceptions<ExPolicy>::call(
                        std::current_exception());
                }

                // Not reachable.
                PIKA_ASSERT(false);
                return partition_helper::call(policy, first, last, pred, proj);
            });

        return f;
    }

    template <typename FwdIter>
    struct partition_algo : public algorithm<partition_algo<FwdIter>, FwdIter>
    {
        partition_algo()
          : partition_algo::algorithm("partition")
        {
        }

        template <typename ExPolicy, typename Sent, typename Pred,
            typename Proj = projection_identity>
        static FwdIter
        sequential(ExPolicy, FwdIter first, Sent last, Pred&& pred, Proj&& proj)
        {
            auto last_iter = detail::advance_to_sentinel(first, last);
            return sequential_partition(first, last_iter,
                PIKA_FORWARD(Pred, pred), PIKA_FORWARD(Proj, proj));
        }

        template <typename ExPolicy, typename Sent, typename Pred,
            typename Proj = projection_identity>
        static typename algorithm_result<ExPolicy, FwdIter>::type
        parallel(ExPolicy&& policy, FwdIter first, Sent last, Pred&& pred,
            Proj&& proj)
        {
            using algorithm_result = algorithm_result<ExPolicy, FwdIter>;
            auto last_iter = detail::advance_to_sentinel(first, last);

            try
            {
                return algorithm_result::get(parallel_partition(
                    PIKA_FORWARD(ExPolicy, policy), first, last_iter,
                    PIKA_FORWARD(Pred, pred), PIKA_FORWARD(Proj, proj)));
            }
            catch (...)
            {
                return algorithm_result::get(
                    detail::handle_exception<ExPolicy, FwdIter>::call(
                        std::current_exception()));
            }
        }
    };
    /// \endcond

    /////////////////////////////////////////////////////////////////////////////
    // partition_copy
    /// \cond NOINTERNAL

    // sequential partition_copy with projection function
    template <typename InIter, typename OutIter1, typename OutIter2,
        typename Pred, typename Proj>
    std::tuple<InIter, OutIter1, OutIter2>
    sequential_partition_copy(InIter first, InIter last, OutIter1 dest_true,
        OutIter2 dest_false, Pred&& pred, Proj&& proj)
    {
        while (first != last)
        {
            if (PIKA_INVOKE(pred, PIKA_INVOKE(proj, *first)))
                *dest_true++ = *first;
            else
                *dest_false++ = *first;
            first++;
        }
        return std::make_tuple(
            PIKA_MOVE(last), PIKA_MOVE(dest_true), PIKA_MOVE(dest_false));
    }

    template <typename IterTuple>
    struct partition_copy
      : public algorithm<partition_copy<IterTuple>, IterTuple>
    {
        partition_copy()
          : partition_copy::algorithm("partition_copy")
        {
        }

        template <typename ExPolicy, typename InIter, typename Sent,
            typename OutIter1, typename OutIter2, typename Pred,
            typename Proj = projection_identity>
        static std::tuple<InIter, OutIter1, OutIter2>
        sequential(ExPolicy, InIter first, Sent last, OutIter1 dest_true,
            OutIter2 dest_false, Pred&& pred, Proj&& proj)
        {
            auto last_iter = detail::advance_to_sentinel(first, last);
            return sequential_partition_copy(first, last_iter, dest_true,
                dest_false, PIKA_FORWARD(Pred, pred), PIKA_FORWARD(Proj, proj));
        }

        template <typename ExPolicy, typename FwdIter1, typename Sent,
            typename FwdIter2, typename FwdIter3, typename Pred,
            typename Proj = projection_identity>
        static typename algorithm_result<ExPolicy,
            std::tuple<FwdIter1, FwdIter2, FwdIter3>>::type
        parallel(ExPolicy&& policy, FwdIter1 first, Sent last,
            FwdIter2 dest_true, FwdIter3 dest_false, Pred&& pred, Proj&& proj)
        {
            using zip_iterator = pika::util::zip_iterator<FwdIter1, bool*>;
            using result = algorithm_result<ExPolicy,
                std::tuple<FwdIter1, FwdIter2, FwdIter3>>;
            using difference_type =
                typename std::iterator_traits<FwdIter1>::difference_type;
            using output_iterator_offset = std::pair<std::size_t, std::size_t>;

            if (first == last)
                return result::get(
                    std::make_tuple(first, dest_true, dest_false));

            auto last_iter = first;
            difference_type count =
                detail::advance_and_get_distance(last_iter, last);

            std::shared_ptr<bool[]> flags(new bool[count]);
            output_iterator_offset init = {0, 0};

            using pika::util::make_zip_iterator;
            using std::get;
            using scan_partitioner_type = scan_partitioner<ExPolicy,
                std::tuple<FwdIter1, FwdIter2, FwdIter3>,
                output_iterator_offset>;

            // Note: replacing the invoke() with PIKA_INVOKE()
            // below makes gcc generate errors
            auto f1 = [pred = PIKA_FORWARD(Pred, pred),
                          proj = PIKA_FORWARD(Proj, proj)](
                          zip_iterator part_begin,
                          std::size_t part_size) -> output_iterator_offset {
                std::size_t true_count = 0;

                // MSVC complains if pred or proj is captured by ref below
                loop_n<std::decay_t<ExPolicy>>(part_begin, part_size,
                    [pred, proj, &true_count](zip_iterator it) mutable -> void {
                        bool f = pika::util::detail::invoke(pred,
                            pika::util::detail::invoke(proj, get<0>(*it)));

                        if ((get<1>(*it) = f))
                            ++true_count;
                    });

                return output_iterator_offset(
                    true_count, part_size - true_count);
            };

            auto f2 = [](output_iterator_offset const& prev_sum,
                          output_iterator_offset const& curr)
                -> output_iterator_offset {
                return output_iterator_offset(get<0>(prev_sum) + get<0>(curr),
                    get<1>(prev_sum) + get<1>(curr));
            };
            auto f3 = [dest_true, dest_false, flags](zip_iterator part_begin,
                          std::size_t part_size,
                          output_iterator_offset val) mutable -> void {
                PIKA_UNUSED(flags);
                output_iterator_offset offset = val;
                std::size_t count_true = get<0>(offset);
                std::size_t count_false = get<1>(offset);
                std::advance(dest_true, count_true);
                std::advance(dest_false, count_false);

                loop_n<std::decay_t<ExPolicy>>(part_begin, part_size,
                    [&dest_true, &dest_false](zip_iterator it) mutable {
                        if (get<1>(*it))
                            *dest_true++ = get<0>(*it);
                        else
                            *dest_false++ = get<0>(*it);
                    });
            };

            auto f4 = [last_iter, dest_true, dest_false, flags](
                          std::vector<output_iterator_offset>&& items,
                          std::vector<pika::future<void>>&&) mutable
                -> std::tuple<FwdIter1, FwdIter2, FwdIter3> {
                PIKA_UNUSED(flags);

                output_iterator_offset count_pair = items.back();
                std::size_t count_true = get<0>(count_pair);
                std::size_t count_false = get<1>(count_pair);
                std::advance(dest_true, count_true);
                std::advance(dest_false, count_false);

                return std::make_tuple(last_iter, dest_true, dest_false);
            };

            return scan_partitioner_type::call(PIKA_FORWARD(ExPolicy, policy),
                make_zip_iterator(first, flags.get()), count, init,
                // step 1 performs first part of scan algorithm
                PIKA_MOVE(f1),
                // step 2 propagates the partition results from left
                // to right
                PIKA_MOVE(f2),
                // step 3 runs final accumulation on each partition
                PIKA_MOVE(f3),
                // step 4 use this return value
                PIKA_MOVE(f4));
        }
    };
    /// \endcond
}    // namespace pika::parallel::detail

namespace pika {
    ///////////////////////////////////////////////////////////////////////////
    // CPO for pika::stable_partition
    inline constexpr struct stable_partition_t final
      : pika::detail::tag_parallel_algorithm<stable_partition_t>
    {
        // clang-format off
        template <typename BidirIter, typename F,
            typename Proj = parallel::detail::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator_v<BidirIter> &&
                parallel::detail::is_projected_v<Proj, BidirIter> &&
                parallel::detail::is_indirect_callable_v<
                    pika::execution::sequenced_policy, F,
                    parallel::detail::projected<Proj, BidirIter>>
        )>
        // clang-format on
        friend BidirIter tag_fallback_invoke(pika::stable_partition_t,
            BidirIter first, BidirIter last, F&& f, Proj&& proj = Proj())
        {
            static_assert(pika::traits::is_bidirectional_iterator_v<BidirIter>,
                "Requires at least bidirectional iterator.");

            return pika::parallel::detail::stable_partition_algo<BidirIter>()
                .call2(pika::execution::seq, std::true_type{}, first, last,
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename BidirIter, typename F,
            typename Proj = parallel::detail::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy_v<ExPolicy> &&
                pika::traits::is_iterator_v<BidirIter> &&
                parallel::detail::is_projected_v<Proj, BidirIter> &&
                parallel::detail::is_indirect_callable_v<ExPolicy, F,
                    parallel::detail::projected<Proj, BidirIter>>
        )>
        // clang-format on
        friend
            typename parallel::detail::algorithm_result_t<ExPolicy, BidirIter>
            tag_fallback_invoke(pika::stable_partition_t, ExPolicy&& policy,
                BidirIter first, BidirIter last, F&& f, Proj&& proj = Proj())
        {
            static_assert(pika::traits::is_bidirectional_iterator_v<BidirIter>,
                "Requires at least bidirectional iterator.");

            using is_seq = std::integral_constant<bool,
                pika::is_sequenced_execution_policy_v<ExPolicy> ||
                    !pika::traits::is_random_access_iterator_v<BidirIter>>;

            return pika::parallel::detail::stable_partition_algo<BidirIter>()
                .call2(PIKA_FORWARD(ExPolicy, policy), is_seq(), first, last,
                    PIKA_FORWARD(F, f), PIKA_FORWARD(Proj, proj));
        }
    } stable_partition{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for pika::partition
    inline constexpr struct partition_t final
      : pika::detail::tag_parallel_algorithm<partition_t>
    {
        // clang-format off
        template <typename FwdIter, typename Pred,
            typename Proj = parallel::detail::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator_v<FwdIter> &&
                parallel::detail::is_projected_v<Proj, FwdIter> &&
                parallel::detail::is_indirect_callable_v<
                    pika::execution::sequenced_policy,
                    Pred, parallel::detail::projected<Proj, FwdIter>>
        )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(pika::partition_t, FwdIter first,
            FwdIter last, Pred&& pred, Proj&& proj = Proj())
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter>,
                "Required at least forward iterator.");

            return pika::parallel::detail::partition_algo<FwdIter>().call(
                pika::execution::seq, first, last, PIKA_FORWARD(Pred, pred),
                PIKA_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Pred,
            typename Proj = parallel::detail::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy_v<ExPolicy> &&
                pika::traits::is_iterator_v<FwdIter> &&
                parallel::detail::is_projected_v<Proj, FwdIter> &&
                parallel::detail::is_indirect_callable_v<ExPolicy,
                    Pred, parallel::detail::projected<Proj, FwdIter>>
        )>
        // clang-format on
        friend typename parallel::detail::algorithm_result_t<ExPolicy, FwdIter>
        tag_fallback_invoke(pika::partition_t, ExPolicy&& policy, FwdIter first,
            FwdIter last, Pred&& pred, Proj&& proj = Proj())
        {
            static_assert(pika::traits::is_forward_iterator_v<FwdIter>,
                "Required at least forward iterator.");

            return pika::parallel::detail::partition_algo<FwdIter>().call(
                PIKA_FORWARD(ExPolicy, policy), first, last,
                PIKA_FORWARD(Pred, pred), PIKA_FORWARD(Proj, proj));
        }
    } partition{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for pika::partition_copy
    inline constexpr struct partition_copy_t final
      : pika::detail::tag_parallel_algorithm<partition_copy_t>
    {
        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            typename FwdIter3, typename Pred,
            typename Proj = parallel::detail::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::traits::is_iterator_v<FwdIter1> &&
                pika::traits::is_iterator_v<FwdIter2> &&
                pika::traits::is_iterator_v<FwdIter3> &&
                parallel::detail::is_projected_v<Proj, FwdIter1> &&
                parallel::detail::is_indirect_callable_v<
                    pika::execution::sequenced_policy, Pred,
                    parallel::detail::projected<Proj, FwdIter1>>
            )>
        // clang-format on
        friend std::pair<FwdIter2, FwdIter3>
        tag_fallback_invoke(pika::partition_copy_t, FwdIter1 first,
            FwdIter1 last, FwdIter2 dest_true, FwdIter3 dest_false, Pred&& pred,
            Proj&& proj = Proj())
        {
            static_assert((pika::traits::is_forward_iterator_v<FwdIter1>),
                "Required at least forward iterator.");
            static_assert((pika::traits::is_forward_iterator_v<FwdIter2>),
                "Requires at least forward iterator.");
            static_assert((pika::traits::is_forward_iterator_v<FwdIter3>),
                "Requires at least forward iterator.");

            using result_type = std::tuple<FwdIter1, FwdIter2, FwdIter3>;

            return parallel::detail::tuple_to_pair(
                parallel::detail::partition_copy<result_type>().call(
                    pika::execution::seq, first, last, dest_true, dest_false,
                    PIKA_FORWARD(Pred, pred), PIKA_FORWARD(Proj, proj)));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename FwdIter3, typename Pred,
            typename Proj = parallel::detail::projection_identity,
            PIKA_CONCEPT_REQUIRES_(
                pika::is_execution_policy_v<ExPolicy> &&
                pika::traits::is_iterator_v<FwdIter1> &&
                pika::traits::is_iterator_v<FwdIter2> &&
                pika::traits::is_iterator_v<FwdIter3> &&
                parallel::detail::is_projected_v<Proj, FwdIter1> &&
                parallel::detail::is_indirect_callable_v<ExPolicy, Pred,
                    parallel::detail::projected<Proj, FwdIter1>>
            )>
        // clang-format on
        friend parallel::detail::algorithm_result_t<ExPolicy,
            std::pair<FwdIter2, FwdIter3>>
        tag_fallback_invoke(pika::partition_copy_t, ExPolicy&& policy,
            FwdIter1 first, FwdIter1 last, FwdIter2 dest_true,
            FwdIter3 dest_false, Pred&& pred, Proj&& proj = Proj())
        {
            static_assert((pika::traits::is_forward_iterator_v<FwdIter1>),
                "Required at least forward iterator.");
            static_assert((pika::traits::is_forward_iterator_v<FwdIter2>),
                "Requires at least forward iterator.");
            static_assert((pika::traits::is_forward_iterator_v<FwdIter3>),
                "Requires at least forward iterator.");

            using result_type = std::tuple<FwdIter1, FwdIter2, FwdIter3>;

            return parallel::detail::tuple_to_pair(
                parallel::detail::partition_copy<result_type>().call(
                    PIKA_FORWARD(ExPolicy, policy), first, last, dest_true,
                    dest_false, PIKA_FORWARD(Pred, pred),
                    PIKA_FORWARD(Proj, proj)));
        }
    } partition_copy{};
}    // namespace pika

#endif    // DOXYGEN
