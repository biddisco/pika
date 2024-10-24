//  Copyright (c) 2020 LiliumAtratum
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// see #4787: `transform_inclusive_scan` gives incorrect results for
//            non-commutative operator

#include <pika/init.hpp>
#include <pika/modules/testing.hpp>
#include <pika/parallel/algorithms/transform_inclusive_scan.hpp>

#include <vector>

struct Elem
{
    int value = 0;
    bool begin = false;
};

bool operator==(Elem lhs, Elem rhs)
{
    return lhs.value == rhs.value;
}

int pika_main()
{
    std::vector<Elem> test{
        Elem{1, true}, Elem{3, false}, Elem{2, true}, Elem{4, false}};
    std::vector<Elem> output(test.size());

    pika::transform_inclusive_scan(
        pika::execution::par, test.cbegin(), test.cend(), output.begin(),
        [](Elem left, Elem right) -> Elem {
            if (right.begin)
            {
                return Elem{right.value, true};
            }
            else
            {
                return Elem{left.value + right.value, left.begin};
            }
        },
        [](Elem el) -> Elem { return el; }, Elem{0, true});

    std::vector<Elem> expected = {Elem{1}, Elem{4}, Elem{2}, Elem{6}};
    PIKA_TEST(output == expected);

    return pika::finalize();
}

int main(int argc, char* argv[])
{
    PIKA_TEST_EQ_MSG(pika::init(pika_main, argc, argv), 0,
        "pika main exited with non-zero status");

    return pika::util::report_errors();
}
