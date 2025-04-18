# Copyright (c) 2023 ETH Zurich
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(example_programs
    any_sender_documentation
    async_rw_mutex_documentation
    drop_operation_state_documentation
    drop_value_documentation
    hello_world_documentation
    init_hpp_documentation
    require_started_documentation
    split_tuple_documentation
    unpack_documentation
    when_all_vector_documentation
)

if(PIKA_WITH_GPU_SUPPORT)
  list(APPEND example_programs cuda_overview_documentation then_with_cublas_documentation
       then_with_stream_documentation
  )
  set(cuda_overview_documentation_GPU ON)
  set(then_with_stream_documentation_GPU ON)
  set(then_with_cublas_documentation_GPU ON)
  set(cuda_overview_documentation_PARAMETERS RUN_SERIAL)
  set(then_with_stream_documentation_PARAMETERS RUN_SERIAL)
  set(then_with_cublas_documentation_PARAMETERS RUN_SERIAL)
endif()

foreach(example_program ${example_programs})
  if(${${example_program}_GPU})
    set(sources ${example_program}.cu)
  else()
    set(sources ${example_program}.cpp)
  endif()

  source_group("Source Files" FILES ${sources})

  pika_add_executable(
    ${example_program} INTERNAL_FLAGS
    SOURCES ${sources} ${${example_program}_FLAGS}
    FOLDER "Examples/Documentation"
  )

  target_link_libraries(${example_program} PRIVATE ${${example_program}_LIBRARIES})
  pika_add_example_target_dependencies("documentation" ${example_program})

  if(PIKA_WITH_TESTS AND PIKA_WITH_TESTS_EXAMPLES)
    pika_add_example_test("documentation" ${example_program} ${${example_program}_PARAMETERS})
  endif()
endforeach()

if(PIKA_WITH_TESTS AND PIKA_WITH_TESTS_EXAMPLES)
  set_tests_properties(
    tests.examples.documentation.require_started_documentation
    PROPERTIES
      PASS_REGULAR_EXPRESSION
      "pika::execution::experimental::~require_started_sender: A require_started sender was never started"
  )
endif()
