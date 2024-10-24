//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/assert.hpp>
#include <pika/modules/errors.hpp>
#include <pika/serialization/binary_filter.hpp>
#include <pika/serialization/container.hpp>
#include <pika/serialization/serialization_chunk.hpp>
#include <pika/serialization/traits/serialization_access_data.hpp>

#include <cstddef>    // for size_t
#include <cstdint>
#include <cstring>    // for memcpy
#include <memory>
#include <vector>

namespace pika { namespace serialization {

    template <typename Container>
    struct input_container : erased_input_container
    {
    private:
        using access_traits = traits::serialization_access_data<Container>;

        std::size_t get_chunk_size(std::size_t chunk) const
        {
            return (*chunks_)[chunk].size_;
        }

        std::uint8_t get_chunk_type(std::size_t chunk) const
        {
            return (*chunks_)[chunk].type_;
        }

        chunk_data get_chunk_data(std::size_t chunk) const
        {
            return (*chunks_)[chunk].data_;
        }

        std::size_t get_num_chunks() const
        {
            return chunks_->size();
        }

    public:
        input_container(Container const& cont, std::size_t inbound_data_size)
          : cont_(cont)
          , current_(0)
          , filter_()
          , decompressed_size_(inbound_data_size)
          , chunks_(nullptr)
          , current_chunk_(std::size_t(-1))
          , current_chunk_size_(0)
        {
        }

        input_container(Container const& cont,
            std::vector<serialization_chunk> const* chunks,
            std::size_t inbound_data_size)
          : cont_(cont)
          , current_(0)
          , filter_()
          , decompressed_size_(inbound_data_size)
          , chunks_(nullptr)
          , current_chunk_(std::size_t(-1))
          , current_chunk_size_(0)
        {
            if (chunks && chunks->size() != 0)
            {
                chunks_ = chunks;
                current_chunk_ = 0;
            }
        }

        void set_filter(binary_filter* filter)    // override
        {
            filter_.reset(filter);
            if (filter)
            {
                current_ = access_traits::init_data(
                    cont_, filter_.get(), current_, decompressed_size_);

                if (decompressed_size_ < current_)
                {
                    PIKA_THROW_EXCEPTION(serialization_error,
                        "input_container::set_filter",
                        "archive data bstream is too short");
                    return;
                }
            }
        }

        void load_binary(void* address, std::size_t count)
        {
            if (filter_)
            {
                filter_->load(address, count);
            }
            else
            {
                std::size_t new_current = current_ + count;
                if (new_current > access_traits::size(cont_))
                {
                    PIKA_THROW_EXCEPTION(serialization_error,
                        "input_container::load_binary",
                        "archive data bstream is too short");
                    return;
                }

                access_traits::read(cont_, count, current_, address);

                current_ = new_current;

                if (chunks_)
                {
                    current_chunk_size_ += count;

                    // make sure we switch to the next serialization_chunk if
                    // necessary
                    std::size_t current_chunk_size =
                        get_chunk_size(current_chunk_);
                    if (current_chunk_size != 0 &&
                        current_chunk_size_ >= current_chunk_size)
                    {
                        // raise an error if we read past the serialization_chunk
                        if (current_chunk_size_ > current_chunk_size)
                        {
                            PIKA_THROW_EXCEPTION(serialization_error,
                                "input_container::load_binary",
                                "archive data bstream structure mismatch");
                            return;
                        }
                        ++current_chunk_;
                        current_chunk_size_ = 0;
                    }
                }
            }
        }

        void load_binary_chunk(void* address, std::size_t count)    // override
        {
            PIKA_ASSERT((std::int64_t) count >= 0);

            if (chunks_ == nullptr ||
                count < PIKA_ZERO_COPY_SERIALIZATION_THRESHOLD || filter_)
            {
                // fall back to serialization_chunk-less archive
                this->input_container::load_binary(address, count);
            }
            else
            {
                PIKA_ASSERT(current_chunk_ != std::size_t(-1));
                PIKA_ASSERT(
                    get_chunk_type(current_chunk_) == chunk_type_pointer);

                if (get_chunk_size(current_chunk_) != count)
                {
                    PIKA_THROW_EXCEPTION(serialization_error,
                        "input_container::load_binary_chunk",
                        "archive data bstream data chunk size mismatch");
                    return;
                }

                // unfortunately we can't implement a zero copy policy on
                // the receiving end
                // as the memory was already allocated by the serialization code
                std::memcpy(
                    address, get_chunk_data(current_chunk_).pos_, count);
                ++current_chunk_;
            }
        }

        Container const& cont_;
        std::size_t current_;
        std::unique_ptr<binary_filter> filter_;
        std::size_t decompressed_size_;

        std::vector<serialization_chunk> const* chunks_;
        std::size_t current_chunk_;
        std::size_t current_chunk_size_;
    };
}}    // namespace pika::serialization
