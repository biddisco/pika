//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>
#include <pika/functional/function.hpp>
#include <pika/program_options/options_description.hpp>
#include <pika/program_options/variables_map.hpp>
#include <pika/runtime_configuration/runtime_configuration.hpp>
#include <pika/util/manage_config.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pika::detail {
    enum class command_line_handling_result
    {
        success,    // All went well, continue starting the runtime
        exit,       // All went well, but should exit (e.g. --pika:help was given)
    };

    struct command_line_handling
    {
        command_line_handling(pika::util::runtime_configuration rtcfg,
            std::vector<std::string> ini_config,
            pika::util::detail::function<int(pika::program_options::variables_map& vm)> pika_main_f)
          : rtcfg_(rtcfg)
          , ini_config_(ini_config)
          // NOLINTNEXTLINE(bugprone-throw-keyword-missing)
          , pika_main_f_(pika_main_f)
          , num_threads_(1)
          , num_cores_(1)
          , pu_step_(1)
          , pu_offset_(std::size_t(-1))
          , numa_sensitive_(0)
          , use_process_mask_(true)
          , cmd_line_parsed_(false)
          , info_printed_(false)
          , version_printed_(false)
        {
        }

        command_line_handling_result call(
            pika::program_options::options_description const& desc_cmdline, int argc,
            char const* const* argv);

        pika::program_options::variables_map vm_;
        pika::util::runtime_configuration rtcfg_;

        std::vector<std::string> ini_config_;
        pika::util::detail::function<int(pika::program_options::variables_map& vm)> pika_main_f_;

        std::size_t num_threads_;
        std::size_t num_cores_;
        std::size_t pu_step_;
        std::size_t pu_offset_;
        std::string scheduler_;
        std::string affinity_domain_;
        std::string affinity_bind_;
        std::size_t numa_sensitive_;
        bool use_process_mask_;
        std::string process_mask_;
        bool cmd_line_parsed_;
        bool info_printed_;
        bool version_printed_;

    protected:
        // Helper functions for checking command line options
        void check_affinity_domain() const;
        void check_affinity_description() const;
        void check_pu_offset() const;
        void check_pu_step() const;

        void handle_arguments(detail::manage_config& cfgmap,
            pika::program_options::variables_map& vm, std::vector<std::string>& ini_config);

        void update_logging_settings(
            pika::program_options::variables_map& vm, std::vector<std::string>& ini_config);

        void store_command_line(int argc, char const* const* argv);
        void store_unregistered_options(
            std::string const& cmd_name, std::vector<std::string> const& unregistered_options);
        bool handle_help_options(pika::program_options::options_description const& help);

        void handle_attach_debugger();

        std::vector<std::string> preprocess_config_settings(int argc, char const* const* argv);
    };
}    // namespace pika::detail
