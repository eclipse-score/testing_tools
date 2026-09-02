// *******************************************************************************
// Copyright (c) 2025 Contributors to the Eclipse Foundation
//
// See the NOTICE file(s) distributed with this work for additional
// information regarding copyright ownership.
//
// This program and the accompanying materials are made available under the
// terms of the Apache License Version 2.0 which is available at
// https://www.apache.org/licenses/LICENSE-2.0
//
// SPDX-License-Identifier: Apache-2.0
// *******************************************************************************
#include "cli.hpp"

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

CliArguments parse_cli_arguments(const std::vector<std::string>& raw_arguments) {
    CliArguments cli_arguments{};

    // Process arguments.
    // First argument (executable name) is skipped.
    auto args_it = raw_arguments.begin();
    while (++args_it < raw_arguments.end()) {
        auto arg{*args_it};
        if (arg == "-n" || arg == "--name") {
            if (++args_it != raw_arguments.end()) {
                cli_arguments.scenario_arguments.name = *args_it;
            } else {
                throw std::runtime_error{"Failed to read name parameter"};
            }
        } else if (arg == "-i" || arg == "--input") {
            if (++args_it != raw_arguments.end()) {
                cli_arguments.scenario_arguments.input = *args_it;
            } else {
                throw std::runtime_error{"Failed to read input parameter"};
            }
        } else if (arg == "-l" || arg == "--list-scenarios") {
            cli_arguments.list_scenarios = true;
        } else if (arg == "-h" || arg == "--help") {
            cli_arguments.help = true;
        } else {
            throw std::runtime_error{"Unknown argument provided"};
        }
    }

    return cli_arguments;
}

void run_cli_app(const std::vector<std::string>& raw_arguments, const TestContext& test_context) {
    // Parse CLI arguments.
    auto cli_arguments{parse_cli_arguments(raw_arguments)};

    // Show help and return.
    if (cli_arguments.help) {
        std::cerr << "Test scenario runner" << std::endl;
        std::cerr << "'-n', '--name' - test scenario name" << std::endl;
        std::cerr << "'-i', '--input' - test scenario input" << std::endl;
        std::cerr << "'-l', '--list-scenarios' - list available scenarios" << std::endl;
        std::cerr << "'-h', '--help' - show help" << std::endl;
        return;
    }

    // List scenarios and return.
    if (cli_arguments.list_scenarios) {
        auto scenario_names{test_context.list_scenarios()};
        for (auto&& scenario_name : scenario_names) {
            std::cout << scenario_name << std::endl;
        }
        return;
    }

    // Find scenario.
    auto scenario{cli_arguments.scenario_arguments};
    if (!scenario.name.has_value()) {
        throw std::runtime_error{"Test scenario name must be provided"};
    } else if (scenario.name.has_value() && scenario.name->empty()) {
        throw std::runtime_error{"Test scenario name must not be empty"};
    }

    // Check input is provided.
    if (!scenario.input.has_value()) {
        throw std::runtime_error{"Test scenario input must be provided"};
    }

    test_context.run(*scenario.name, *scenario.input);
}
