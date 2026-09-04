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
#pragma once

#include "test_context.hpp"

/// @brief Test scenario arguments.
struct ScenarioArguments {
    /// @brief Test scenario name.
    std::optional<std::string> name;

    /// @brief Test scenario input.
    std::optional<std::string> input;
};

/// @brief CLI arguments.
struct CliArguments {
    /// @brief Test scenario arguments.
    ScenarioArguments scenario_arguments;

    /// @brief List scenarios.
    bool list_scenarios;

    /// @brief Show help.
    bool help;
};

/// @brief Parse CLI arguments.
/// @param raw_arguments Collected arguments from `argv`.
/// @return Parsed CLI arguments.
CliArguments parse_cli_arguments(const std::vector<std::string>& raw_arguments);

/// @brief Runs CLI application based on provided arguments and test context.
/// @param raw_arguments Collected arguments from `argv`.
/// @param test_context Test context to use.
void run_cli_app(const std::vector<std::string>& raw_arguments, const TestContext& test_context);
