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
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "cli.hpp"
#include "common.hpp"
#include "scenario.hpp"
#include "test_context.hpp"

TEST(parse_cli_arguments, empty) {
    std::vector<std::string> raw_arguments;
    auto cli_arguments{parse_cli_arguments(raw_arguments)};

    // Default values are expected.
    ASSERT_FALSE(cli_arguments.scenario_arguments.name.has_value());
    ASSERT_FALSE(cli_arguments.scenario_arguments.input.has_value());
    ASSERT_FALSE(cli_arguments.list_scenarios);
    ASSERT_FALSE(cli_arguments.help);
}

TEST(parse_cli_arguments, executable_name_only) {
    std::string exe_name{"exe_name"};
    std::vector<std::string> raw_arguments{exe_name};
    auto cli_arguments{parse_cli_arguments(raw_arguments)};

    // Default values are expected.
    ASSERT_FALSE(cli_arguments.scenario_arguments.name.has_value());
    ASSERT_FALSE(cli_arguments.scenario_arguments.input.has_value());
    ASSERT_FALSE(cli_arguments.list_scenarios);
    ASSERT_FALSE(cli_arguments.help);
}

TEST(parse_cli_arguments, name_ok) {
    std::vector<std::string> args{"-n", "--name"};
    for (auto&& arg : args) {
        std::string exe_name{"exe_name"};
        std::string example_name{"example_name"};
        std::vector<std::string> raw_arguments{exe_name, arg, example_name};
        auto cli_arguments{parse_cli_arguments(raw_arguments)};

        ASSERT_TRUE(cli_arguments.scenario_arguments.name.has_value());
        ASSERT_EQ(*cli_arguments.scenario_arguments.name, example_name);
        ASSERT_FALSE(cli_arguments.scenario_arguments.input.has_value());
        ASSERT_FALSE(cli_arguments.list_scenarios);
        ASSERT_FALSE(cli_arguments.help);
    }
}

TEST(parse_cli_arguments, name_missing) {
    std::string exe_name{"exe_name"};
    std::vector<std::string> raw_arguments{exe_name, "--name"};
    SHOULD_THROW_RE(auto cli_arguments{parse_cli_arguments(raw_arguments)},
                    "Failed to read name parameter");
}

TEST(parse_cli_arguments, input_ok) {
    std::vector<std::string> args{"-i", "--input"};
    for (auto&& arg : args) {
        std::string exe_name{"exe_name"};
        std::string example_input{"example_input"};
        std::vector<std::string> raw_arguments{exe_name, arg, example_input};
        auto cli_arguments{parse_cli_arguments(raw_arguments)};

        ASSERT_FALSE(cli_arguments.scenario_arguments.name.has_value());
        ASSERT_TRUE(cli_arguments.scenario_arguments.input.has_value());
        ASSERT_EQ(*cli_arguments.scenario_arguments.input, example_input);
        ASSERT_FALSE(cli_arguments.list_scenarios);
        ASSERT_FALSE(cli_arguments.help);
    }
}

TEST(parse_cli_arguments, input_missing) {
    std::string exe_name{"exe_name"};
    std::vector<std::string> raw_arguments{exe_name, "--input"};
    SHOULD_THROW_RE(auto cli_arguments{parse_cli_arguments(raw_arguments)},
                    "Failed to read input parameter");
}

TEST(parse_cli_arguments, list_scenarios) {
    std::vector<std::string> args{"-l", "--list-scenarios"};
    for (auto&& arg : args) {
        std::string exe_name{"exe_name"};
        std::vector<std::string> raw_arguments{exe_name, arg};
        auto cli_arguments{parse_cli_arguments(raw_arguments)};

        ASSERT_FALSE(cli_arguments.scenario_arguments.name.has_value());
        ASSERT_FALSE(cli_arguments.scenario_arguments.input.has_value());
        ASSERT_TRUE(cli_arguments.list_scenarios);
        ASSERT_FALSE(cli_arguments.help);
    }
}

TEST(parse_cli_arguments, help) {
    std::vector<std::string> args{"-h", "--help"};
    for (auto&& arg : args) {
        std::string exe_name{"exe_name"};
        std::vector<std::string> raw_arguments{exe_name, arg};
        auto cli_arguments{parse_cli_arguments(raw_arguments)};

        ASSERT_FALSE(cli_arguments.scenario_arguments.name.has_value());
        ASSERT_FALSE(cli_arguments.scenario_arguments.input.has_value());
        ASSERT_FALSE(cli_arguments.list_scenarios);
        ASSERT_TRUE(cli_arguments.help);
    }
}

TEST(parse_cli_arguments, unknown_argument) {
    std::string exe_name{"exe_name"};
    std::vector<std::string> raw_arguments{exe_name, "--invalid-arg"};
    SHOULD_THROW_RE(auto cli_arguments{parse_cli_arguments(raw_arguments)},
                    "Unknown argument provided");
}

TEST(parse_cli_arguments, all) {
    std::string exe_name{"exe_name"};
    std::string example_name{"example_name"};
    std::string example_input{"example_input"};
    std::vector<std::string> raw_arguments{
        exe_name, "--help", "--list-scenarios", "--input", example_input, "--name", example_name,
    };
    auto cli_arguments{parse_cli_arguments(raw_arguments)};

    ASSERT_TRUE(cli_arguments.scenario_arguments.name.has_value());
    ASSERT_EQ(*cli_arguments.scenario_arguments.name, example_name);
    ASSERT_TRUE(cli_arguments.scenario_arguments.input.has_value());
    ASSERT_EQ(*cli_arguments.scenario_arguments.input, example_input);
    ASSERT_TRUE(cli_arguments.list_scenarios);
    ASSERT_TRUE(cli_arguments.help);
}

TEST(run_cli_app, show_help) {
    std::string exe_name{"exe_name"};
    std::vector<std::string> raw_arguments{exe_name, "--help"};
    std::vector<Scenario::Ptr> scenarios;
    std::vector<ScenarioGroup::Ptr> groups;
    ScenarioGroup::Ptr root_group{new ScenarioGroupImpl{"root", scenarios, groups}};
    TestContext test_context{root_group};

    run_cli_app(raw_arguments, test_context);
    // TODO: capture stderr.
}

TEST(run_cli_app, list_scenarios) {
    std::string exe_name{"exe_name"};
    std::vector<std::string> raw_arguments{exe_name, "--list-scenarios"};
    std::vector<Scenario::Ptr> scenarios;
    std::vector<ScenarioGroup::Ptr> groups;
    ScenarioGroup::Ptr root_group{new ScenarioGroupImpl{"root", scenarios, groups}};
    TestContext test_context{root_group};

    run_cli_app(raw_arguments, test_context);
    // TODO: capture stderr.
}

TEST(run_cli_app, ok) {
    std::string exe_name{"exe_name"};
    std::string scenario_name{"example_scenario"};
    std::vector<std::string> raw_arguments{
        exe_name, "--name", scenario_name, "--input", "ok",
    };
    Scenario::Ptr scenario{new ScenarioStub{scenario_name}};
    std::vector<Scenario::Ptr> scenarios{scenario};
    std::vector<ScenarioGroup::Ptr> groups;
    ScenarioGroup::Ptr root_group{new ScenarioGroupImpl{"root", scenarios, groups}};
    TestContext test_context{root_group};

    run_cli_app(raw_arguments, test_context);
}

TEST(run_cli_app, error) {
    std::string exe_name{"exe_name"};
    std::string scenario_name{"example_scenario"};
    std::vector<std::string> raw_arguments{
        exe_name, "--name", scenario_name, "--input", "error",
    };
    Scenario::Ptr scenario{new ScenarioStub{scenario_name}};
    std::vector<Scenario::Ptr> scenarios{scenario};
    std::vector<ScenarioGroup::Ptr> groups;
    ScenarioGroup::Ptr root_group{new ScenarioGroupImpl{"root", scenarios, groups}};
    TestContext test_context{root_group};

    // It's expected that test will fail due to error from `ScenarioStub`, not from `run_cli_app`.
    SHOULD_THROW_RE(run_cli_app(raw_arguments, test_context), "Requested error");
}

TEST(run_cli_app, missing_input) {
    std::string exe_name{"exe_name"};
    std::string scenario_name{"example_scenario"};
    std::vector<std::string> raw_arguments{exe_name, "--name", scenario_name};
    Scenario::Ptr scenario{new ScenarioStub{scenario_name}};
    std::vector<Scenario::Ptr> scenarios{scenario};
    std::vector<ScenarioGroup::Ptr> groups;
    ScenarioGroup::Ptr root_group{new ScenarioGroupImpl{"root", scenarios, groups}};
    TestContext test_context{root_group};

    SHOULD_THROW_RE(run_cli_app(raw_arguments, test_context),
                    "Test scenario input must be provided");
}

TEST(run_cli_app, missing_name) {
    std::string exe_name{"exe_name"};
    std::string scenario_name{"example_scenario"};
    std::vector<std::string> raw_arguments{exe_name};
    Scenario::Ptr scenario{new ScenarioStub{scenario_name}};
    std::vector<Scenario::Ptr> scenarios{scenario};
    std::vector<ScenarioGroup::Ptr> groups;
    ScenarioGroup::Ptr root_group{new ScenarioGroupImpl{"root", scenarios, groups}};
    TestContext test_context{root_group};

    SHOULD_THROW_RE(run_cli_app(raw_arguments, test_context),
                    "Test scenario name must be provided");
}

TEST(run_cli_app, empty_name) {
    std::string exe_name{"exe_name"};
    std::string scenario_name{"example_scenario"};
    std::vector<std::string> raw_arguments{exe_name, "--name", ""};
    Scenario::Ptr scenario{new ScenarioStub{scenario_name}};
    std::vector<Scenario::Ptr> scenarios{scenario};
    std::vector<ScenarioGroup::Ptr> groups;
    ScenarioGroup::Ptr root_group{new ScenarioGroupImpl{"root", scenarios, groups}};
    TestContext test_context{root_group};

    SHOULD_THROW_RE(run_cli_app(raw_arguments, test_context),
                    "Test scenario name must not be empty");
}

TEST(run_cli_app, invalid_name) {
    std::string exe_name{"exe_name"};
    std::string scenario_name{"example_scenario"};
    std::vector<std::string> raw_arguments{exe_name, "--name", "invalid_scenario", "--input", ""};
    Scenario::Ptr scenario{new ScenarioStub{scenario_name}};
    std::vector<Scenario::Ptr> scenarios{scenario};
    std::vector<ScenarioGroup::Ptr> groups;
    ScenarioGroup::Ptr root_group{new ScenarioGroupImpl{"root", scenarios, groups}};
    TestContext test_context{root_group};

    // It's expected that test will fail due to error from `TestContext`, not from `run_cli_app`.
    SHOULD_THROW_RE(run_cli_app(raw_arguments, test_context),
                    "Scenario invalid_scenario not found");
}

#undef SHOULD_THROW_RE
#undef SHOULD_THROW
