// *******************************************************************************
// Copyright (c) 2026 Contributors to the Eclipse Foundation
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

#include <cli.hpp>
#include <scenario.hpp>
#include <test_context.hpp>
#include <tracing.hpp>

#include <nlohmann/json.hpp>

#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const std::string kTargetName{"examples::basic::list"};

std::vector<std::string> parse_items(const std::string& input) {
    nlohmann::json parsed;
    try {
        parsed = nlohmann::json::parse(input);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error{"invalid input: " + std::string{e.what()}};
    }
    if (!parsed.contains("items")) {
        throw std::runtime_error{"invalid input: missing 'items' field"};
    }
    return parsed.at("items").get<std::vector<std::string>>();
}

// Logs each item with its index via structured tracing.
class EnumerateScenario final : public Scenario {
   public:
    std::string name() const override { return "enumerate"; }

    void run(const std::string& input) const override {
        auto items{parse_items(input)};
        for (std::size_t index = 0; index < items.size(); ++index) {
            TRACING_INFO(kTargetName, std::pair{std::string{"index"}, index},
                         std::pair{std::string{"item"}, items[index]});
        }
    }
};

// Fails if the input has no items.
class RequireNonEmptyScenario final : public Scenario {
   public:
    std::string name() const override { return "require_non_empty"; }

    void run(const std::string& input) const override {
        auto items{parse_items(input)};
        if (items.empty()) {
            throw std::runtime_error{"items must not be empty"};
        }
        TRACING_INFO(kTargetName, std::pair{std::string{"count"}, items.size()});
    }
};

}  // namespace

int main(int argc, char* argv[]) {
    std::vector<std::string> raw_arguments{argv, argv + argc};

    ScenarioGroup::Ptr list_group{new ScenarioGroupImpl{
        "list",
        std::vector<Scenario::Ptr>{std::make_shared<EnumerateScenario>(),
                                    std::make_shared<RequireNonEmptyScenario>()},
        std::vector<ScenarioGroup::Ptr>{}}};
    ScenarioGroup::Ptr root_group{new ScenarioGroupImpl{
        "root", std::vector<Scenario::Ptr>{}, std::vector<ScenarioGroup::Ptr>{list_group}}};
    TestContext test_context{root_group};

    try {
        run_cli_app(raw_arguments, test_context);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
