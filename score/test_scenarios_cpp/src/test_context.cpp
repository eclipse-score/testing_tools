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
#include "test_context.hpp"

#include <sstream>

namespace {

std::string join_name(const std::string& left, const std::string& right) {
    if (!left.empty()) {
        std::stringstream ss;
        ss << left << "." << right;
        return ss.str();
    } else {
        return right;
    }
}

std::vector<std::string> list_scenarios_recursive(ScenarioGroup::Ptr scenario_group,
                                                  std::string prefix) {
    std::vector<std::string> names;

    auto groups{scenario_group->groups()};
    for (auto&& group : groups) {
        auto new_prefix{join_name(prefix, group->name())};
        auto result{list_scenarios_recursive(group, new_prefix)};
        names.insert(names.end(), result.begin(), result.end());
    }

    auto scenarios{scenario_group->scenarios()};
    for (auto&& scenario : scenarios) {
        auto scenario_name{join_name(prefix, scenario->name())};
        names.push_back(scenario_name);
    }

    return names;
}
}  // namespace

TestContext::TestContext(ScenarioGroup::Ptr root_group) : root_group_{root_group} {}

void TestContext::run(const std::string& name, const std::string& input) const {
    auto scenario{root_group_->find_scenario(name)};
    if (!scenario.has_value()) {
        std::stringstream ss;
        ss << "Scenario " << name << " not found";
        throw std::runtime_error{ss.str()};
    }
    scenario->get()->run(input);
}

std::vector<std::string> TestContext::list_scenarios() const {
    return list_scenarios_recursive(root_group_, "");
}
