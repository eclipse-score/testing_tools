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

#include "common.hpp"
#include "scenario.hpp"

namespace {

ScenarioGroup::Ptr init_group() {
    Scenario::Ptr scenario_inner{new ScenarioStub{"inner_scenario"}};
    std::vector<Scenario::Ptr> scenarios_inner{scenario_inner};
    std::vector<ScenarioGroup::Ptr> groups_inner;
    ScenarioGroup::Ptr group_inner{
        new ScenarioGroupImpl{"inner_group", scenarios_inner, groups_inner}};

    Scenario::Ptr scenario_outer{new ScenarioStub{"outer_scenario"}};
    std::vector<Scenario::Ptr> scenarios_outer{scenario_outer};
    std::vector<ScenarioGroup::Ptr> groups_outer{group_inner};
    ScenarioGroup::Ptr group_outer{
        new ScenarioGroupImpl{"outer_group", scenarios_outer, groups_outer}};

    return group_outer;
}

}  // namespace

TEST(group_impl, group_name_ok) {
    auto group{init_group()};
    ASSERT_EQ(group->name(), "outer_group");
}

TEST(group_impl, groups_ok) {
    auto group{init_group()};

    auto groups_result{group->groups()};
    ASSERT_EQ(groups_result.size(), 1);
    ASSERT_EQ(groups_result[0]->name(), "inner_group");

    auto scenarios_result{groups_result[0]->scenarios()};
    ASSERT_EQ(scenarios_result.size(), 1);
    ASSERT_EQ(scenarios_result[0]->name(), "inner_scenario");
}

TEST(group_impl, scenarios_ok) {
    auto group{init_group()};

    auto groups_result{group->groups()};
    auto scenarios_result{groups_result[0]->scenarios()};
    ASSERT_EQ(scenarios_result.size(), 1);
    ASSERT_EQ(scenarios_result[0]->name(), "inner_scenario");
}

TEST(group_impl, find_scenario_ok) {
    auto group{init_group()};
    auto scenario1{group->find_scenario("inner_group.inner_scenario")};
    ASSERT_TRUE(scenario1.has_value());
    ASSERT_EQ((*scenario1)->name(), "inner_scenario");
    auto scenario2{group->find_scenario("outer_scenario")};
    ASSERT_TRUE(scenario2.has_value());
    ASSERT_EQ((*scenario2)->name(), "outer_scenario");
}

TEST(group_impl, find_scenario_empty_input) {
    auto group{init_group()};
    auto scenario{group->find_scenario("")};
    ASSERT_FALSE(scenario.has_value());
}

TEST(group_impl, find_scenario_invalid_name) {
    auto group{init_group()};
    auto scenario{group->find_scenario("invalid_group.invalid_scenario")};
    ASSERT_FALSE(scenario.has_value());
}
