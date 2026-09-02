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

#include <memory>
#include <string>
#include <vector>

#include "scenario.hpp"

/// @brief Test context. Responsible for listing and running scenarios.
class TestContext {
   public:
    /// @brief Create test context.
    /// @param root_group Root test scenario group.
    explicit TestContext(ScenarioGroup::Ptr root_group);

    /// @brief Run test scenario.
    /// @param name Test scenario name.
    /// @param input Test scenario input.
    void run(const std::string& name, const std::string& input) const;

    /// @brief List available scenarios.
    /// @return Names of available scenarios.
    std::vector<std::string> list_scenarios() const;

   private:
    ScenarioGroup::Ptr root_group_;
};
