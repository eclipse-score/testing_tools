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
#include <optional>
#include <string>
#include <vector>

/// @brief Scenario definition.
class Scenario {
   public:
    /// @brief `shared_ptr` alias.
    using Ptr = std::shared_ptr<const Scenario>;

    virtual ~Scenario() = 0;

    /// @brief Get scenario name.
    /// @return Scenario name.
    virtual std::string name() const = 0;

    /// @brief Run test scenario.
    /// @param input Test scenario input.
    virtual void run(const std::string& input) const = 0;
};

/// @brief Scenario group definition.
class ScenarioGroup {
   public:
    /// @brief `shared_ptr` alias.
    using Ptr = std::shared_ptr<const ScenarioGroup>;

    virtual ~ScenarioGroup() = 0;

    /// @brief Get scenario group name.
    /// @return Scenario group name.
    virtual std::string name() const = 0;

    /// @brief List groups from this group.
    /// @return Groups.
    virtual const std::vector<ScenarioGroup::Ptr>& groups() const = 0;

    /// @brief List scenarios from this group.
    /// @return Scenarios.
    virtual const std::vector<Scenario::Ptr>& scenarios() const = 0;

    /// @brief Find scenario by name.
    /// @param name Name of scenario to find.
    /// @return Found scenario. Empty if not found.
    virtual std::optional<Scenario::Ptr> find_scenario(const std::string& name) const = 0;
};

/// @brief Common scenario group definition.
class ScenarioGroupImpl : public ScenarioGroup {
   public:
    ScenarioGroupImpl(const std::string& name, const std::vector<Scenario::Ptr>& scenarios,
                      const std::vector<ScenarioGroup::Ptr>& groups);

    ~ScenarioGroupImpl() override;

    virtual std::string name() const override;

    virtual const std::vector<ScenarioGroup::Ptr>& groups() const override;

    virtual const std::vector<Scenario::Ptr>& scenarios() const override;

    virtual std::optional<Scenario::Ptr> find_scenario(const std::string& name) const override;

   protected:
    std::string name_;
    std::vector<Scenario::Ptr> scenarios_;
    std::vector<ScenarioGroup::Ptr> groups_;
};
