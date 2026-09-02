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
/// Scenario definition.
pub trait Scenario {
    /// Get scenario name.
    fn name(&self) -> &str;

    /// Run test scenario.
    ///
    /// * `input` - Test scenario input.
    fn run(&self, input: &str) -> Result<(), String>;
}

/// Scenario group definition.
pub trait ScenarioGroup {
    /// Get scenario group name.
    fn name(&self) -> &str;

    /// List groups from this group.
    fn groups(&self) -> &Vec<Box<dyn ScenarioGroup>>;

    /// List scenarios from this group.
    fn scenarios(&self) -> &Vec<Box<dyn Scenario>>;

    /// Find scenario by name.
    ///
    /// * `name` - Name of the scenario to find.
    fn find_scenario(&self, name: &str) -> Option<&dyn Scenario>;
}

/// Common scenario group definition.
pub struct ScenarioGroupImpl {
    name: String,
    scenarios: Vec<Box<dyn Scenario>>,
    groups: Vec<Box<dyn ScenarioGroup>>,
}

impl ScenarioGroupImpl {
    /// Create common scenario group definition.
    ///
    /// * `name` - Name of the scenario group.
    /// * `scenario` - Scenarios in this group.
    /// * `groups` - Groups in this group.
    pub fn new(
        name: &str,
        scenarios: Vec<Box<dyn Scenario>>,
        groups: Vec<Box<dyn ScenarioGroup>>,
    ) -> Self {
        ScenarioGroupImpl {
            name: name.to_string(),
            scenarios,
            groups,
        }
    }
}

impl ScenarioGroup for ScenarioGroupImpl {
    fn name(&self) -> &str {
        self.name.as_str()
    }

    fn groups(&self) -> &Vec<Box<dyn ScenarioGroup>> {
        &self.groups
    }

    fn scenarios(&self) -> &Vec<Box<dyn Scenario>> {
        &self.scenarios
    }

    fn find_scenario(&self, name: &str) -> Option<&dyn Scenario> {
        let split: Vec<&str> = name.split('.').collect();
        if split.len() == 1 {
            for scenario in &self.scenarios {
                if scenario.name() == name {
                    return Some(scenario.as_ref());
                }
            }
        } else {
            for group in &self.groups {
                if group.name() == split[0] {
                    return group.find_scenario(split[1..].join(".").as_str());
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use crate::scenario::{Scenario, ScenarioGroup, ScenarioGroupImpl};

    struct ScenarioStub {
        name: String,
    }

    impl Scenario for ScenarioStub {
        fn name(&self) -> &str {
            &self.name
        }

        fn run(&self, _input: &str) -> Result<(), String> {
            Ok(())
        }
    }

    fn init_group() -> Box<dyn ScenarioGroup> {
        let scenario_inner = ScenarioStub {
            name: "inner_scenario".to_string(),
        };
        let group_inner =
            ScenarioGroupImpl::new("inner_group", vec![Box::new(scenario_inner)], vec![]);
        let scenario_outer = ScenarioStub {
            name: "outer_scenario".to_string(),
        };
        let group_outer = ScenarioGroupImpl::new(
            "outer_group",
            vec![Box::new(scenario_outer)],
            vec![Box::new(group_inner)],
        );

        Box::new(group_outer)
    }

    #[test]
    fn test_group_name_ok() {
        let group = init_group();
        assert_eq!(group.name(), "outer_group");
    }

    #[test]
    fn test_groups_ok() {
        let group = init_group();

        let groups_result = group.groups();
        assert_eq!(groups_result.len(), 1);
        assert_eq!(groups_result[0].name(), "inner_group");

        let scenarios_result = groups_result[0].scenarios();
        assert_eq!(scenarios_result.len(), 1);
        assert_eq!(scenarios_result[0].name(), "inner_scenario");
    }

    #[test]
    fn test_scenarios_ok() {
        let group = init_group();

        let groups_result = group.groups();
        let scenarios_result = groups_result[0].scenarios();
        assert_eq!(scenarios_result.len(), 1);
        assert_eq!(scenarios_result[0].name(), "inner_scenario");
    }

    #[test]
    fn test_find_scenario_ok() {
        let group = init_group();
        let scenario1 = group.find_scenario("inner_group.inner_scenario");
        assert!(scenario1.is_some_and(|s| s.name() == "inner_scenario"));
        let scenario2 = group.find_scenario("outer_scenario");
        assert!(scenario2.is_some_and(|s| s.name() == "outer_scenario"));
    }

    #[test]
    fn test_find_scenario_empty_input() {
        let group = init_group();
        let scenario = group.find_scenario("");
        assert!(scenario.is_none());
    }

    #[test]
    fn test_find_scenario_invalid_name() {
        let group = init_group();
        let scenario = group.find_scenario("invalid_group.invalid_scenario");
        assert!(scenario.is_none());
    }
}
