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
use crate::scenario::ScenarioGroup;

fn join_name(left: &str, right: &str) -> String {
    if !left.is_empty() {
        format!("{left}.{right}")
    } else {
        right.to_string()
    }
}

fn list_scenarios_recursive(group: &dyn ScenarioGroup, prefix: String) -> Vec<String> {
    let mut names = Vec::new();

    let groups = group.groups();
    for group in groups {
        let new_prefix = join_name(&prefix, group.name());
        let result = list_scenarios_recursive(group.as_ref(), new_prefix);
        names.extend(result);
    }

    let scenarios = group.scenarios();
    for scenario in scenarios {
        let scenario_name = join_name(&prefix, scenario.name());
        names.push(scenario_name);
    }

    names
}

/// Test context. Responsible for listing and running scenarios.
pub struct TestContext {
    root_group: Box<dyn ScenarioGroup>,
}

impl TestContext {
    /// Create test context.
    ///
    /// * `root_group` - Root test scenario group.
    pub fn new(root_group: Box<dyn ScenarioGroup>) -> Self {
        TestContext { root_group }
    }

    /// Run test scenario.
    ///
    /// * `name` - Name of the scenario to run.
    /// * `input` - Test scenario input.
    pub fn run(&self, name: &str, input: &str) -> Result<(), String> {
        let scenario = self.root_group.find_scenario(name);
        match scenario {
            Some(scenario) => scenario.run(input),
            None => Err(format!("Scenario {name} not found")),
        }
    }

    /// List available scenarios.
    pub fn list_scenarios(&self) -> Vec<String> {
        list_scenarios_recursive(self.root_group.as_ref(), "".to_string())
    }
}

#[cfg(test)]
mod tests {
    use crate::scenario::{Scenario, ScenarioGroup, ScenarioGroupImpl};
    use crate::test_context::TestContext;

    struct ScenarioStub {
        name: String,
    }

    impl Scenario for ScenarioStub {
        fn name(&self) -> &str {
            &self.name
        }

        fn run(&self, input: &str) -> Result<(), String> {
            match input {
                "ok" => Ok(()),
                "error" => Err("Requested error".to_string()),
                _ => Err("Missing input".to_string()),
            }
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
    fn test_run_some_input_ok() {
        let root_group = init_group();
        let context = TestContext::new(root_group);
        let result = context.run("inner_group.inner_scenario", "ok");

        assert!(result.is_ok());
    }

    #[test]
    fn test_run_some_input_err() {
        let root_group = init_group();
        let context = TestContext::new(root_group);
        let result = context.run("inner_group.inner_scenario", "error");

        assert!(result.is_err_and(|e| e == "Requested error"));
    }

    #[test]
    fn test_run_not_found() {
        let root_group = init_group();
        let context = TestContext::new(root_group);
        let result = context.run("some_scenario", "");

        assert!(result.is_err_and(|e| e == "Scenario some_scenario not found"));
    }

    #[test]
    fn test_list_scenarios_ok() {
        let root_group = init_group();
        let context = TestContext::new(root_group);
        let result = context.list_scenarios();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "inner_group.inner_scenario");
        assert_eq!(result[1], "outer_scenario");
    }

    #[test]
    fn test_list_scenarios_empty() {
        let root_group = ScenarioGroupImpl::new("root", vec![], vec![]);
        let context = TestContext::new(Box::new(root_group));
        let result = context.list_scenarios();

        assert_eq!(result.len(), 0);
    }
}
