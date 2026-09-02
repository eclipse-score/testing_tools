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
use crate::monotonic_clock::MonotonicClock;
use crate::test_context::TestContext;
use tracing::Level;
use tracing_subscriber::fmt::format::{Format, JsonFields};
use tracing_subscriber::FmtSubscriber;

/// Create a tracing subscriber that outputs logs in JSON format with monotonic timestamps.
/// # Returns
/// configured `FmtSubscriber`.
pub fn create_tracing_subscriber(
) -> FmtSubscriber<JsonFields, Format<tracing_subscriber::fmt::format::Json, MonotonicClock>> {
    FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .with_thread_ids(true)
        .with_timer(MonotonicClock::new())
        .json()
        .finish()
}

/// Test scenario arguments.
#[derive(Default)]
struct ScenarioArguments {
    /// Test scenario name.
    name: Option<String>,

    /// Test scenario input.
    input: Option<String>,
}

/// CLI arguments.
#[derive(Default)]
struct CliArguments {
    /// Test scenario arguments.
    scenario_arguments: ScenarioArguments,

    /// List scenarios.
    list_scenarios: bool,

    /// Show help.
    help: bool,
}

/// Parse CLI arguments.
///
/// * `raw_arguments` - Collected arguments from `std::env::args()`.
fn parse_cli_arguments(raw_arguments: &[String]) -> Result<CliArguments, String> {
    let mut cli_arguments = CliArguments::default();

    // Process arguments.
    // First argument (executable name) is skipped.
    let mut args_it = raw_arguments.iter().skip(1);
    while let Some(arg) = args_it.next() {
        match arg.as_str() {
            "-n" | "--name" => {
                if let Some(value) = args_it.next() {
                    cli_arguments.scenario_arguments.name = Some(value.clone());
                } else {
                    return Err("Failed to read name parameter".to_string());
                }
            }
            "-i" | "--input" => {
                if let Some(value) = args_it.next() {
                    cli_arguments.scenario_arguments.input = Some(value.clone());
                } else {
                    return Err("Failed to read input parameter".to_string());
                }
            }
            "-l" | "--list-scenarios" => {
                cli_arguments.list_scenarios = true;
            }
            "-h" | "--help" => {
                cli_arguments.help = true;
            }
            _ => {
                return Err(format!("Unknown argument provided: {arg}"));
            }
        }
    }

    Ok(cli_arguments)
}

/// Runs CLI application based on provided arguments and test context.
///
/// * `raw_arguments` - Collected arguments from `std::env::args()`.
/// * `test_context` - Test context to use.
///
/// # Examples
///
/// ```rust
/// use test_scenarios_rust::test_context::TestContext;
/// use test_scenarios_rust::scenario::ScenarioGroupImpl;
/// use test_scenarios_rust::cli::run_cli_app;
///
/// let raw_arguments = Vec::from(["example".to_string(), "--list-scenarios".to_string()]);
/// let root_group = ScenarioGroupImpl::new("root", Vec::new(), Vec::new());
/// let test_context = TestContext::new(Box::new(root_group));
///
/// let result = run_cli_app(&raw_arguments, &test_context);
/// ```
pub fn run_cli_app(raw_arguments: &[String], test_context: &TestContext) -> Result<(), String> {
    // Parse CLI arguments.
    let cli_arguments = parse_cli_arguments(raw_arguments)?;

    // Show help and return.
    if cli_arguments.help {
        eprintln!("Test scenario runner");
        eprintln!("'-n', '--name' - test scenario name");
        eprintln!("'-i', '--input' - test scenario input");
        eprintln!("'-l', '--list-scenarios' - list available scenarios");
        eprintln!("'-h', '--help' - show help");
        return Ok(());
    }

    // List scenarios and return.
    if cli_arguments.list_scenarios {
        let scenario_names = test_context.list_scenarios();
        for scenario_name in scenario_names {
            println!("{scenario_name}");
        }
        return Ok(());
    }

    // Find scenario.
    let scenario = cli_arguments.scenario_arguments;
    let scenario_name = match scenario.name {
        Some(n) => {
            if n.is_empty() {
                return Err("Test scenario name must not be empty".to_string());
            } else {
                n
            }
        }
        None => return Err("Test scenario name must be provided".to_string()),
    };

    // Check input is provided.
    let scenario_input = match scenario.input {
        Some(input) => input,
        None => return Err("Test scenario input must be provided".to_string()),
    };

    test_context.run(&scenario_name, &scenario_input)
}

#[cfg(test)]
mod tests {
    use crate::cli::{parse_cli_arguments, run_cli_app};
    use crate::scenario::{Scenario, ScenarioGroupImpl};
    use crate::test_context::TestContext;

    struct ScenarioStub {
        name: String,
    }

    impl ScenarioStub {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    impl Scenario for ScenarioStub {
        fn name(&self) -> &str {
            &self.name
        }

        fn run(&self, input: &str) -> Result<(), String> {
            match input {
                "ok" => Ok(()),
                "error" => Err("Requested error".to_string()),
                _ => Err("Unknown value".to_string()),
            }
        }
    }

    #[test]
    fn test_parse_cli_arguments_empty() {
        let raw_arguments = vec![];
        let cli_arguments = parse_cli_arguments(&raw_arguments).unwrap();

        // Default values are expected.
        assert!(cli_arguments.scenario_arguments.name.is_none());
        assert!(cli_arguments.scenario_arguments.input.is_none());
        assert!(!cli_arguments.list_scenarios);
        assert!(!cli_arguments.help);
    }

    #[test]
    fn test_parse_cli_arguments_executable_name_only() {
        let exe_name = "exe_name".to_string();
        let raw_arguments = vec![exe_name];
        let cli_arguments = parse_cli_arguments(&raw_arguments).unwrap();

        // Default values are expected.
        assert!(cli_arguments.scenario_arguments.name.is_none());
        assert!(cli_arguments.scenario_arguments.input.is_none());
        assert!(!cli_arguments.list_scenarios);
        assert!(!cli_arguments.help);
    }

    #[test]
    fn test_parse_cli_arguments_name_ok() {
        let exe_name = "exe_name".to_string();
        for arg in ["-n", "--name"] {
            let example_name = "example_name".to_string();
            let raw_arguments = vec![exe_name.clone(), arg.to_string(), example_name.clone()];
            let cli_arguments = parse_cli_arguments(&raw_arguments).unwrap();

            assert!(cli_arguments
                .scenario_arguments
                .name
                .is_some_and(|n| n == example_name));
            assert!(cli_arguments.scenario_arguments.input.is_none());
            assert!(!cli_arguments.list_scenarios);
            assert!(!cli_arguments.help);
        }
    }

    #[test]
    fn test_parse_cli_arguments_name_missing() {
        let exe_name = "exe_name".to_string();
        let raw_arguments = [exe_name, "--name".to_string()];
        let result = parse_cli_arguments(&raw_arguments);
        assert!(result.is_err_and(|e| e == "Failed to read name parameter"))
    }

    #[test]
    fn test_parse_cli_arguments_input_ok() {
        for arg in ["-i", "--input"] {
            let exe_name = "exe_name".to_string();
            let example_input = "example_input".to_string();
            let raw_arguments = [exe_name.clone(), arg.to_string(), example_input.clone()];
            let cli_arguments = parse_cli_arguments(&raw_arguments).unwrap();

            assert!(cli_arguments.scenario_arguments.name.is_none());
            assert!(cli_arguments
                .scenario_arguments
                .input
                .is_some_and(|n| n == example_input));
            assert!(!cli_arguments.list_scenarios);
            assert!(!cli_arguments.help);
        }
    }

    #[test]
    fn test_parse_cli_arguments_input_missing() {
        let exe_name = "exe_name".to_string();
        let raw_arguments = [exe_name, "--input".to_string()];
        let result = parse_cli_arguments(&raw_arguments);
        assert!(result.is_err_and(|e| e == "Failed to read input parameter"))
    }

    #[test]
    fn test_parse_cli_arguments_list_scenarios() {
        let exe_name = "exe_name".to_string();
        for arg in ["-l", "--list-scenarios"] {
            let raw_arguments = [exe_name.clone(), arg.to_string()];
            let cli_arguments = parse_cli_arguments(&raw_arguments).unwrap();

            assert!(cli_arguments.scenario_arguments.name.is_none());
            assert!(cli_arguments.scenario_arguments.input.is_none());
            assert!(cli_arguments.list_scenarios);
            assert!(!cli_arguments.help);
        }
    }

    #[test]
    fn test_parse_cli_arguments_help() {
        let exe_name = "exe_name".to_string();
        for arg in ["-h", "--help"] {
            let raw_arguments = [exe_name.clone(), arg.to_string()];
            let cli_arguments = parse_cli_arguments(&raw_arguments).unwrap();

            assert!(cli_arguments.scenario_arguments.name.is_none());
            assert!(cli_arguments.scenario_arguments.input.is_none());
            assert!(!cli_arguments.list_scenarios);
            assert!(cli_arguments.help);
        }
    }

    #[test]
    fn test_parse_cli_arguments_unknown_argument() {
        let exe_name = "exe_name".to_string();
        let raw_arguments = [exe_name, "--invalid-arg".to_string()];
        let result = parse_cli_arguments(&raw_arguments);
        assert!(result.is_err_and(|e| e == "Unknown argument provided: --invalid-arg"));
    }

    #[test]
    fn test_parse_cli_arguments_all() {
        let exe_name = "exe_name".to_string();
        let example_name = "example_name".to_string();
        let example_input = "example_input".to_string();
        let raw_arguments = [
            exe_name,
            "--help".to_string(),
            "--list-scenarios".to_string(),
            "--input".to_string(),
            example_input.clone(),
            "--name".to_string(),
            example_name.clone(),
        ];
        let cli_arguments = parse_cli_arguments(&raw_arguments).unwrap();

        assert!(cli_arguments
            .scenario_arguments
            .name
            .is_some_and(|n| n == example_name));
        assert!(cli_arguments
            .scenario_arguments
            .input
            .is_some_and(|n| n == example_input));
        assert!(cli_arguments.list_scenarios);
        assert!(cli_arguments.help);
    }

    #[test]
    fn test_run_cli_app_show_help() {
        let exe_name = "exe_name".to_string();
        let raw_arguments = vec![exe_name, "--help".to_string()];
        let root_group = ScenarioGroupImpl::new("root", vec![], vec![]);
        let test_context = TestContext::new(Box::new(root_group));

        let result = run_cli_app(&raw_arguments, &test_context);
        assert!(result.is_ok());
        // It's not possible to check stderr without unstable feature.
    }

    #[test]
    fn test_run_cli_app_list_scenarios() {
        let exe_name = "exe_name".to_string();
        let raw_arguments = vec![exe_name, "--list-scenarios".to_string()];
        let root_group = ScenarioGroupImpl::new("root", vec![], vec![]);
        let test_context = TestContext::new(Box::new(root_group));

        let result = run_cli_app(&raw_arguments, &test_context);
        assert!(result.is_ok());
        // It's not possible to check stdout without unstable feature.
    }

    #[test]
    fn test_run_cli_app_ok() {
        let exe_name = "exe_name".to_string();
        let scenario_name = "example_scenario";
        let raw_arguments = [
            exe_name,
            "--name".to_string(),
            scenario_name.to_string(),
            "--input".to_string(),
            "ok".to_string(),
        ];
        let scenario = ScenarioStub::new(scenario_name);
        let root_group = ScenarioGroupImpl::new("root", vec![Box::new(scenario)], vec![]);
        let test_context = TestContext::new(Box::new(root_group));

        let result = run_cli_app(&raw_arguments, &test_context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_cli_app_error() {
        let exe_name = "exe_name".to_string();
        let scenario_name = "example_scenario";
        let raw_arguments = [
            exe_name,
            "--name".to_string(),
            scenario_name.to_string(),
            "--input".to_string(),
            "error".to_string(),
        ];
        let scenario = ScenarioStub::new(scenario_name);
        let root_group = ScenarioGroupImpl::new("root", vec![Box::new(scenario)], vec![]);
        let test_context = TestContext::new(Box::new(root_group));

        // It's expected that test will fail due to error from `ScenarioStub`, not from `run_cli_app`.
        let result = run_cli_app(&raw_arguments, &test_context);
        assert!(result.is_err_and(|e| e == "Requested error"));
    }

    #[test]
    fn test_run_cli_app_missing_input() {
        let exe_name = "exe_name".to_string();
        let scenario_name = "example_scenario";
        let raw_arguments = [exe_name, "--name".to_string(), scenario_name.to_string()];
        let scenario = ScenarioStub::new(scenario_name);
        let root_group = ScenarioGroupImpl::new("root", vec![Box::new(scenario)], vec![]);
        let test_context = TestContext::new(Box::new(root_group));

        // It's expected that test will fail due to error from `ScenarioStub`, not from `run_cli_app`.
        let result = run_cli_app(&raw_arguments, &test_context);
        assert!(result.is_err_and(|e| e == "Test scenario input must be provided"));
    }

    #[test]
    fn test_run_cli_app_missing_name() {
        let exe_name = "exe_name".to_string();
        let scenario_name = "example_scenario";
        let raw_arguments = vec![exe_name];
        let scenario = ScenarioStub::new(scenario_name);
        let root_group = ScenarioGroupImpl::new("root", vec![Box::new(scenario)], vec![]);
        let test_context = TestContext::new(Box::new(root_group));

        let result = run_cli_app(&raw_arguments, &test_context);
        assert!(result.is_err_and(|e| e == "Test scenario name must be provided"));
    }

    #[test]
    fn test_run_cli_app_empty_name() {
        let exe_name = "exe_name".to_string();
        let scenario_name = "example_scenario";
        let raw_arguments = vec![exe_name, "--name".to_string(), String::new()];
        let scenario = ScenarioStub::new(scenario_name);
        let root_group = ScenarioGroupImpl::new("root", vec![Box::new(scenario)], vec![]);
        let test_context = TestContext::new(Box::new(root_group));

        let result = run_cli_app(&raw_arguments, &test_context);
        assert!(result.is_err_and(|e| e == "Test scenario name must not be empty"));
    }

    #[test]
    fn test_run_cli_app_invalid_name() {
        let exe_name = "exe_name".to_string();
        let scenario_name = "example_scenario";
        let raw_arguments = [
            exe_name,
            "--name".to_string(),
            "invalid_scenario".to_string(),
            "--input".to_string(),
            "".to_string(),
        ];
        let scenario = ScenarioStub::new(scenario_name);
        let root_group = ScenarioGroupImpl::new("root", vec![Box::new(scenario)], vec![]);
        let test_context = TestContext::new(Box::new(root_group));

        // It's expected that test will fail due to error from `TestContext`, not from `run_cli_app`.
        let result = run_cli_app(&raw_arguments, &test_context);
        assert!(result.is_err_and(|e| e == "Scenario invalid_scenario not found"));
    }
}
