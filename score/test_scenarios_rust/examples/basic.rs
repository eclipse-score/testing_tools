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

use std::process::ExitCode;

use serde::Deserialize;
use tracing::info;

use test_scenarios_rust::cli::{create_tracing_subscriber, run_cli_app};
use test_scenarios_rust::scenario::{Scenario, ScenarioGroupImpl};
use test_scenarios_rust::test_context::TestContext;

#[derive(Deserialize)]
struct ItemsInput {
    items: Vec<String>,
}

impl ItemsInput {
    fn parse(input: &str) -> Result<Self, String> {
        serde_json::from_str(input).map_err(|e| format!("invalid input: {e}"))
    }
}

/// Logs each item with its index via structured tracing.
struct EnumerateScenario;

impl Scenario for EnumerateScenario {
    fn name(&self) -> &str {
        "enumerate"
    }

    fn run(&self, input: &str) -> Result<(), String> {
        let parsed = ItemsInput::parse(input)?;
        for (index, item) in parsed.items.iter().enumerate() {
            info!(index, item = item.as_str());
        }
        Ok(())
    }
}

/// Fails if the input has no items.
struct RequireNonEmptyScenario;

impl Scenario for RequireNonEmptyScenario {
    fn name(&self) -> &str {
        "require_non_empty"
    }

    fn run(&self, input: &str) -> Result<(), String> {
        let parsed = ItemsInput::parse(input)?;
        if parsed.items.is_empty() {
            return Err("items must not be empty".to_string());
        }
        info!(count = parsed.items.len());
        Ok(())
    }
}

fn main() -> ExitCode {
    tracing::subscriber::set_global_default(create_tracing_subscriber())
        .expect("Setting default subscriber failed!");

    let raw_arguments: Vec<String> = std::env::args().collect();

    let list_group = ScenarioGroupImpl::new(
        "list",
        vec![Box::new(EnumerateScenario), Box::new(RequireNonEmptyScenario)],
        Vec::new(),
    );
    let root_group = ScenarioGroupImpl::new("root", Vec::new(), vec![Box::new(list_group)]);
    let test_context = TestContext::new(Box::new(root_group));

    match run_cli_app(&raw_arguments, &test_context) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
