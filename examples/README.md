# testing_tools Examples

Example programs demonstrating `test_scenarios_cpp` and `test_scenarios_rust`. C++ examples are
located in [score/test_scenarios_cpp/examples](../score/test_scenarios_cpp/examples), and Rust
examples in [score/test_scenarios_rust/examples](../score/test_scenarios_rust/examples). In this
directory there are Bazel aliases for convenience.

## Running Examples

```bash
bazel run //examples:cpp_basic -- --list-scenarios
bazel run //examples:cpp_basic -- --name list.enumerate --input '{"items":["a","b","c"]}'
bazel run //examples:cpp_basic -- --name list.require_non_empty --input '{"items":["a"]}'
bazel run //examples:cpp_basic -- --name list.require_non_empty --input '{"items":[]}'

bazel run //examples:rust_basic -- --list-scenarios
bazel run //examples:rust_basic -- --name list.enumerate --input '{"items":["a","b","c"]}'
bazel run //examples:rust_basic -- --name list.require_non_empty --input '{"items":["a"]}'
bazel run //examples:rust_basic -- --name list.require_non_empty --input '{"items":[]}'
```

## Available Examples

### basic (C++ and Rust)

A foundational example demonstrating the core building blocks of the library:

- Implementing `Scenario` — both scenarios parse a JSON `{"items": [...]}` input (failing
  genuinely on malformed JSON or a missing `items` field). `list.enumerate` logs each item with
  its index via structured tracing; `list.require_non_empty` fails genuinely if `items` is empty,
  otherwise logs the count
- Structured logging via the library's own tracing support (`TRACING_INFO` in C++,
  `tracing::info!` over a subscriber built with `create_tracing_subscriber()` in Rust), with
  timestamps from the library's monotonic clock
- Grouping scenarios with `ScenarioGroupImpl`, including a nested group (`list`) to show how
  nesting produces dotted scenario names (`list.enumerate`, `list.require_non_empty`)
- Wiring a `TestContext` and running it through `run_cli_app`, which provides
  `--list-scenarios`, `--name`, and `--input` for free

**Key concepts:** `Scenario`, `ScenarioGroupImpl` (including nested groups), `TestContext`,
`run_cli_app`, structured tracing with a monotonic clock, JSON input parsing, CLI argument
handling, error propagation
