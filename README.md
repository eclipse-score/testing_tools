# `testing-utils`

Test framework tools and helpers for performance stack project.

## Overview

This repository provided utilities to assist with test automation, log handling, and result parsing.
It is designed to be a set of helper libraries for test frameworks or custom test runners.

## Features

- **Test scenarios libraries**: Rust and C++ libraries for implementing test scenarios.
- **Build tools**: Utilities for interacting with Bazel and Cargo.
- **Log container**: A container for storing and querying logs.
- **`ResultEntry`** and subclasses: Structured representation of test log entries.
- **Scenario**: Utilities for defining and running test scenarios.

## Installation

`virtualenv` usage is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install `testing-utils`:

```bash
pip install . --config-settings editable_mode=strict
```

Install `testing-utils` in editable mode with additional dev dependencies:

```bash
pip install -e .[dev] --config-settings editable_mode=strict
```

> `--config-settings editable_mode=strict` is required by Pylance plugin in VS Code.
> Package will work without it, but autocompletion won't work properly.

## Usage

### Test scenarios utilities

Libraries should be included as a dependency in `Cargo.toml` (Rust) or `BUILD` (C++).

Main components:

- `TestContext` - responsible for listing and running scenarios.
- `Scenario` and `ScenarioGroup` - base classes for defining scenarios and groups.
- `run_cli_app` - runs CLI application based on provided arguments and test context.

### Build tools

#### Get Cargo metadata

Cargo metadata is obtained using "cargo metadata" command.
CWD must be set to Cargo project.

```python
from typing import Any
from testing_utils import cargo_metadata

metadata: dict[str, Any] = cargo_metadata()
```

#### Find target path

Find path to executable based on provided target name.

```python
from pathlib import Path
from testing_utils import BazelTools

target_name = "target_name"
build_tools = BazelTools()
target_path: Path = build_tools.find_target_path(target_name)
...
```

#### Select executable

This feature is to ensure flexible usage in pytest context.
Additional configuration is required.

Expected flags depend on implementation, refer to `select_target_path` docs.
Add options to `conftest.py`:

```python
from pathlib import Path

def pytest_addoption(parser):
    parser.addoption(
        "--target-path",
        type=Path,
        help="Path to test scenarios executable. Search is performed by default.",
    )
    parser.addoption(
        "--target-name",
        type=str,
        default="rust_test_scenarios",
        help='Test scenario executable name. Overwritten by "--target-path".',
    )
```

Usage:

```python
import pytest
from pathlib import Path
from testing_utils import BazelTools

def test_example(request: pytest.FeatureRequest) -> None:
    build_tools = BazelTools()
    target_path: Path = build_tools.select_target_path(request.config)
    ...
```

#### Run build

Run build for selected target.

```python
from testing_utils import BazelTools

target_name = "target_name"
build_tools = BazelTools()
target_path: Path = build_tools.build(target_name)
...
```

### Log container and `ResultEntry` example

Usage as container:

```python
from testing_utils import LogContainer, ResultEntry

lc = LogContainer()
lc.add_log(
    ResultEntry({
        "timestamp": "2025-06-05T07:46:11.796134Z",
        "level": "DEBUG",
        "fields": {"message": "Debug message"},
        "target": "target::DEBUG_message",
        "threadId": "ThreadId(1)",
    })
)
logs = lc.get_logs()
# Output: debug message.
print(logs[0].message)
```

Usage as JSON log trace parser:

```python
from testing_utils import LogContainer, ResultEntry

# "messages" is a list of JSON logs.
logs = [ResultEntry(msg) for msg in messages]
lc = LogContainer(logs)
lc.contains_log(field="message", pattern="SomeExampleAction")  # True
```

Usage as log filter:

```python
from testing_utils import LogContainer, ResultEntry

# "messages" is a list of JSON logs.
logs = [ResultEntry(msg) for msg in messages]
lc = LogContainer(logs)
lc_only_info = lc.get_logs_by_field(field="level", pattern="INFO")
```

### Scenario example

`Scenario` is a base class containing basic test behavior.
Test execution results are provided using two fixtures:

- `results` - executable run results
- `logs` - logs from run

`build_tools`, `scenario_name` and `test_config` are marked as abstract and must be implemented.

Example implementation:

```python
import pytest
from testing_utils import Scenario, ScenarioResult, LogContainer, CargoTools, BuildTools

class TestExample(Scenario):
    @pytest.fixture(scope="class")
    def build_tools(self) -> BuildTools:
        return CargoTools()

    @pytest.fixture(scope="class")
    def scenario_name(self) -> str:
        return "example_scenario_name"

    @pytest.fixture(scope="class")
    def test_config(self) -> dict[str, Any]:
        return {"runtime": {"task_queue_size": 256, "workers": 1}}

    def test_example(self, results: ScenarioResult, logs: LogContainer) -> None:
        ...
```

Execution timeout uses `"--default-execution-timeout"` set in `conftest.py`, or is set to 5 seconds by default.

`stderr` is shown, but not captured by default.
To capture `stderr` use:

```python
class TestExample(Scenario):
    def capture_stderr(self) -> bool:
        return True

    ...
```

Methods can be overridden to utilize test-specific fixtures:

```python
import pytest
from testing_utils import Scenario

class TestExample(Scenario)
    @pytest.fixture(scope="class", params=[1, 4, 256])
    def queue_size(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture(scope="class")
    def test_config(self, queue_size: int) -> dict[str, Any]:
        return {"runtime": {"task_queue_size": queue_size, "workers": 1}}

    ...
```

## Development

- Python 3.12+ required.
- Code style is enforced with [ruff](https://github.com/astral-sh/ruff).

To run the tests, use:

```bash
pytest -vs .
```
