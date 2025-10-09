# *******************************************************************************
# Copyright (c) 2025 Contributors to the Eclipse Foundation
#
# See the NOTICE file(s) distributed with this work for additional
# information regarding copyright ownership.
#
# This program and the accompanying materials are made available under the
# terms of the Apache License Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
# *******************************************************************************
"""
Utilities for defining and running test scenarios.
"""

__all__ = ["ScenarioResult", "Scenario"]

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired
from typing import Any

import pytest

from .build_tools import BuildTools
from .log_container import LogContainer
from .result_entry import ResultEntry


@dataclass
class ScenarioResult:
    """
    Test scenario executable result.
    """

    stdout: str
    stderr: str | None
    return_code: int | None
    hang: bool

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        field_reprs: list[str] = []
        for name, value in self.__dict__.items():
            if isinstance(value, str):
                value = f"{value[:47]}..."
            field_reprs.append(f"{name}={repr(value)}")
        return f"{class_name}({', '.join(field_reprs)})"


class Scenario(ABC):
    """
    Base test scenario definition.
    """

    @pytest.fixture(scope="class")
    @abstractmethod
    def build_tools(self, *args, **kwargs) -> BuildTools:
        """
        Build tools used to handle test scenario.
        """

    @pytest.fixture(scope="class")
    @abstractmethod
    def scenario_name(self, *args, **kwargs) -> str:
        """
        Name of a test scenario to run.
        """

    @pytest.fixture(scope="class")
    @abstractmethod
    def test_config(self, *args, **kwargs) -> dict[str, Any]:
        """
        Test configuration.
        """

    @pytest.fixture(scope="class")
    def execution_timeout(self, request: pytest.FixtureRequest, *args, **kwargs) -> float:
        """
        Test execution timeout in seconds.

        Parameters
        ----------
        request : pytest.FixtureRequest
            Test request built-in fixture.
        """
        timeout = request.config.getoption("--default-execution-timeout")
        if isinstance(timeout, float):
            return timeout
        return 5.0

    def capture_stderr(self, *args, **kwargs) -> bool:
        """
        Capture or display stderr during execution.
        """
        return False

    @pytest.fixture(scope="class")
    def target_path(self, build_tools: BuildTools, request: pytest.FixtureRequest) -> Path:
        """
        Return path to test scenario executable.

        Parameters
        ----------
        request : pytest.FixtureRequest
            Test request built-in fixture.
        """
        return build_tools.select_target_path(request.config, expect_exists=True)

    @pytest.fixture(scope="class")
    def command(self, target_path: Path | str, scenario_name: str, test_config: dict[str, Any]) -> list[str]:
        """
        Command to invoke.

        Parameters
        ----------
        target_path : Path | str
            Path to test scenarios executable.
        scenario_name : str
            Name of a test scenario to run.
        test_config : dict[str, Any]
            Test configuration.
        """
        # Dump test configuration to string.
        test_config_str = json.dumps(test_config)
        return [str(target_path), "--name", scenario_name, "--input", test_config_str]

    def _run_command(self, command: list[str], execution_timeout: float, *args, **kwargs) -> ScenarioResult:
        """
        Execute test scenario executable.

        Parameters
        ----------
        command : list[str]
            Command to invoke.
        execution_timeout : float
            Test execution timeout in seconds.
        """
        hang = False
        stderr_param = PIPE if self.capture_stderr() else None
        with Popen(command, stdout=PIPE, stderr=stderr_param, text=True) as p:
            try:
                stdout, stderr = p.communicate(timeout=execution_timeout)
            except TimeoutExpired:
                hang = True
                p.kill()
                stdout, stderr = p.communicate()

        return ScenarioResult(stdout, stderr, p.returncode, hang)

    @pytest.fixture(scope="class")
    def results(
        self,
        command: list[str],
        execution_timeout: float,
        *args,
        **kwargs,
    ) -> ScenarioResult:
        """
        Execute test scenario executable and return results.

        Parameters
        ----------
        command : list[str]
            Command to invoke.
        execution_timeout : float
            Test execution timeout in seconds.
        """
        return self._run_command(command, execution_timeout, args, kwargs)

    @pytest.fixture(scope="class")
    def logs(self, results: ScenarioResult, *args, **kwargs) -> LogContainer:
        """
        Execute test scenario executable and return logs.

        Parameters
        ----------
        results : ScenarioResult
            Scenario results fixture.
        """
        # Split into lines.
        text_lines = results.stdout.strip().split("\n")

        # Filter out non-JSON messages.
        messages = filter(lambda m: m.startswith("{") and m.endswith("}"), text_lines)
        messages = list(map(json.loads, messages))

        # Convert timestamp from microseconds to timedelta.
        for msg in messages:
            msg["timestamp"] = timedelta(microseconds=int(msg["timestamp"]))

        # Sort messages into chronological order.
        messages.sort(key=lambda m: m["timestamp"])

        # Convert messages to list of ResultEntry and create log container.
        result_entries = [ResultEntry(msg) for msg in messages]
        return LogContainer(result_entries)
